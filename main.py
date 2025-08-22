"""FastAPI application for AI trends analysis."""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import pandas as pd
import io

from ai_trends.core.analyzer import AITrendsAnalyzer
from ai_trends.models.schemas import (
    AnalysisRequest, AnalysisResult, AISkillRecord, PredictionRecord,
    ComprehensiveReport
)
from ai_trends.utils.cache import CacheManager
from ai_trends.utils.exceptions import AITrendsException
from ai_trends.utils.logging import setup_logging
from config.settings import settings

# Initialize logging
setup_logging()

# Initialize cache manager
cache_manager = CacheManager()

# Initialize analyzer
analyzer = AITrendsAnalyzer()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request tracking
active_requests: Dict[str, dict] = {}


# Dependency for request tracking
async def track_request():
    """Track active requests for monitoring."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    active_requests[request_id] = {
        "start_time": start_time,
        "status": "processing"
    }
    
    try:
        yield request_id
    finally:
        if request_id in active_requests:
            del active_requests[request_id]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Trends Analyzer Pro API",
        "version": settings.api_version,
        "environment": settings.environment.value,
        "docs_url": "/docs" if settings.enable_docs else "disabled",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if analyzer is working
        test_result = analyzer.get_system_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.api_version,
            "environment": settings.environment.value,
            "active_requests": len(active_requests),
            "system_status": test_result
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_trends(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(track_request)
) -> AnalysisResult:
    """Generate AI trends analysis."""
    try:
        logger.info(f"Starting analysis request {request_id}")
        start_time = time.time()
        
        # Check cache first
        cache_key = f"analysis_{hash(str(request.dict()))}"
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached result for request {request_id}")
            return AnalysisResult.parse_obj(cached_result)
        
        # Validate request
        if request.end_year <= request.start_year:
            raise HTTPException(
                status_code=400, 
                detail="End year must be after start year"
            )
        
        # Perform analysis
        result = await analyzer.analyze_trends(
            sample_size=request.sample_size,
            start_year=request.start_year,
            end_year=request.end_year,
            prediction_years=request.prediction_years,
            states=request.states,
            skills=request.skills,
            include_visualizations=request.include_visualizations
        )
        
        # Add metadata
        execution_time = time.time() - start_time
        result.request_id = request_id
        result.timestamp = datetime.utcnow()
        result.execution_time_seconds = execution_time
        
        # Cache result in background
        background_tasks.add_task(
            cache_manager.set, 
            cache_key, 
            result.dict(), 
            ttl=settings.cache_ttl
        )
        
        logger.info(f"Analysis completed for request {request_id} in {execution_time:.2f}s")
        return result
        
    except AITrendsException as e:
        logger.error(f"Analysis failed for request {request_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/historical-data", response_model=List[AISkillRecord])
async def get_historical_data(
    sample_size: int = 1000,
    start_year: int = 2019,
    end_year: int = 2024,
    states: Optional[str] = None,
    skills: Optional[str] = None,
    request_id: str = Depends(track_request)
) -> List[AISkillRecord]:
    """Get historical AI trends data."""
    try:
        logger.info(f"Generating historical data for request {request_id}")
        
        # Parse comma-separated lists
        states_list = states.split(",") if states else None
        skills_list = skills.split(",") if skills else None
        
        # Generate data
        historical_data = await analyzer.generate_historical_data(
            sample_size=sample_size,
            start_year=start_year,
            end_year=end_year,
            states=states_list,
            skills=skills_list
        )
        
        logger.info(f"Generated {len(historical_data)} historical records")
        return historical_data
        
    except AITrendsException as e:
        logger.error(f"Historical data generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/predictions", response_model=List[PredictionRecord])
async def get_predictions(
    sample_size: int = 1000,
    start_year: int = 2019,
    end_year: int = 2024,
    prediction_years: int = 6,
    request_id: str = Depends(track_request)
) -> List[PredictionRecord]:
    """Get future trend predictions."""
    try:
        logger.info(f"Generating predictions for request {request_id}")
        
        # Generate historical data first
        historical_data = await analyzer.generate_historical_data(
            sample_size=sample_size,
            start_year=start_year,
            end_year=end_year
        )
        
        # Generate predictions
        predictions = await analyzer.generate_predictions(
            historical_data=historical_data,
            prediction_years=prediction_years
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
        
    except AITrendsException as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/report", response_model=ComprehensiveReport)
async def get_comprehensive_report(
    sample_size: int = 1000,
    start_year: int = 2019,
    end_year: int = 2024,
    prediction_years: int = 6,
    request_id: str = Depends(track_request)
) -> ComprehensiveReport:
    """Get comprehensive analysis report."""
    try:
        logger.info(f"Generating comprehensive report for request {request_id}")
        
        report = await analyzer.generate_comprehensive_report(
            sample_size=sample_size,
            start_year=start_year,
            end_year=end_year,
            prediction_years=prediction_years
        )
        
        logger.info(f"Generated comprehensive report with {report.total_records} records")
        return report
        
    except AITrendsException as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
async def get_metrics():
    """Get API metrics and system status."""
    try:
        system_status = analyzer.get_system_status()
        
        return {
            "active_requests": len(active_requests),
            "system_status": system_status,
            "cache_stats": await cache_manager.get_stats(),
            "uptime_seconds": time.time() - start_time,
            "environment": settings.environment.value,
            "version": settings.api_version
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.get("/states")
async def get_supported_states():
    """Get list of supported Indian states."""
    return {
        "states": settings.indian_states,
        "count": len(settings.indian_states)
    }


@app.get("/skills")
async def get_supported_skills():
    """Get list of supported AI skills."""
    return {
        "skills": settings.ai_skills,
        "count": len(settings.ai_skills),
        "categories": {
            "traditional_ai": [skill for skill in settings.ai_skills if any(term in skill for term in ["Machine Learning", "Deep Learning", "Computer Vision", "NLP"])],
            "agent_technologies": [skill for skill in settings.ai_skills if any(term in skill for term in ["Agent", "RAG", "Agentic", "Multi-Agent"])],
            "tools_frameworks": [skill for skill in settings.ai_skills if any(term in skill for term in ["TensorFlow", "PyTorch", "LangChain", "CrewAI"])]
        }
    }


@app.get("/job-roles")
async def get_supported_job_roles():
    """Get list of supported job roles."""
    return {
        "job_roles": settings.job_roles,
        "count": len(settings.job_roles)
    }


@app.post("/upload-csv")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload and analyze CSV file from external sources like Kaggle."""
    request_id = str(uuid.uuid4())
    logger.info(f"Processing CSV upload for request {request_id}")
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Map CSV columns to our schema (flexible mapping)
        mapped_data, found_columns = await _map_csv_to_schema(df)
        
        if not mapped_data:
            raise HTTPException(
                status_code=400, 
                detail="Unable to map CSV columns to AI trends schema. Please ensure your CSV contains relevant AI/tech job data."
            )
        
        # For large datasets, limit to a reasonable sample for analysis
        sample_size = min(len(mapped_data), 5000)  # Limit to 5000 records for analysis
        if len(mapped_data) > sample_size:
            logger.info(f"Large dataset detected. Sampling {sample_size} records from {len(mapped_data)} for analysis")
            # Take a random sample
            import random
            mapped_data_sample = random.sample(mapped_data, sample_size)
        else:
            mapped_data_sample = mapped_data
        
        # Convert to our format and analyze
        logger.info(f"Starting analysis of {len(mapped_data_sample)} records")
        analysis_result = await analyzer.analyze_trends_from_data(mapped_data_sample)
        logger.info("Analysis completed successfully")
        
        return {
            "request_id": request_id,
            "file_info": {
                "filename": file.filename,
                "original_rows": len(df),
                "processed_rows": len(mapped_data),
                "sampled_rows": len(mapped_data_sample),
                "columns": list(df.columns),
                "found_mappings": found_columns
            },
            "analysis": analysis_result.dict() if hasattr(analysis_result, 'dict') else analysis_result,
            "message": f"Successfully processed {len(mapped_data)} records (analyzed {len(mapped_data_sample)}) from uploaded CSV"
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        logger.error(f"CSV upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")


async def _map_csv_to_schema(df: pd.DataFrame) -> List[AISkillRecord]:
    """Map CSV columns to our AISkillRecord schema with flexible column mapping."""
    try:
        mapped_records = []
        
        # Enhanced column name mappings for various dataset formats
        column_mappings = {
            'skill': ['skill', 'skills', 'skill_name', 'technology', 'tech_skill', 'ai_skill', 'required_skills'],
            'salary': ['salary', 'avg_salary', 'salary_inr', 'salary_usd', 'compensation', 'pay'],
            'location': ['location', 'state', 'city', 'region', 'indian_state', 'company_location', 'employee_residence'],
            'experience': ['experience', 'exp_level', 'experience_level', 'seniority', 'years_experience'],
            'company_size': ['company_size', 'company_type', 'org_size'],
            'role': ['role', 'job_role', 'job_title', 'position'],
            'year': ['year', 'date', 'timestamp', 'posting_date'],
            'industry': ['industry', 'sector', 'domain'],
            'remote_work': ['remote', 'remote_work', 'work_type', 'remote_percentage', 'remote_ratio'],
            'employment_type': ['employment_type', 'job_type', 'work_type'],
            'company_name': ['company_name', 'company', 'employer'],
            'benefits': ['benefits_score', 'benefits', 'perks']
        }
        
        # Find matching columns
        found_columns = {}
        for field, possible_names in column_mappings.items():
            for col in df.columns:
                if any(name.lower() in col.lower() for name in possible_names):
                    found_columns[field] = col
                    break
        
        # If we don't have basic required fields, return empty
        if 'skill' not in found_columns and 'role' not in found_columns:
            logger.warning("No skill or role columns found in CSV")
            return []
        
        # Process each row with progress logging
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows from CSV")
        
        for idx, row in df.iterrows():
            try:
                # Log progress every 1000 rows
                if idx % 1000 == 0:
                    logger.info(f"Processing row {idx}/{total_rows} ({(idx/total_rows)*100:.1f}%)")
                # Extract and clean skill information
                skill = str(row.get(found_columns.get('skill', ''), '')).strip()
                if not skill or skill.lower() in ['nan', 'none', '', 'null']:
                    # Fallback to job title if no specific skill column
                    job_title = str(row.get(found_columns.get('role', ''), 'AI Specialist')).strip()
                    # Extract AI-related keywords from job title
                    ai_keywords = ['AI', 'ML', 'Machine Learning', 'Deep Learning', 'Data Science', 'Python', 'TensorFlow', 'PyTorch']
                    skill = next((kw for kw in ai_keywords if kw.lower() in job_title.lower()), 'General AI')
                
                # Extract year from posting_date or years_experience
                posting_year = 2024  # default
                if 'year' in found_columns:
                    try:
                        year_val = row.get(found_columns['year'])
                        if pd.isna(year_val):
                            posting_year = 2024
                        else:
                            year_str = str(year_val)
                            if '-' in year_str:
                                posting_year = int(year_str.split('-')[0])
                            else:
                                year_int = int(float(year_str))
                                # If year is less than 100, it's probably years of experience, not actual year
                                if year_int < 100:
                                    # Convert experience years to realistic posting year
                                    posting_year = max(2020, min(2024, 2024 - year_int))
                                else:
                                    posting_year = year_int
                    except:
                        posting_year = 2024
                
                # Ensure year is within valid range
                posting_year = max(2020, min(2024, posting_year))
                
                # Map experience levels
                exp_level = str(row.get(found_columns.get('experience', ''), 'Mid')).strip()
                experience_mapping = {
                    'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Lead',
                    'Entry': 'Entry', 'Mid': 'Mid', 'Senior': 'Senior', 'Lead': 'Lead',
                    '0': 'Entry', '1': 'Entry', '2': 'Mid', '3': 'Mid', '4': 'Senior', '5': 'Senior'
                }
                mapped_experience = experience_mapping.get(exp_level, 'Mid')
                
                # Map company sizes
                comp_size = str(row.get(found_columns.get('company_size', ''), 'M')).strip()
                size_mapping = {'S': 'Startup', 'M': 'Medium', 'L': 'Large', 'Small': 'Startup', 'Medium': 'Medium', 'Large': 'Large'}
                mapped_size = size_mapping.get(comp_size, 'Medium')
                
                # Extract location and map to Indian states (with international fallback)
                location = str(row.get(found_columns.get('location', ''), 'Karnataka')).strip()
                indian_states = ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Telangana', 'Delhi', 'West Bengal', 'Gujarat', 'Rajasthan', 'Uttar Pradesh', 'Kerala']
                # If location contains an Indian state, use it; otherwise default to Karnataka
                mapped_state = next((state for state in indian_states if state.lower() in location.lower()), 'Karnataka')
                
                # Process salary
                salary = 50000  # default
                if 'salary' in found_columns:
                    try:
                        salary_val = str(row.get(found_columns['salary'], '50000')).replace(',', '').replace('$', '')
                        salary = max(10000, float(salary_val))
                    except:
                        salary = 50000
                
                # Process remote work ratio
                remote_ratio = 50  # default
                if 'remote_work' in found_columns:
                    try:
                        remote_val = str(row.get(found_columns['remote_work'], '50')).replace('%', '')
                        remote_ratio = min(100, max(0, float(remote_val)))
                    except:
                        remote_ratio = 50
                
                # Map industry to our allowed values
                industry_raw = str(row.get(found_columns.get('industry', ''), 'IT Services')).strip()
                industry_mapping = {
                    'Technology': 'IT Services', 'Finance': 'FinTech', 'Financial': 'FinTech', 
                    'Education': 'EdTech', 'Healthcare': 'Healthcare', 'Health': 'Healthcare',
                    'E-commerce': 'E-commerce', 'Ecommerce': 'E-commerce', 'Retail': 'E-commerce',
                    'Government': 'Government', 'Research': 'Research', 'Consulting': 'IT Services',
                    'Media': 'IT Services', 'Gaming': 'IT Services', 'Automotive': 'IT Services',
                    'Manufacturing': 'IT Services', 'Energy': 'IT Services', 'Real Estate': 'IT Services',
                    'Transportation': 'IT Services', 'Telecommunications': 'IT Services'
                }
                mapped_industry = industry_mapping.get(industry_raw, 'IT Services')
                
                # Calculate quarter properly
                quarter_num = ((posting_year - 2020) % 4) + 1
                
                # Default values for required fields
                record_data = {
                    'date': datetime.now(),
                    'year': posting_year,
                    'quarter': f'Q{quarter_num}',
                    'skill_category': skill,
                    'job_role': str(row.get(found_columns.get('role', ''), skill)).strip(),
                    'indian_state': mapped_state,
                    'job_postings': max(1, int(idx / 10) + 1),  # Estimate based on data size
                    'avg_salary_usd': salary,
                    'avg_salary_inr': salary * 83,  # Convert to INR
                    'skill_demand_score': min(100, max(10, 50 + (len(skill) * 2))),  # Score based on skill complexity
                    'training_programs': max(1, len(skill.split()) * 2),  # Programs based on skill complexity
                    'certification_completion_rate': min(100, max(30, 70 + (remote_ratio / 10))),  # Higher for remote roles
                    'remote_work_percentage': remote_ratio,
                    'experience_level': mapped_experience,
                    'company_size': mapped_size,
                    'industry': mapped_industry,
                    'ai_adoption_maturity': 'Intermediate',
                    'focus_area': 'Implementation'
                }
                
                # Convert USD to INR if needed
                if record_data['avg_salary_usd'] > 0:
                    record_data['avg_salary_inr'] = record_data['avg_salary_usd'] * 83
                
                # Create record
                mapped_records.append(AISkillRecord(**record_data))
                
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {str(e)}")
                continue
        
        logger.info(f"Successfully mapped {len(mapped_records)} records from CSV")
        return mapped_records, found_columns
        
    except Exception as e:
        logger.error(f"CSV mapping failed: {str(e)}")
        return [], {}


# Error handlers
@app.exception_handler(AITrendsException)
async def ai_trends_exception_handler(request, exc: AITrendsException):
    """Handle AI trends specific exceptions."""
    logger.error(f"AI Trends error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": type(exc).__name__}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "InternalServerError"}
    )


# Startup/shutdown events
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting AI Trends Analyzer Pro API")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize cache if configured
    if settings.redis_url:
        await cache_manager.initialize(settings.redis_url)
        logger.info("Cache manager initialized")
    
    # Initialize analyzer
    await analyzer.initialize()
    logger.info("AI Trends Analyzer initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Trends Analyzer Pro API")
    
    # Cleanup cache
    await cache_manager.close()
    
    # Cleanup analyzer
    await analyzer.cleanup()
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ai_trends.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.value.lower(),
        reload=settings.debug
    )