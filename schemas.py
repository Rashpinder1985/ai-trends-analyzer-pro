"""Data models and schemas for AI trends analysis."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ExperienceLevel(str, Enum):
    """Experience levels."""
    
    ENTRY = "Entry"
    MID = "Mid"
    SENIOR = "Senior"
    LEAD = "Lead"


class CompanySize(str, Enum):
    """Company sizes."""
    
    STARTUP = "Startup"
    MEDIUM = "Medium"
    LARGE = "Large"
    ENTERPRISE = "Enterprise"


class Industry(str, Enum):
    """Industries."""
    
    EDTECH = "EdTech"
    FINTECH = "FinTech"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    IT_SERVICES = "IT Services"
    GOVERNMENT = "Government"
    RESEARCH = "Research"


class AIAdoptionMaturity(str, Enum):
    """AI adoption maturity levels."""
    
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class FocusArea(str, Enum):
    """Focus areas."""
    
    RESEARCH = "Research"
    PRODUCT_DEVELOPMENT = "Product Development"
    CONSULTING = "Consulting"
    EDUCATION = "Education"
    IMPLEMENTATION = "Implementation"


class AISkillRecord(BaseModel):
    """Individual AI skill record."""
    
    id: Optional[str] = None
    date: datetime
    year: int
    quarter: str
    skill_category: str = Field(..., description="AI skill category")
    job_role: str = Field(..., description="Job role")
    indian_state: str = Field(..., description="Indian state")
    job_postings: int = Field(..., ge=0, description="Number of job postings")
    avg_salary_usd: float = Field(..., ge=0, description="Average salary in USD")
    avg_salary_inr: float = Field(..., ge=0, description="Average salary in INR")
    skill_demand_score: float = Field(..., ge=0, le=100, description="Skill demand score")
    training_programs: int = Field(..., ge=0, description="Number of training programs")
    certification_completion_rate: float = Field(..., ge=0, le=100, description="Certification completion rate")
    remote_work_percentage: float = Field(..., ge=0, le=100, description="Remote work percentage")
    experience_level: ExperienceLevel
    company_size: CompanySize
    industry: Industry
    ai_adoption_maturity: AIAdoptionMaturity
    focus_area: FocusArea

    @validator("quarter")
    def validate_quarter(cls, v: str) -> str:
        """Validate quarter format."""
        if not v.startswith("Q") or v[1:] not in ["1", "2", "3", "4"]:
            raise ValueError("Quarter must be in format Q1, Q2, Q3, or Q4")
        return v

    @validator("year")
    def validate_year(cls, v: int) -> int:
        """Validate year range."""
        current_year = datetime.now().year
        if v < 2015 or v > current_year + 10:
            raise ValueError(f"Year must be between 2015 and {current_year + 10}")
        return v


class PredictionRecord(BaseModel):
    """Prediction record for future trends."""
    
    year: int
    avg_salary_inr: float = Field(..., ge=0)
    avg_salary_usd: float = Field(..., ge=0)
    job_postings: float = Field(..., ge=0)
    skill_demand_score: float = Field(..., ge=0, le=100)
    training_programs: float = Field(..., ge=0)
    certification_completion_rate: float = Field(..., ge=0, le=100)
    remote_work_percentage: float = Field(..., ge=0, le=100)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    
    sample_size: int = Field(default=1000, ge=100, le=10000, description="Sample size")
    start_year: int = Field(default=2019, ge=2015, description="Start year")
    end_year: int = Field(default=2024, description="End year")
    prediction_years: int = Field(default=6, ge=1, le=10, description="Years to predict")
    states: Optional[List[str]] = Field(default=None, description="States to include")
    skills: Optional[List[str]] = Field(default=None, description="Skills to include")
    include_visualizations: bool = Field(default=True, description="Include visualizations")

    @validator("end_year")
    def validate_end_year(cls, v: int, values: dict) -> int:
        """Validate end year is after start year."""
        if "start_year" in values and v <= values["start_year"]:
            raise ValueError("End year must be after start year")
        return v


class AnalysisResult(BaseModel):
    """Result model for analysis."""
    
    request_id: str
    timestamp: datetime
    historical_data: List[AISkillRecord]
    predictions: List[PredictionRecord]
    summary_statistics: dict
    visualizations: Optional[dict] = None
    execution_time_seconds: float


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    
    metric: str
    historical_growth_rate: float
    cagr: float  # Compound Annual Growth Rate
    trend_direction: str  # "increasing", "decreasing", "stable"
    confidence_score: float = Field(..., ge=0, le=1)


class StateAnalysis(BaseModel):
    """State-wise analysis results."""
    
    state: str
    total_job_postings: int
    avg_salary_inr: float
    market_share_percentage: float
    growth_rate: float
    top_skills: List[str]


class SkillAnalysis(BaseModel):
    """Skill-wise analysis results."""
    
    skill: str
    demand_score: float
    avg_salary_premium: float  # Premium over average
    growth_rate: float
    adoption_trend: str
    market_maturity: str


class ComprehensiveReport(BaseModel):
    """Comprehensive analysis report."""
    
    report_id: str
    generated_at: datetime
    period: str
    total_records: int
    
    # Overview
    key_insights: List[str]
    market_overview: dict
    
    # Detailed Analysis
    trend_analysis: List[TrendAnalysis]
    state_analysis: List[StateAnalysis]
    skill_analysis: List[SkillAnalysis]
    
    # Predictions
    future_outlook: dict
    recommendations: List[str]
    
    # Metadata
    data_quality_score: float = Field(..., ge=0, le=1)
    confidence_level: float = Field(..., ge=0, le=1)