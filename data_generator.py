"""Enhanced data generator with proper validation and error handling."""

import random
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ai_trends.models.schemas import AISkillRecord, ExperienceLevel, CompanySize, Industry, AIAdoptionMaturity, FocusArea
from ai_trends.utils.exceptions import DataGenerationError, ValidationError
from config.settings import settings


class DataGenerator:
    """Production-ready data generator for AI trends analysis."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize data generator with optional random seed."""
        self.random_seed = random_seed or settings.random_seed
        self._setup_random_state()
        self._initialize_constants()
        logger.info(f"DataGenerator initialized with seed: {self.random_seed}")

    def _setup_random_state(self) -> None:
        """Set up random state for reproducibility."""
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def _initialize_constants(self) -> None:
        """Initialize constants for data generation."""
        # Skill multipliers for salary calculation
        self.skill_multipliers = {
            "Machine Learning": 1.2, "Deep Learning": 1.4, "MLOps": 1.3,
            "AI Ethics": 0.9, "Computer Vision": 1.1, "Natural Language Processing": 1.15,
            "AI Agents": 1.5, "RAG": 1.6, "Agentic AI": 1.7, "LangChain": 1.4,
            "CrewAI": 1.3, "Large Language Models": 1.5, "Generative AI": 1.6,
            "Prompt Engineering": 1.2, "Vector Databases": 1.3, "Multi-Agent Systems": 1.6
        }

        # Role multipliers
        self.role_multipliers = {
            "AI Research Scientist": 1.5, "Machine Learning Engineer": 1.3,
            "AI Architect": 1.4, "Data Scientist": 1.1, "AI Agent Developer": 1.6,
            "RAG Engineer": 1.5, "Agentic AI Specialist": 1.7, "LLM Engineer": 1.5,
            "Prompt Engineer": 1.2, "Multi-Agent Systems Developer": 1.6
        }

        # State multipliers (tech hub premiums with market variations)
        self.state_multipliers = {
            "Karnataka": 1.3, "Maharashtra": 1.25, "Tamil Nadu": 1.2,
            "Telangana": 1.25, "Delhi": 1.3, "West Bengal": 1.0,
            "Gujarat": 1.1, "Rajasthan": 1.0, "Uttar Pradesh": 1.1, "Kerala": 1.05
        }
        
        # Skill market weights (base popularity/demand distribution)
        self.skill_base_weights = {
            # Foundational skills (high weight - established market)
            "Python": 1.0, "Machine Learning": 0.9, "SQL": 0.8, "TensorFlow": 0.7,
            "PyTorch": 0.6, "Scikit-learn": 0.5, "Deep Learning": 0.7,
            "Natural Language Processing": 0.4, "Computer Vision": 0.4,
            
            # Emerging AI (year-dependent explosive growth)
            "AI Agents": 0.05, "Multi-Agent Systems": 0.03, "RAG": 0.04, "Agentic AI": 0.02,
            "LangChain": 0.03, "CrewAI": 0.02, "AutoGen": 0.01, "Agent Frameworks": 0.02,
            "Vector Databases": 0.03, "Embedding Models": 0.02, "Prompt Engineering": 0.08,
            "Large Language Models": 0.06, "Generative AI": 0.1, "ChatGPT Integration": 0.04,
            
            # Cloud & Tools (corporate adoption)
            "AWS AI": 0.3, "Azure AI": 0.25, "Google AI": 0.2, "Spark": 0.2,
            "R": 0.15, "Keras": 0.3, "OpenCV": 0.25,
            
            # Specialized (niche but important)
            "Reinforcement Learning": 0.1, "MLOps": 0.2, "AI Ethics": 0.05
        }
        
        # State-wise market maturity and growth patterns
        self.state_market_patterns = {
            "Karnataka": {"maturity": 0.9, "growth_volatility": 0.2, "startup_density": 0.8},
            "Maharashtra": {"maturity": 0.85, "growth_volatility": 0.15, "startup_density": 0.7},
            "Tamil Nadu": {"maturity": 0.7, "growth_volatility": 0.25, "startup_density": 0.6},
            "Telangana": {"maturity": 0.75, "growth_volatility": 0.2, "startup_density": 0.65},
            "Delhi": {"maturity": 0.8, "growth_volatility": 0.18, "startup_density": 0.75},
            "West Bengal": {"maturity": 0.5, "growth_volatility": 0.3, "startup_density": 0.4},
            "Gujarat": {"maturity": 0.6, "growth_volatility": 0.25, "startup_density": 0.5},
            "Rajasthan": {"maturity": 0.45, "growth_volatility": 0.35, "startup_density": 0.3},
            "Uttar Pradesh": {"maturity": 0.55, "growth_volatility": 0.3, "startup_density": 0.45},
            "Kerala": {"maturity": 0.65, "growth_volatility": 0.22, "startup_density": 0.55}
        }

        # Experience level multipliers
        self.experience_multipliers = {
            ExperienceLevel.ENTRY: 0.7,
            ExperienceLevel.MID: 1.0,
            ExperienceLevel.SENIOR: 1.4,
            ExperienceLevel.LEAD: 1.8
        }

        # Company size multipliers
        self.company_size_multipliers = {
            CompanySize.STARTUP: 0.9,
            CompanySize.MEDIUM: 1.0,
            CompanySize.LARGE: 1.2,
            CompanySize.ENTERPRISE: 1.4
        }

    def generate_historical_data(
        self,
        sample_size: int = None,
        start_year: int = None,
        end_year: int = None,
        states: Optional[List[str]] = None,
        skills: Optional[List[str]] = None
    ) -> List[AISkillRecord]:
        """Generate historical AI trends data with validation."""
        try:
            sample_size = sample_size or settings.default_sample_size
            start_year = start_year or settings.start_year
            end_year = end_year or settings.end_year
            states = states or settings.indian_states
            skills = skills or settings.ai_skills

            self._validate_inputs(sample_size, start_year, end_year, states, skills)

            logger.info(f"Generating {sample_size} historical records from {start_year} to {end_year}")

            records = []
            for i in range(sample_size):
                try:
                    record = self._generate_single_record(
                        start_year, end_year, states, skills, settings.job_roles
                    )
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to generate record {i}: {str(e)}")
                    continue

            if len(records) < sample_size * 0.9:  # Less than 90% success rate
                raise DataGenerationError(
                    f"Only generated {len(records)} out of {sample_size} records"
                )

            logger.info(f"Successfully generated {len(records)} historical records")
            return records

        except Exception as e:
            logger.error(f"Data generation failed: {str(e)}")
            raise DataGenerationError(f"Failed to generate historical data: {str(e)}")

    def _validate_inputs(
        self,
        sample_size: int,
        start_year: int,
        end_year: int,
        states: List[str],
        skills: List[str]
    ) -> None:
        """Validate input parameters."""
        if sample_size <= 0:
            raise ValidationError("Sample size must be positive")
        
        if start_year >= end_year:
            raise ValidationError("Start year must be before end year")
        
        current_year = datetime.now().year
        if end_year > current_year:
            raise ValidationError(f"End year cannot be in the future (current: {current_year})")
        
        if not states:
            raise ValidationError("States list cannot be empty")
        
        if not skills:
            raise ValidationError("Skills list cannot be empty")

    def _select_weighted_skill(self, skills: List[str], year_factor: float) -> str:
        """Select skill based on weighted probabilities and year-dependent growth."""
        weights = []
        
        for skill in skills:
            base_weight = self.skill_base_weights.get(skill, 0.1)  # Default small weight
            
            # Apply year-dependent growth for emerging technologies
            if any(tech in skill for tech in ['Agent', 'RAG', 'Agentic', 'LLM', 'ChatGPT', 'Vector']):
                # Exponential growth starting around 2022 (year_factor ~0.6)
                if year_factor > 0.5:  # After 2021
                    growth_multiplier = 1 + (year_factor - 0.5) ** 2 * 20  # Explosive growth
                else:
                    growth_multiplier = 0.1  # Very low before 2022
                final_weight = base_weight * growth_multiplier
            elif any(tech in skill for tech in ['Machine Learning', 'Python', 'SQL']):
                # Established tech: steady high demand with slight saturation
                saturation_factor = 1.2 - (year_factor * 0.3)  # Slight decline as market matures
                final_weight = base_weight * saturation_factor
            elif any(tech in skill for tech in ['Deep Learning', 'TensorFlow', 'PyTorch']):
                # Mature tech: peaked earlier, now stable
                maturity_curve = 1 + np.sin(year_factor * np.pi) * 0.3  # Peak in middle, stable later
                final_weight = base_weight * maturity_curve
            elif any(tech in skill for tech in ['AWS', 'Azure', 'Google']):
                # Cloud tech: corporate adoption cycles
                corporate_adoption = 0.5 + 0.8 * (1 - np.exp(-year_factor * 4))  # S-curve adoption
                final_weight = base_weight * corporate_adoption
            else:
                # Other skills: moderate growth
                final_weight = base_weight * (1 + year_factor * 0.5)
            
            # Add some randomness to avoid perfect predictability
            final_weight *= np.random.uniform(0.8, 1.2)
            weights.append(max(final_weight, 0.001))  # Ensure positive weight
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select skill based on weighted probabilities
        return np.random.choice(skills, p=probabilities)

    def _select_weighted_state(self, states: List[str], year_factor: float) -> str:
        """Select state based on weighted probabilities reflecting tech hub development."""
        weights = []
        
        for state in states:
            if state in self.state_market_patterns:
                pattern = self.state_market_patterns[state]
                # Tech hubs grow faster over time
                base_weight = pattern["maturity"] * pattern["startup_density"]
                # Add growth over time for developing markets
                growth_factor = 1 + (year_factor * (1 - pattern["maturity"]) * 2)
                final_weight = base_weight * growth_factor
            else:
                final_weight = 0.1  # Default low weight for unlisted states
            
            # Add some randomness
            final_weight *= np.random.uniform(0.9, 1.1)
            weights.append(max(final_weight, 0.01))
        
        # Normalize to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        return np.random.choice(states, p=probabilities)

    def _generate_single_record(
        self,
        start_year: int,
        end_year: int,
        states: List[str],
        skills: List[str],
        job_roles: List[str]
    ) -> AISkillRecord:
        """Generate a single AI skill record."""
        # Generate random date
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        # Calculate year progression factor
        year_factor = (random_date.year - start_year) / (end_year - start_year)

        # Select skill and state using weighted probabilities
        skill = self._select_weighted_skill(skills, year_factor)
        role = random.choice(job_roles)
        state = self._select_weighted_state(states, year_factor)
        experience_level = random.choice(list(ExperienceLevel))
        company_size = random.choice(list(CompanySize))
        industry = random.choice(list(Industry))
        ai_maturity = random.choice(list(AIAdoptionMaturity))
        focus_area = random.choice(list(FocusArea))

        # Calculate salary with realistic growth and variations
        base_salary_inr = self._calculate_base_salary(year_factor)
        salary_multiplier = self._calculate_salary_multiplier(
            skill, role, state, experience_level, company_size
        )
        
        # Add some randomness
        salary_variation = np.random.normal(0, base_salary_inr * 0.15)
        final_salary_inr = max(base_salary_inr * salary_multiplier + salary_variation, 400000)
        final_salary_usd = final_salary_inr / 83  # Approximate conversion

        # Calculate other metrics with enhanced state and skill dependencies
        job_postings = self._calculate_job_postings(year_factor, random_date.month, state)
        skill_demand_score = self._calculate_skill_demand(year_factor, skill)
        training_programs = self._calculate_training_programs(year_factor, state, skill)
        cert_completion_rate = self._calculate_certification_rate(year_factor, skill, state)
        remote_work_percentage = self._calculate_remote_work(year_factor, random_date.year, company_size, industry)

        return AISkillRecord(
            date=random_date,
            year=random_date.year,
            quarter=f"Q{((random_date.month - 1) // 3) + 1}",
            skill_category=skill,
            job_role=role,
            indian_state=state,
            job_postings=job_postings,
            avg_salary_usd=round(final_salary_usd, 2),
            avg_salary_inr=round(final_salary_inr, 2),
            skill_demand_score=round(skill_demand_score, 1),
            training_programs=training_programs,
            certification_completion_rate=round(cert_completion_rate, 1),
            remote_work_percentage=round(remote_work_percentage, 1),
            experience_level=experience_level,
            company_size=company_size,
            industry=industry,
            ai_adoption_maturity=ai_maturity,
            focus_area=focus_area
        )

    def _calculate_base_salary(self, year_factor: float) -> float:
        """Calculate base salary with realistic market volatility."""
        # Non-linear growth with market cycles
        base_salary = 800000
        
        # Technology adoption S-curve (slow start, rapid middle, plateau)
        growth_curve = 1.2 * (1 / (1 + np.exp(-6 * (year_factor - 0.5))))
        
        # Market volatility cycles
        volatility = np.sin(year_factor * 8) * 0.15 + np.sin(year_factor * 3.5) * 0.08
        
        # Economic events impact (random market shocks)
        market_shock = np.random.choice([1.0, 0.85, 1.15, 0.9, 1.1], p=[0.6, 0.15, 0.15, 0.05, 0.05])
        
        total_growth = base_salary * (1 + growth_curve + volatility) * market_shock
        return max(total_growth, 400000)  # Minimum salary floor

    def _calculate_salary_multiplier(
        self,
        skill: str,
        role: str,
        state: str,
        experience: ExperienceLevel,
        company_size: CompanySize
    ) -> float:
        """Calculate combined salary multiplier."""
        skill_mult = self.skill_multipliers.get(skill, 1.0)
        role_mult = self.role_multipliers.get(role, 1.0)
        state_mult = self.state_multipliers.get(state, 1.0)
        exp_mult = self.experience_multipliers.get(experience, 1.0)
        company_mult = self.company_size_multipliers.get(company_size, 1.0)
        
        return skill_mult * role_mult * state_mult * exp_mult * company_mult

    def _calculate_job_postings(self, year_factor: float, month: int, state: str = None) -> int:
        """Calculate job postings with seasonal variations and state-specific patterns."""
        base_postings = 500 + (year_factor * 1500)
        
        # Enhanced seasonal patterns (hiring cycles)
        seasonal_factor = np.sin((month - 1) * np.pi / 6) * 100  # Peak in Jan-Mar, Sept-Oct
        quarter_end_boost = 50 if month in [3, 6, 9, 12] else 0  # Quarter-end hiring pushes
        
        # State-specific market activity
        state_factor = 1.0
        if state and state in self.state_market_patterns:
            pattern = self.state_market_patterns[state]
            # Tech hubs have more consistent posting volumes
            state_factor = 0.5 + (pattern["startup_density"] * 0.8)
            # Market volatility affects posting frequency
            market_volatility = np.random.normal(0, base_postings * pattern["growth_volatility"])
        else:
            market_volatility = np.random.normal(0, 150)
        
        # Economic cycles (boom/bust periods)
        economic_cycle = np.sin(year_factor * 4) * 200  # 4-year economic cycles
        
        total_postings = (base_postings + seasonal_factor + quarter_end_boost + 
                         economic_cycle + market_volatility) * state_factor
        
        return max(int(total_postings), 25)

    def _calculate_skill_demand(self, year_factor: float, skill: str) -> float:
        """Calculate skill demand with realistic technology adoption curves."""
        # Different adoption patterns for different technologies
        adoption_patterns = {
            # Emerging tech (late adopters, exponential growth)
            'emerging': ['Agent', 'RAG', 'Agentic', 'Vector', 'LangChain', 'CrewAI', 'AutoGen'],
            # Established tech (early adoption, steady growth)
            'established': ['Machine Learning', 'Python', 'TensorFlow', 'PyTorch', 'SQL'],
            # Mature tech (early adoption, plateau)
            'mature': ['Deep Learning', 'Computer Vision', 'Natural Language Processing'],
            # Cloud tech (corporate adoption cycles)
            'cloud': ['AWS AI', 'Azure AI', 'Google AI']
        }
        
        # Determine technology category
        tech_type = 'established'  # default
        for category, techs in adoption_patterns.items():
            if any(tech in skill for tech in techs):
                tech_type = category
                break
        
        # Different growth curves for different tech types
        if tech_type == 'emerging':
            # S-curve with late start (ChatGPT effect)
            base_demand = 10 + 80 * (1 / (1 + np.exp(-8 * (year_factor - 0.7))))
        elif tech_type == 'mature':
            # Early adoption then plateau
            base_demand = 60 + 25 * year_factor * (1 - year_factor)
        elif tech_type == 'cloud':
            # Corporate adoption cycles (step-wise growth)
            base_demand = 30 + 50 * (1 - np.exp(-3 * year_factor))
        else:  # established
            # Steady linear growth with saturation
            base_demand = 25 + 60 * year_factor * (1.2 - year_factor)
        
        # Add market noise and regional variations
        market_noise = np.random.normal(0, 8)
        seasonal_factor = np.sin(year_factor * 12) * 5  # Hiring cycles
        
        final_demand = base_demand + market_noise + seasonal_factor
        return min(max(final_demand, 5), 100)

    def _calculate_training_programs(self, year_factor: float, state: str = None, skill: str = None) -> int:
        """Calculate number of training programs with market-driven variations."""
        base_programs = 30 + year_factor * 100
        
        # Technology-specific training demand
        tech_boost = 1.0
        if skill:
            if any(tech in skill for tech in ['Agent', 'RAG', 'Agentic', 'LLM', 'ChatGPT']):
                # Exponential growth for emerging AI
                tech_boost = 1.5 + (year_factor ** 2) * 2
            elif any(tech in skill for tech in ['Machine Learning', 'Deep Learning']):
                # Steady established growth
                tech_boost = 1.2 + year_factor * 0.5
            elif any(tech in skill for tech in ['Python', 'SQL']):
                # Mature tech with plateau
                tech_boost = 1.1 + year_factor * 0.3 * (1 - year_factor * 0.5)
        
        # State-specific education infrastructure
        state_factor = 1.0
        if state and state in self.state_market_patterns:
            pattern = self.state_market_patterns[state]
            # More mature markets have more training infrastructure
            state_factor = 0.6 + (pattern["maturity"] * 0.8)
            # Government initiatives in different states
            govt_boost = np.random.choice([1.0, 1.3, 0.8], p=[0.7, 0.2, 0.1])  # Policy initiatives
        else:
            govt_boost = 1.0
        
        # Market demand cycles
        demand_cycle = 1 + np.sin(year_factor * 6) * 0.3  # Faster cycles for training
        
        # Corporate training budget cycles
        budget_factor = np.random.choice([0.8, 1.0, 1.4], p=[0.2, 0.6, 0.2])  # Budget constraints
        
        total_programs = (base_programs * tech_boost * state_factor * 
                         demand_cycle * budget_factor * govt_boost)
        
        random_factor = np.random.normal(0, total_programs * 0.2)
        return max(int(total_programs + random_factor), 3)

    def _calculate_certification_rate(self, year_factor: float, skill: str = None, state: str = None) -> float:
        """Calculate certification completion rate with realistic market factors."""
        base_rate = 55 + year_factor * 30
        
        # Skill-specific certification difficulty and value
        skill_factor = 1.0
        if skill:
            if any(tech in skill for tech in ['Agent', 'RAG', 'Agentic', 'LLM']):
                # New tech: fewer certs available, but high motivation
                skill_factor = 0.7 + year_factor * 0.6  # Growing cert availability
            elif any(tech in skill for tech in ['AWS', 'Azure', 'Google']):
                # Cloud certs: well-established, high completion rates
                skill_factor = 1.3
            elif any(tech in skill for tech in ['Python', 'SQL']):
                # Foundational skills: many options, moderate completion
                skill_factor = 1.1
            elif any(tech in skill for tech in ['Ethics', 'MLOps']):
                # Specialized but important: moderate rates
                skill_factor = 0.9
        
        # State-specific education culture and infrastructure
        state_factor = 1.0
        if state and state in self.state_market_patterns:
            pattern = self.state_market_patterns[state]
            # More mature markets have better certification infrastructure
            state_factor = 0.8 + (pattern["maturity"] * 0.4)
            # Economic factors affect certification affordability
            economic_factor = np.random.normal(1.0, pattern["growth_volatility"] * 0.5)
        else:
            economic_factor = 1.0
        
        # Market saturation effects
        saturation_effect = 1.2 - (year_factor * 0.3)  # Easier to get certified early, harder later
        
        # Seasonal completion patterns (end of quarters, end of year)
        current_month = np.random.randint(1, 13)
        seasonal_boost = 1.1 if current_month in [3, 6, 9, 12] else 1.0
        
        # Corporate training budget cycles
        corporate_factor = np.random.choice([0.85, 1.0, 1.25], p=[0.25, 0.5, 0.25])
        
        total_rate = (base_rate * skill_factor * state_factor * saturation_effect * 
                     seasonal_boost * corporate_factor * economic_factor)
        
        random_factor = np.random.normal(0, 8)
        return min(max(total_rate + random_factor, 25), 95)

    def _calculate_remote_work(self, year_factor: float, year: int, company_size: CompanySize = None, industry: Industry = None) -> float:
        """Calculate remote work percentage with COVID impact and company/industry factors."""
        base_remote = 15 + year_factor * 60
        
        # COVID impact with realistic adoption curves
        if year >= 2020:
            # Non-linear COVID adoption
            covid_years = year - 2019
            covid_boost = 25 * (1 - np.exp(-covid_years * 0.8))  # Exponential adoption
            # Post-COVID normalization (some return to office)
            if year >= 2023:
                return_to_office = (year - 2022) * 5  # Gradual return
                covid_boost = max(covid_boost - return_to_office, 15)
        else:
            covid_boost = 0
        
        # Company size factors
        company_factor = 1.0
        if company_size:
            if company_size == CompanySize.STARTUP:
                company_factor = 1.4  # Startups more remote-friendly
            elif company_size == CompanySize.MEDIUM:
                company_factor = 1.2
            elif company_size == CompanySize.LARGE:
                company_factor = 0.9  # Large companies slower to adopt
            elif company_size == CompanySize.ENTERPRISE:
                company_factor = 0.7  # Enterprises most conservative
        
        # Industry-specific remote work patterns
        industry_factor = 1.0
        if industry:
            if industry in [Industry.IT_SERVICES, Industry.FINTECH]:
                industry_factor = 1.5  # Tech companies lead remote work
            elif industry == Industry.EDTECH:
                industry_factor = 1.4  # EdTech companies are remote-friendly
            elif industry == Industry.HEALTHCARE:
                industry_factor = 0.6  # Healthcare requires physical presence
            elif industry == Industry.ECOMMERCE:
                industry_factor = 1.2  # E-commerce moderate remote work
            elif industry == Industry.GOVERNMENT:
                industry_factor = 0.5  # Government requires physical presence
            elif industry == Industry.RESEARCH:
                industry_factor = 1.3  # Research adapts quickly
        
        # Market maturity and infrastructure factors
        infrastructure_factor = 0.7 + year_factor * 0.6  # Better infrastructure over time
        
        # Seasonal variations (less remote work in certain months)
        current_month = np.random.randint(1, 13)
        seasonal_factor = 0.95 if current_month in [6, 7, 8] else 1.0  # Summer months
        
        total_remote = (base_remote + covid_boost) * company_factor * industry_factor * infrastructure_factor * seasonal_factor
        
        random_factor = np.random.normal(0, total_remote * 0.15)
        return min(max(total_remote + random_factor, 5), 85)

    def export_to_dataframe(self, records: List[AISkillRecord]) -> pd.DataFrame:
        """Export records to pandas DataFrame."""
        try:
            data = [record.dict() for record in records]
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            logger.info(f"Exported {len(df)} records to DataFrame")
            return df
        except Exception as e:
            logger.error(f"Failed to export to DataFrame: {str(e)}")
            raise DataGenerationError(f"Export failed: {str(e)}")

    def validate_data_quality(self, records: List[AISkillRecord]) -> dict:
        """Validate data quality and return metrics."""
        try:
            total_records = len(records)
            
            if total_records == 0:
                return {"quality_score": 0.0, "issues": ["No records to validate"]}

            issues = []
            
            # Check for duplicates
            dates = [r.date for r in records]
            if len(dates) != len(set(dates)):
                issues.append("Duplicate dates found")

            # Check salary ranges
            salaries_inr = [r.avg_salary_inr for r in records]
            if any(s < 400000 or s > 8000000 for s in salaries_inr):
                issues.append("Salary values out of realistic range")

            # Check skill demand scores
            demand_scores = [r.skill_demand_score for r in records]
            if any(s < 0 or s > 100 for s in demand_scores):
                issues.append("Skill demand scores out of range")

            quality_score = max(0.0, 1.0 - len(issues) * 0.2)
            
            return {
                "total_records": total_records,
                "quality_score": quality_score,
                "issues": issues,
                "salary_range_inr": {"min": min(salaries_inr), "max": max(salaries_inr)},
                "demand_score_range": {"min": min(demand_scores), "max": max(demand_scores)}
            }
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {"quality_score": 0.0, "issues": [f"Validation error: {str(e)}"]}