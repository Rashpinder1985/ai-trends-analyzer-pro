"""Configuration management for AI Trends Analyzer."""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environments."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings with validation."""

    # Application
    app_name: str = Field(default="AI Trends Analyzer Pro", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Database/Cache
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format"
    )

    # Data Generation
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    default_sample_size: int = Field(default=1000, description="Default sample size")
    start_year: int = Field(default=2019, description="Historical data start year")
    end_year: int = Field(default=2024, description="Historical data end year")
    prediction_years: int = Field(default=6, description="Years to predict")

    # Indian States
    indian_states: List[str] = Field(
        default=[
            "Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Delhi",
            "West Bengal", "Gujarat", "Rajasthan", "Uttar Pradesh", "Kerala"
        ],
        description="Indian states to analyze"
    )

    # AI Skills
    ai_skills: List[str] = Field(
        default=[
            "Machine Learning", "Deep Learning", "Natural Language Processing",
            "Computer Vision", "Reinforcement Learning", "MLOps", "AI Ethics",
            "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "OpenCV",
            "Python", "R", "SQL", "Spark", "AWS AI", "Azure AI", "Google AI",
            "AI Agents", "Multi-Agent Systems", "RAG", "Agentic AI",
            "LangChain", "CrewAI", "AutoGen", "Agent Frameworks",
            "Vector Databases", "Embedding Models", "Prompt Engineering",
            "Large Language Models", "Generative AI", "ChatGPT Integration"
        ],
        description="AI skills to analyze"
    )

    # Job Roles
    job_roles: List[str] = Field(
        default=[
            "Machine Learning Engineer", "Data Scientist", "AI Research Scientist",
            "MLOps Engineer", "AI Product Manager", "AI Consultant",
            "Computer Vision Engineer", "NLP Engineer", "AI Architect",
            "AI Agent Developer", "RAG Engineer", "Agentic AI Specialist",
            "Conversational AI Engineer", "LLM Engineer", "Prompt Engineer",
            "AI Integration Specialist", "Multi-Agent Systems Developer"
        ],
        description="Job roles to analyze"
    )

    # File paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    # API
    api_title: str = Field(default="AI Trends Analyzer API", description="API title")
    api_description: str = Field(
        default="Production-ready API for AI trends analysis in Indian market",
        description="API description"
    )
    api_version: str = Field(default="1.0.0", description="API version")
    enable_docs: bool = Field(default=True, description="Enable API documentation")

    # Security
    cors_origins: List[str] = Field(
        default=[""https://rashpinder1985.github.io""],
        description="CORS allowed origins"
    )
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Max request size")

    @field_validator("data_dir", "output_dir", "logs_dir", mode="before")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> Environment:
        """Validate environment value."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    class Config:
        """Pydantic config."""
        
        env_file = ".env"
        env_prefix = "AI_TRENDS_"
        case_sensitive = False


# Global settings instance
settings = Settings()
