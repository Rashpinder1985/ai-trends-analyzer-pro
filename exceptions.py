"""Custom exceptions for AI trends analyzer."""


class AITrendsException(Exception):
    """Base exception for AI trends analyzer."""
    pass


class ValidationError(AITrendsException):
    """Raised when validation fails."""
    pass


class DataGenerationError(AITrendsException):
    """Raised when data generation fails."""
    pass


class PredictionError(AITrendsException):
    """Raised when prediction fails."""
    pass


class VisualizationError(AITrendsException):
    """Raised when visualization creation fails."""
    pass


class ConfigurationError(AITrendsException):
    """Raised when configuration is invalid."""
    pass


class CacheError(AITrendsException):
    """Raised when cache operations fail."""
    pass


class APIError(AITrendsException):
    """Raised when API operations fail."""
    pass