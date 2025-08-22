"""Centralized logging configuration."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_json: bool = False
) -> None:
    """Set up centralized logging configuration."""
    
    # Remove default handler
    logger.remove()
    
    # Use settings if not provided
    log_level = log_level or settings.log_level.value
    log_file = log_file or (settings.logs_dir / "ai_trends.log")
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        level=log_level,
        format=settings.log_format,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler
    logger.add(
        log_file,
        level=log_level,
        format=settings.log_format,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # JSON handler for production (optional)
    if enable_json:
        json_file = settings.logs_dir / "ai_trends.json"
        logger.add(
            json_file,
            level=log_level,
            serialize=True,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    # Error handler
    error_file = settings.logs_dir / "errors.log"
    logger.add(
        error_file,
        level="ERROR",
        format=settings.log_format,
        rotation="10 MB",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> "logger":
    """Get a logger instance for a specific module."""
    return logger.bind(module=name)


# Performance logging decorator
def log_performance(func_name: Optional[str] = None):
    """Decorator to log function performance."""
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Function {name} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {name} failed after {execution_time:.4f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator


# Request logging middleware
class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Log request
            method = scope["method"]
            path = scope["path"]
            logger.info(f"Request started: {method} {path}")
            
            # Process request
            try:
                await self.app(scope, receive, send)
                execution_time = time.time() - start_time
                logger.info(f"Request completed: {method} {path} in {execution_time:.4f}s")
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Request failed: {method} {path} after {execution_time:.4f}s: {str(e)}")
                raise
        else:
            await self.app(scope, receive, send)