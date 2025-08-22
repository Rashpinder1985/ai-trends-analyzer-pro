"""Advanced prediction engine with proper ML practices."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from ai_trends.models.schemas import AISkillRecord, PredictionRecord
from ai_trends.utils.exceptions import PredictionError, ValidationError
from ai_trends.utils.logging import log_performance
from config.settings import settings


class TrendPredictor:
    """Advanced trend prediction with proper ML validation."""

    def __init__(self):
        """Initialize predictor with default settings."""
        self.models: Dict[str, Pipeline] = {}
        self.model_scores: Dict[str, Dict[str, float]] = {}
        self.is_trained = False
        logger.info("TrendPredictor initialized")

    @log_performance("trend_prediction")
    def predict_trends(
        self,
        historical_data: List[AISkillRecord],
        prediction_years: int = None,
        confidence_level: float = 0.95
    ) -> List[PredictionRecord]:
        """Predict future trends with confidence intervals."""
        try:
            prediction_years = prediction_years or settings.prediction_years
            
            if not historical_data:
                raise ValidationError("Historical data cannot be empty")

            logger.info(f"Predicting trends for {prediction_years} years")

            # Convert to DataFrame for easier processing
            df = pd.DataFrame([record.dict() for record in historical_data])
            
            # Prepare yearly aggregated data
            yearly_data = self._prepare_yearly_data(df)
            
            # Train models
            self._train_models(yearly_data)
            
            # Generate predictions
            predictions = self._generate_predictions(yearly_data, prediction_years, confidence_level)
            
            logger.info(f"Generated {len(predictions)} prediction records")
            return predictions

        except Exception as e:
            logger.error(f"Trend prediction failed: {str(e)}")
            raise PredictionError(f"Failed to predict trends: {str(e)}")

    def _prepare_yearly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare yearly aggregated data for training."""
        try:
            df['date'] = pd.to_datetime(df['date'])
            
            yearly_data = df.groupby('year').agg({
                'avg_salary_inr': 'mean',
                'avg_salary_usd': 'mean',
                'job_postings': 'sum',
                'skill_demand_score': 'mean',
                'training_programs': 'sum',
                'certification_completion_rate': 'mean',
                'remote_work_percentage': 'mean'
            }).reset_index()
            
            logger.info(f"Prepared yearly data with {len(yearly_data)} years")
            return yearly_data

        except Exception as e:
            raise PredictionError(f"Failed to prepare yearly data: {str(e)}")

    def _train_models(self, yearly_data: pd.DataFrame) -> None:
        """Train prediction models with validation."""
        try:
            X = yearly_data['year'].values.reshape(-1, 1)
            
            metrics_to_predict = [
                'avg_salary_inr', 'avg_salary_usd', 'job_postings',
                'skill_demand_score', 'training_programs',
                'certification_completion_rate', 'remote_work_percentage'
            ]
            
            for metric in metrics_to_predict:
                logger.info(f"Training model for {metric}")
                
                y = yearly_data[metric].values
                
                # Create pipeline with polynomial features
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                    ('regressor', LinearRegression())
                ])
                
                # Train model
                model.fit(X, y)
                
                # Validate model performance
                scores = self._validate_model(model, X, y, metric)
                
                # Store model and scores
                self.models[metric] = model
                self.model_scores[metric] = scores

            self.is_trained = True
            logger.info("All models trained successfully")

        except Exception as e:
            raise PredictionError(f"Model training failed: {str(e)}")

    def _validate_model(
        self,
        model: Pipeline,
        X: np.ndarray,
        y: np.ndarray,
        metric_name: str
    ) -> Dict[str, float]:
        """Validate model performance with cross-validation."""
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)), scoring='r2')
            
            # Train-test split for additional metrics
            if len(X) > 3:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=settings.random_seed
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                # Not enough data for train-test split
                y_pred = model.predict(X)
                mae = mean_absolute_error(y, y_pred)
                r2 = model.score(X, y)

            scores = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'r2_score': r2,
                'mae': mae,
                'rmse': np.sqrt(np.mean((y - model.predict(X)) ** 2))
            }
            
            logger.info(f"Model validation for {metric_name}: R²={r2:.3f}, MAE={mae:.2f}")
            return scores

        except Exception as e:
            logger.warning(f"Model validation failed for {metric_name}: {str(e)}")
            return {'r2_score': 0.0, 'mae': float('inf'), 'rmse': float('inf')}

    def _generate_predictions(
        self,
        yearly_data: pd.DataFrame,
        prediction_years: int,
        confidence_level: float
    ) -> List[PredictionRecord]:
        """Generate predictions with confidence intervals."""
        try:
            if not self.is_trained:
                raise PredictionError("Models not trained yet")

            last_year = yearly_data['year'].max()
            future_years = list(range(last_year + 1, last_year + prediction_years + 1))
            
            predictions = []
            
            for year in future_years:
                pred_data = {'year': year}
                
                X_future = np.array([[year]])
                
                for metric, model in self.models.items():
                    try:
                        # Base prediction
                        pred_value = model.predict(X_future)[0]
                        
                        # Apply constraints
                        pred_value = self._apply_constraints(metric, pred_value, year)
                        
                        # Calculate confidence interval (simplified)
                        model_score = self.model_scores[metric]
                        uncertainty = model_score.get('rmse', 0) * 1.96  # 95% CI approximation
                        
                        pred_data[metric] = pred_value
                        pred_data[f'{metric}_confidence_lower'] = max(0, pred_value - uncertainty)
                        pred_data[f'{metric}_confidence_upper'] = pred_value + uncertainty
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for {metric} in year {year}: {str(e)}")
                        pred_data[metric] = 0.0

                prediction = PredictionRecord(
                    year=year,
                    avg_salary_inr=pred_data.get('avg_salary_inr', 0),
                    avg_salary_usd=pred_data.get('avg_salary_usd', 0),
                    job_postings=pred_data.get('job_postings', 0),
                    skill_demand_score=pred_data.get('skill_demand_score', 0),
                    training_programs=pred_data.get('training_programs', 0),
                    certification_completion_rate=pred_data.get('certification_completion_rate', 0),
                    remote_work_percentage=pred_data.get('remote_work_percentage', 0),
                    confidence_interval_lower=pred_data.get('avg_salary_inr_confidence_lower'),
                    confidence_interval_upper=pred_data.get('avg_salary_inr_confidence_upper')
                )
                
                predictions.append(prediction)

            return predictions

        except Exception as e:
            raise PredictionError(f"Failed to generate predictions: {str(e)}")

    def _apply_constraints(self, metric: str, value: float, year: int) -> float:
        """Apply realistic constraints to predictions."""
        constraints = {
            'avg_salary_inr': (500000, 15000000),  # 5-150 LPA
            'avg_salary_usd': (6000, 180000),     # Corresponding USD
            'job_postings': (0, 1000000),         # Max job postings
            'skill_demand_score': (0, 100),       # 0-100 scale
            'training_programs': (0, 50000),      # Max training programs
            'certification_completion_rate': (0, 100),  # 0-100%
            'remote_work_percentage': (0, 95)     # Max 95% remote
        }
        
        if metric in constraints:
            min_val, max_val = constraints[metric]
            return max(min_val, min(max_val, value))
        
        return value

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all trained models."""
        if not self.is_trained:
            return {}
        
        return self.model_scores.copy()

    def calculate_growth_rates(self, historical_data: List[AISkillRecord]) -> Dict[str, float]:
        """Calculate compound annual growth rates (CAGR)."""
        try:
            df = pd.DataFrame([record.dict() for record in historical_data])
            yearly_data = self._prepare_yearly_data(df)
            
            if len(yearly_data) < 2:
                return {}
            
            growth_rates = {}
            years = yearly_data['year'].max() - yearly_data['year'].min()
            
            metrics = ['avg_salary_inr', 'job_postings', 'skill_demand_score']
            
            for metric in metrics:
                start_value = yearly_data[yearly_data['year'] == yearly_data['year'].min()][metric].iloc[0]
                end_value = yearly_data[yearly_data['year'] == yearly_data['year'].max()][metric].iloc[0]
                
                if start_value > 0 and years > 0:
                    cagr = ((end_value / start_value) ** (1/years) - 1) * 100
                    growth_rates[f'{metric}_cagr'] = round(cagr, 2)
            
            logger.info(f"Calculated CAGR for {len(growth_rates)} metrics")
            return growth_rates

        except Exception as e:
            logger.error(f"CAGR calculation failed: {str(e)}")
            return {}

    def analyze_trends(self, historical_data: List[AISkillRecord]) -> Dict[str, str]:
        """Analyze trend directions (increasing, decreasing, stable)."""
        try:
            df = pd.DataFrame([record.dict() for record in historical_data])
            yearly_data = self._prepare_yearly_data(df)
            
            trends = {}
            metrics = ['avg_salary_inr', 'job_postings', 'skill_demand_score', 'remote_work_percentage']
            
            for metric in metrics:
                values = yearly_data[metric].values
                if len(values) >= 2:
                    # Simple trend analysis using linear regression slope
                    X = np.arange(len(values)).reshape(-1, 1)
                    model = LinearRegression().fit(X, values)
                    slope = model.coef_[0]
                    
                    if slope > values.std() * 0.1:  # Significant positive trend
                        trends[metric] = "increasing"
                    elif slope < -values.std() * 0.1:  # Significant negative trend
                        trends[metric] = "decreasing"
                    else:
                        trends[metric] = "stable"
            
            return trends

        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            return {}