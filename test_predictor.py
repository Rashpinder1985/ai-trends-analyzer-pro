"""Unit tests for trend predictor."""

import pytest
import numpy as np

from ai_trends.core.data_generator import DataGenerator
from ai_trends.core.predictor import TrendPredictor
from ai_trends.models.schemas import AISkillRecord, PredictionRecord
from ai_trends.utils.exceptions import ValidationError, PredictionError


class TestTrendPredictor:
    """Test cases for TrendPredictor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = TrendPredictor()
        self.generator = DataGenerator(random_seed=42)

    def test_initialization(self):
        """Test predictor initialization."""
        assert not self.predictor.is_trained
        assert len(self.predictor.models) == 0
        assert len(self.predictor.model_scores) == 0

    def test_predict_trends_empty_data(self):
        """Test prediction with empty historical data."""
        with pytest.raises(ValidationError, match="Historical data cannot be empty"):
            self.predictor.predict_trends([])

    def test_predict_trends_basic(self):
        """Test basic trend prediction."""
        # Generate some historical data
        historical_data = self.generator.generate_historical_data(sample_size=20)
        
        # Predict trends
        predictions = self.predictor.predict_trends(
            historical_data, 
            prediction_years=3
        )
        
        assert len(predictions) == 3
        assert all(isinstance(pred, PredictionRecord) for pred in predictions)
        assert self.predictor.is_trained
        
        # Check prediction years are sequential and in future
        last_historical_year = max(record.year for record in historical_data)
        for i, pred in enumerate(predictions):
            assert pred.year == last_historical_year + i + 1

    def test_predict_trends_validation(self):
        """Test prediction validation and constraints."""
        historical_data = self.generator.generate_historical_data(sample_size=15)
        predictions = self.predictor.predict_trends(historical_data, prediction_years=2)
        
        for pred in predictions:
            # Check salary constraints
            assert 500000 <= pred.avg_salary_inr <= 15000000
            assert 6000 <= pred.avg_salary_usd <= 180000
            
            # Check percentage constraints
            assert 0 <= pred.skill_demand_score <= 100
            assert 0 <= pred.certification_completion_rate <= 100
            assert 0 <= pred.remote_work_percentage <= 95
            
            # Check positive values
            assert pred.job_postings >= 0
            assert pred.training_programs >= 0

    def test_prepare_yearly_data(self):
        """Test yearly data preparation."""
        historical_data = self.generator.generate_historical_data(sample_size=25)
        df = self.generator.export_to_dataframe(historical_data)
        
        yearly_data = self.predictor._prepare_yearly_data(df)
        
        assert 'year' in yearly_data.columns
        assert 'avg_salary_inr' in yearly_data.columns
        assert 'job_postings' in yearly_data.columns
        assert len(yearly_data) <= 6  # Max years in data (2019-2024)

    def test_train_models(self):
        """Test model training."""
        historical_data = self.generator.generate_historical_data(sample_size=30)
        df = self.generator.export_to_dataframe(historical_data)
        yearly_data = self.predictor._prepare_yearly_data(df)
        
        self.predictor._train_models(yearly_data)
        
        assert self.predictor.is_trained
        assert len(self.predictor.models) > 0
        assert len(self.predictor.model_scores) > 0
        
        # Check that all expected metrics have models
        expected_metrics = [
            'avg_salary_inr', 'avg_salary_usd', 'job_postings',
            'skill_demand_score', 'training_programs',
            'certification_completion_rate', 'remote_work_percentage'
        ]
        
        for metric in expected_metrics:
            assert metric in self.predictor.models
            assert metric in self.predictor.model_scores

    def test_validate_model(self):
        """Test model validation."""
        # Create simple test data
        X = np.array([[2019], [2020], [2021], [2022], [2023]])
        y = np.array([1000, 1100, 1200, 1300, 1400])  # Linear trend
        
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        
        scores = self.predictor._validate_model(model, X, y, "test_metric")
        
        assert 'r2_score' in scores
        assert 'mae' in scores
        assert 'rmse' in scores
        assert scores['r2_score'] > 0.8  # Should fit linear trend well

    def test_apply_constraints(self):
        """Test constraint application."""
        # Test salary constraints
        constrained_salary = self.predictor._apply_constraints('avg_salary_inr', 20000000, 2025)
        assert constrained_salary <= 15000000
        
        constrained_low_salary = self.predictor._apply_constraints('avg_salary_inr', 100000, 2025)
        assert constrained_low_salary >= 500000
        
        # Test percentage constraints
        constrained_percentage = self.predictor._apply_constraints('skill_demand_score', 150, 2025)
        assert constrained_percentage <= 100
        
        constrained_negative = self.predictor._apply_constraints('skill_demand_score', -10, 2025)
        assert constrained_negative >= 0

    def test_get_model_performance(self):
        """Test model performance retrieval."""
        # Before training
        performance = self.predictor.get_model_performance()
        assert len(performance) == 0
        
        # After training
        historical_data = self.generator.generate_historical_data(sample_size=20)
        self.predictor.predict_trends(historical_data, prediction_years=2)
        
        performance = self.predictor.get_model_performance()
        assert len(performance) > 0
        
        for metric, scores in performance.items():
            assert 'r2_score' in scores
            assert 'mae' in scores

    def test_calculate_growth_rates(self):
        """Test CAGR calculation."""
        historical_data = self.generator.generate_historical_data(sample_size=30)
        growth_rates = self.predictor.calculate_growth_rates(historical_data)
        
        assert len(growth_rates) > 0
        
        # Check that we get CAGR for expected metrics
        expected_cagr_metrics = ['avg_salary_inr_cagr', 'job_postings_cagr', 'skill_demand_score_cagr']
        
        for metric in expected_cagr_metrics:
            assert metric in growth_rates
            assert isinstance(growth_rates[metric], (int, float))

    def test_analyze_trends(self):
        """Test trend direction analysis."""
        historical_data = self.generator.generate_historical_data(sample_size=25)
        trends = self.predictor.analyze_trends(historical_data)
        
        assert len(trends) > 0
        
        # Check that trend directions are valid
        valid_directions = ["increasing", "decreasing", "stable"]
        for metric, direction in trends.items():
            assert direction in valid_directions

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        historical_data = self.generator.generate_historical_data(sample_size=20)
        predictions = self.predictor.predict_trends(historical_data, prediction_years=2)
        
        for pred in predictions:
            if pred.confidence_interval_lower is not None and pred.confidence_interval_upper is not None:
                # Lower bound should be less than upper bound
                assert pred.confidence_interval_lower <= pred.confidence_interval_upper
                
                # Main prediction should be within bounds
                assert pred.confidence_interval_lower <= pred.avg_salary_inr <= pred.confidence_interval_upper

    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        # Create minimal data that might cause issues
        historical_data = self.generator.generate_historical_data(sample_size=2)
        
        # Should still work but might have lower quality predictions
        predictions = self.predictor.predict_trends(historical_data, prediction_years=1)
        assert len(predictions) == 1

    def test_reproducibility(self):
        """Test that predictions are reproducible."""
        historical_data = self.generator.generate_historical_data(sample_size=20)
        
        predictor1 = TrendPredictor()
        predictor2 = TrendPredictor()
        
        predictions1 = predictor1.predict_trends(historical_data, prediction_years=2)
        predictions2 = predictor2.predict_trends(historical_data, prediction_years=2)
        
        # Should produce identical predictions with same data
        for p1, p2 in zip(predictions1, predictions2):
            assert p1.year == p2.year
            assert abs(p1.avg_salary_inr - p2.avg_salary_inr) < 1  # Account for floating point precision

    def test_long_term_predictions(self):
        """Test long-term predictions stability."""
        historical_data = self.generator.generate_historical_data(sample_size=30)
        predictions = self.predictor.predict_trends(historical_data, prediction_years=10)
        
        assert len(predictions) == 10
        
        # Check that predictions don't go completely unrealistic
        salaries = [pred.avg_salary_inr for pred in predictions]
        
        # Should show some growth but not exponential explosion
        for i in range(1, len(salaries)):
            growth_factor = salaries[i] / salaries[i-1]
            assert 0.8 <= growth_factor <= 2.0  # Reasonable year-over-year growth