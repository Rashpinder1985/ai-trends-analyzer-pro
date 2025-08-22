"""Unit tests for data generator."""

import pytest
from datetime import datetime

from ai_trends.core.data_generator import DataGenerator
from ai_trends.models.schemas import AISkillRecord, ExperienceLevel
from ai_trends.utils.exceptions import ValidationError, DataGenerationError


class TestDataGenerator:
    """Test cases for DataGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = DataGenerator(random_seed=42)

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.random_seed == 42
        assert hasattr(self.generator, 'skill_multipliers')
        assert hasattr(self.generator, 'state_multipliers')

    def test_generate_historical_data_default(self):
        """Test data generation with default parameters."""
        records = self.generator.generate_historical_data(sample_size=10)
        
        assert len(records) == 10
        assert all(isinstance(record, AISkillRecord) for record in records)
        
        # Check data ranges
        for record in records:
            assert 2019 <= record.year <= 2024
            assert record.avg_salary_inr >= 400000
            assert 0 <= record.skill_demand_score <= 100
            assert 0 <= record.certification_completion_rate <= 100

    def test_generate_historical_data_custom_params(self):
        """Test data generation with custom parameters."""
        records = self.generator.generate_historical_data(
            sample_size=5,
            start_year=2020,
            end_year=2022,
            states=["Karnataka", "Maharashtra"],
            skills=["Machine Learning", "Deep Learning"]
        )
        
        assert len(records) == 5
        assert all(2020 <= record.year <= 2022 for record in records)
        assert all(record.indian_state in ["Karnataka", "Maharashtra"] for record in records)
        assert all(record.skill_category in ["Machine Learning", "Deep Learning"] for record in records)

    def test_validate_inputs_invalid_sample_size(self):
        """Test validation with invalid sample size."""
        with pytest.raises(ValidationError, match="Sample size must be positive"):
            self.generator._validate_inputs(-1, 2019, 2024, ["Karnataka"], ["ML"])

    def test_validate_inputs_invalid_years(self):
        """Test validation with invalid year range."""
        with pytest.raises(ValidationError, match="Start year must be before end year"):
            self.generator._validate_inputs(100, 2024, 2019, ["Karnataka"], ["ML"])

    def test_validate_inputs_future_year(self):
        """Test validation with future end year."""
        future_year = datetime.now().year + 5
        with pytest.raises(ValidationError, match="End year cannot be in the future"):
            self.generator._validate_inputs(100, 2019, future_year, ["Karnataka"], ["ML"])

    def test_validate_inputs_empty_states(self):
        """Test validation with empty states list."""
        with pytest.raises(ValidationError, match="States list cannot be empty"):
            self.generator._validate_inputs(100, 2019, 2024, [], ["ML"])

    def test_validate_inputs_empty_skills(self):
        """Test validation with empty skills list."""
        with pytest.raises(ValidationError, match="Skills list cannot be empty"):
            self.generator._validate_inputs(100, 2019, 2024, ["Karnataka"], [])

    def test_calculate_base_salary(self):
        """Test base salary calculation."""
        # Test year factor 0 (start year)
        salary_start = self.generator._calculate_base_salary(0.0)
        assert salary_start == 800000

        # Test year factor 1 (end year)
        salary_end = self.generator._calculate_base_salary(1.0)
        assert salary_end == 2000000

        # Test middle year
        salary_mid = self.generator._calculate_base_salary(0.5)
        assert 800000 < salary_mid < 2000000

    def test_calculate_salary_multiplier(self):
        """Test salary multiplier calculation."""
        multiplier = self.generator._calculate_salary_multiplier(
            skill="AI Agents",
            role="AI Agent Developer",
            state="Karnataka",
            experience=ExperienceLevel.SENIOR,
            company_size="Enterprise"
        )
        
        # Should be significant multiplier due to premium skill/role/state/experience
        assert multiplier > 2.0

    def test_calculate_job_postings(self):
        """Test job postings calculation."""
        # Test start year
        postings_start = self.generator._calculate_job_postings(0.0, 6)  # June
        assert postings_start >= 25  # Minimum constraint

        # Test end year
        postings_end = self.generator._calculate_job_postings(1.0, 6)
        assert postings_end > postings_start

    def test_calculate_skill_demand(self):
        """Test skill demand calculation."""
        # Regular skill
        demand_regular = self.generator._calculate_skill_demand(0.5, "Python")
        assert 10 <= demand_regular <= 100

        # Agent skill (should have boost)
        demand_agent = self.generator._calculate_skill_demand(0.5, "AI Agents")
        assert demand_agent >= demand_regular

    def test_calculate_remote_work_covid_impact(self):
        """Test remote work calculation with COVID impact."""
        # Pre-COVID
        remote_2019 = self.generator._calculate_remote_work(0.0, 2019)
        
        # Post-COVID
        remote_2021 = self.generator._calculate_remote_work(0.4, 2021)
        
        # Should show significant increase
        assert remote_2021 > remote_2019

    def test_export_to_dataframe(self):
        """Test DataFrame export."""
        records = self.generator.generate_historical_data(sample_size=10)
        df = self.generator.export_to_dataframe(records)
        
        assert len(df) == 10
        assert 'date' in df.columns
        assert 'avg_salary_inr' in df.columns
        assert df['date'].dtype == 'datetime64[ns]'

    def test_validate_data_quality_good_data(self):
        """Test data quality validation with good data."""
        records = self.generator.generate_historical_data(sample_size=10)
        quality_report = self.generator.validate_data_quality(records)
        
        assert quality_report['total_records'] == 10
        assert quality_report['quality_score'] > 0.8
        assert len(quality_report['issues']) == 0

    def test_validate_data_quality_empty_data(self):
        """Test data quality validation with empty data."""
        quality_report = self.generator.validate_data_quality([])
        
        assert quality_report['quality_score'] == 0.0
        assert "No records to validate" in quality_report['issues']

    def test_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        generator1 = DataGenerator(random_seed=123)
        generator2 = DataGenerator(random_seed=123)
        
        records1 = generator1.generate_historical_data(sample_size=5)
        records2 = generator2.generate_historical_data(sample_size=5)
        
        # Should generate identical data
        for r1, r2 in zip(records1, records2):
            assert r1.date == r2.date
            assert r1.avg_salary_inr == r2.avg_salary_inr
            assert r1.skill_category == r2.skill_category

    def test_error_handling_invalid_data_generation(self):
        """Test error handling during data generation."""
        # Mock a scenario where record generation fails frequently
        original_method = self.generator._generate_single_record
        
        def failing_method(*args, **kwargs):
            raise Exception("Simulated failure")
        
        self.generator._generate_single_record = failing_method
        
        with pytest.raises(DataGenerationError):
            self.generator.generate_historical_data(sample_size=10)
        
        # Restore original method
        self.generator._generate_single_record = original_method