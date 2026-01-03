"""
Tests for HuggingFace collectors.

Validates pre-LLM filtering and human verification logic.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from corpus.collectors.huggingface import (
    HuggingFaceNYTCollector,
    HuggingFaceGenericCollector
)
from corpus.schema import Domain, Subdomain


class TestHuggingFaceNYTCollector:
    """Test HuggingFace NYT collector."""

    @pytest.fixture
    def collector(self):
        """Create test collector without loading dataset."""
        with patch('corpus.collectors.huggingface.HAS_DATASETS', True):
            collector = HuggingFaceNYTCollector()
            collector._dataset = Mock()  # Mock dataset to avoid loading
            return collector

    def test_verify_human_written_explicit_flag(self, collector):
        """Test human verification with explicit is_human flag."""
        # Human sample
        sample = {'is_human': True, 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is True

        # AI sample
        sample = {'is_human': False, 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is False

    def test_verify_human_written_source_field(self, collector):
        """Test human verification with source field."""
        # Human source
        sample = {'source': 'human', 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is True

        # AI source
        sample = {'source': 'GPT-4', 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is False

        sample = {'source': 'ai_generated', 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is False

    def test_verify_human_written_model_field(self, collector):
        """Test human verification rejects samples with model field."""
        # Sample with model field is AI
        sample = {'model': 'gpt-4', 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is False

        # Sample with generated_by field is AI
        sample = {'generated_by': 'claude', 'text': 'Sample text'}
        assert collector._verify_human_written(sample) is False

    def test_extract_text(self, collector):
        """Test text extraction from various fields."""
        # Standard text field
        sample = {'text': 'Sample article text'}
        assert collector._extract_text(sample) == 'Sample article text'

        # Content field
        sample = {'content': 'Sample article content'}
        assert collector._extract_text(sample) == 'Sample article content'

        # Article field
        sample = {'article': 'Sample article'}
        assert collector._extract_text(sample) == 'Sample article'

        # No text field
        sample = {'title': 'Just a title'}
        assert collector._extract_text(sample) is None

    def test_extract_date(self, collector):
        """Test date extraction and parsing."""
        # ISO format
        sample = {'date': '2022-05-15'}
        result = collector._extract_date(sample)
        assert result == date(2022, 5, 15)

        # With time
        sample = {'publication_date': '2021-12-31T23:59:59'}
        result = collector._extract_date(sample)
        assert result == date(2021, 12, 31)

        # Different field name
        sample = {'pub_date': '2020-01-01'}
        result = collector._extract_date(sample)
        assert result == date(2020, 1, 1)

        # No date field
        sample = {'text': 'No date'}
        assert collector._extract_date(sample) is None

    def test_llm_cutoff_filtering(self, collector):
        """Test that post-LLM dates are rejected."""
        # Pre-LLM date (should pass)
        pre_llm = date(2022, 10, 15)
        assert pre_llm < collector.LLM_CUTOFF

        # Post-LLM date (should fail)
        post_llm = date(2022, 11, 15)
        assert post_llm >= collector.LLM_CUTOFF

        # Cutoff date itself (should fail)
        cutoff = date(2022, 11, 1)
        assert cutoff >= collector.LLM_CUTOFF

    def test_create_sample(self, collector):
        """Test CorpusSample creation."""
        sample_data = {
            'id': '12345',
            'title': 'Test Article',
            'url': 'https://example.com/article',
            'text': 'Article content'
        }

        pub_date = date(2022, 5, 15)
        text = 'Article content'

        corpus_sample = collector._create_sample(sample_data, 0, pub_date, text)

        assert corpus_sample.domain == Domain.JOURNALISTIC
        assert corpus_sample.subdomain == Subdomain.NEWS
        assert corpus_sample.text == text
        assert corpus_sample.source.content_date == pub_date
        assert corpus_sample.source.verified_pre_llm is True
        assert corpus_sample.source.author_type == "organization"
        assert corpus_sample.source.url == 'https://example.com/article'

    def test_create_sample_without_date(self, collector):
        """Test CorpusSample creation without publication date."""
        sample_data = {
            'id': '12345',
            'text': 'Article content'
        }

        corpus_sample = collector._create_sample(sample_data, 0, None, 'Article content')

        assert corpus_sample.source.content_date is None
        assert corpus_sample.source.verified_pre_llm is False
        assert corpus_sample.source.content_date_precision == "unknown"


class TestHuggingFaceGenericCollector:
    """Test generic HuggingFace collector."""

    def test_initialization(self):
        """Test collector initialization with custom parameters."""
        with patch('corpus.collectors.huggingface.HAS_DATASETS', True):
            collector = HuggingFaceGenericCollector(
                dataset_name="test/dataset",
                domain=Domain.ACADEMIC,
                subdomain=Subdomain.ESSAYS,
                text_field="content",
                date_field="created",
                human_filter=lambda x: x.get('human') is True
            )

            assert collector.dataset_name == "test/dataset"
            assert collector.DOMAIN == Domain.ACADEMIC
            assert collector.SUBDOMAIN == Subdomain.ESSAYS
            assert collector.text_field == "content"
            assert collector.date_field == "created"
            assert collector.human_filter({'human': True}) is True
            assert collector.human_filter({'human': False}) is False

    def test_requires_datasets_library(self):
        """Test that ImportError is raised when datasets not installed."""
        with patch('corpus.collectors.huggingface.HAS_DATASETS', False):
            with pytest.raises(ImportError, match="datasets library required"):
                HuggingFaceGenericCollector(
                    dataset_name="test/dataset",
                    domain=Domain.ACADEMIC,
                    subdomain=Subdomain.ESSAYS
                )


class TestPreLLMSafety:
    """Test pre-LLM safety guarantees."""

    @pytest.fixture
    def collector(self):
        """Create test collector."""
        with patch('corpus.collectors.huggingface.HAS_DATASETS', True):
            return HuggingFaceNYTCollector()

    def test_nov_1_2022_cutoff(self, collector):
        """Verify LLM cutoff is exactly Nov 1, 2022."""
        assert collector.LLM_CUTOFF == date(2022, 11, 1)

    def test_october_31_2022_passes(self, collector):
        """Last pre-LLM date should pass."""
        last_safe_date = date(2022, 10, 31)
        assert last_safe_date < collector.LLM_CUTOFF

    def test_november_1_2022_fails(self, collector):
        """First potentially-LLM date should fail."""
        first_unsafe_date = date(2022, 11, 1)
        assert first_unsafe_date >= collector.LLM_CUTOFF

    def test_verified_pre_llm_flag(self, collector):
        """Test verified_pre_llm flag is set correctly."""
        # Pre-LLM sample
        sample = {}
        pre_llm_date = date(2022, 5, 15)
        corpus_sample = collector._create_sample(sample, 0, pre_llm_date, 'text')
        assert corpus_sample.source.verified_pre_llm is True

        # Post-LLM sample
        post_llm_date = date(2023, 1, 1)
        corpus_sample = collector._create_sample(sample, 0, post_llm_date, 'text')
        assert corpus_sample.source.verified_pre_llm is False

        # No date sample
        corpus_sample = collector._create_sample(sample, 0, None, 'text')
        assert corpus_sample.source.verified_pre_llm is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
