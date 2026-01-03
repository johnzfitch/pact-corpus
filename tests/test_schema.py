"""
PACT Corpus Tests

Run with: pytest tests/
"""

import pytest
from datetime import date

from corpus.schema import (
    CorpusSample, 
    Domain, 
    Subdomain, 
    SourceMetadata,
    DOMAIN_CONFIGS,
)
from corpus.registry import DomainRegistry, BaseCollector


class TestSchema:
    """Test core data structures."""
    
    def test_corpus_sample_creation(self):
        """Test creating a basic corpus sample."""
        source = SourceMetadata(
            corpus_name="test",
            corpus_version="1.0",
            original_id="test_001",
            content_date=date(2020, 1, 1),
            verified_pre_llm=True,
        )
        
        sample = CorpusSample(
            sample_id="academic_essays_test_001",
            domain=Domain.ACADEMIC,
            subdomain=Subdomain.ESSAYS,
            source=source,
            text="This is a test essay about testing.",
        )
        
        assert sample.domain == Domain.ACADEMIC
        assert sample.is_original == True
        assert sample.word_count == 7
        assert len(sample.content_hash) == 16
    
    def test_sample_id_generation(self):
        """Test canonical ID format."""
        sample_id = CorpusSample.create_id(
            Domain.ACADEMIC,
            Subdomain.ESSAYS,
            "asap",
            "12345"
        )
        assert sample_id == "academic_essays_asap_12345"
    
    def test_pre_llm_verification(self):
        """Test pre-LLM date verification."""
        # Pre-LLM date
        source_pre = SourceMetadata(
            corpus_name="test",
            corpus_version="1.0",
            original_id="1",
            content_date=date(2020, 6, 15),
            verified_pre_llm=True,
        )
        assert source_pre.verified_pre_llm == True
        
        # Post-LLM date should not be verified
        source_post = SourceMetadata(
            corpus_name="test",
            corpus_version="1.0",
            original_id="2",
            content_date=date(2023, 6, 15),
            verified_pre_llm=False,
        )
        assert source_post.verified_pre_llm == False
    
    def test_domain_configs_exist(self):
        """Verify all domains have configurations."""
        for domain in Domain:
            assert domain in DOMAIN_CONFIGS
            config = DOMAIN_CONFIGS[domain]
            assert config.target_samples > 0
            assert len(config.subdomains) > 0


class TestRegistry:
    """Test collector registry."""
    
    def test_list_collectors(self):
        """Test listing registered collectors."""
        # Import collectors to register them
        import corpus.collectors
        
        collectors = DomainRegistry.list_collectors()
        assert len(collectors) > 0
        
        # Check we have collectors for major domains
        domains_with_collectors = set(c[0] for c in collectors)
        assert Domain.ACADEMIC in domains_with_collectors
        assert Domain.CONVERSATIONAL in domains_with_collectors
        assert Domain.LEGAL in domains_with_collectors
    
    def test_get_collector(self):
        """Test retrieving a specific collector."""
        import corpus.collectors
        
        collector_class = DomainRegistry.get(
            Domain.ACADEMIC,
            Subdomain.ESSAYS,
            "asap"
        )
        
        assert collector_class is not None
        assert issubclass(collector_class, BaseCollector)


class TestSampleSerialization:
    """Test serialization/deserialization."""
    
    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        source = SourceMetadata(
            corpus_name="test",
            corpus_version="1.0",
            original_id="test_001",
            content_date=date(2020, 1, 1),
            verified_pre_llm=True,
        )
        
        original = CorpusSample(
            sample_id="test_sample",
            domain=Domain.ACADEMIC,
            subdomain=Subdomain.ESSAYS,
            source=source,
            text="Test content here.",
        )
        
        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data['domain'] == 'academic'
        
        # Deserialize
        restored = CorpusSample.from_dict(data)
        assert restored.sample_id == original.sample_id
        assert restored.domain == original.domain
        assert restored.text == original.text
    
    def test_json_serialization(self):
        """Test JSON export."""
        import json
        
        source = SourceMetadata(
            corpus_name="test",
            corpus_version="1.0",
            original_id="test_001",
        )
        
        sample = CorpusSample(
            sample_id="test_sample",
            domain=Domain.ACADEMIC,
            subdomain=Subdomain.ESSAYS,
            source=source,
            text="Test.",
        )
        
        json_str = sample.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['sample_id'] == 'test_sample'
        assert parsed['domain'] == 'academic'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
