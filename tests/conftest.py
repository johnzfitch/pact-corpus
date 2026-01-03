"""
Pytest configuration for PACT Corpus tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_text():
    """Sample human-written text for testing."""
    return """
    The rapid advancement of artificial intelligence has fundamentally 
    transformed how we interact with technology. Machine learning algorithms 
    now power everything from search engines to medical diagnostics, 
    demonstrating both remarkable capabilities and significant limitations.
    
    However, as these systems become more sophisticated, questions about 
    transparency and accountability become increasingly urgent. How do we 
    ensure that AI systems make decisions we can understand and trust?
    """


@pytest.fixture
def sample_academic_text():
    """Sample academic essay text."""
    return """
    This essay examines the relationship between economic development and 
    environmental sustainability in developing nations. The central argument 
    is that traditional models of industrialization, which prioritize rapid 
    economic growth over environmental protection, are ultimately 
    self-defeating. Evidence from recent case studies in Southeast Asia 
    demonstrates that countries which invest early in sustainable practices 
    actually achieve stronger long-term economic performance.
    
    The methodology employed in this analysis combines quantitative economic 
    indicators with qualitative assessments of environmental impact. By 
    triangulating these data sources, we can develop a more nuanced 
    understanding of the complex tradeoffs involved in development policy.
    """


@pytest.fixture
def sample_reddit_text():
    """Sample conversational Reddit text."""
    return """
    Honestly I've been thinking about this a lot lately. The whole situation 
    with my roommate is getting ridiculous - like who leaves dishes in the 
    sink for THREE WEEKS? I've tried talking to them about it but they just 
    shrug and say they'll get to it eventually.
    
    Anyone else dealt with something like this? Starting to wonder if I 
    should just move out when the lease is up. The rent is great but my 
    sanity is worth something too lol
    """


@pytest.fixture  
def temp_corpus_dir(tmp_path):
    """Create a temporary corpus directory."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    
    (corpus_dir / "academic.jsonl").touch()
    (corpus_dir / "conversational.jsonl").touch()
    
    return corpus_dir
