# PACT Corpus

**Provenance, Attestation, and Cognitive Trust** - A framework for AI text detection and verification.

This repository provides tools for:
1. **Collecting** human-written text across multiple domains (academic, conversational, legal, etc.)
2. **Generating** AI rewrites using multiple models (GPT-4, Claude, Gemini, Grok)
3. **Analyzing** both with SpecHO for watermark detection
4. **Building** baselines for cross-domain transfer validation

## The PACT Framework

PACT defines a **bilateral trust layer** for human-AI interaction:

```
                    ATTESTATION AXIS
                         ↑
    [Model Claims]       |       [Verified Identity]
    "I am Claude"        |       Behavioral fingerprint matches
                         |
    ─────────────────────┼─────────────────────────→ PROVENANCE AXIS
                         |
    [Unknown Origin]     |       [Cryptographic Chain]
    Could be anyone      |       Traceable lineage
```

| Layer | Human Side | AI Side | Verification |
|-------|------------|---------|--------------|
| **P**rovenance | "This text is mine" | "This output is from model X" | SpecHO watermark detection |
| **A**ttestation | "I am human" | "I match my model card" | Behavioral probes |
| **C**ognitive | "I understand this AI" | "I understand this human" | Alignment verification |
| **T**rust | "I trust this AI" | "I trust this request" | AURORA handshake |

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/pact-corpus
cd pact-corpus
pip install -r requirements.txt
```

### Collect Human Text

```bash
# List available collectors
python -m corpus.cli list

# Collect all domains
python -m corpus.cli collect --output ./data/corpus

# Collect specific domain
python -m corpus.cli collect --output ./data/corpus --domain academic --limit 1000
```

### Generate AI Rewrites

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="..."

# Generate rewrites
python -m corpus.cli generate ./data/corpus/academic.jsonl --output ./data/rewrites --runs 3
```

### Build Splits

```bash
python -m corpus.cli split ./data/corpus --stratify domain
```

## Corpus Structure

```
data/
├── corpus/                    # Human-written originals
│   ├── academic.jsonl
│   ├── conversational.jsonl
│   ├── legal.jsonl
│   └── collection_stats.json
│
├── rewrites/                  # AI-generated variants
│   ├── rewrites_academic.jsonl
│   └── generation_stats.json
│
└── splits/                    # Train/val/test
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## Sample Schema

Each sample contains:

```json
{
  "sample_id": "academic_essays_asap_1234",
  "domain": "academic",
  "subdomain": "essays",
  "text": "...",
  "source": {
    "corpus_name": "asap",
    "content_date": "2012-01-01",
    "verified_pre_llm": true
  },
  "is_original": true,
  
  // For AI variants:
  "parent_id": "academic_essays_asap_1234",
  "model": "claude-sonnet-4-20250514",
  "prompt_id": 2,
  "prompt_style": "improvement",
  
  // After SpecHO analysis:
  "specho_score": 0.73,
  "specho_confidence": 0.89,
  "echo_phonetic": 0.65,
  "echo_structural": 0.78,
  "echo_semantic": 0.71
}
```

## Domains and Sources

| Domain | Subdomains | Sources | Pre-LLM Verification |
|--------|------------|---------|---------------------|
| Academic | essays, abstracts, reviews | ASAP, Persuade 2.0, arXiv, Semantic Scholar | ✓ Dated content |
| Conversational | reddit, forums, discord | Pushshift, Stack Overflow | ✓ Timestamps |
| Legal | court_filings, legislation | CourtListener, Congress.gov | ✓ Filing dates |
| Journalistic | news, editorials | Common Crawl News | ✓ Publication dates |
| Technical | docs, tutorials | Stack Overflow, ReadTheDocs | ✓ Edit history |
| Creative | fiction, screenplays | Gutenberg, IMSDB | ✓ Publication dates |
| Business | emails, reports | Enron, SEC | ✓ Filing dates |
| Social | tweets, reviews | Twitter archive, Yelp | ✓ Post dates |

## Rewrite Prompts

The generator uses 5 prompt styles to test watermark stability:

| ID | Style | Prompt | Hypothesis |
|----|-------|--------|------------|
| 0 | minimal | "Rewrite this:" | Minimal instruction still triggers watermark |
| 1 | neutral | "Rewrite in your own words..." | Baseline detection scenario |
| 2 | improvement | "Rewrite to improve clarity..." | May increase stylistic changes |
| 3 | formal | "Rewrite in academic tone..." | Register constraints affect echoes |
| 4 | casual | "Rewrite conversationally..." | Casual register reduces echoing |

## Models

Default configuration tests:
- `gpt-4o-2024-08-06` (OpenAI)
- `claude-sonnet-4-20250514` (Anthropic)
- `gemini-1.5-pro` (Google)
- `grok-2` (xAI)
- `Llama-3.1-70B-Instruct` (Meta, via Together)

Each at temperatures 0.7 and 1.0, with 3 runs per configuration.

## Integration with SpecHO

After collection and generation:

```python
from specHO.detector import SpecHODetector
from corpus.pipeline import CollectionPipeline

# Load corpus
pipeline = CollectionPipeline("./data/corpus")

# Analyze with SpecHO
detector = SpecHODetector()

for sample in pipeline.load_samples():
    result = detector.analyze(sample.text)
    sample.specho_score = result.final_score
    sample.specho_confidence = result.confidence
    # ... save updated sample
```

## Building Domain Baselines

SpecHO needs domain-specific baselines for accurate detection:

```python
from specHO.validator import BaselineBuilder

# Build baseline from human-only samples
builder = BaselineBuilder()

for sample in pipeline.load_samples(domain=Domain.ACADEMIC):
    if sample.is_original:
        builder.add_sample(sample.text)

baseline = builder.build()
baseline.save("baselines/academic.json")
```

## Data Sources

### Academic
- **ASAP**: https://www.kaggle.com/c/asap-aes/data
- **Persuade 2.0**: https://github.com/scrosseern/persuade_corpus_2.0
- **arXiv**: https://arxiv.org/ (API)
- **Semantic Scholar**: https://api.semanticscholar.org/

### Conversational
- **Pushshift Reddit**: https://files.pushshift.io/reddit/
- **Stack Overflow**: https://archive.org/details/stackexchange

### Legal
- **CourtListener**: https://www.courtlistener.com/api/
- **Congress.gov**: https://api.congress.gov/

## Cost Estimates

For full corpus generation (16K originals × 5 models × 5 prompts × 2 temps × 3 runs):

| Model | Est. Cost |
|-------|-----------|
| GPT-4o | ~$9,000 |
| Claude | ~$9,000 |
| Gemini | ~$3,000 |
| Grok | ~$6,000 |
| Llama (Together) | ~$1,200 |
| **Total** | **~$28,000** |

Reduce by: fewer runs, fewer temperatures, fewer models, smaller corpus.

## License

MIT License. 

Note: Individual data sources have their own licenses:
- ASAP: Kaggle competition terms
- arXiv: arXiv license
- Reddit/Pushshift: Reddit user agreement
- CourtListener: Public domain (court opinions)
- Stack Overflow: CC BY-SA 4.0

## Citation

```bibtex
@software{pact_corpus,
  title = {PACT Corpus: Provenance, Attestation, and Cognitive Trust},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pact-corpus}
}
```

## Related Projects

- **SpecHO**: Echo Rule watermark detection - [link]
- **AURORA**: AI agent trust protocol - [link]
- **definitelynot.ai**: Public AI detection service - https://definitelynot.ai
