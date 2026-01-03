# PACT Corpus Data Directory

This directory stores collected and generated data.

## Structure

```
data/
├── corpus/              # Human-written originals
│   ├── academic.jsonl
│   ├── conversational.jsonl
│   ├── legal.jsonl
│   └── ...
│
├── rewrites/            # AI-generated variants
│   ├── rewrites_academic.jsonl
│   └── ...
│
└── splits/              # Train/val/test splits
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## Data Sources

Download source data and place in `~/.pact_corpus/cache/<source>/`:

### Academic
- **ASAP essays**: https://www.kaggle.com/c/asap-aes/data
  - Download `training_set_rel3.tsv`
  - Place in `~/.pact_corpus/cache/asap/`

- **Persuade 2.0**: https://github.com/scrosseern/persuade_corpus_2.0
  - Clone repo, copy CSV file
  - Place in `~/.pact_corpus/cache/persuade2/`

### Conversational  
- **Pushshift Reddit**: https://files.pushshift.io/reddit/
  - Download `RC_*.zst` files for comments
  - Download `RS_*.zst` files for submissions
  - Place in `~/.pact_corpus/cache/pushshift/`

- **Stack Overflow**: https://archive.org/details/stackexchange
  - Download and extract `Posts.xml`
  - Place in `~/.pact_corpus/cache/stackoverflow/`

### Legal
- **CourtListener**: https://www.courtlistener.com/api/
  - API access, no download needed
  - Get API token for higher rate limits

- **Congress.gov**: https://api.congress.gov/
  - Sign up for API key
  - Set `CONGRESS_API_KEY` environment variable

## File Format

All data files use JSONL (JSON Lines) format with one sample per line.

Example sample:
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
  "is_original": true
}
```
