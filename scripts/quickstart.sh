#!/usr/bin/env bash
#
# PACT Corpus Quickstart
# 
# Sets up the environment and runs a test collection.
#

set -e

echo "======================================"
echo "PACT Corpus - Quickstart Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -e ".[dev]"

# Download spaCy model (for SpecHO integration later)
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm -q 2>/dev/null || true

# Create data directories
mkdir -p data/corpus data/rewrites data/splits

# Run tests
echo ""
echo "Running tests..."
pytest tests/ -v --tb=short

# List available collectors
echo ""
echo "======================================"
echo "Available Collectors:"
echo "======================================"
python -m corpus.cli list

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Download data sources:"
echo "   - ASAP essays: https://www.kaggle.com/c/asap-aes/data"
echo "   - Pushshift Reddit: https://files.pushshift.io/reddit/"
echo "   - Place in ~/.pact_corpus/cache/<source>/"
echo ""
echo "2. Set API keys for generation:"
echo "   export OPENAI_API_KEY='sk-...'"
echo "   export ANTHROPIC_API_KEY='sk-ant-...'"
echo "   export GOOGLE_API_KEY='...'"
echo "   export XAI_API_KEY='...'"
echo ""
echo "3. Collect human text:"
echo "   python -m corpus.cli collect --output ./data/corpus --limit 100"
echo ""
echo "4. Generate AI rewrites:"
echo "   python -m corpus.cli generate ./data/corpus/academic.jsonl -o ./data/rewrites"
echo ""
