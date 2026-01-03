#!/usr/bin/env bash
# Convenience wrapper for PACT corpus CLI

cd "$(dirname "$0")"
PYTHONPATH=src python -m corpus.cli "$@"
