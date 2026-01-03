#!/usr/bin/env python3
"""
Quick validation script for return_rate metric.

Checks if 0.0 values are legitimate or an edge case artifact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specHO"))

from fingerprint.trajectory import TrajectoryAnalyzer


def test_return_rate():
    """Test return_rate on different text lengths."""

    analyzer = TrajectoryAnalyzer(embedding_model="all-MiniLM-L6-v2")

    test_cases = [
        ("Single sentence", "This is a single sentence about machine learning."),
        ("Two sentences different topics",
         "Machine learning is transforming industries. "
         "The weather today is quite pleasant."),
        ("Three sentences with return",
         "Machine learning is transforming industries. "
         "The weather today is quite pleasant. "
         "Deep learning, a subset of machine learning, is particularly powerful."),
        ("Long text with repetition",
         "Artificial intelligence has revolutionized many fields. "
         "Machine learning is a subset of AI that enables systems to learn. "
         "Deep learning is a subset of machine learning using neural networks. "
         "These AI techniques are becoming increasingly important. "
         "The field of artificial intelligence continues to grow rapidly. "
         "Machine learning applications are everywhere today."),
    ]

    print("=" * 80)
    print("RETURN_RATE VALIDATION")
    print("=" * 80)

    for name, text in test_cases:
        result = analyzer.analyze(text)

        print(f"\n{name}:")
        print(f"  Text length: {len(text)} chars")
        print(f"  Sentences: {text.count('.')}")
        print(f"  return_rate: {result.return_rate:.3f}")

        if result.return_rate == 0.0:
            print(f"  ⚠️  Zero return rate - check if legitimate")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("- return_rate = 0.0 is EXPECTED for:")
    print("  • Single sentence texts (can't return to previous)")
    print("  • Texts with all unique topics (no semantic revisitation)")
    print("- return_rate > 0.0 indicates:")
    print("  • Topic revisitation (e.g., 'AI... weather... AI again')")
    print("  • Conceptual loops in discourse")
    print("=" * 80)


if __name__ == "__main__":
    test_return_rate()
