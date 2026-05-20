#!/usr/bin/env python3
"""CI/local entry point — run from repo root: python run_feature_pipeline.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.feature_pipeline import run

if __name__ == "__main__":
    run()
