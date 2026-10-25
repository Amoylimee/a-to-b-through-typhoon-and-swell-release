#!/usr/bin/env python3
"""Run the E008 monthly aggregation and its validation gates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monthly_release import main

if __name__ == "__main__":
    main()
