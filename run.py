#!/usr/bin/env python3
"""Convenience entry point - runs the demo or forwards to main."""

import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).parent
sys.path.insert(0, str(root))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        from scripts.run_demo import main
        main()
    else:
        from src.main import main
        main()
