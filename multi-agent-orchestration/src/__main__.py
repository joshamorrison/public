#!/usr/bin/env python3
"""
Multi-Agent Orchestration Platform - Main Entry Point
"""

import sys
import os
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the quick_start demo from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("Multi-Agent Orchestration Platform")
    print("=" * 50)
    print("For quick start demo, run: python quick_start.py")
    print("For platform usage examples, see examples/ directory")