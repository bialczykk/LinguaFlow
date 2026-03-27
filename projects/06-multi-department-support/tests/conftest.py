"""Pytest configuration for the multi-department-support test suite.

Adds the project root to sys.path so that modules like `nodes`, `graph`,
and `models` can be imported directly from anywhere in the test suite.
"""

import os
import sys

# Insert the project root (one level above this tests/ directory) so that
# all project modules are importable without a package install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
