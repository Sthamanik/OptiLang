"""
OptiLang - A Python-inspired interpreter with real-time code analysis
and optimization suggestions.
"""

# optilang/__init__.py
from optilang.executor import execute
from optilang.parser import parse

__version__ = "0.1.0"
__all__ = ["execute", "parse"]