"""
OptiLang - A Python-inspired interpreter with real-time code analysis
and optimization suggestions.
"""

# optilang/__init__.py
from optilang.executor import execute
from optilang.parser import parse
from optilang.scoring import calculate_score, ScoreReport

__version__ = "0.2.0"
__all__ = ["execute", "parse", "calculate_score", 'ScoreReport']
