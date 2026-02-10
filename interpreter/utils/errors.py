"""
Error classes for OptiLang.
"""


class OptiLangError(Exception):
    """Base exception for OptiLang."""
    pass


class LexerError(OptiLangError):
    """Lexer-related errors."""
    pass


class ParserError(OptiLangError):
    """Parser-related errors."""
    pass


class RuntimeError(OptiLangError):
    """Runtime execution errors."""
    pass


class TimeoutError(OptiLangError):
    """Execution timeout errors."""
    pass