"""
Error classes for OptiLang.

All exceptions inherit from OptiLangError for easy catching.
Each error type includes location information when available.
"""

from typing import Optional, Any


class OptiLangError(Exception):
    """Base exception for all OptiLang errors."""

    def __init__(
        self, message: str, line: Optional[int] = None, column: Optional[int] = None
    ):
        self.message = message
        self.line = line
        self.column = column

        if line is not None and column is not None:
            super().__init__(f"Line {line}, Column {column}: {message}")
        elif line is not None:
            super().__init__(f"Line {line}: {message}")
        else:
            super().__init__(message)


class LexerError(OptiLangError):
    """Exception raised for lexical errors during tokenization."""

    def __init__(
        self, message: str, line: int, column: int, illegal_char: Optional[str] = None
    ):
        self.illegal_char = illegal_char
        if illegal_char:
            full_message = f"{message} (found '{illegal_char}')"
        else:
            full_message = message
        super().__init__(full_message, line, column)


class ParserError(OptiLangError):
    """Exception raised for syntax errors during parsing."""

    def __init__(
        self,
        message: str,
        token: Optional[Any] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ):
        self.token = token

        # If token provided, extract line/column from it
        if token and hasattr(token, "line") and hasattr(token, "column"):
            line = token.line
            column = token.column
            if hasattr(token, "value"):
                message = f"{message} (got '{token.value}')"
            elif hasattr(token, "type"):
                message = f"{message} (got {token.type})"

        super().__init__(message, line, column)


class RuntimeError(OptiLangError):
    """Runtime execution errors (undefined variables, type errors, etc.)."""

    def __init__(
        self, message: str, line: Optional[int] = None, node: Optional[Any] = None
    ):
        # Extract line from AST node if available
        if node and hasattr(node, "line"):
            line = node.line
        super().__init__(message, line)


class TimeoutError(OptiLangError):
    """Execution timeout errors (infinite loops, excessive runtime)."""

    def __init__(self, timeout_seconds: float, line: Optional[int] = None):
        message = f"Execution timeout: exceeded {timeout_seconds} seconds"
        super().__init__(message, line)
        self.timeout_seconds = timeout_seconds


class NameError(RuntimeError):
    """Undefined variable or function name."""

    def __init__(self, name: str, line: Optional[int] = None):
        super().__init__(f"Name '{name}' is not defined", line)
        self.name = name


class TypeError(RuntimeError):
    """Type mismatch errors."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        expected: Optional[str] = None,
        got: Optional[str] = None,
    ):
        if expected and got:
            full_message = f"{message} (expected {expected}, got {got})"
        else:
            full_message = message
        super().__init__(full_message, line)
        self.expected = expected
        self.got = got


class ValueError(RuntimeError):
    """Invalid value errors."""

    def __init__(
        self, message: str, line: Optional[int] = None, value: Optional[Any] = None
    ):
        super().__init__(message, line)
        self.value = value


class ZeroDivisionError(RuntimeError):
    """Division by zero."""

    def __init__(self, line: Optional[int] = None):
        super().__init__("Division by zero", line)


class IndexError(RuntimeError):
    """List/string index out of range."""

    def __init__(self, index: int, length: int, line: Optional[int] = None):
        super().__init__(f"Index {index} out of range (length {length})", line)
        self.index = index
        self.length = length


class KeyError(RuntimeError):
    """Dictionary key not found."""

    def __init__(self, key: Any, line: Optional[int] = None):
        super().__init__(f"Key '{key}' not found in dictionary", line)
        self.key = key


class RecursionError(RuntimeError):
    """Maximum recursion depth exceeded."""

    def __init__(self, max_depth: int = 1000, line: Optional[int] = None):
        super().__init__(f"Maximum recursion depth ({max_depth}) exceeded", line)
        self.max_depth = max_depth


class ArgumentError(RuntimeError):
    """Function argument errors."""

    def __init__(
        self, func_name: str, expected: int, got: int, line: Optional[int] = None
    ):
        super().__init__(
            f"Function '{func_name}' expected {expected} arguments, got {got}", line
        )
        self.func_name = func_name
        self.expected = expected
        self.got = got


class AttributeError(RuntimeError):
    """Attribute not found on object."""

    def __init__(self, obj_type: str, attr_name: str, line: Optional[int] = None):
        super().__init__(f"'{obj_type}' object has no attribute '{attr_name}'", line)
        self.obj_type = obj_type
        self.attr_name = attr_name
