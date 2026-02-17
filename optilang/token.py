from dataclasses import dataclass
from typing import Any


@dataclass
class Token:
    """Represents a single token"""

    type: str  # Token type (e.g., 'NUMBER', 'IDENTIFIER')
    value: Any  # Token value
    line: int  # Line num in source code
    column: int  # Column num in source code

    def __repr__(self) -> str:
        """For easy debugging"""
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.column})"


# Token types
class TokenType:
    NUMBER = "NUMBER"
    STRING = "STRING"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NONE = "NONE"

    # Identifiers and Keywords
    IDENTIFIER = "IDENTIFIER"

    # Keywords
    IF = "IF"
    ELSE = "ELSE"
    ELIF = "ELIF"
    WHILE = "WHILE"
    FOR = "FOR"
    IN = "IN"
    DEF = "DEF"
    RETURN = "RETURN"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    PASS = "PASS"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    TRY = "TRY"
    EXCEPT = "EXCEPT"
    FINALLY = "FINALLY"
    LAMBDA = "LAMBDA"

    # Operations
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    POWER = "POWER"
    FLOOR_DIVIDE = "FLOOR_DIVIDE"

    # Comparision operators
    EQ = "EQ"  # ==
    NE = "NE"  # !=
    LT = "LT"  # <
    LE = "LE"  # <=
    GT = "GT"  # >
    GE = "GE"  # >=

    # Assignment
    ASSIGN = "ASSIGN"  # =

    # Augmented assignment
    PLUS_ASSIGN = "PLUS_ASSIGN"  # +=
    MINUS_ASSIGN = "MINUS_ASSIGN"  # -=
    MULTIPLY_ASSIGN = "MULTIPLY_ASSIGN"  # *=
    DIVIDE_ASSIGN = "DIVIDE_ASSIGN"  # /=

    # Delimiters
    LPAREN = "LPAREN"  # (
    RPAREN = "RPAREN"  # )
    LBRACKET = "LBRACKET"  # [
    RBRACKET = "RBRACKET"  # ]
    LBRACE = "LBRACE"  # {
    RBRACE = "RBRACE"  # }
    COMMA = "COMMA"  # ,
    COLON = "COLON"  # :
    DOT = "DOT"  # .

    # Special
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    EOF = "EOF"

    # All keywords
    KEYWORDS = {
        "if": IF,
        "else": ELSE,
        "elif": ELIF,
        "while": WHILE,
        "for": FOR,
        "in": IN,
        "def": DEF,
        "return": RETURN,
        "break": BREAK,
        "continue": CONTINUE,
        "pass": PASS,
        "and": AND,
        "or": OR,
        "not": NOT,
        "True": TRUE,
        "False": FALSE,
        "None": NONE,
        "try": TRY,
        "except": EXCEPT,
        "finally": FINALLY,
        "lambda": LAMBDA,
    }
