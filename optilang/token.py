from dataclasses import dataclass
from enum import Enum
from typing import Any


class TokenType(str, Enum):
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

    # Comparison operators
    EQ = "EQ"
    NE = "NE"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"

    # Assignment
    ASSIGN = "ASSIGN"

    # Augmented assignment
    PLUS_ASSIGN = "PLUS_ASSIGN"
    MINUS_ASSIGN = "MINUS_ASSIGN"
    MULTIPLY_ASSIGN = "MULTIPLY_ASSIGN"
    DIVIDE_ASSIGN = "DIVIDE_ASSIGN"
    FLOOR_DIVIDE_ASSIGN = "FLOOR_DIVIDE_ASSIGN"
    MODULO_ASSIGN = "MODULO_ASSIGN"
    POWER_ASSIGN = "POWER_ASSIGN"

    # Delimiters
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    COLON = "COLON"
    DOT = "DOT"

    # Special
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    EOF = "EOF"


KEYWORDS: dict[str, TokenType] = {
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "elif": TokenType.ELIF,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "def": TokenType.DEF,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "pass": TokenType.PASS,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "None": TokenType.NONE,
    "try": TokenType.TRY,
    "except": TokenType.EXCEPT,
    "finally": TokenType.FINALLY,
    "lambda": TokenType.LAMBDA,
}


@dataclass
class Token:
    """Represents a single token."""

    type: TokenType
    value: Any
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.column})"
