"""
Converts source code into tokens
"""

from typing import List, Optional, Any
from .token import Token, TokenType
from .utils.errors import LexerError


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0  # Current position in source
        self.line = 1  # Current line number
        self.column = 1  # Current column number
        self.tokens: List[Token] = []
        self.indent_stack = [0]  # Stack to track indentation levels

    def current_char(self) -> Optional[str]:
        """Get the current character without advancing"""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at a character ahead without advancing"""
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]

    def advance(self) -> Optional[str]:
        """Move to the next character and return it"""
        if self.pos >= len(self.source):
            return None

        char = self.source[self.pos]
        self.pos += 1

        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def skip_whitespace(self) -> None:
        """Skip spaces and tabs (but not newlines)"""
        while self.current_char() in (" ", "\t"):
            self.advance()

    def skip_comment(self) -> None:
        """Skip single-line comments starting with #"""
        char = self.current_char()
        if char == "#":
            while True:
                char = self.current_char()
                if not char or char == "\n":
                    break
                self.advance()

    def read_number(self) -> Token:
        """Read a number token (integer or float)"""
        start_line = self.line
        start_column = self.column
        num_str = ""
        has_dot = False

        char = self.current_char()
        while char and (char.isdigit() or char == "."):
            if char == ".":
                if has_dot:
                    raise LexerError(
                        "Invalid number format: multiple decimal points",
                        start_line,
                        start_column,
                    )
                has_dot = True
            num_str += char
            self.advance()
            char = self.current_char()

        # Check if number ends with a dot
        if num_str.endswith("."):
            raise LexerError(
                "Invalid number format: ends with decimal point",
                start_line,
                start_column,
            )

        value = float(num_str) if has_dot else int(num_str)
        return Token(TokenType.NUMBER, value, start_line, start_column)

    def read_string(self, quote: str) -> Token:
        """Read a string token (single or double quoted)"""
        start_line = self.line
        start_column = self.column

        self.advance()  # Skip opening quote
        string_value = ""

        char = self.current_char()
        while char and char != quote:
            if char == "\\":
                self.advance()
                # Handle escape sequences
                escape_char = self.current_char()
                if escape_char == "n":
                    string_value += "\n"
                elif escape_char == "t":
                    string_value += "\t"
                elif escape_char == "r":
                    string_value += "\r"
                elif escape_char == "\\":
                    string_value += "\\"
                elif escape_char == quote:
                    string_value += quote
                elif escape_char:
                    string_value += escape_char
                self.advance()
            else:
                string_value += char
                self.advance()
            char = self.current_char()

        if not char:
            raise LexerError("Unterminated string", start_line, start_column)

        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, string_value, start_line, start_column)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword token"""
        start_line = self.line
        start_column = self.column
        identifier = ""

        char = self.current_char()
        while char and (char.isalnum() or char == "_"):
            identifier += char
            self.advance()
            char = self.current_char()

        # Check if it's a keyword
        token_type = TokenType.KEYWORDS.get(identifier, TokenType.IDENTIFIER)

        # For boolean and None literals, convert to actual values
        value: Any
        if token_type == TokenType.TRUE:
            value = True
        elif token_type == TokenType.FALSE:
            value = False
        elif token_type == TokenType.NONE:
            value = None
        else:
            value = identifier

        return Token(token_type, value, start_line, start_column)

    def handle_indentation(self, line_start: bool) -> List[Token]:
        """
        Handle indentation at the start of a line
        """
        if not line_start:
            return []

        indent_level = 0
        start_column = self.column

        # Count spaces (4 spaces = 1 indent level)
        char = self.current_char()
        while char in (" ", "\t"):
            if char == " ":
                indent_level += 1
            else:  # Tab counts as 4 spaces
                indent_level += 4
            self.advance()
            char = self.current_char()

        # Normalize to indent units (4 spaces = 1 indent)
        indent_units = indent_level // 4

        # Check if indentation is valid (must be multiple of 4)
        if indent_level % 4 != 0:
            raise LexerError(
                "Indentation must be multiple of 4 spaces", self.line, start_column
            )

        tokens = []
        current_indent = self.indent_stack[-1]

        if indent_units > current_indent:
            # Indent
            self.indent_stack.append(indent_units)
            tokens.append(
                Token(TokenType.INDENT, indent_units, self.line, start_column)
            )
        elif indent_units < current_indent:
            # Dedent (possibly multiple levels)
            while self.indent_stack[-1] > indent_units:
                self.indent_stack.pop()
                tokens.append(
                    Token(
                        TokenType.DEDENT, self.indent_stack[-1], self.line, start_column
                    )
                )

            # Check if we dedented to a valid level
            if self.indent_stack[-1] != indent_units:
                raise LexerError("Invalid indentation level", self.line, start_column)

        return tokens

    def tokenize(self) -> List[Token]:
        """
        Main tokenization method
        Converts entire source code into tokens

        Returns list of tokens
        """
        self.tokens = []
        line_start = True  # Track if we're at the start of a line

        while self.pos < len(self.source):
            # Skip whitespace (except at line start for indentation)
            if not line_start:
                self.skip_whitespace()

            # Skip comments
            self.skip_comment()

            char = self.current_char()

            if not char:
                break

            # Handle indentation at line start
            if line_start and char in (" ", "\t"):
                indent_tokens = self.handle_indentation(True)
                self.tokens.extend(indent_tokens)
                char = self.current_char()
                line_start = False
            elif line_start and char not in ("\n", "\r"):
                # No indentation, but not a blank line
                indent_tokens = self.handle_indentation(True)
                self.tokens.extend(indent_tokens)
                line_start = False

            if not char:
                break

            start_line = self.line
            start_column = self.column

            # Newline
            if char == "\n":
                # Only add NEWLINE if previous token wasn't NEWLINE
                if not self.tokens or self.tokens[-1].type != TokenType.NEWLINE:
                    self.tokens.append(
                        Token(TokenType.NEWLINE, "\n", start_line, start_column)
                    )
                self.advance()
                line_start = True
                continue

            # Skip carriage return
            if char == "\r":
                self.advance()
                continue

            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue

            # Strings
            if char in ('"', "'"):
                self.tokens.append(self.read_string(char))
                continue

            # Identifiers and keywords
            if char.isalpha() or char == "_":
                self.tokens.append(self.read_identifier())
                continue

            # Two-character operators
            if char == "=" and self.peek_char() == "=":
                self.tokens.append(Token(TokenType.EQ, "==", start_line, start_column))
                self.advance()
                self.advance()
                continue

            if char == "!" and self.peek_char() == "=":
                self.tokens.append(Token(TokenType.NE, "!=", start_line, start_column))
                self.advance()
                self.advance()
                continue

            if char == "<" and self.peek_char() == "=":
                self.tokens.append(Token(TokenType.LE, "<=", start_line, start_column))
                self.advance()
                self.advance()
                continue

            if char == ">" and self.peek_char() == "=":
                self.tokens.append(Token(TokenType.GE, ">=", start_line, start_column))
                self.advance()
                self.advance()
                continue

            if char == "*" and self.peek_char() == "*":
                self.tokens.append(
                    Token(TokenType.POWER, "**", start_line, start_column)
                )
                self.advance()
                self.advance()
                continue

            if char == "/" and self.peek_char() == "/":
                self.tokens.append(
                    Token(TokenType.FLOOR_DIVIDE, "//", start_line, start_column)
                )
                self.advance()
                self.advance()
                continue

            if char == "+" and self.peek_char() == "=":
                self.tokens.append(
                    Token(TokenType.PLUS_ASSIGN, "+=", start_line, start_column)
                )
                self.advance()
                self.advance()
                continue

            if char == "-" and self.peek_char() == "=":
                self.tokens.append(
                    Token(TokenType.MINUS_ASSIGN, "-=", start_line, start_column)
                )
                self.advance()
                self.advance()
                continue

            if char == "*" and self.peek_char() == "=":
                self.tokens.append(
                    Token(TokenType.MULTIPLY_ASSIGN, "*=", start_line, start_column)
                )
                self.advance()
                self.advance()
                continue

            if char == "/" and self.peek_char() == "=":
                self.tokens.append(
                    Token(TokenType.DIVIDE_ASSIGN, "/=", start_line, start_column)
                )
                self.advance()
                self.advance()
                continue

            # Single-character operators and delimiters
            single_char_tokens = {
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.MULTIPLY,
                "/": TokenType.DIVIDE,
                "%": TokenType.MODULO,
                "=": TokenType.ASSIGN,
                "<": TokenType.LT,
                ">": TokenType.GT,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                "[": TokenType.LBRACKET,
                "]": TokenType.RBRACKET,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                ",": TokenType.COMMA,
                ":": TokenType.COLON,
                ".": TokenType.DOT,
            }

            if char in single_char_tokens:
                self.tokens.append(
                    Token(single_char_tokens[char], char, start_line, start_column)
                )
                self.advance()
                continue

            # Unknown character
            raise LexerError(
                f"Unexpected character: '{char}'", start_line, start_column
            )

        # Add remaining DEDENT tokens for unclosed indents
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(
                Token(TokenType.DEDENT, self.indent_stack[-1], self.line, self.column)
            )

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))

        return self.tokens

    def tokenize_to_string(self) -> str:
        """
        Tokenize and return a formatted string representation
        Useful for debugging
        """
        tokens = self.tokenize()
        return "\n".join(str(token) for token in tokens)


def tokenize(source: str) -> List[Token]:
    """
    Function to tokenize source code

    Args:
        source: source code

    Returns: list of tokens
    """
    lexer = Lexer(source)
    return lexer.tokenize()
