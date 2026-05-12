"""
DFA-Based Lexer for PyLite (OptiLang)

Implements a Deterministic Finite Automaton (DFA) for tokenizing PyLite source code.

Theory:
    A DFA is a 5-tuple M = (Q, Σ, δ, q0, F) where:
        Q  = finite set of states (LexerState enum)
        Σ  = input alphabet (ASCII characters)
        δ  = transition function (transition() method)
        q0 = start state (LexerState.START)
        F  = set of accepting states (states that emit tokens)

    Each character consumed drives a state transition.
    When an accepting state is reached, a token is emitted
    and the DFA resets to START.

State Diagram (simplified):
    START ──digit──────────→ IN_NUMBER ──digit──→ (loop)
          ──alpha/_─────────→ IN_IDENTIFIER
          ──"──────────────→ IN_STRING_DOUBLE
          ──'──────────────→ IN_STRING_SINGLE
          ──#──────────────→ IN_COMMENT
          ──=──────────────→ SAW_EQ ──=──→ emit EQ
                                    ──*──→ emit ASSIGN
          ──<──────────────→ SAW_LT ──=──→ emit LE
          ──>──────────────→ SAW_GT ──=──→ emit GE
          ──!──────────────→ SAW_BANG ──=──→ emit NE
          ──+──────────────→ SAW_PLUS ──=──→ emit PLUS_ASSIGN
          ──-──────────────→ SAW_MINUS ──=──→ emit MINUS_ASSIGN
          ──*──────────────→ SAW_STAR ──*──→ emit POWER
                                      ──=──→ emit MULTIPLY_ASSIGN
          ──/──────────────→ SAW_SLASH ──/──→ emit FLOOR_DIVIDE
                                       ──=──→ emit DIVIDE_ASSIGN
          ──single_char────→ emit single-char token
"""

from __future__ import annotations
from enum import Enum, auto
from typing import List, Optional, Any
from .token import Token, TokenType, KEYWORDS
from .utils.errors import LexerError

# ─────────────────────────────────────────────
#  DFA States
# ─────────────────────────────────────────────


class LexerState(Enum):
    """
    Explicit DFA states for the PyLite lexer.

    Each state represents where the automaton currently is
    during its scan of the input string.
    """

    # ── Entry point ──────────────────────────
    START = auto()  # q0: initial / reset state

    # ── Numeric literals ─────────────────────
    IN_NUMBER = auto()  # reading integer digits
    IN_FLOAT = auto()  # reading digits after decimal point

    # ── Identifiers & keywords ───────────────
    IN_IDENTIFIER = auto()  # reading alphanumeric / underscore chars

    # ── String literals ──────────────────────
    IN_STRING_DOUBLE = auto()  # inside "..." string
    IN_STRING_SINGLE = auto()  # inside '...' string
    IN_STRING_ESCAPE = auto()  # just saw backslash inside a string

    # ── Comments ─────────────────────────────
    IN_COMMENT = auto()  # inside # ... comment, skip to EOL

    # ── Newline / indentation ────────────────
    IN_NEWLINE = auto()  # processing \n, will check indentation next

    # ── Two-character operator lookahead ─────
    # Each state means "I saw the first character; peeking at next"
    SAW_EQ = auto()  # '='  → could be '==' or '='
    SAW_LT = auto()  # '<'  → could be '<=' or '<'
    SAW_GT = auto()  # '>'  → could be '>=' or '>'
    SAW_BANG = auto()  # '!'  → must be '!=' (else error)
    SAW_PLUS = auto()  # '+'  → could be '+=' or '+'
    SAW_MINUS = auto()  # '-'  → could be '-=' or '-'
    SAW_STAR = auto()  # '*'  → could be '**', '*=', or '*'
    SAW_STAR_STAR = auto()  # '**' → could be '**=' or '**'
    SAW_SLASH = auto()  # '/'  → could be '//', '//=', '/=', or '/'
    SAW_SLASH_SLASH = auto()  # '//' → could be '//=' or '//'
    SAW_PERCENT = auto()  # '%'  → could be '%=' or '%'

    # ── Indentation handling ─────────────────
    IN_INDENT = auto()  # counting spaces/tabs at line start

    # ── Terminal states ──────────────────────
    DONE = auto()  # accepted: token emitted, reset to START
    ERROR = auto()  # rejected: raise LexerError


# ─────────────────────────────────────────────
#  Lexer
# ─────────────────────────────────────────────


class Lexer:
    """
    DFA-based lexer for PyLite source code.

    Drives the automaton character-by-character through
    transition() until EOF, collecting tokens.
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.indent_stack = [0]  # tracks open indentation levels

        # ── per-token working state ──────────
        self._state = LexerState.START
        self._buffer = ""  # characters accumulated for current token
        self._tok_line = 1  # line where current token started
        self._tok_col = 1  # column where current token started
        self._string_quote = ""  # which quote char opened current string
        self._string_escape_state: Optional[LexerState] = (
            None  # state to return to after escape
        )

    # ─────────────────────────────────────────
    #  Low-level character access
    # ─────────────────────────────────────────

    def _current(self) -> Optional[str]:
        """Return the character at the current position, or None at EOF."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def _advance(self) -> Optional[str]:
        """Consume and return the current character, updating line/column."""
        if self.pos >= len(self.source):
            return None
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    # ─────────────────────────────────────────
    #  Token emission helpers
    # ─────────────────────────────────────────

    def _emit(self, token_type: TokenType, value: Any) -> Token:
        """Create a token and append it to the output list."""
        tok = Token(token_type, value, self._tok_line, self._tok_col)
        self.tokens.append(tok)
        return tok

    def _reset(self) -> None:
        """Reset working state after emitting a token."""
        self._buffer = ""
        self._state = LexerState.START

    def _mark_start(self) -> None:
        """Record the start position of the next token."""
        self._tok_line = self.line
        self._tok_col = self.column

    # ─────────────────────────────────────────
    #  Indentation helpers
    # ─────────────────────────────────────────

    def _handle_indent(self, indent_level: int) -> None:
        """
        Compare indent_level against the indent stack and emit
        INDENT / DEDENT tokens as needed.

        This is called once per non-blank, non-comment line.
        """
        if indent_level % 4 != 0:
            raise LexerError("Indentation must be a multiple of 4 spaces", self.line, 1)

        indent_units = indent_level // 4
        current_indent = self.indent_stack[-1]

        if indent_units > current_indent:
            self.indent_stack.append(indent_units)
            self._emit(TokenType.INDENT, indent_units)

        elif indent_units < current_indent:
            while self.indent_stack[-1] > indent_units:
                self.indent_stack.pop()
                self._emit(TokenType.DEDENT, self.indent_stack[-1])

            if self.indent_stack[-1] != indent_units:
                raise LexerError("Invalid indentation level", self.line, 1)

    # ─────────────────────────────────────────
    #  Core DFA transition function  δ(state, char)
    # ─────────────────────────────────────────

    def _transition(self, state: LexerState, ch: Optional[str]) -> LexerState:
        """
        Transition function δ: Q × Σ → Q

        Given the current state and input character, advance the
        automaton, optionally buffering characters or emitting tokens.

        Returns the next state.
        """

        # ── START ────────────────────────────────────────────────────────
        if state == LexerState.START:
            if ch is None:
                return LexerState.DONE  # EOF

            self._mark_start()

            if ch in (" ", "\t"):
                self._advance()
                return LexerState.START  # skip whitespace between tokens

            if ch == "#":
                self._advance()
                return LexerState.IN_COMMENT

            if ch == "\r":
                self._advance()
                return LexerState.START

            if ch == "\n":
                self._advance()
                # Only emit NEWLINE if the last token wasn't already NEWLINE
                if not self.tokens or self.tokens[-1].type != TokenType.NEWLINE:
                    self._emit(TokenType.NEWLINE, "\n")
                return LexerState.IN_INDENT

            if ch.isdigit():
                self._buffer += ch
                self._advance()
                return LexerState.IN_NUMBER

            if ch.isalpha() or ch == "_":
                self._buffer += ch
                self._advance()
                return LexerState.IN_IDENTIFIER

            if ch == '"':
                self._string_quote = '"'
                self._advance()
                return LexerState.IN_STRING_DOUBLE

            if ch == "'":
                self._string_quote = "'"
                self._advance()
                return LexerState.IN_STRING_SINGLE

            # Two-char operator lead characters
            if ch == "=":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_EQ
            if ch == "<":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_LT
            if ch == ">":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_GT
            if ch == "!":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_BANG
            if ch == "+":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_PLUS
            if ch == "-":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_MINUS
            if ch == "*":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_STAR
            if ch == "/":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_SLASH

            # Single-character tokens
            single = {
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
            if ch == "%":
                self._buffer += ch
                self._advance()
                return LexerState.SAW_PERCENT
            if ch in single:
                self._advance()
                self._emit(single[ch], ch)
                return LexerState.START

            raise LexerError(f"Unexpected character: '{ch}'", self.line, self.column)

        # ── IN_INDENT ────────────────────────────────────────────────────
        # Count leading whitespace at the start of a new line, then
        # delegate to _handle_indent() before processing the real token.
        if state == LexerState.IN_INDENT:
            indent_level = 0

            # Consume all leading spaces / tabs
            while self._current() in (" ", "\t"):
                c = self._current()
                indent_level += 1 if c == " " else 4  # tab = 4 spaces
                self._advance()

            next_ch = self._current()

            # Blank line or comment line – skip indentation check entirely
            if next_ch in ("\n", "\r", "#", None):
                return LexerState.START

            self._handle_indent(indent_level)
            return LexerState.START

        # ── IN_NUMBER ────────────────────────────────────────────────────
        if state == LexerState.IN_NUMBER:
            if ch and ch.isdigit():
                self._buffer += ch
                self._advance()
                return LexerState.IN_NUMBER

            if ch == ".":
                # Peek at the char after the dot
                next_ch = (
                    self.source[self.pos + 1]
                    if self.pos + 1 < len(self.source)
                    else None
                )
                if next_ch and next_ch.isdigit():
                    # Valid float: e.g. "3.14"
                    self._buffer += ch
                    self._advance()
                    return LexerState.IN_FLOAT
                # Trailing dot is invalid: e.g. "1." raises an error
                raise LexerError(
                    f"Invalid number format: '{self._buffer}.' ends with decimal point",
                    self._tok_line,
                    self._tok_col,
                )

            # Accepting state: emit integer
            self._emit(TokenType.NUMBER, int(self._buffer))
            self._reset()
            return LexerState.START

        # ── IN_FLOAT ─────────────────────────────────────────────────────
        if state == LexerState.IN_FLOAT:
            if ch and ch.isdigit():
                self._buffer += ch
                self._advance()
                return LexerState.IN_FLOAT

            if ch == ".":
                raise LexerError(
                    "Invalid number: multiple decimal points",
                    self._tok_line,
                    self._tok_col,
                )

            # Accepting state: emit float
            self._emit(TokenType.NUMBER, float(self._buffer))
            self._reset()
            return LexerState.START

        # ── IN_IDENTIFIER ────────────────────────────────────────────────
        if state == LexerState.IN_IDENTIFIER:
            if ch and (ch.isalnum() or ch == "_"):
                self._buffer += ch
                self._advance()
                return LexerState.IN_IDENTIFIER

            # Accepting state: keyword or identifier?
            word = self._buffer
            tok_type = KEYWORDS.get(word, TokenType.IDENTIFIER)
            value: Any = word

            if tok_type == TokenType.TRUE:
                value = True
            elif tok_type == TokenType.FALSE:
                value = False
            elif tok_type == TokenType.NONE:
                value = None

            self._emit(tok_type, value)
            self._reset()
            return LexerState.START

        # ── IN_STRING_DOUBLE / IN_STRING_SINGLE ──────────────────────────
        if state in (LexerState.IN_STRING_DOUBLE, LexerState.IN_STRING_SINGLE):
            if ch is None:
                raise LexerError(
                    "Unterminated string literal", self._tok_line, self._tok_col
                )
            if ch == "\\":
                # Enter escape sub-state, remembering which string state to return to
                self._string_escape_state = state
                self._advance()
                return LexerState.IN_STRING_ESCAPE

            if ch == self._string_quote:
                # Accepting state: emit the completed string
                self._advance()  # consume closing quote
                self._emit(TokenType.STRING, self._buffer)
                self._reset()
                return LexerState.START

            self._buffer += ch
            self._advance()
            return state  # stay in same string state

        # ── IN_STRING_ESCAPE ─────────────────────────────────────────────
        if state == LexerState.IN_STRING_ESCAPE:
            escape_map = {
                "n": "\n",
                "t": "\t",
                "r": "\r",
                "\\": "\\",
                "'": "'",
                '"': '"',
            }
            self._buffer += escape_map.get(ch, ch) if ch else ""
            if ch:
                self._advance()
            # Return to whichever string state we came from
            return self._string_escape_state  # type: ignore[return-value]

        # ── IN_COMMENT ───────────────────────────────────────────────────
        if state == LexerState.IN_COMMENT:
            if ch is None or ch == "\n":
                return LexerState.START  # comment ends at EOL (don't consume \n)
            self._advance()
            return LexerState.IN_COMMENT

        # ── TWO-CHARACTER OPERATOR STATES ─────────────────────────────────

        # SAW_EQ: already buffered '='
        if state == LexerState.SAW_EQ:
            if ch == "=":
                self._advance()
                self._emit(TokenType.EQ, "==")
            else:
                self._emit(TokenType.ASSIGN, "=")
            self._reset()
            return LexerState.START

        # SAW_LT: already buffered '<'
        if state == LexerState.SAW_LT:
            if ch == "=":
                self._advance()
                self._emit(TokenType.LE, "<=")
            else:
                self._emit(TokenType.LT, "<")
            self._reset()
            return LexerState.START

        # SAW_GT: already buffered '>'
        if state == LexerState.SAW_GT:
            if ch == "=":
                self._advance()
                self._emit(TokenType.GE, ">=")
            else:
                self._emit(TokenType.GT, ">")
            self._reset()
            return LexerState.START

        # SAW_BANG: already buffered '!' — only valid next char is '='
        if state == LexerState.SAW_BANG:
            if ch == "=":
                self._advance()
                self._emit(TokenType.NE, "!=")
                self._reset()
                return LexerState.START
            raise LexerError(
                f"Expected '=' after '!' but got '{ch}'", self._tok_line, self._tok_col
            )

        # SAW_PLUS: already buffered '+'
        if state == LexerState.SAW_PLUS:
            if ch == "=":
                self._advance()
                self._emit(TokenType.PLUS_ASSIGN, "+=")
            else:
                self._emit(TokenType.PLUS, "+")
            self._reset()
            return LexerState.START

        # SAW_MINUS: already buffered '-'
        if state == LexerState.SAW_MINUS:
            if ch == "=":
                self._advance()
                self._emit(TokenType.MINUS_ASSIGN, "-=")
            else:
                self._emit(TokenType.MINUS, "-")
            self._reset()
            return LexerState.START

        # SAW_STAR: already buffered '*' — could be '**', '**=', '*=', or '*'
        if state == LexerState.SAW_STAR:
            if ch == "*":
                self._advance()
                return LexerState.SAW_STAR_STAR
            elif ch == "=":
                self._advance()
                self._emit(TokenType.MULTIPLY_ASSIGN, "*=")
            else:
                self._emit(TokenType.MULTIPLY, "*")
            self._reset()
            return LexerState.START

        # SAW_STAR_STAR: already buffered '**' — could be '**=' or '**'
        if state == LexerState.SAW_STAR_STAR:
            if ch == "=":
                self._advance()
                self._emit(TokenType.POWER_ASSIGN, "**=")
            else:
                self._emit(TokenType.POWER, "**")
            self._reset()
            return LexerState.START

        # SAW_SLASH: already buffered '/' — could be '//', '//=', '/=', or '/'
        if state == LexerState.SAW_SLASH:
            if ch == "/":
                self._advance()
                return LexerState.SAW_SLASH_SLASH
            elif ch == "=":
                self._advance()
                self._emit(TokenType.DIVIDE_ASSIGN, "/=")
            else:
                self._emit(TokenType.DIVIDE, "/")
            self._reset()
            return LexerState.START

        # SAW_SLASH_SLASH: already buffered '//' — could be '//=' or '//'
        if state == LexerState.SAW_SLASH_SLASH:
            if ch == "=":
                self._advance()
                self._emit(TokenType.FLOOR_DIVIDE_ASSIGN, "//=")
            else:
                self._emit(TokenType.FLOOR_DIVIDE, "//")
            self._reset()
            return LexerState.START

        # SAW_PERCENT: already buffered '%' — could be '%=' or '%'
        if state == LexerState.SAW_PERCENT:
            if ch == "=":
                self._advance()
                self._emit(TokenType.MODULO_ASSIGN, "%=")
            else:
                self._emit(TokenType.MODULO, "%")
            self._reset()
            return LexerState.START

        # Should never reach here
        raise LexerError(
            f"DFA reached undefined state: {state}", self.line, self.column
        )

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        """
        Run the DFA over the entire source string.

        Drives _transition() in a loop until EOF,
        then closes any open indentation levels and
        appends the EOF token.

        Returns:
            List[Token]: the complete token stream
        """
        self.tokens = []
        self._state = LexerState.START
        self._buffer = ""

        # ── Main DFA loop ────────────────────
        while True:
            ch = self._current()

            # Drive one transition
            next_state = self._transition(self._state, ch)
            self._state = next_state

            # DONE is the accepting halt state — stop immediately
            if self._state == LexerState.DONE:
                break

            # Safety: ERROR state means unrecoverable scan failure
            if self._state == LexerState.ERROR:
                raise LexerError("DFA entered ERROR state", self.line, self.column)

        # ── Close remaining indentation levels ───
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self._emit(TokenType.DEDENT, self.indent_stack[-1])

        # ── EOF token ────────────────────────────
        self._emit(TokenType.EOF, None)

        return self.tokens

    def tokenize_to_string(self) -> str:
        """
        Tokenize and return a human-readable token listing.
        Useful for debugging and unit tests.
        """
        tokens = self.tokenize()
        return "\n".join(str(tok) for tok in tokens)


# ─────────────────────────────────────────────
#  Module-level convenience function
# ─────────────────────────────────────────────


def tokenize(source: str) -> List[Token]:
    """
    Tokenize PyLite source code using the DFA-based lexer.

    Args:
        source: Raw PyLite source code string.

    Returns:
        List[Token]: Ordered list of tokens including EOF.

    Raises:
        LexerError: On any invalid character or malformed token.

    Example:
        >>> from optilang.lexer import tokenize
        >>> tokens = tokenize("x = 5 + 3")
        >>> [t.type for t in tokens]
        [IDENTIFIER, ASSIGN, NUMBER, PLUS, NUMBER, EOF]
    """
    return Lexer(source).tokenize()
