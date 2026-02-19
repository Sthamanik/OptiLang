"""
Comprehensive test suite for OptiLang Lexer (optilang/lexer.py)

Tests cover:
- Number literals (integer, float, invalid)
- String literals (single/double quoted, escape sequences, unterminated)
- Boolean and None literals
- All keywords
- All operators (single and two-character)
- Identifiers
- Indentation (INDENT / DEDENT)
- Comments
- Newlines
- Unknown/illegal characters
- Token metadata (line, column)
"""

import pytest
from optilang.lexer import tokenize
from optilang.token import TokenType
from optilang.utils.errors import LexerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_types(source: str):
    """Return just the token types for a source string."""
    return [t.type for t in tokenize(source)]


def first_token(source: str):
    """Return the first token of the source."""
    return tokenize(source)[0]


# ===========================================================================
# 1. NUMBER LITERALS
# ===========================================================================

class TestNumbers:

    def test_integer_type(self):
        token = first_token("42")
        assert token.type == TokenType.NUMBER

    def test_integer_value(self):
        token = first_token("42")
        assert token.value == 42

    def test_integer_is_int_not_float(self):
        token = first_token("10")
        assert isinstance(token.value, int)

    def test_zero(self):
        token = first_token("0")
        assert token.value == 0

    def test_float_type(self):
        token = first_token("3.14")
        assert token.type == TokenType.NUMBER

    def test_float_value(self):
        token = first_token("3.14")
        assert abs(token.value - 3.14) < 1e-9

    def test_float_is_float_not_int(self):
        token = first_token("3.14")
        assert isinstance(token.value, float)

    def test_float_starting_with_zero(self):
        token = first_token("0.5")
        assert abs(token.value - 0.5) < 1e-9

    def test_multiple_decimal_points_raises_error(self):
        with pytest.raises(LexerError):
            tokenize("3.1.4")

    def test_float_ending_with_decimal_raises_error(self):
        with pytest.raises(LexerError):
            tokenize("3.")

    def test_large_integer(self):
        token = first_token("99999")
        assert token.value == 99999

    def test_number_line_number(self):
        token = first_token("42")
        assert token.line == 1

    def test_number_column_number(self):
        token = first_token("42")
        assert token.column == 1


# ===========================================================================
# 2. STRING LITERALS
# ===========================================================================

class TestStrings:

    def test_double_quoted_string_type(self):
        token = first_token('"hello"')
        assert token.type == TokenType.STRING

    def test_double_quoted_string_value(self):
        token = first_token('"hello"')
        assert token.value == "hello"

    def test_single_quoted_string_type(self):
        token = first_token("'hello'")
        assert token.type == TokenType.STRING

    def test_single_quoted_string_value(self):
        token = first_token("'hello'")
        assert token.value == "hello"

    def test_empty_string(self):
        token = first_token('""')
        assert token.value == ""

    def test_string_with_spaces(self):
        token = first_token('"hello world"')
        assert token.value == "hello world"

    def test_escape_newline(self):
        token = first_token('"hello\\nworld"')
        assert token.value == "hello\nworld"

    def test_escape_tab(self):
        token = first_token('"hello\\tworld"')
        assert token.value == "hello\tworld"

    def test_escape_backslash(self):
        token = first_token('"hello\\\\world"')
        assert token.value == "hello\\world"

    def test_escape_double_quote_inside_double_quoted(self):
        token = first_token('"say \\"hello\\""')
        assert token.value == 'say "hello"'

    def test_unterminated_string_raises_error(self):
        with pytest.raises(LexerError):
            tokenize('"unterminated')

    def test_unterminated_single_quote_raises_error(self):
        with pytest.raises(LexerError):
            tokenize("'unterminated")

    def test_string_with_numbers(self):
        token = first_token('"abc123"')
        assert token.value == "abc123"


# ===========================================================================
# 3. BOOLEAN AND NONE LITERALS
# ===========================================================================

class TestBooleanAndNone:

    def test_true_type(self):
        token = first_token("True")
        assert token.type == TokenType.TRUE

    def test_true_value(self):
        token = first_token("True")
        assert token.value is True

    def test_false_type(self):
        token = first_token("False")
        assert token.type == TokenType.FALSE

    def test_false_value(self):
        token = first_token("False")
        assert token.value is False

    def test_none_type(self):
        token = first_token("None")
        assert token.type == TokenType.NONE

    def test_none_value(self):
        token = first_token("None")
        assert token.value is None

    def test_true_lowercase_is_identifier(self):
        token = first_token("true")
        assert token.type == TokenType.IDENTIFIER

    def test_false_lowercase_is_identifier(self):
        token = first_token("false")
        assert token.type == TokenType.IDENTIFIER


# ===========================================================================
# 4. KEYWORDS
# ===========================================================================

class TestKeywords:

    @pytest.mark.parametrize("keyword, expected_type", [
        ("if",       TokenType.IF),
        ("else",     TokenType.ELSE),
        ("elif",     TokenType.ELIF),
        ("while",    TokenType.WHILE),
        ("for",      TokenType.FOR),
        ("in",       TokenType.IN),
        ("def",      TokenType.DEF),
        ("return",   TokenType.RETURN),
        ("break",    TokenType.BREAK),
        ("continue", TokenType.CONTINUE),
        ("pass",     TokenType.PASS),
        ("and",      TokenType.AND),
        ("or",       TokenType.OR),
        ("not",      TokenType.NOT),
        ("try",      TokenType.TRY),
        ("except",   TokenType.EXCEPT),
        ("finally",  TokenType.FINALLY),
    ])
    def test_keyword_token_type(self, keyword, expected_type):
        token = first_token(keyword)
        assert token.type == expected_type

    def test_keyword_prefix_is_identifier(self):
        """'iffy' should be IDENTIFIER not IF."""
        token = first_token("iffy")
        assert token.type == TokenType.IDENTIFIER

    def test_keyword_suffix_is_identifier(self):
        """'define' should be IDENTIFIER not DEF."""
        token = first_token("define")
        assert token.type == TokenType.IDENTIFIER

    def test_keyword_with_numbers_is_identifier(self):
        """'if1' should be IDENTIFIER not IF."""
        token = first_token("if1")
        assert token.type == TokenType.IDENTIFIER


# ===========================================================================
# 5. IDENTIFIERS
# ===========================================================================

class TestIdentifiers:

    def test_simple_identifier(self):
        token = first_token("x")
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "x"

    def test_multi_char_identifier(self):
        token = first_token("total")
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "total"

    def test_identifier_with_underscore(self):
        token = first_token("my_var")
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "my_var"

    def test_identifier_starting_with_underscore(self):
        token = first_token("_private")
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "_private"

    def test_identifier_with_numbers(self):
        token = first_token("var1")
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "var1"

    def test_identifier_cannot_start_with_number(self):
        """'1var' should tokenize as NUMBER then IDENTIFIER."""
        tokens = tokenize("1var")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[1].type == TokenType.IDENTIFIER

    def test_all_uppercase_identifier(self):
        token = first_token("TOTAL")
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "TOTAL"


# ===========================================================================
# 6. OPERATORS — SINGLE CHARACTER
# ===========================================================================

class TestSingleCharOperators:

    @pytest.mark.parametrize("source, expected_type", [
        ("+",  TokenType.PLUS),
        ("-",  TokenType.MINUS),
        ("*",  TokenType.MULTIPLY),
        ("/",  TokenType.DIVIDE),
        ("%",  TokenType.MODULO),
        ("=",  TokenType.ASSIGN),
        ("<",  TokenType.LT),
        (">",  TokenType.GT),
        ("(",  TokenType.LPAREN),
        (")",  TokenType.RPAREN),
        ("[",  TokenType.LBRACKET),
        ("]",  TokenType.RBRACKET),
        ("{",  TokenType.LBRACE),
        ("}",  TokenType.RBRACE),
        (",",  TokenType.COMMA),
        (":",  TokenType.COLON),
    ])
    def test_single_char_operator(self, source, expected_type):
        token = first_token(source)
        assert token.type == expected_type


# ===========================================================================
# 7. OPERATORS — TWO CHARACTER
# ===========================================================================

class TestTwoCharOperators:

    @pytest.mark.parametrize("source, expected_type", [
        ("==", TokenType.EQ),
        ("!=", TokenType.NE),
        ("<=", TokenType.LE),
        (">=", TokenType.GE),
        ("**", TokenType.POWER),
        ("//", TokenType.FLOOR_DIVIDE),
        ("+=", TokenType.PLUS_ASSIGN),
        ("-=", TokenType.MINUS_ASSIGN),
        ("*=", TokenType.MULTIPLY_ASSIGN),
        ("/=", TokenType.DIVIDE_ASSIGN),
    ])
    def test_two_char_operator(self, source, expected_type):
        token = first_token(source)
        assert token.type == expected_type

    def test_eq_not_confused_with_assign(self):
        tokens = tokenize("==")
        assert tokens[0].type == TokenType.EQ

    def test_assign_not_confused_with_eq(self):
        tokens = tokenize("=")
        assert tokens[0].type == TokenType.ASSIGN

    def test_power_not_confused_with_multiply(self):
        tokens = tokenize("**")
        assert tokens[0].type == TokenType.POWER

    def test_floor_divide_not_confused_with_divide(self):
        tokens = tokenize("//x")
        assert tokens[0].type == TokenType.FLOOR_DIVIDE

    def test_plus_assign_not_confused_with_plus(self):
        tokens = tokenize("+=")
        assert tokens[0].type == TokenType.PLUS_ASSIGN

    def test_ne_token_value(self):
        token = first_token("!=")
        assert token.value == "!="


# ===========================================================================
# 8. INDENTATION
# ===========================================================================

class TestIndentation:

    def test_indent_produced_on_block(self):
        source = "if True:\n    x = 1"
        types = get_types(source)
        assert TokenType.INDENT in types

    def test_dedent_produced_after_block(self):
        source = "if True:\n    x = 1\ny = 2"
        types = get_types(source)
        assert TokenType.DEDENT in types

    def test_nested_indent_produces_two_indents(self):
        source = "if True:\n    if True:\n        x = 1"
        types = get_types(source)
        assert types.count(TokenType.INDENT) == 2

    def test_nested_dedent_produces_two_dedents(self):
        source = "if True:\n    if True:\n        x = 1\ny = 2"
        types = get_types(source)
        assert types.count(TokenType.DEDENT) == 2

    def test_invalid_indentation_not_multiple_of_4_raises_error(self):
        source = "if True:\n   x = 1"  # 3 spaces
        with pytest.raises(LexerError):
            tokenize(source)

    def test_two_space_indent_raises_error(self):
        source = "if True:\n  x = 1"  # 2 spaces
        with pytest.raises(LexerError):
            tokenize(source)

    def test_valid_8_space_double_nested_indent(self):
        source = "if True:\n    if True:\n        x = 1"
        tokens = tokenize(source)
        assert tokens is not None

    def test_tab_treated_as_indent(self):
        source = "if True:\n\tx = 1"
        types = get_types(source)
        assert TokenType.INDENT in types

    def test_no_indent_at_top_level(self):
        source = "x = 1\ny = 2"
        types = get_types(source)
        assert TokenType.INDENT not in types
        assert TokenType.DEDENT not in types


# ===========================================================================
# 9. COMMENTS
# ===========================================================================

class TestComments:

    def test_comment_only_produces_eof(self):
        tokens = tokenize("# this is a comment")
        assert tokens[-1].type == TokenType.EOF
        assert len(tokens) == 1

    def test_comment_after_code_ignored(self):
        types = get_types("x = 5 # assign x")
        assert TokenType.NUMBER in types
        assert TokenType.ASSIGN in types

    def test_comment_does_not_produce_tokens(self):
        source = "x = 1\n# comment\ny = 2"
        tokens = tokenize(source)
        values = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        assert "x" in values
        assert "y" in values

    def test_hash_inside_string_is_not_comment(self):
        token = first_token('"hello # world"')
        assert token.type == TokenType.STRING
        assert token.value == "hello # world"

    def test_multiple_comment_lines(self):
        source = "# line 1\n# line 2\nx = 1"
        tokens = tokenize(source)
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert len(identifiers) == 1
        assert identifiers[0].value == "x"


# ===========================================================================
# 10. NEWLINES
# ===========================================================================

class TestNewlines:

    def test_newline_token_produced(self):
        source = "x = 1\ny = 2"
        types = get_types(source)
        assert TokenType.NEWLINE in types

    def test_multiple_blank_lines_produce_single_newline(self):
        """Consecutive newlines should not produce duplicate NEWLINE tokens."""
        source = "x = 1\n\n\ny = 2"
        types = get_types(source)
        # No two consecutive NEWLINEs
        for i in range(len(types) - 1):
            if types[i] == TokenType.NEWLINE:
                assert types[i + 1] != TokenType.NEWLINE

    def test_no_trailing_newline_still_works(self):
        tokens = tokenize("x = 1")
        assert tokens[-1].type == TokenType.EOF

    def test_newline_between_statements(self):
        source = "x = 1\ny = 2"
        types = get_types(source)
        newline_idx = types.index(TokenType.NEWLINE)
        assert newline_idx > 0  # something before it


# ===========================================================================
# 11. TOKEN METADATA (line, column)
# ===========================================================================

class TestTokenMetadata:

    def test_first_token_is_line_1(self):
        token = first_token("x")
        assert token.line == 1

    def test_first_token_is_column_1(self):
        token = first_token("x")
        assert token.column == 1

    def test_token_on_second_line(self):
        tokens = tokenize("x = 1\ny = 2")
        y_token = next(t for t in tokens if t.value == "y")
        assert y_token.line == 2

    def test_token_column_after_spaces(self):
        tokens = tokenize("x = 42")
        num_token = next(t for t in tokens if t.type == TokenType.NUMBER)
        assert num_token.column > 1

    def test_error_contains_line_number(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize('"unterminated')
        assert exc_info.value.line is not None

    def test_error_contains_column_number(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize('"unterminated')
        assert exc_info.value.column is not None


# ===========================================================================
# 12. ILLEGAL / UNKNOWN CHARACTERS
# ===========================================================================

class TestIllegalCharacters:

    def test_at_sign_raises_lexer_error(self):
        with pytest.raises(LexerError):
            tokenize("@")

    def test_dollar_sign_raises_error(self):
        with pytest.raises(LexerError):
            tokenize("$x")

    def test_hash_is_not_illegal_it_is_comment(self):
        """# starts a comment, should not raise error."""
        tokens = tokenize("# comment")
        assert tokens[-1].type == TokenType.EOF

    def test_exclamation_alone_raises_error(self):
        """! alone (not !=) should raise error."""
        with pytest.raises(LexerError):
            tokenize("!")


# ===========================================================================
# 13. EOF TOKEN
# ===========================================================================

class TestEOF:

    def test_eof_always_present(self):
        tokens = tokenize("x = 1")
        assert tokens[-1].type == TokenType.EOF

    def test_empty_source_produces_only_eof(self):
        tokens = tokenize("")
        assert tokens[-1].type == TokenType.EOF

    def test_whitespace_only_produces_eof(self):
        tokens = tokenize("   ")
        assert tokens[-1].type == TokenType.EOF

    def test_eof_value_is_none(self):
        tokens = tokenize("x = 1")
        assert tokens[-1].value is None


# ===========================================================================
# 14. COMPLETE MULTI-TOKEN SEQUENCES
# ===========================================================================

class TestCompleteExpressions:

    def test_simple_assignment_token_sequence(self):
        types = get_types("x = 42")
        assert types[0] == TokenType.IDENTIFIER
        assert types[1] == TokenType.ASSIGN
        assert types[2] == TokenType.NUMBER

    def test_function_call_token_sequence(self):
        types = get_types("print(x)")
        assert types[0] == TokenType.IDENTIFIER
        assert types[1] == TokenType.LPAREN
        assert types[2] == TokenType.IDENTIFIER
        assert types[3] == TokenType.RPAREN

    def test_for_loop_token_sequence(self):
        types = get_types("for i in range(10):")
        assert TokenType.FOR in types
        assert TokenType.IN in types
        assert TokenType.COLON in types

    def test_if_statement_token_sequence(self):
        types = get_types("if x == 5:")
        assert TokenType.IF in types
        assert TokenType.EQ in types
        assert TokenType.COLON in types

    def test_def_statement_token_sequence(self):
        types = get_types("def foo(x, y):")
        assert TokenType.DEF in types
        assert TokenType.IDENTIFIER in types
        assert TokenType.LPAREN in types
        assert TokenType.COMMA in types
        assert TokenType.RPAREN in types
        assert TokenType.COLON in types

    def test_augmented_assignment_sequence(self):
        types = get_types("x += 1")
        assert types[0] == TokenType.IDENTIFIER
        assert types[1] == TokenType.PLUS_ASSIGN
        assert types[2] == TokenType.NUMBER

    def test_list_literal_sequence(self):
        types = get_types("[1, 2, 3]")
        assert types[0] == TokenType.LBRACKET
        assert TokenType.COMMA in types
        assert TokenType.RBRACKET in types

    def test_dict_literal_sequence(self):
        types = get_types('{"key": "value"}')
        assert types[0] == TokenType.LBRACE
        assert TokenType.COLON in types
        assert TokenType.RBRACE in types