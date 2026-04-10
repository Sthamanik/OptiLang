from __future__ import annotations

from optilang.ast_nodes import NumberNode
from optilang.token import Token, TokenType
from optilang.utils.errors import (
    AttributeError,
    LexerError,
    ParserError,
    RecursionError,
    RuntimeError,
    TypeError,
    ValueError,
)


class TokenWithoutValue:
    def __init__(self) -> None:
        self.line = 3
        self.column = 7
        self.type = TokenType.DEF


class TestErrors:

    def test_lexer_error_formats_message_with_and_without_illegal_char(self) -> None:
        with_char = LexerError("Illegal character", 2, 4, illegal_char="@")
        assert with_char.illegal_char == "@"
        assert str(with_char) == "Line 2, Column 4: Illegal character (found '@')"

        without_char = LexerError("Illegal character", 2, 4)
        assert without_char.illegal_char is None
        assert str(without_char) == "Line 2, Column 4: Illegal character"

    def test_parser_error_uses_token_value_or_type_metadata(self) -> None:
        valued = ParserError(
            "Expected expression",
            token=Token(TokenType.NUMBER, 42, 5, 9),
        )
        assert str(valued) == "Line 5, Column 9: Expected expression (got '42')"

        typed = ParserError("Expected expression", token=TokenWithoutValue())
        assert str(typed) == "Line 3, Column 7: Expected expression (got TokenType.DEF)"

    def test_runtime_error_uses_node_line_when_provided(self) -> None:
        err = RuntimeError("boom", node=NumberNode(line=9, column=1, value=1))
        assert err.line == 9
        assert str(err) == "Line 9: boom"

    def test_type_value_recursion_and_attribute_errors_store_context(self) -> None:
        type_err = TypeError("Type mismatch", line=4, expected="int", got="str")
        assert type_err.expected == "int"
        assert type_err.got == "str"
        assert str(type_err) == "Line 4: Type mismatch (expected int, got str)"

        value_err = ValueError("Invalid value", line=6, value="abc")
        assert value_err.value == "abc"
        assert str(value_err) == "Line 6: Invalid value"

        recursion_err = RecursionError(max_depth=42, line=8)
        assert recursion_err.max_depth == 42
        assert str(recursion_err) == "Line 8: Maximum recursion depth (42) exceeded"

        attribute_err = AttributeError("list", "appendd", line=10)
        assert attribute_err.obj_type == "list"
        assert attribute_err.attr_name == "appendd"
        assert str(attribute_err) == "Line 10: 'list' object has no attribute 'appendd'"
