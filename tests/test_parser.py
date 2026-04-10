"""
Comprehensive test suite for OptiLang Parser (optilang/parser.py)

Tests cover:
- Literal nodes (number, string, boolean, none)
- Identifier nodes
- Binary operations and precedence
- Unary operations
- Assignment and augmented assignment
- Control flow (if/elif/else, while, for)
- Functions (def, call, return)
- Data structures (list, dict, index)
- Exception handling (try/except/finally)
- Parser error cases
"""

import pytest
from optilang.lexer import tokenize
from optilang.parser import Parser, parse
from optilang.token import Token, TokenType
from optilang.ast_nodes import (
    ProgramNode,
    NumberNode,
    StringNode,
    BooleanNode,
    NullNode,
    IdentifierNode,
    BinaryOpNode,
    UnaryOpNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    IfNode,
    WhileNode,
    ForNode,
    BreakNode,
    ContinueNode,
    PassNode,
    FunctionDefNode,
    FunctionCallNode,
    ReturnNode,
    ListNode,
    DictNode,
    IndexNode,
    TryNode,
    ASTNode,
)
from optilang.utils.errors import ParserError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_source(source: str) -> ProgramNode:
    """Tokenize and parse source, return ProgramNode."""
    tokens = tokenize(source)
    return parse(tokens)


def first_stmt(source: str) -> ASTNode:
    """Return first statement of parsed source."""
    return parse_source(source).statements[0]


def first_expr(source: str) -> ASTNode:
    """Parse a single expression statement and return the expression node."""
    return first_stmt(source)


# ===========================================================================
# 1. PROGRAM NODE
# ===========================================================================


class TestProgramNode:

    def test_empty_program_produces_program_node(self) -> None:
        tree = parse_source("")
        assert isinstance(tree, ProgramNode)

    def test_empty_program_has_no_statements(self) -> None:
        tree = parse_source("")
        assert tree.statements == []

    def test_single_statement_program(self) -> None:
        tree = parse_source("x = 1")
        assert len(tree.statements) == 1

    def test_multiple_statements(self) -> None:
        tree = parse_source("x = 1\ny = 2\nz = 3")
        assert len(tree.statements) == 3

    def test_program_node_line_is_1(self) -> None:
        tree = parse_source("x = 1")
        assert tree.line == 1


# ===========================================================================
# 2. LITERAL NODES
# ===========================================================================


class TestLiteralNodes:

    def test_integer_literal_produces_number_node(self) -> None:
        node = first_expr("42")
        assert isinstance(node, NumberNode)

    def test_integer_literal_value(self) -> None:
        node = first_expr("42")
        assert isinstance(node, NumberNode)
        assert node.value == 42

    def test_float_literal_produces_number_node(self) -> None:
        node = first_expr("3.14")
        assert isinstance(node, NumberNode)

    def test_float_literal_value(self) -> None:
        node = first_expr("3.14")
        assert isinstance(node, NumberNode)
        assert abs(node.value - 3.14) < 1e-9

    def test_string_literal_produces_string_node(self) -> None:
        node = first_expr('"hello"')
        assert isinstance(node, StringNode)

    def test_string_literal_value(self) -> None:
        node = first_expr('"hello"')
        assert isinstance(node, StringNode)
        assert node.value == "hello"

    def test_true_literal_produces_boolean_node(self) -> None:
        node = first_expr("True")
        assert isinstance(node, BooleanNode)

    def test_true_literal_value(self) -> None:
        node = first_expr("True")
        assert isinstance(node, BooleanNode)
        assert node.value is True

    def test_false_literal_produces_boolean_node(self) -> None:
        node = first_expr("False")
        assert isinstance(node, BooleanNode)

    def test_false_literal_value(self) -> None:
        node = first_expr("False")
        assert isinstance(node, BooleanNode)
        assert node.value is False

    def test_none_literal_produces_null_node(self) -> None:
        node = first_expr("None")
        assert isinstance(node, NullNode)


# ===========================================================================
# 3. IDENTIFIER NODES
# ===========================================================================


class TestIdentifierNodes:

    def test_identifier_produces_identifier_node(self) -> None:
        node = first_expr("x")
        assert isinstance(node, IdentifierNode)

    def test_identifier_name(self) -> None:
        node = first_expr("my_var")
        assert isinstance(node, IdentifierNode)
        assert node.name == "my_var"


# ===========================================================================
# 4. BINARY OPERATIONS
# ===========================================================================


class TestBinaryOperations:

    @pytest.mark.parametrize(
        "source, expected_op",
        [
            ("1 + 2", "+"),
            ("1 - 2", "-"),
            ("1 * 2", "*"),
            ("1 / 2", "/"),
            ("1 % 2", "%"),
            ("1 ** 2", "**"),
            ("1 // 2", "//"),
            ("1 == 2", "=="),
            ("1 != 2", "!="),
            ("1 < 2", "<"),
            ("1 <= 2", "<="),
            ("1 > 2", ">"),
            ("1 >= 2", ">="),
            ("1 and 2", "and"),
            ("1 or 2", "or"),
        ],
    )
    def test_binary_op_operator(self, source: str, expected_op: str) -> None:
        node = first_expr(source)
        assert isinstance(node, BinaryOpNode)
        assert node.operator == expected_op

    def test_binary_op_left_node(self) -> None:
        node = first_expr("3 + 4")
        assert isinstance(node, BinaryOpNode)
        assert isinstance(node.left, NumberNode)
        assert node.left.value == 3

    def test_binary_op_right_node(self) -> None:
        node = first_expr("3 + 4")
        assert isinstance(node, BinaryOpNode)
        assert isinstance(node.right, NumberNode)
        assert node.right.value == 4

    def test_operator_precedence_multiply_before_add(self) -> None:
        """2 + 3 * 4 → BinaryOp(2, +, BinaryOp(3, *, 4))"""
        node = first_expr("2 + 3 * 4")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "+"
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.operator == "*"

    def test_operator_precedence_power_before_multiply(self) -> None:
        """2 * 3 ** 2 → BinaryOp(2, *, BinaryOp(3, **, 2))"""
        node = first_expr("2 * 3 ** 2")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "*"
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.operator == "**"

    def test_power_is_right_associative(self) -> None:
        """2 ** 3 ** 2 → BinaryOp(2, **, BinaryOp(3, **, 2))"""
        node = first_expr("2 ** 3 ** 2")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "**"
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.operator == "**"

    def test_parentheses_override_precedence(self) -> None:
        """(2 + 3) * 4 → BinaryOp(BinaryOp(2, +, 3), *, 4)"""
        node = first_expr("(2 + 3) * 4")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "*"
        assert isinstance(node.left, BinaryOpNode)
        assert node.left.operator == "+"

    def test_comparison_precedence_below_arithmetic(self) -> None:
        """1 + 2 == 3 → BinaryOp(BinaryOp(1,+,2), ==, 3)"""
        node = first_expr("1 + 2 == 3")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "=="
        assert isinstance(node.left, BinaryOpNode)
        assert node.left.operator == "+"

    def test_and_precedence_below_comparison(self) -> None:
        """a < b and c > d"""
        node = first_expr("a < b and c > d")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "and"

    def test_or_precedence_below_and(self) -> None:
        """a and b or c and d → BinaryOp(BinaryOp(a,and,b), or, BinaryOp(c,and,d))"""
        node = first_expr("a and b or c and d")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "or"


# ===========================================================================
# 5. UNARY OPERATIONS
# ===========================================================================


class TestUnaryOperations:

    def test_unary_minus_produces_unary_node(self) -> None:
        node = first_expr("-5")
        assert isinstance(node, UnaryOpNode)

    def test_unary_minus_operator(self) -> None:
        node = first_expr("-5")
        assert isinstance(node, UnaryOpNode)
        assert node.operator == "-"

    def test_unary_minus_operand(self) -> None:
        node = first_expr("-5")
        assert isinstance(node, UnaryOpNode)
        assert isinstance(node.operand, NumberNode)
        assert node.operand.value == 5

    def test_unary_not_produces_unary_node(self) -> None:
        node = first_expr("not True")
        assert isinstance(node, UnaryOpNode)

    def test_unary_not_operator(self) -> None:
        node = first_expr("not True")
        assert isinstance(node, UnaryOpNode)
        assert node.operator == "not"

    def test_unary_not_operand_is_boolean(self) -> None:
        node = first_expr("not True")
        assert isinstance(node, UnaryOpNode)
        assert isinstance(node.operand, BooleanNode)

    def test_double_unary_minus(self) -> None:
        node = first_expr("--5")
        assert isinstance(node, UnaryOpNode)
        assert isinstance(node.operand, UnaryOpNode)


# ===========================================================================
# 6. ASSIGNMENT
# ===========================================================================


class TestAssignment:

    def test_assignment_produces_assignment_node(self) -> None:
        node = first_stmt("x = 5")
        assert isinstance(node, AssignmentNode)

    def test_assignment_target_name(self) -> None:
        node = first_stmt("x = 5")
        assert isinstance(node, AssignmentNode)
        assert node.target.name == "x"

    def test_assignment_value_is_number_node(self) -> None:
        node = first_stmt("x = 5")
        assert isinstance(node, AssignmentNode)
        assert isinstance(node.value, NumberNode)
        assert node.value.value == 5

    def test_assignment_with_expression_value(self) -> None:
        node = first_stmt("x = 2 + 3")
        assert isinstance(node, AssignmentNode)
        assert isinstance(node.value, BinaryOpNode)

    def test_assignment_with_string_value(self) -> None:
        node = first_stmt('name = "alice"')
        assert isinstance(node, AssignmentNode)
        assert isinstance(node.value, StringNode)
        assert node.value.value == "alice"


# ===========================================================================
# 7. AUGMENTED ASSIGNMENT
# ===========================================================================


class TestAugmentedAssignment:

    @pytest.mark.parametrize(
        "source, expected_op",
        [
            ("x += 1", "+="),
            ("x -= 1", "-="),
            ("x *= 2", "*="),
            ("x /= 2", "/="),
        ],
    )
    def test_augmented_assignment_operator(self, source: str, expected_op: str) -> None:
        node = first_stmt(source)
        assert isinstance(node, AugmentedAssignmentNode)
        assert node.operator == expected_op

    def test_augmented_assignment_target(self) -> None:
        node = first_stmt("x += 1")
        assert isinstance(node, AugmentedAssignmentNode)
        assert node.target.name == "x"

    def test_augmented_assignment_value(self) -> None:
        node = first_stmt("x += 5")
        assert isinstance(node, AugmentedAssignmentNode)
        assert isinstance(node.value, NumberNode)
        assert node.value.value == 5


# ===========================================================================
# 8. IF / ELIF / ELSE
# ===========================================================================


class TestIfStatement:

    def test_if_produces_if_node(self) -> None:
        source = "if x:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)

    def test_if_condition(self) -> None:
        source = "if x:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert isinstance(node.condition, IdentifierNode)
        assert node.condition.name == "x"

    def test_if_block_has_statements(self) -> None:
        source = "if True:\n    x = 1"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert len(node.if_block) == 1

    def test_if_else_has_else_block(self) -> None:
        source = "if True:\n    x = 1\nelse:\n    x = 2"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert node.else_block is not None
        assert len(node.else_block) == 1

    def test_if_without_else_has_none_else_block(self) -> None:
        source = "if True:\n    x = 1"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert node.else_block is None

    def test_elif_parts_populated(self) -> None:
        source = "if x:\n    pass\nelif y:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert len(node.elif_parts) == 1

    def test_multiple_elif_parts(self) -> None:
        source = "if x:\n    pass\nelif y:\n    pass\nelif z:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert len(node.elif_parts) == 2

    def test_nested_if(self) -> None:
        source = "if True:\n    if False:\n        pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert isinstance(node.if_block[0], IfNode)


# ===========================================================================
# 9. WHILE LOOP
# ===========================================================================


class TestWhileLoop:

    def test_while_produces_while_node(self) -> None:
        source = "while True:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)

    def test_while_condition(self) -> None:
        source = "while x:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)
        assert isinstance(node.condition, IdentifierNode)

    def test_while_body_has_statements(self) -> None:
        source = "while True:\n    x = 1"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)
        assert len(node.body) == 1

    def test_while_body_multiple_statements(self) -> None:
        source = "while True:\n    x = 1\n    y = 2"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)
        assert len(node.body) == 2


# ===========================================================================
# 10. FOR LOOP
# ===========================================================================


class TestForLoop:

    def test_for_produces_for_node(self) -> None:
        source = "for i in range(10):\n    pass"
        node = first_stmt(source)
        assert isinstance(node, ForNode)

    def test_for_iterator_name(self) -> None:
        source = "for i in range(10):\n    pass"
        node = first_stmt(source)
        assert isinstance(node, ForNode)
        assert node.iterator.name == "i"

    def test_for_iterable_is_function_call(self) -> None:
        source = "for i in range(10):\n    pass"
        node = first_stmt(source)
        assert isinstance(node, ForNode)
        assert isinstance(node.iterable, FunctionCallNode)

    def test_for_body_has_statements(self) -> None:
        source = "for i in range(5):\n    x = i"
        node = first_stmt(source)
        assert isinstance(node, ForNode)
        assert len(node.body) == 1

    def test_for_over_identifier(self) -> None:
        source = "for item in my_list:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, ForNode)
        assert isinstance(node.iterable, IdentifierNode)
        assert node.iterable.name == "my_list"


# ===========================================================================
# 11. BREAK / CONTINUE / PASS
# ===========================================================================


class TestJumpStatements:

    def test_break_produces_break_node(self) -> None:
        source = "while True:\n    break"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)
        assert isinstance(node.body[0], BreakNode)

    def test_continue_produces_continue_node(self) -> None:
        source = "while True:\n    continue"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)
        assert isinstance(node.body[0], ContinueNode)

    def test_pass_produces_pass_node(self) -> None:
        source = "if True:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)
        assert isinstance(node.if_block[0], PassNode)


# ===========================================================================
# 12. FUNCTION DEFINITION
# ===========================================================================


class TestFunctionDefinition:

    def test_def_produces_function_def_node(self) -> None:
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)

    def test_function_name(self) -> None:
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)
        assert node.name.name == "foo"

    def test_no_parameters(self) -> None:
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)
        assert node.parameters == []

    def test_single_parameter(self) -> None:
        source = "def foo(x):\n    pass"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)
        assert len(node.parameters) == 1
        assert node.parameters[0].name == "x"

    def test_multiple_parameters(self) -> None:
        source = "def foo(x, y, z):\n    pass"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)
        assert len(node.parameters) == 3
        assert [p.name for p in node.parameters] == ["x", "y", "z"]

    def test_function_body_has_statements(self) -> None:
        source = "def foo():\n    x = 1"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)
        assert len(node.body) == 1


# ===========================================================================
# 13. RETURN STATEMENT
# ===========================================================================


class TestReturnStatement:

    def test_return_with_value_produces_return_node(self) -> None:
        source = "def foo():\n    return 5"
        func_node = first_stmt(source)
        assert isinstance(func_node, FunctionDefNode)
        return_node = func_node.body[0]
        assert isinstance(return_node, ReturnNode)

    def test_return_value(self) -> None:
        source = "def foo():\n    return 5"
        func_node = first_stmt(source)
        assert isinstance(func_node, FunctionDefNode)
        return_node = func_node.body[0]
        assert isinstance(return_node, ReturnNode)
        assert isinstance(return_node.value, NumberNode)
        assert return_node.value.value == 5

    def test_return_without_value(self) -> None:
        source = "def foo():\n    return"
        func_node = first_stmt(source)
        assert isinstance(func_node, FunctionDefNode)
        return_node = func_node.body[0]
        assert isinstance(return_node, ReturnNode)
        assert return_node.value is None

    def test_return_expression(self) -> None:
        source = "def foo():\n    return x + 1"
        func_node = first_stmt(source)
        assert isinstance(func_node, FunctionDefNode)
        return_node = func_node.body[0]
        assert isinstance(return_node, ReturnNode)
        assert isinstance(return_node.value, BinaryOpNode)


# ===========================================================================
# 14. FUNCTION CALL
# ===========================================================================


class TestFunctionCall:

    def test_function_call_produces_function_call_node(self) -> None:
        node = first_expr("foo()")
        assert isinstance(node, FunctionCallNode)

    def test_function_call_name(self) -> None:
        node = first_expr("foo()")
        assert isinstance(node, FunctionCallNode)
        assert node.function.name == "foo"

    def test_function_call_no_arguments(self) -> None:
        node = first_expr("foo()")
        assert isinstance(node, FunctionCallNode)
        assert node.arguments == []

    def test_function_call_single_argument(self) -> None:
        node = first_expr("foo(x)")
        assert isinstance(node, FunctionCallNode)
        assert len(node.arguments) == 1
        assert isinstance(node.arguments[0], IdentifierNode)

    def test_function_call_multiple_arguments(self) -> None:
        node = first_expr("foo(x, y, z)")
        assert isinstance(node, FunctionCallNode)
        assert len(node.arguments) == 3

    def test_function_call_with_expression_argument(self) -> None:
        node = first_expr("foo(1 + 2)")
        assert isinstance(node, FunctionCallNode)
        assert isinstance(node.arguments[0], BinaryOpNode)

    def test_nested_function_call(self) -> None:
        node = first_expr("foo(bar())")
        assert isinstance(node, FunctionCallNode)
        assert isinstance(node.arguments[0], FunctionCallNode)

    def test_print_call(self) -> None:
        node = first_expr('print("hello")')
        assert isinstance(node, FunctionCallNode)
        assert node.function.name == "print"


# ===========================================================================
# 15. DATA STRUCTURES
# ===========================================================================


class TestDataStructures:

    def test_empty_list_produces_list_node(self) -> None:
        node = first_expr("[]")
        assert isinstance(node, ListNode)

    def test_empty_list_has_no_elements(self) -> None:
        node = first_expr("[]")
        assert isinstance(node, ListNode)
        assert node.elements == []

    def test_list_with_elements(self) -> None:
        node = first_expr("[1, 2, 3]")
        assert isinstance(node, ListNode)
        assert len(node.elements) == 3

    def test_list_element_values(self) -> None:
        node = first_expr("[1, 2, 3]")
        assert isinstance(node, ListNode)
        values = [e.value for e in node.elements if isinstance(e, NumberNode)]
        assert values == [1, 2, 3]

    def test_nested_list(self) -> None:
        node = first_expr("[[1, 2], [3, 4]]")
        assert isinstance(node, ListNode)
        assert isinstance(node.elements[0], ListNode)

    def test_empty_dict_produces_dict_node(self) -> None:
        node = first_expr("{}")
        assert isinstance(node, DictNode)

    def test_empty_dict_has_no_pairs(self) -> None:
        node = first_expr("{}")
        assert isinstance(node, DictNode)
        assert node.pairs == []

    def test_dict_with_pairs(self) -> None:
        node = first_expr('{"a": 1, "b": 2}')
        assert isinstance(node, DictNode)
        assert len(node.pairs) == 2

    def test_dict_key_and_value(self) -> None:
        node = first_expr('{"key": 42}')
        assert isinstance(node, DictNode)
        key, value = node.pairs[0]
        assert isinstance(key, StringNode)
        assert isinstance(value, NumberNode)

    def test_index_access_produces_index_node(self) -> None:
        node = first_expr("a[0]")
        assert isinstance(node, IndexNode)

    def test_index_access_collection(self) -> None:
        node = first_expr("a[0]")
        assert isinstance(node, IndexNode)
        assert isinstance(node.collection, IdentifierNode)
        assert node.collection.name == "a"

    def test_index_access_index(self) -> None:
        node = first_expr("a[0]")
        assert isinstance(node, IndexNode)
        assert isinstance(node.index, NumberNode)
        assert node.index.value == 0

    def test_string_key_index(self) -> None:
        node = first_expr('d["key"]')
        assert isinstance(node, IndexNode)
        assert isinstance(node.index, StringNode)

    def test_nested_index_access(self) -> None:
        node = first_expr("a[0][1]")
        assert isinstance(node, IndexNode)
        assert isinstance(node.collection, IndexNode)


# ===========================================================================
# 16. EXCEPTION HANDLING
# ===========================================================================


class TestExceptionHandling:

    def test_try_except_produces_try_node(self) -> None:
        source = "try:\n    x = 1\nexcept:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, TryNode)

    def test_try_block_has_statements(self) -> None:
        source = "try:\n    x = 1\nexcept:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, TryNode)
        assert len(node.try_block) == 1

    def test_except_block_has_statements(self) -> None:
        source = "try:\n    x = 1\nexcept:\n    x = 0"
        node = first_stmt(source)
        assert isinstance(node, TryNode)
        assert node.except_block is not None
        assert len(node.except_block) == 1

    def test_try_without_except_has_none_except(self) -> None:
        source = "try:\n    x = 1\nfinally:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, TryNode)
        assert node.except_block is None

    def test_finally_block_has_statements(self) -> None:
        source = "try:\n    x = 1\nexcept:\n    pass\nfinally:\n    y = 2"
        node = first_stmt(source)
        assert isinstance(node, TryNode)
        assert node.finally_block is not None
        assert len(node.finally_block) == 1

    def test_try_without_finally_has_none_finally(self) -> None:
        source = "try:\n    x = 1\nexcept:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, TryNode)
        assert node.finally_block is None


# ===========================================================================
# 17. PARSER ERROR CASES
# ===========================================================================


class TestParserErrors:

    def test_missing_colon_after_if(self) -> None:
        with pytest.raises(ParserError):
            parse_source("if True\n    pass")

    def test_missing_in_keyword_in_for(self) -> None:
        with pytest.raises(ParserError):
            parse_source("for i range(10):\n    pass")

    def test_unclosed_parenthesis(self) -> None:
        with pytest.raises(ParserError):
            parse_source("foo(1, 2")

    def test_missing_colon_after_def(self) -> None:
        with pytest.raises(ParserError):
            parse_source("def foo()\n    pass")

    def test_missing_colon_after_while(self) -> None:
        with pytest.raises(ParserError):
            parse_source("while True\n    pass")


# ===========================================================================
# 18. NODE LINE/COLUMN METADATA
# ===========================================================================


class TestNodeMetadata:

    def test_assignment_node_has_line(self) -> None:
        node = first_stmt("x = 1")
        assert node.line == 1

    def test_assignment_node_has_column(self) -> None:
        node = first_stmt("x = 1")
        assert node.column is not None

    def test_if_node_has_line(self) -> None:
        source = "if True:\n    pass"
        node = first_stmt(source)
        assert node.line == 1

    def test_function_def_node_has_line(self) -> None:
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert node.line == 1


class TestParserInternals:

    def test_peek_out_of_bounds_returns_none(self) -> None:
        parser = Parser(tokenize("x"))
        assert parser.peek(10) is None

    def test_advance_past_end_of_file_raises(self) -> None:
        parser = Parser([])
        with pytest.raises(ParserError, match="Cannot advance past end of file"):
            parser.advance()

    def test_expect_reaches_end_of_file_when_current_token_is_none(self) -> None:
        parser = Parser([])
        with pytest.raises(ParserError, match="reached end of file"):
            parser.expect(TokenType.NUMBER)

    def test_match_returns_false_when_current_token_is_none(self) -> None:
        assert Parser([]).match(TokenType.NUMBER) is False

    def test_parse_wraps_unexpected_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = Parser(tokenize("x = 1"))

        def boom() -> list[ASTNode]:
            raise ValueError("boom")

        monkeypatch.setattr(parser, "parse_statements", boom)

        with pytest.raises(ParserError, match="Unexpected error during parsing: boom"):
            parser.parse()

    def test_parse_statements_skips_blank_lines(self) -> None:
        tree = parse_source("x = 1\n\n\ny = 2\n")
        assert len(tree.statements) == 2

    def test_parse_statements_skips_newline_after_none_statement(self) -> None:
        parser = Parser(
            [
                Token(TokenType.NEWLINE, None, 1, 1),
                Token(TokenType.IDENTIFIER, "x", 2, 1),
                Token(TokenType.EOF, None, 2, 2),
            ]
        )
        parser.skip_newlines = lambda: None  # type: ignore[method-assign]

        def fake_parse_statement() -> ASTNode:
            parser.current_token = parser.tokens[2]
            return NumberNode(line=2, column=1, value=1)

        parser.parse_statement = fake_parse_statement  # type: ignore[method-assign]

        statements = parser.parse_statements()

        assert len(statements) == 1
        assert isinstance(statements[0], NumberNode)

    def test_augmented_assignment_operator_missing_at_end_of_file(self) -> None:
        parser = Parser([Token(TokenType.IDENTIFIER, "x", 1, 1)])
        with pytest.raises(
            ParserError,
            match="augmented assignment operator but reached end of file",
        ):
            parser.parse_augmented_assignment()

    def test_augmented_assignment_operator_rejects_non_augmented_token(self) -> None:
        parser = Parser(
            [
                Token(TokenType.IDENTIFIER, "x", 1, 1),
                Token(TokenType.ASSIGN, "=", 1, 3),
                Token(TokenType.NUMBER, 1, 1, 5),
                Token(TokenType.EOF, None, 1, 6),
            ]
        )

        with pytest.raises(ParserError, match="Expected augmented assignment operator"):
            parser.parse_augmented_assignment()

    def test_parse_unary_none_operator_branch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parser = Parser([])
        monkeypatch.setattr(parser, "match", lambda *args: True)
        parser.current_token = None

        with pytest.raises(ParserError, match="Expected unary operator but got None"):
            parser.parse_unary()

    def test_none_literal_can_be_used_in_postfix_expression(self) -> None:
        node = first_expr("None[0]")
        assert isinstance(node, IndexNode)
        assert isinstance(node.collection, NullNode)

    def test_unexpected_token_in_expression_raises(self) -> None:
        with pytest.raises(ParserError, match="Unexpected token in expression"):
            parse_source("]")

    def test_list_literal_allows_trailing_comma(self) -> None:
        node = first_expr("[1, 2,]")
        assert isinstance(node, ListNode)
        assert len(node.elements) == 2

    def test_dict_literal_allows_trailing_comma(self) -> None:
        node = first_expr("{'a': 1,}")
        assert isinstance(node, DictNode)
        assert len(node.pairs) == 1
