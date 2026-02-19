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
from optilang.parser import parse
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
)
from optilang.utils.errors import ParserError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_source(source: str) -> ProgramNode:
    """Tokenize and parse source, return ProgramNode."""
    tokens = tokenize(source)
    return parse(tokens)


def first_stmt(source: str):
    """Return first statement of parsed source."""
    return parse_source(source).statements[0]


def first_expr(source: str):
    """Parse a single expression statement and return the expression node."""
    return first_stmt(source)


# ===========================================================================
# 1. PROGRAM NODE
# ===========================================================================

class TestProgramNode:

    def test_empty_program_produces_program_node(self):
        tree = parse_source("")
        assert isinstance(tree, ProgramNode)

    def test_empty_program_has_no_statements(self):
        tree = parse_source("")
        assert tree.statements == []

    def test_single_statement_program(self):
        tree = parse_source("x = 1")
        assert len(tree.statements) == 1

    def test_multiple_statements(self):
        tree = parse_source("x = 1\ny = 2\nz = 3")
        assert len(tree.statements) == 3

    def test_program_node_line_is_1(self):
        tree = parse_source("x = 1")
        assert tree.line == 1


# ===========================================================================
# 2. LITERAL NODES
# ===========================================================================

class TestLiteralNodes:

    def test_integer_literal_produces_number_node(self):
        node = first_expr("42")
        assert isinstance(node, NumberNode)

    def test_integer_literal_value(self):
        node = first_expr("42")
        assert node.value == 42

    def test_float_literal_produces_number_node(self):
        node = first_expr("3.14")
        assert isinstance(node, NumberNode)

    def test_float_literal_value(self):
        node = first_expr("3.14")
        assert abs(node.value - 3.14) < 1e-9

    def test_string_literal_produces_string_node(self):
        node = first_expr('"hello"')
        assert isinstance(node, StringNode)

    def test_string_literal_value(self):
        node = first_expr('"hello"')
        assert node.value == "hello"

    def test_true_literal_produces_boolean_node(self):
        node = first_expr("True")
        assert isinstance(node, BooleanNode)

    def test_true_literal_value(self):
        node = first_expr("True")
        assert node.value is True

    def test_false_literal_produces_boolean_node(self):
        node = first_expr("False")
        assert isinstance(node, BooleanNode)

    def test_false_literal_value(self):
        node = first_expr("False")
        assert node.value is False

    def test_none_literal_produces_null_node(self):
        node = first_expr("None")
        assert isinstance(node, NullNode)


# ===========================================================================
# 3. IDENTIFIER NODES
# ===========================================================================

class TestIdentifierNodes:

    def test_identifier_produces_identifier_node(self):
        node = first_expr("x")
        assert isinstance(node, IdentifierNode)

    def test_identifier_name(self):
        node = first_expr("my_var")
        assert node.name == "my_var"


# ===========================================================================
# 4. BINARY OPERATIONS
# ===========================================================================

class TestBinaryOperations:

    @pytest.mark.parametrize("source, expected_op", [
        ("1 + 2",   "+"),
        ("1 - 2",   "-"),
        ("1 * 2",   "*"),
        ("1 / 2",   "/"),
        ("1 % 2",   "%"),
        ("1 ** 2",  "**"),
        ("1 // 2",  "//"),
        ("1 == 2",  "=="),
        ("1 != 2",  "!="),
        ("1 < 2",   "<"),
        ("1 <= 2",  "<="),
        ("1 > 2",   ">"),
        ("1 >= 2",  ">="),
        ("1 and 2", "and"),
        ("1 or 2",  "or"),
    ])
    def test_binary_op_operator(self, source, expected_op):
        node = first_expr(source)
        assert isinstance(node, BinaryOpNode)
        assert node.operator == expected_op

    def test_binary_op_left_node(self):
        node = first_expr("3 + 4")
        assert isinstance(node.left, NumberNode)
        assert node.left.value == 3

    def test_binary_op_right_node(self):
        node = first_expr("3 + 4")
        assert isinstance(node.right, NumberNode)
        assert node.right.value == 4

    def test_operator_precedence_multiply_before_add(self):
        """2 + 3 * 4 → BinaryOp(2, +, BinaryOp(3, *, 4))"""
        node = first_expr("2 + 3 * 4")
        assert isinstance(node, BinaryOpNode)
        assert node.operator == "+"
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.operator == "*"

    def test_operator_precedence_power_before_multiply(self):
        """2 * 3 ** 2 → BinaryOp(2, *, BinaryOp(3, **, 2))"""
        node = first_expr("2 * 3 ** 2")
        assert node.operator == "*"
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.operator == "**"

    def test_power_is_right_associative(self):
        """2 ** 3 ** 2 → BinaryOp(2, **, BinaryOp(3, **, 2))"""
        node = first_expr("2 ** 3 ** 2")
        assert node.operator == "**"
        assert isinstance(node.right, BinaryOpNode)
        assert node.right.operator == "**"

    def test_parentheses_override_precedence(self):
        """(2 + 3) * 4 → BinaryOp(BinaryOp(2, +, 3), *, 4)"""
        node = first_expr("(2 + 3) * 4")
        assert node.operator == "*"
        assert isinstance(node.left, BinaryOpNode)
        assert node.left.operator == "+"

    def test_comparison_precedence_below_arithmetic(self):
        """1 + 2 == 3 → BinaryOp(BinaryOp(1,+,2), ==, 3)"""
        node = first_expr("1 + 2 == 3")
        assert node.operator == "=="
        assert isinstance(node.left, BinaryOpNode)
        assert node.left.operator == "+"

    def test_and_precedence_below_comparison(self):
        """a < b and c > d"""
        node = first_expr("a < b and c > d")
        assert node.operator == "and"

    def test_or_precedence_below_and(self):
        """a and b or c and d → BinaryOp(BinaryOp(a,and,b), or, BinaryOp(c,and,d))"""
        node = first_expr("a and b or c and d")
        assert node.operator == "or"


# ===========================================================================
# 5. UNARY OPERATIONS
# ===========================================================================

class TestUnaryOperations:

    def test_unary_minus_produces_unary_node(self):
        node = first_expr("-5")
        assert isinstance(node, UnaryOpNode)

    def test_unary_minus_operator(self):
        node = first_expr("-5")
        assert node.operator == "-"

    def test_unary_minus_operand(self):
        node = first_expr("-5")
        assert isinstance(node.operand, NumberNode)
        assert node.operand.value == 5

    def test_unary_not_produces_unary_node(self):
        node = first_expr("not True")
        assert isinstance(node, UnaryOpNode)

    def test_unary_not_operator(self):
        node = first_expr("not True")
        assert node.operator == "not"

    def test_unary_not_operand_is_boolean(self):
        node = first_expr("not True")
        assert isinstance(node.operand, BooleanNode)

    def test_double_unary_minus(self):
        node = first_expr("--5")
        assert isinstance(node, UnaryOpNode)
        assert isinstance(node.operand, UnaryOpNode)


# ===========================================================================
# 6. ASSIGNMENT
# ===========================================================================

class TestAssignment:

    def test_assignment_produces_assignment_node(self):
        node = first_stmt("x = 5")
        assert isinstance(node, AssignmentNode)

    def test_assignment_target_name(self):
        node = first_stmt("x = 5")
        assert node.target.name == "x"

    def test_assignment_value_is_number_node(self):
        node = first_stmt("x = 5")
        assert isinstance(node.value, NumberNode)
        assert node.value.value == 5

    def test_assignment_with_expression_value(self):
        node = first_stmt("x = 2 + 3")
        assert isinstance(node.value, BinaryOpNode)

    def test_assignment_with_string_value(self):
        node = first_stmt('name = "alice"')
        assert isinstance(node.value, StringNode)
        assert node.value.value == "alice"


# ===========================================================================
# 7. AUGMENTED ASSIGNMENT
# ===========================================================================

class TestAugmentedAssignment:

    @pytest.mark.parametrize("source, expected_op", [
        ("x += 1",  "+="),
        ("x -= 1",  "-="),
        ("x *= 2",  "*="),
        ("x /= 2",  "/="),
    ])
    def test_augmented_assignment_operator(self, source, expected_op):
        node = first_stmt(source)
        assert isinstance(node, AugmentedAssignmentNode)
        assert node.operator == expected_op

    def test_augmented_assignment_target(self):
        node = first_stmt("x += 1")
        assert node.target.name == "x"

    def test_augmented_assignment_value(self):
        node = first_stmt("x += 5")
        assert isinstance(node.value, NumberNode)
        assert node.value.value == 5


# ===========================================================================
# 8. IF / ELIF / ELSE
# ===========================================================================

class TestIfStatement:

    def test_if_produces_if_node(self):
        source = "if x:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, IfNode)

    def test_if_condition(self):
        source = "if x:\n    pass"
        node = first_stmt(source)
        assert isinstance(node.condition, IdentifierNode)
        assert node.condition.name == "x"

    def test_if_block_has_statements(self):
        source = "if True:\n    x = 1"
        node = first_stmt(source)
        assert len(node.if_block) == 1

    def test_if_else_has_else_block(self):
        source = "if True:\n    x = 1\nelse:\n    x = 2"
        node = first_stmt(source)
        assert node.else_block is not None
        assert len(node.else_block) == 1

    def test_if_without_else_has_none_else_block(self):
        source = "if True:\n    x = 1"
        node = first_stmt(source)
        assert node.else_block is None

    def test_elif_parts_populated(self):
        source = "if x:\n    pass\nelif y:\n    pass"
        node = first_stmt(source)
        assert len(node.elif_parts) == 1

    def test_multiple_elif_parts(self):
        source = "if x:\n    pass\nelif y:\n    pass\nelif z:\n    pass"
        node = first_stmt(source)
        assert len(node.elif_parts) == 2

    def test_nested_if(self):
        source = "if True:\n    if False:\n        pass"
        node = first_stmt(source)
        assert isinstance(node.if_block[0], IfNode)


# ===========================================================================
# 9. WHILE LOOP
# ===========================================================================

class TestWhileLoop:

    def test_while_produces_while_node(self):
        source = "while True:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, WhileNode)

    def test_while_condition(self):
        source = "while x:\n    pass"
        node = first_stmt(source)
        assert isinstance(node.condition, IdentifierNode)

    def test_while_body_has_statements(self):
        source = "while True:\n    x = 1"
        node = first_stmt(source)
        assert len(node.body) == 1

    def test_while_body_multiple_statements(self):
        source = "while True:\n    x = 1\n    y = 2"
        node = first_stmt(source)
        assert len(node.body) == 2


# ===========================================================================
# 10. FOR LOOP
# ===========================================================================

class TestForLoop:

    def test_for_produces_for_node(self):
        source = "for i in range(10):\n    pass"
        node = first_stmt(source)
        assert isinstance(node, ForNode)

    def test_for_iterator_name(self):
        source = "for i in range(10):\n    pass"
        node = first_stmt(source)
        assert node.iterator.name == "i"

    def test_for_iterable_is_function_call(self):
        source = "for i in range(10):\n    pass"
        node = first_stmt(source)
        assert isinstance(node.iterable, FunctionCallNode)

    def test_for_body_has_statements(self):
        source = "for i in range(5):\n    x = i"
        node = first_stmt(source)
        assert len(node.body) == 1

    def test_for_over_identifier(self):
        source = "for item in my_list:\n    pass"
        node = first_stmt(source)
        assert isinstance(node.iterable, IdentifierNode)
        assert node.iterable.name == "my_list"


# ===========================================================================
# 11. BREAK / CONTINUE / PASS
# ===========================================================================

class TestJumpStatements:

    def test_break_produces_break_node(self):
        source = "while True:\n    break"
        node = first_stmt(source)
        assert isinstance(node.body[0], BreakNode)

    def test_continue_produces_continue_node(self):
        source = "while True:\n    continue"
        node = first_stmt(source)
        assert isinstance(node.body[0], ContinueNode)

    def test_pass_produces_pass_node(self):
        source = "if True:\n    pass"
        node = first_stmt(source)
        assert isinstance(node.if_block[0], PassNode)


# ===========================================================================
# 12. FUNCTION DEFINITION
# ===========================================================================

class TestFunctionDefinition:

    def test_def_produces_function_def_node(self):
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert isinstance(node, FunctionDefNode)

    def test_function_name(self):
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert node.name.name == "foo"

    def test_no_parameters(self):
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert node.parameters == []

    def test_single_parameter(self):
        source = "def foo(x):\n    pass"
        node = first_stmt(source)
        assert len(node.parameters) == 1
        assert node.parameters[0].name == "x"

    def test_multiple_parameters(self):
        source = "def foo(x, y, z):\n    pass"
        node = first_stmt(source)
        assert len(node.parameters) == 3
        assert [p.name for p in node.parameters] == ["x", "y", "z"]

    def test_function_body_has_statements(self):
        source = "def foo():\n    x = 1"
        node = first_stmt(source)
        assert len(node.body) == 1


# ===========================================================================
# 13. RETURN STATEMENT
# ===========================================================================

class TestReturnStatement:

    def test_return_with_value_produces_return_node(self):
        source = "def foo():\n    return 5"
        func_node = first_stmt(source)
        return_node = func_node.body[0]
        assert isinstance(return_node, ReturnNode)

    def test_return_value(self):
        source = "def foo():\n    return 5"
        func_node = first_stmt(source)
        return_node = func_node.body[0]
        assert isinstance(return_node.value, NumberNode)
        assert return_node.value.value == 5

    def test_return_without_value(self):
        source = "def foo():\n    return"
        func_node = first_stmt(source)
        return_node = func_node.body[0]
        assert isinstance(return_node, ReturnNode)
        assert return_node.value is None

    def test_return_expression(self):
        source = "def foo():\n    return x + 1"
        func_node = first_stmt(source)
        return_node = func_node.body[0]
        assert isinstance(return_node.value, BinaryOpNode)


# ===========================================================================
# 14. FUNCTION CALL
# ===========================================================================

class TestFunctionCall:

    def test_function_call_produces_function_call_node(self):
        node = first_expr("foo()")
        assert isinstance(node, FunctionCallNode)

    def test_function_call_name(self):
        node = first_expr("foo()")
        assert node.function.name == "foo"

    def test_function_call_no_arguments(self):
        node = first_expr("foo()")
        assert node.arguments == []

    def test_function_call_single_argument(self):
        node = first_expr("foo(x)")
        assert len(node.arguments) == 1
        assert isinstance(node.arguments[0], IdentifierNode)

    def test_function_call_multiple_arguments(self):
        node = first_expr("foo(x, y, z)")
        assert len(node.arguments) == 3

    def test_function_call_with_expression_argument(self):
        node = first_expr("foo(1 + 2)")
        assert isinstance(node.arguments[0], BinaryOpNode)

    def test_nested_function_call(self):
        node = first_expr("foo(bar())")
        assert isinstance(node.arguments[0], FunctionCallNode)

    def test_print_call(self):
        node = first_expr('print("hello")')
        assert isinstance(node, FunctionCallNode)
        assert node.function.name == "print"


# ===========================================================================
# 15. DATA STRUCTURES
# ===========================================================================

class TestDataStructures:

    def test_empty_list_produces_list_node(self):
        node = first_expr("[]")
        assert isinstance(node, ListNode)

    def test_empty_list_has_no_elements(self):
        node = first_expr("[]")
        assert node.elements == []

    def test_list_with_elements(self):
        node = first_expr("[1, 2, 3]")
        assert isinstance(node, ListNode)
        assert len(node.elements) == 3

    def test_list_element_values(self):
        node = first_expr("[1, 2, 3]")
        values = [e.value for e in node.elements]
        assert values == [1, 2, 3]

    def test_nested_list(self):
        node = first_expr("[[1, 2], [3, 4]]")
        assert isinstance(node, ListNode)
        assert isinstance(node.elements[0], ListNode)

    def test_empty_dict_produces_dict_node(self):
        node = first_expr("{}")
        assert isinstance(node, DictNode)

    def test_empty_dict_has_no_pairs(self):
        node = first_expr("{}")
        assert node.pairs == []

    def test_dict_with_pairs(self):
        node = first_expr('{"a": 1, "b": 2}')
        assert isinstance(node, DictNode)
        assert len(node.pairs) == 2

    def test_dict_key_and_value(self):
        node = first_expr('{"key": 42}')
        key, value = node.pairs[0]
        assert isinstance(key, StringNode)
        assert isinstance(value, NumberNode)

    def test_index_access_produces_index_node(self):
        node = first_expr("a[0]")
        assert isinstance(node, IndexNode)

    def test_index_access_collection(self):
        node = first_expr("a[0]")
        assert isinstance(node.collection, IdentifierNode)
        assert node.collection.name == "a"

    def test_index_access_index(self):
        node = first_expr("a[0]")
        assert isinstance(node.index, NumberNode)
        assert node.index.value == 0

    def test_string_key_index(self):
        node = first_expr('d["key"]')
        assert isinstance(node.index, StringNode)

    def test_nested_index_access(self):
        node = first_expr("a[0][1]")
        assert isinstance(node, IndexNode)
        assert isinstance(node.collection, IndexNode)


# ===========================================================================
# 16. EXCEPTION HANDLING
# ===========================================================================

class TestExceptionHandling:

    def test_try_except_produces_try_node(self):
        source = "try:\n    x = 1\nexcept:\n    pass"
        node = first_stmt(source)
        assert isinstance(node, TryNode)

    def test_try_block_has_statements(self):
        source = "try:\n    x = 1\nexcept:\n    pass"
        node = first_stmt(source)
        assert len(node.try_block) == 1

    def test_except_block_has_statements(self):
        source = "try:\n    x = 1\nexcept:\n    x = 0"
        node = first_stmt(source)
        assert node.except_block is not None
        assert len(node.except_block) == 1

    def test_try_without_except_has_none_except(self):
        source = "try:\n    x = 1\nfinally:\n    pass"
        node = first_stmt(source)
        assert node.except_block is None

    def test_finally_block_has_statements(self):
        source = "try:\n    x = 1\nexcept:\n    pass\nfinally:\n    y = 2"
        node = first_stmt(source)
        assert node.finally_block is not None
        assert len(node.finally_block) == 1

    def test_try_without_finally_has_none_finally(self):
        source = "try:\n    x = 1\nexcept:\n    pass"
        node = first_stmt(source)
        assert node.finally_block is None


# ===========================================================================
# 17. PARSER ERROR CASES
# ===========================================================================

class TestParserErrors:

    def test_missing_colon_after_if(self):
        with pytest.raises(ParserError):
            parse_source("if True\n    pass")

    def test_missing_in_keyword_in_for(self):
        with pytest.raises(ParserError):
            parse_source("for i range(10):\n    pass")

    def test_unclosed_parenthesis(self):
        with pytest.raises(ParserError):
            parse_source("foo(1, 2")

    def test_missing_colon_after_def(self):
        with pytest.raises(ParserError):
            parse_source("def foo()\n    pass")

    def test_missing_colon_after_while(self):
        with pytest.raises(ParserError):
            parse_source("while True\n    pass")


# ===========================================================================
# 18. NODE LINE/COLUMN METADATA
# ===========================================================================

class TestNodeMetadata:

    def test_assignment_node_has_line(self):
        node = first_stmt("x = 1")
        assert node.line == 1

    def test_assignment_node_has_column(self):
        node = first_stmt("x = 1")
        assert node.column is not None

    def test_if_node_has_line(self):
        source = "if True:\n    pass"
        node = first_stmt(source)
        assert node.line == 1

    def test_function_def_node_has_line(self):
        source = "def foo():\n    pass"
        node = first_stmt(source)
        assert node.line == 1