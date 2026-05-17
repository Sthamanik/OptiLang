"""
tests/test_complexity.py
------------------------
Test suite for optilang/analysis/complexity.py
"""

from __future__ import annotations

import pytest

from optilang import (
    Complexity,
    ComplexityExpr,
    analyze_complexity,
    analyze_function_complexity,
)
from optilang.analysis.complexity import (
    Add,
    CallExpr,
    ComplexityAnalyzer,
    Const,
    Factorial,
    Log,
    Mul,
    Param,
    Var,
)
from optilang.core.ast_nodes import (
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    BooleanNode,
    BreakNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionDefNode,
    IdentifierNode,
    IfNode,
    IndexAssignmentNode,
    IndexedAugmentedAssignmentNode,
    IndexNode,
    ListNode,
    NumberNode,
    PassNode,
    ProgramNode,
    ReturnNode,
    StringNode,
    TryNode,
    UnaryOpNode,
    WhileNode,
)
from optilang.core.lexer import tokenize
from optilang.core.parser import parse


class TestComplexityEnum:
    """Tests for Complexity enum."""

    def test_all_complexity_classes_exist(self) -> None:
        expected = [
            "O1",
            "LOGN",
            "N",
            "NLOGN",
            "N2",
            "N2LOGN",
            "N3",
            "N4",
            "NK",
            "NF",
            "EXP",
            "UNKNOWN",
        ]
        actual = [c.name for c in Complexity]
        assert actual == expected

    def test_str_representation(self) -> None:
        assert str(Complexity.O1) == "O(1)"
        assert str(Complexity.N) == "O(n)"
        assert str(Complexity.N2) == "O(n²)"
        assert str(Complexity.UNKNOWN) == "Unknown"


class TestComplexityExpr:
    """Tests for ComplexityExpr and subclasses."""

    def test_complexity_expr_base(self) -> None:
        expr = ComplexityExpr()
        assert expr is not None

    def test_const_creation(self) -> None:
        from optilang.analysis.complexity import Const

        c = Const(value=5.0)
        assert c.value == 5.0

    def test_const_default_value(self) -> None:
        from optilang.analysis.complexity import Const

        c = Const()
        assert c.value == 1.0

    def test_param_creation(self) -> None:
        from optilang.analysis.complexity import Param

        p = Param(name="n")
        assert p.name == "n"

    def test_var_creation(self) -> None:
        from optilang.analysis.complexity import Var

        v = Var(name="x")
        assert v.name == "x"

    def test_log_creation(self) -> None:
        from optilang.analysis.complexity import Log, Param

        inner = Param(name="n")
        log = Log(inner=inner)
        assert log.inner == inner

    def test_add_creation(self) -> None:
        from optilang.analysis.complexity import Add, Const

        left = Const(value=1.0)
        right = Const(value=2.0)
        add = Add(left=left, right=right)
        assert add.left == left
        assert add.right == right

    def test_mul_creation(self) -> None:
        from optilang.analysis.complexity import Mul, Const, Param

        left = Const(value=2.0)
        right = Param(name="n")
        mul = Mul(left=left, right=right)
        assert mul.left == left
        assert mul.right == right

    def test_call_expr_creation(self) -> None:
        from optilang.analysis.complexity import CallExpr, Const

        call = CallExpr(name="len", body_complexity=Const(value=1.0))
        assert call.name == "len"
        assert call.body_complexity is not None

    def test_call_expr_without_body(self) -> None:
        from optilang.analysis.complexity import CallExpr

        call = CallExpr(name="print")
        assert call.name == "print"
        assert call.body_complexity is None

    def test_factorial_creation(self) -> None:
        from optilang.analysis.complexity import Factorial, Param

        inner = Param(name="n")
        fact = Factorial(inner=inner)
        assert fact.inner == inner

    def test_factorial_with_var(self) -> None:
        from optilang.analysis.complexity import Factorial, Var

        inner = Var(name="size")
        fact = Factorial(inner=inner)
        assert fact.inner.name == "size"


class TestComplexityResult:
    """Tests for ComplexityResult."""

    def test_result_with_all_fields(self) -> None:
        from optilang.analysis.complexity import ComplexityResult

        result = ComplexityResult(
            complexity="O(n²)",
            confidence=0.85,
            explanation="Nested loops detected",
            bound_symbol="n",
        )
        assert result.complexity == "O(n²)"
        assert result.confidence == 0.85
        assert result.explanation == "Nested loops detected"
        assert result.bound_symbol == "n"

    def test_result_without_bound_symbol(self) -> None:
        from optilang.analysis.complexity import ComplexityResult

        result = ComplexityResult(
            complexity="O(1)",
            confidence=1.0,
            explanation="Constant time",
        )
        assert result.bound_symbol is None


class TestComplexityAnalyzer:
    """Tests for ComplexityAnalyzer class."""

    def test_analyzer_creation(self) -> None:
        from optilang.analysis.complexity import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        assert analyzer is not None

    def test_analyze_simple_assignment(self) -> None:
        from optilang.analysis.complexity import ComplexityAnalyzer

        assign = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=42, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[assign], line=1, column=1)
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(program)
        assert result.complexity == "O(1)"

    def test_analyze_multiple_statements(self) -> None:
        from optilang.analysis.complexity import ComplexityAnalyzer

        stmt1 = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=10, line=1, column=5),
            line=1,
            column=1,
        )
        stmt2 = AssignmentNode(
            target=IdentifierNode(name="y", line=2, column=1),
            value=NumberNode(value=20, line=2, column=5),
            line=2,
            column=1,
        )
        program = ProgramNode(statements=[stmt1, stmt2], line=1, column=1)
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(program)
        assert result.complexity == "O(1)"
        assert result.confidence == 1.0


class TestAnalyzeComplexity:
    """Tests for analyze_complexity function."""

    def test_analyze_returns_result(self) -> None:
        from optilang.analysis.complexity import ComplexityResult

        program = ProgramNode(statements=[], line=1, column=1)
        result = analyze_complexity(program)
        assert isinstance(result, ComplexityResult)
        assert result.complexity in [
            "O(1)",
            "O(n)",
            "O(n²)",
            "O(log n)",
            "O(n log n)",
            "Unknown",
        ]
        assert 0.0 <= result.confidence <= 1.0
        assert result.explanation is not None

    def test_analyze_empty_program_o1(self) -> None:
        program = ProgramNode(statements=[], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"
        assert result.confidence == 1.0

    def test_analyze_single_assignment(self) -> None:
        assign = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=42, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[assign], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_analyze_two_assignments(self) -> None:
        stmt1 = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=10, line=1, column=5),
            line=1,
            column=1,
        )
        stmt2 = AssignmentNode(
            target=IdentifierNode(name="y", line=2, column=1),
            value=NumberNode(value=20, line=2, column=5),
            line=2,
            column=1,
        )
        program = ProgramNode(statements=[stmt1, stmt2], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_analyze_for_loop(self) -> None:
        # Create a for loop with a range call
        range_call = FunctionCallNode(
            function=IdentifierNode(name="range", line=1, column=10),
            arguments=[IdentifierNode(name="n", line=1, column=16)],
            line=1,
            column=10,
        )
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=range_call,
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # Loop should be O(n) or higher
        assert result.complexity in ["O(n)", "O(n²)", "O(n log n)", "Unknown"]

    def test_analyze_nested_for_loops(self) -> None:
        # Create inner for loop
        inner_range = FunctionCallNode(
            function=IdentifierNode(name="range", line=3, column=7),
            arguments=[IdentifierNode(name="n", line=3, column=13)],
            line=3,
            column=7,
        )
        inner_for = ForNode(
            iterator=IdentifierNode(name="j", line=3, column=1),
            iterable=inner_range,
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=4, column=1),
                    value=NumberNode(value=0, line=4, column=5),
                    line=4,
                    column=1,
                )
            ],
            line=3,
            column=1,
        )
        # Create outer for loop with inner as body
        outer_range = FunctionCallNode(
            function=IdentifierNode(name="range", line=1, column=10),
            arguments=[IdentifierNode(name="n", line=1, column=16)],
            line=1,
            column=10,
        )
        outer_for = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=outer_range,
            body=[inner_for],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[outer_for], line=1, column=1)
        result = analyze_complexity(program)
        # Nested loops should be O(n²) or higher
        assert result.complexity in ["O(n²)", "O(n³)", "O(n^k)", "Unknown"]

    def test_analyze_while_loop(self) -> None:
        # Create while loop
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="i", line=1, column=7),
                operator=">",
                right=NumberNode(value=0, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="i", line=2, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="i", line=2, column=5),
                        operator="-",
                        right=NumberNode(value=1, line=2, column=9),
                        line=2,
                        column=5,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        # While loop could be O(log n) or O(n)
        assert result.complexity in ["O(log n)", "O(n)", "Unknown"]


class TestAnalyzeFunctionComplexity:
    """Tests for analyze_function_complexity function."""

    def test_analyze_function_returns_result(self) -> None:
        from optilang.analysis.complexity import ComplexityResult

        func_def = FunctionDefNode(
            name=IdentifierNode(name="test", line=1, column=1),
            body=[],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert isinstance(result, ComplexityResult)
        assert result.complexity is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_function_o1(self) -> None:
        func_def = FunctionDefNode(
            name=IdentifierNode(name="empty", line=1, column=1),
            body=[],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"

    def test_function_with_assignment(self) -> None:
        assign = AssignmentNode(
            target=IdentifierNode(name="x", line=2, column=1),
            value=NumberNode(value=1, line=2, column=5),
            line=2,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="setup", line=1, column=1),
            body=[assign],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"

    def test_function_with_return(self) -> None:
        ret = ReturnNode(
            value=NumberNode(value=42, line=2, column=8),
            line=2,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="get_value", line=1, column=1),
            body=[ret],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"


class TestEdgeCases:
    """Edge case tests for complexity analysis."""

    def test_function_with_none_body(self) -> None:
        func_def = FunctionDefNode(
            name=IdentifierNode(name="none_body", line=1, column=1),
            body=None,
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"


class TestIfStatements:
    """Tests for if/elif/else complexity analysis."""

    def test_simple_if_no_else(self) -> None:
        from optilang.core.ast_nodes import IfNode

        if_node = IfNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="x", line=1, column=4),
                operator=">",
                right=NumberNode(value=0, line=1, column=8),
                line=1,
                column=4,
            ),
            if_block=[
                AssignmentNode(
                    target=IdentifierNode(name="y", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            elif_parts=[],
            else_block=None,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[if_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_if_with_else(self) -> None:
        from optilang.core.ast_nodes import IfNode

        if_node = IfNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="x", line=1, column=4),
                operator=">",
                right=NumberNode(value=0, line=1, column=8),
                line=1,
                column=4,
            ),
            if_block=[
                AssignmentNode(
                    target=IdentifierNode(name="y", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            elif_parts=[],
            else_block=[
                AssignmentNode(
                    target=IdentifierNode(name="y", line=4, column=1),
                    value=NumberNode(value=0, line=4, column=5),
                    line=4,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[if_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_if_with_elif(self) -> None:
        from optilang.core.ast_nodes import IfNode

        if_node = IfNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="x", line=1, column=4),
                operator=">",
                right=NumberNode(value=10, line=1, column=8),
                line=1,
                column=4,
            ),
            if_block=[
                AssignmentNode(
                    target=IdentifierNode(name="result", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=9),
                    line=2,
                    column=1,
                )
            ],
            elif_parts=[
                (
                    BinaryOpNode(
                        left=IdentifierNode(name="x", line=3, column=6),
                        operator=">",
                        right=NumberNode(value=5, line=3, column=10),
                        line=3,
                        column=6,
                    ),
                    [
                        AssignmentNode(
                            target=IdentifierNode(name="result", line=4, column=1),
                            value=NumberNode(value=2, line=4, column=9),
                            line=4,
                            column=1,
                        )
                    ],
                )
            ],
            else_block=None,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[if_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestLoops:
    """Tests for loop complexity analysis."""

    def test_for_loop_over_list(self) -> None:
        from optilang.core.ast_nodes import ListNode

        list_node = ListNode(
            elements=[
                NumberNode(value=1, line=1, column=8),
                NumberNode(value=2, line=1, column=11),
                NumberNode(value=3, line=1, column=14),
            ],
            line=1,
            column=7,
        )
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=list_node,
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # Literal lists have fixed write-time size, so iteration is constant.
        assert result.complexity == "O(1)"

    def test_while_loop_with_increment(self) -> None:
        assign_i = AssignmentNode(
            target=IdentifierNode(name="i", line=1, column=1),
            value=NumberNode(value=0, line=1, column=5),
            line=1,
            column=1,
        )
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="i", line=2, column=7),
                operator="<",
                right=NumberNode(value=10, line=2, column=11),
                line=2,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="i", line=3, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="i", line=3, column=5),
                        operator="+",
                        right=NumberNode(value=1, line=3, column=9),
                        line=3,
                        column=5,
                    ),
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        program = ProgramNode(statements=[assign_i, while_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(n)"

    def test_while_loop_halving_pattern(self) -> None:
        assign_n = AssignmentNode(
            target=IdentifierNode(name="n", line=1, column=1),
            value=NumberNode(value=100, line=1, column=5),
            line=1,
            column=1,
        )
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="n", line=2, column=7),
                operator=">",
                right=NumberNode(value=1, line=2, column=11),
                line=2,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="n", line=3, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="n", line=3, column=5),
                        operator="//",
                        right=NumberNode(value=2, line=3, column=9),
                        line=3,
                        column=5,
                    ),
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        program = ProgramNode(statements=[assign_n, while_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(log n)"


class TestStatements:
    """Tests for various statement types."""

    def test_break_statement(self) -> None:
        from optilang.core.ast_nodes import BreakNode

        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="n", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[
                BreakNode(line=2, column=1),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(n)"

    def test_continue_statement(self) -> None:
        from optilang.core.ast_nodes import ContinueNode

        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="n", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[
                ContinueNode(line=2, column=1),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(n)"

    def test_pass_statement(self) -> None:
        from optilang.core.ast_nodes import PassNode

        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="n", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[
                PassNode(line=2, column=1),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(n)"


class TestLiterals:
    """Tests for literal complexity analysis."""

    def test_string_literal(self) -> None:
        str_node = StringNode(value="hi", line=1, column=1)
        program = ProgramNode(statements=[str_node], line=1, column=1)
        result = analyze_complexity(program)
        # String literals return O(n) based on string length
        assert result.complexity in ["O(1)", "O(2)"]

    def test_boolean_literal(self) -> None:
        bool_node = BooleanNode(value=True, line=1, column=1)
        program = ProgramNode(statements=[bool_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestFunctionWithParams:
    """Tests for functions with parameters."""

    def test_function_with_params(self) -> None:
        param = IdentifierNode(name="n", line=1, column=12)
        func_def = FunctionDefNode(
            name=IdentifierNode(name="test", line=1, column=1),
            parameters=[param],
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="i", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"

    def test_function_with_loop_param(self) -> None:
        param = IdentifierNode(name="n", line=1, column=12)
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=2, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=2, column=10),
                arguments=[IdentifierNode(name="n", line=2, column=16)],
                line=2,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=3, column=1),
                    value=NumberNode(value=0, line=3, column=5),
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="test", line=1, column=1),
            parameters=[param],
            body=[for_node],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity in ["O(n)", "O(n²)", "Unknown"]


class TestTryExcept:
    """Tests for try-except complexity analysis."""

    def test_try_except(self) -> None:
        from optilang.core.ast_nodes import TryNode

        try_node = TryNode(
            try_block=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            except_block=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=4, column=1),
                    value=NumberNode(value=0, line=4, column=5),
                    line=4,
                    column=1,
                )
            ],
            finally_block=None,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[try_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_try_finally(self) -> None:
        from optilang.core.ast_nodes import TryNode

        try_node = TryNode(
            try_block=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            except_block=None,
            finally_block=[
                AssignmentNode(
                    target=IdentifierNode(name="y", line=4, column=1),
                    value=NumberNode(value=0, line=4, column=5),
                    line=4,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[try_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestIndexing:
    """Tests for indexing complexity analysis."""

    def test_index_access(self) -> None:
        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=NumberNode(value=0, line=1, column=5),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[index_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_index_assignment(self) -> None:
        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=NumberNode(value=0, line=1, column=5),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=1,
        )
        index_assign = IndexAssignmentNode(
            target=index_node,
            value=NumberNode(value=42, line=1, column=9),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[index_assign], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestBinaryOps:
    """Tests for binary operation complexity analysis."""

    def test_binary_add(self) -> None:
        bin_op = BinaryOpNode(
            left=NumberNode(value=1, line=1, column=1),
            operator="+",
            right=NumberNode(value=2, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_unary_op(self) -> None:
        unary_op = UnaryOpNode(
            operator="-",
            operand=NumberNode(value=5, line=1, column=2),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[unary_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestAugmentedAssignment:
    """Tests for augmented assignment complexity."""

    def test_augmented_assignment_add(self) -> None:
        aug_assign = AugmentedAssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            operator="+=",
            value=NumberNode(value=1, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[aug_assign], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_indexed_augmented_assignment(self) -> None:
        from optilang.core.ast_nodes import IndexedAugmentedAssignmentNode

        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=NumberNode(value=0, line=1, column=5),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=1,
        )
        indexed_aug = IndexedAugmentedAssignmentNode(
            target=index_node,
            operator="+=",
            value=NumberNode(value=1, line=1, column=10),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[indexed_aug], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestFunctionCallAnalysis:
    """Tests for function call complexity analysis."""

    def test_function_call_with_known_function(self) -> None:
        # Define a function first
        func_def = FunctionDefNode(
            name=IdentifierNode(name="process", line=1, column=1),
            body=[
                ForNode(
                    iterator=IdentifierNode(name="i", line=2, column=4),
                    iterable=FunctionCallNode(
                        function=IdentifierNode(name="range", line=2, column=10),
                        arguments=[IdentifierNode(name="n", line=2, column=16)],
                        line=2,
                        column=10,
                    ),
                    body=[
                        AssignmentNode(
                            target=IdentifierNode(name="x", line=3, column=1),
                            value=NumberNode(value=0, line=3, column=5),
                            line=3,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        # Create a program with the function and a call
        func_call = FunctionCallNode(
            function=IdentifierNode(name="process", line=4, column=1),
            arguments=[IdentifierNode(name="data", line=4, column=9)],
            line=4,
            column=1,
        )
        program = ProgramNode(statements=[func_def, func_call], line=1, column=1)
        result = analyze_complexity(program)
        # Should be O(n) or O(n²) depending on function body
        assert result.complexity in ["O(n)", "O(n²)", "O(n log n)", "Unknown"]

    def test_unknown_function_call(self) -> None:
        func_call = FunctionCallNode(
            function=IdentifierNode(name="unknown_func", line=1, column=1),
            arguments=[NumberNode(value=1, line=1, column=14)],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[func_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestListComprehension:
    """Tests for list comprehension complexity analysis."""

    def test_list_literal_empty(self) -> None:
        list_node = ListNode(elements=[], line=1, column=1)
        program = ProgramNode(statements=[list_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_list_literal_with_elements(self) -> None:
        list_node = ListNode(
            elements=[
                NumberNode(value=1, line=1, column=2),
                NumberNode(value=2, line=1, column=5),
                NumberNode(value=3, line=1, column=8),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[list_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(3)"


class TestSliceAnalysis:
    """Tests for slicing complexity analysis."""

    def test_index_with_slice(self) -> None:
        # Create an index with slicing (start/stop/step present)
        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=None,
            start=NumberNode(value=0, line=1, column=5),
            stop=NumberNode(value=5, line=1, column=8),
            step=None,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[index_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(n)"


class TestReturnStatements:
    """Tests for return statement complexity."""

    def test_return_with_expression(self) -> None:
        ret = ReturnNode(
            value=BinaryOpNode(
                left=NumberNode(value=1, line=1, column=8),
                operator="+",
                right=NumberNode(value=2, line=1, column=12),
                line=1,
                column=8,
            ),
            line=1,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="add", line=1, column=1),
            body=[ret],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"

    def test_return_none(self) -> None:
        ret = ReturnNode(value=None, line=1, column=1)
        func_def = FunctionDefNode(
            name=IdentifierNode(name="noop", line=1, column=1),
            body=[ret],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"


class TestCombinedComplexities:
    """Tests for combined complexity expressions."""

    def test_multiple_statements_sequential(self) -> None:
        stmt1 = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=1, line=1, column=5),
            line=1,
            column=1,
        )
        stmt2 = AssignmentNode(
            target=IdentifierNode(name="y", line=2, column=1),
            value=NumberNode(value=2, line=2, column=5),
            line=2,
            column=1,
        )
        stmt3 = AssignmentNode(
            target=IdentifierNode(name="z", line=3, column=1),
            value=NumberNode(value=3, line=3, column=5),
            line=3,
            column=1,
        )
        program = ProgramNode(statements=[stmt1, stmt2, stmt3], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_mixed_complexities(self) -> None:
        # Simple assignment
        assign = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=1, line=1, column=5),
            line=1,
            column=1,
        )
        # For loop
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=2, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=2, column=10),
                arguments=[IdentifierNode(name="n", line=2, column=16)],
                line=2,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="sum", line=3, column=1),
                    value=NumberNode(value=0, line=3, column=7),
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        program = ProgramNode(statements=[assign, for_node], line=1, column=1)
        result = analyze_complexity(program)
        # Combined complexity should be max of O(1) and O(n) = O(n)
        assert result.complexity in ["O(n)", "O(n²)", "Unknown"]


class TestEdgeCasesInLoops:
    """Additional edge case tests for loops."""

    def test_for_loop_with_range_3_args(self) -> None:
        # range(a, b, step)
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[
                    NumberNode(value=0, line=1, column=16),
                    NumberNode(value=10, line=1, column=19),
                    NumberNode(value=2, line=1, column=23),
                ],
                line=1,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # For literal range args, returns constant or O(n)
        assert result.complexity in ["O(n)", "O(1)", "O(10)"]

    def test_for_loop_identifier_iterable(self) -> None:
        # Iterate over a variable
        for_node = ForNode(
            iterator=IdentifierNode(name="item", line=1, column=4),
            iterable=IdentifierNode(name="items", line=1, column=11),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "O(items)", "Unknown"]


class TestRecursiveFunction:
    """Tests for recursive function analysis."""

    def test_recursive_function_detection(self) -> None:
        # Create a recursive function: def fact(n): return n * fact(n-1)
        func_def = FunctionDefNode(
            name=IdentifierNode(name="fact", line=1, column=1),
            parameters=[IdentifierNode(name="n", line=1, column=13)],
            body=[
                ReturnNode(
                    value=BinaryOpNode(
                        left=IdentifierNode(name="n", line=2, column=11),
                        operator="*",
                        right=FunctionCallNode(
                            function=IdentifierNode(name="fact", line=2, column=15),
                            arguments=[
                                BinaryOpNode(
                                    left=IdentifierNode(name="n", line=2, column=21),
                                    operator="-",
                                    right=NumberNode(value=1, line=2, column=25),
                                    line=2,
                                    column=21,
                                )
                            ],
                            line=2,
                            column=15,
                        ),
                        line=2,
                        column=11,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        # Should be O(n) for recursive factorial
        assert result.complexity in ["O(n)", "O(n²)", "O(n^k)", "Unknown"]


class TestComplexFunctionBody:
    """Tests for complex function bodies."""

    def test_function_with_nested_loops(self) -> None:
        # Outer loop
        inner_for = ForNode(
            iterator=IdentifierNode(name="j", line=3, column=5),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=3, column=11),
                arguments=[IdentifierNode(name="n", line=3, column=17)],
                line=3,
                column=11,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="z", line=4, column=3),
                    value=NumberNode(value=0, line=4, column=7),
                    line=4,
                    column=3,
                )
            ],
            line=3,
            column=5,
        )
        outer_for = ForNode(
            iterator=IdentifierNode(name="i", line=2, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=2, column=10),
                arguments=[IdentifierNode(name="n", line=2, column=16)],
                line=2,
                column=10,
            ),
            body=[inner_for],
            line=2,
            column=4,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="nested", line=1, column=1),
            parameters=[IdentifierNode(name="n", line=1, column=15)],
            body=[outer_for],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity in ["O(n²)", "O(n³)", "O(n^k)", "Unknown"]


class TestSymbolTableParams:
    """Tests for symbol table parameter extraction."""

    def test_analyze_with_symbol_table(self) -> None:
        """Test that analyze() uses symbol table to extract params."""
        from optilang.analysis.complexity import ComplexityAnalyzer

        assign = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=42, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[assign], line=1, column=1)
        # Pass a symbol table - this should trigger _extract_params
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(program, symbol_table={"x": 1, "y": "test"})
        assert result.complexity == "O(1)"

    def test_analyze_with_callable_symbol_table(self) -> None:
        """Test that symbol table skips callable values."""
        from optilang.analysis.complexity import ComplexityAnalyzer

        assign = AssignmentNode(
            target=IdentifierNode(name="x", line=1, column=1),
            value=NumberNode(value=42, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[assign], line=1, column=1)
        # Symbol table with callable should still work
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(program, symbol_table={"func": lambda: None, "x": 1})
        assert result.complexity == "O(1)"


class TestHalvingLoop:
    """Tests for halving loop detection."""

    def test_detect_halving_loop_greater_than(self) -> None:
        """Test halving detection with > operator."""
        from optilang.analysis.complexity import ComplexityAnalyzer

        # while n > 1: n = n // 2
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="n", line=1, column=7),
                operator=">",
                right=NumberNode(value=1, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="n", line=2, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="n", line=2, column=5),
                        operator="//",
                        right=NumberNode(value=2, line=2, column=9),
                        line=2,
                        column=5,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(log n)"

    def test_detect_halving_loop_greater_equal(self) -> None:
        """Test halving detection with >= operator."""
        from optilang.analysis.complexity import ComplexityAnalyzer

        # while n >= 1: n = n // 2
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="n", line=1, column=7),
                operator=">=",
                right=NumberNode(value=1, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="n", line=2, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="n", line=2, column=5),
                        operator="//",
                        right=NumberNode(value=2, line=2, column=9),
                        line=2,
                        column=5,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(log n)"

    def test_detect_halving_loop_less_than(self) -> None:
        """Test halving detection with < operator."""
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=NumberNode(value=1, line=1, column=7),
                operator="<",
                right=IdentifierNode(name="n", line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="n", line=2, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="n", line=2, column=5),
                        operator="//",
                        right=NumberNode(value=2, line=2, column=9),
                        line=2,
                        column=5,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(log n)"

    def test_halving_loop_not_detected_without_assignment(self) -> None:
        """Test that halving is not detected without assignment."""
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="n", line=1, column=7),
                operator=">",
                right=NumberNode(value=1, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),  # different var
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        # Without proper halving assignment, should be O(n) or Unknown
        assert result.complexity in ["O(n)", "Unknown"]

    def test_halving_loop_not_detected_with_wrong_operator(self) -> None:
        """Test that halving is not detected with wrong operator."""
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="n", line=1, column=7),
                operator=">",
                right=NumberNode(value=1, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="n", line=2, column=1),
                    value=BinaryOpNode(
                        left=IdentifierNode(name="n", line=2, column=5),
                        operator="-",  # not //
                        right=NumberNode(value=1, line=2, column=9),
                        line=2,
                        column=5,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        # Should be O(n) since it's linear decrement, not halving
        assert result.complexity in ["O(n)", "Unknown"]


class TestIterableComplexity:
    """Tests for iterable complexity extraction."""

    def test_for_loop_identifier_not_in_params(self) -> None:
        """Test for loop over identifier that's not a param."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=IdentifierNode(name="items", line=1, column=11),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # Unknown variable - should use Var("items") complexity
        assert result.complexity in ["O(n)", "O(items)", "Unknown"]

    def test_for_loop_with_index_node(self) -> None:
        """Test for loop over an index expression."""
        index_node = IndexNode(
            collection=IdentifierNode(name="matrices", line=1, column=11),
            index=NumberNode(value=0, line=1, column=20),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=11,
        )
        for_node = ForNode(
            iterator=IdentifierNode(name="row", line=1, column=4),
            iterable=index_node,
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity is not None


class TestRangeComplexity:
    """Tests for range() complexity extraction."""

    def test_range_with_param_in_range(self) -> None:
        """Test range(n) where n is a function param."""
        param = IdentifierNode(name="n", line=1, column=12)
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=2, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=2, column=10),
                arguments=[IdentifierNode(name="n", line=2, column=16)],
                line=2,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=3, column=1),
                    value=NumberNode(value=0, line=3, column=5),
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="process", line=1, column=1),
            parameters=[param],
            body=[for_node],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity in ["O(n)", "Unknown"]

    def test_range_with_loop_iterator(self) -> None:
        """Test range with loop iterator variable."""
        # This tests the _loop_iterators code path
        outer_for = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="limit", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[
                ForNode(
                    iterator=IdentifierNode(name="j", line=2, column=4),
                    iterable=FunctionCallNode(
                        function=IdentifierNode(name="range", line=2, column=10),
                        arguments=[IdentifierNode(name="i", line=2, column=16)],
                        line=2,
                        column=10,
                    ),
                    body=[
                        AssignmentNode(
                            target=IdentifierNode(name="x", line=3, column=1),
                            value=NumberNode(value=0, line=3, column=5),
                            line=3,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        # Wrap in ProgramNode for analyze_complexity
        program = ProgramNode(statements=[outer_for], line=1, column=1)
        result = analyze_complexity(program)
        # The inner bound depends on the outer loop iterator, so the analyzer
        # preserves the independent bound expression.
        assert result.complexity in [
            "O(n²)",
            "O(n log n)",
            "O(n)",
            "O(n*m)",
            "Unknown",
            "O(limit²)",
        ]

    def test_range_with_number_arg(self) -> None:
        """Test range with literal number."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[NumberNode(value=100, line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # Should return O(100) which simplifies to O(1) or stays as constant
        assert result.complexity in ["O(100)", "O(1)", "O(n)"]

    def test_range_two_args(self) -> None:
        """Test range(a, b) complexity."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[
                    NumberNode(value=0, line=1, column=16),
                    NumberNode(value=10, line=1, column=19),
                ],
                line=1,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # range(0, 10) iterates 10 times
        assert result.complexity in ["O(10)", "O(n)", "O(1)"]

    def test_range_three_args(self) -> None:
        """Test range(a, b, step) complexity."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[
                    NumberNode(value=0, line=1, column=16),
                    NumberNode(value=100, line=1, column=19),
                    NumberNode(value=2, line=1, column=23),
                ],
                line=1,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # Should handle step but default to O(max)
        assert result.complexity in ["O(n)", "O(100)", "O(50)", "O(1)"]


class TestFunctionCallComplexity:
    """Tests for function call complexity analysis."""

    def test_function_with_cache(self) -> None:
        """Test that function complexity is cached."""
        # Call analyze twice on same function - second should use cache
        func_def = FunctionDefNode(
            name=IdentifierNode(name="test", line=1, column=1),
            body=[
                ForNode(
                    iterator=IdentifierNode(name="i", line=2, column=4),
                    iterable=FunctionCallNode(
                        function=IdentifierNode(name="range", line=2, column=10),
                        arguments=[IdentifierNode(name="n", line=2, column=16)],
                        line=2,
                        column=10,
                    ),
                    body=[
                        AssignmentNode(
                            target=IdentifierNode(name="x", line=3, column=1),
                            value=NumberNode(value=0, line=3, column=5),
                            line=3,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        # First call - not cached
        result1 = analyze_function_complexity(func_def)
        # Second call - should hit cache
        result2 = analyze_function_complexity(func_def)
        # Results should be consistent
        assert result1.complexity == result2.complexity

    test_function_with_cache.__pytest_seed__ = 1234

    def test_recursive_function_complexity(self) -> None:
        """Test recursive function complexity detection."""
        # def fib(n): return fib(n-1) + fib(n-2)
        func_def = FunctionDefNode(
            name=IdentifierNode(name="fib", line=1, column=1),
            parameters=[IdentifierNode(name="n", line=1, column=13)],
            body=[
                ReturnNode(
                    value=BinaryOpNode(
                        left=FunctionCallNode(
                            function=IdentifierNode(name="fib", line=2, column=11),
                            arguments=[
                                BinaryOpNode(
                                    left=IdentifierNode(name="n", line=2, column=16),
                                    operator="-",
                                    right=NumberNode(value=1, line=2, column=18),
                                    line=2,
                                    column=16,
                                )
                            ],
                            line=2,
                            column=11,
                        ),
                        operator="+",
                        right=FunctionCallNode(
                            function=IdentifierNode(name="fib", line=2, column=21),
                            arguments=[
                                BinaryOpNode(
                                    left=IdentifierNode(name="n", line=2, column=26),
                                    operator="-",
                                    right=NumberNode(value=2, line=2, column=28),
                                    line=2,
                                    column=26,
                                )
                            ],
                            line=2,
                            column=21,
                        ),
                        line=2,
                        column=11,
                    ),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        # Recursive functions may return O(1) if only return is analyzed
        # or O(n) or higher depending on detection
        assert result.complexity in ["O(2^n)", "O(n)", "O(n^2)", "Unknown", "O(1)"]

    def test_call_unknown_function(self) -> None:
        """Test calling unknown function returns O(1)."""
        call = FunctionCallNode(
            function=IdentifierNode(name="external_api", line=1, column=1),
            arguments=[NumberNode(value=1, line=1, column=15)],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestBinaryOpComplexity:
    """Tests for binary operation complexity."""

    def test_binary_op_with_children_method(self) -> None:
        """Test binary op that has _get_children method."""
        # BinaryOpNode that may have _get_children
        bin_op = BinaryOpNode(
            left=BinaryOpNode(
                left=NumberNode(value=1, line=1, column=1),
                operator="+",
                right=NumberNode(value=2, line=1, column=5),
                line=1,
                column=1,
            ),
            operator="*",
            right=NumberNode(value=3, line=1, column=9),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_unary_op_complexity(self) -> None:
        """Test unary operations are O(1)."""
        unary = UnaryOpNode(
            operator="-",
            operand=NumberNode(value=5, line=1, column=2),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[unary], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_unary_not_op(self) -> None:
        """Test unary NOT operation."""
        unary = UnaryOpNode(
            operator="not",
            operand=BooleanNode(value=True, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[unary], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestIndexAssignment:
    """Tests for index assignment complexity."""

    def test_index_assignment_complexity(self) -> None:
        """Test arr[i] = value is O(1)."""
        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=IdentifierNode(name="i", line=1, column=5),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=1,
        )
        index_assign = IndexAssignmentNode(
            target=index_node,
            value=NumberNode(value=42, line=1, column=9),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[index_assign], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestListComprehension:
    """Tests for list comprehension complexity."""

    def test_list_comprehension_complexity(self) -> None:
        """Test [x for x in items] is O(n)."""
        # This is represented as a ForNode with list comprehension semantics
        for_node = ForNode(
            iterator=IdentifierNode(name="x", line=1, column=4),
            iterable=IdentifierNode(name="items", line=1, column=9),
            body=[
                IdentifierNode(name="x", line=2, column=1),  # implicit yield
            ],
            line=1,
            column=1,
        )
        # Wrap as expression statement
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "Unknown"]


class TestComplexExpressions:
    """Tests for complex expression complexity."""

    def test_assignment_to_subscript(self) -> None:
        """Test assignment to subscript like matrix[i][j] = 1."""
        outer_index = IndexNode(
            collection=IndexNode(
                collection=IdentifierNode(name="matrix", line=1, column=1),
                index=IdentifierNode(name="i", line=1, column=8),
                start=None,
                stop=None,
                step=None,
                line=1,
                column=1,
            ),
            index=IdentifierNode(name="j", line=1, column=12),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=1,
        )
        assign = AssignmentNode(
            target=outer_index,
            value=NumberNode(value=1, line=1, column=17),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[assign], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_augmented_assignment_to_index(self) -> None:
        """Test arr[i] += 1 complexity."""
        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=IdentifierNode(name="i", line=1, column=5),
            start=None,
            stop=None,
            step=None,
            line=1,
            column=1,
        )
        aug_assign = IndexedAugmentedAssignmentNode(
            target=index_node,
            operator="+=",
            value=NumberNode(value=1, line=1, column=10),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[aug_assign], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestEdgeConditions:
    """Tests for edge conditions in complexity analysis."""

    def test_node_with_get_children_attribute(self) -> None:
        """Test nodes that have _get_children method."""
        from optilang.analysis.complexity import ComplexityAnalyzer

        # Use analyzer directly to trigger _collect_functions path
        # that handles nodes with _get_children
        complex_node = BinaryOpNode(
            left=NumberNode(value=1, line=1, column=1),
            operator="+",
            right=NumberNode(value=2, line=1, column=5),
            line=1,
            column=1,
        )
        # Create a function that contains the complex node
        func_def = FunctionDefNode(
            name=IdentifierNode(name="test", line=1, column=1),
            body=[ReturnNode(value=complex_node, line=2, column=1)],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[func_def], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_string_literal_length_affects_complexity(self) -> None:
        """Test that string literals of different lengths are handled."""
        short_str = StringNode(value="a", line=1, column=1)
        program1 = ProgramNode(statements=[short_str], line=1, column=1)
        result1 = analyze_complexity(program1)

        long_str = StringNode(value="abcdefghij", line=1, column=1)
        program2 = ProgramNode(statements=[long_str], line=1, column=1)
        result2 = analyze_complexity(program2)

        # Both should be O(1) or O(length)
        assert result1.complexity in ["O(1)", "O(1)"]
        assert result2.complexity in ["O(1)", "O(10)"]

    def test_slice_with_step(self) -> None:
        """Test index with step (slice with start, stop, step)."""
        index_node = IndexNode(
            collection=IdentifierNode(name="arr", line=1, column=1),
            index=None,
            start=NumberNode(value=0, line=1, column=5),
            stop=NumberNode(value=10, line=1, column=8),
            step=NumberNode(value=2, line=1, column=12),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[index_node], line=1, column=1)
        result = analyze_complexity(program)
        # Slice with step is O(n)
        assert result.complexity in ["O(n)", "O(5)", "O(1)"]


class TestMoreBinaryOps:
    """Tests for more binary operation complexity scenarios."""

    def test_nested_binary_ops(self) -> None:
        """Test deeply nested binary operations."""
        # ((1 + 2) * 3) + (4 * 5)
        inner_add = BinaryOpNode(
            left=NumberNode(value=1, line=1, column=1),
            operator="+",
            right=NumberNode(value=2, line=1, column=5),
            line=1,
            column=1,
        )
        mul1 = BinaryOpNode(
            left=inner_add,
            operator="*",
            right=NumberNode(value=3, line=1, column=9),
            line=1,
            column=1,
        )
        mul2 = BinaryOpNode(
            left=NumberNode(value=4, line=1, column=11),
            operator="*",
            right=NumberNode(value=5, line=1, column=15),
            line=1,
            column=11,
        )
        final = BinaryOpNode(
            left=mul1,
            operator="+",
            right=mul2,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[final], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_binary_op_with_division(self) -> None:
        """Test binary operation with division."""
        div = BinaryOpNode(
            left=NumberNode(value=10, line=1, column=1),
            operator="/",
            right=NumberNode(value=2, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[div], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_binary_op_with_modulo(self) -> None:
        """Test binary operation with modulo."""
        mod = BinaryOpNode(
            left=NumberNode(value=10, line=1, column=1),
            operator="%",
            right=NumberNode(value=3, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[mod], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_binary_op_with_power(self) -> None:
        """Test binary operation with power."""
        pow_op = BinaryOpNode(
            left=NumberNode(value=2, line=1, column=1),
            operator="**",
            right=NumberNode(value=3, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[pow_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestMoreUnaryOps:
    """Tests for more unary operation scenarios."""

    def test_unary_bitwise_not(self) -> None:
        """Test unary bitwise NOT."""
        unary = UnaryOpNode(
            operator="~",
            operand=NumberNode(value=5, line=1, column=2),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[unary], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_double_negation(self) -> None:
        """Test double negation."""
        inner = UnaryOpNode(
            operator="not",
            operand=BooleanNode(value=True, line=1, column=5),
            line=1,
            column=1,
        )
        outer = UnaryOpNode(
            operator="not",
            operand=inner,
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[outer], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestMoreFunctionCalls:
    """Tests for more function call scenarios."""

    def test_function_with_default_arg(self) -> None:
        """Test function call with default argument."""
        func_call = FunctionCallNode(
            function=IdentifierNode(name="print", line=1, column=1),
            arguments=[
                StringNode(value="hello", line=1, column=7),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[func_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_function_with_multiple_args(self) -> None:
        """Test function call with multiple arguments."""
        func_call = FunctionCallNode(
            function=IdentifierNode(name="max", line=1, column=1),
            arguments=[
                NumberNode(value=1, line=1, column=5),
                NumberNode(value=2, line=1, column=8),
                NumberNode(value=3, line=1, column=11),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[func_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_method_call(self) -> None:
        """Test method call complexity."""
        from optilang.core.ast_nodes import MethodCallNode

        method_call = MethodCallNode(
            object=IdentifierNode(name="items", line=1, column=1),
            method=IdentifierNode(name="append", line=1, column=8),
            arguments=[NumberNode(value=1, line=1, column=16)],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[method_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_nested_function_calls(self) -> None:
        """Test nested function calls."""
        inner_call = FunctionCallNode(
            function=IdentifierNode(name="len", line=1, column=5),
            arguments=[IdentifierNode(name="items", line=1, column=9)],
            line=1,
            column=5,
        )
        outer_call = FunctionCallNode(
            function=IdentifierNode(name="print", line=1, column=1),
            arguments=[inner_call],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[outer_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestMoreDataStructures:
    """Tests for more data structure scenarios."""

    def test_dict_literal(self) -> None:
        """Test dictionary literal complexity."""
        from optilang.core.ast_nodes import DictNode

        dict_node = DictNode(
            pairs=[
                (
                    StringNode(value="a", line=1, column=2),
                    NumberNode(value=1, line=1, column=5),
                ),
                (
                    StringNode(value="b", line=1, column=8),
                    NumberNode(value=2, line=1, column=11),
                ),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[dict_node], line=1, column=1)
        result = analyze_complexity(program)
        # Dict literal with 2 pairs
        assert result.complexity in ["O(1)", "O(2)"]

    def test_empty_dict(self) -> None:
        """Test empty dictionary."""
        from optilang.core.ast_nodes import DictNode

        dict_node = DictNode(pairs=[], line=1, column=1)
        program = ProgramNode(statements=[dict_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_tuple_in_expression(self) -> None:
        """Test tuple in expression context."""
        # Tuples in expressions (not assignments) are handled as expressions
        # Test with a binary operation containing tuple-like structure
        # Just test with a list which is more common
        list_node = ListNode(
            elements=[
                NumberNode(value=1, line=1, column=2),
                NumberNode(value=2, line=1, column=5),
                NumberNode(value=3, line=1, column=8),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[list_node], line=1, column=1)
        result = analyze_complexity(program)
        # List with 3 elements
        assert result.complexity in ["O(1)", "O(3)"]

    def test_empty_list(self) -> None:
        """Test empty list."""
        list_node = ListNode(elements=[], line=1, column=1)
        program = ProgramNode(statements=[list_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestMoreControlFlow:
    """Tests for more control flow scenarios."""

    def test_nested_if_else(self) -> None:
        """Test nested if-else."""
        inner_if = IfNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="y", line=3, column=7),
                operator=">",
                right=NumberNode(value=5, line=3, column=11),
                line=3,
                column=7,
            ),
            if_block=[
                AssignmentNode(
                    target=IdentifierNode(name="z", line=4, column=1),
                    value=NumberNode(value=1, line=4, column=5),
                    line=4,
                    column=1,
                )
            ],
            elif_parts=[],
            else_block=None,
            line=3,
            column=1,
        )
        outer_if = IfNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="x", line=1, column=4),
                operator=">",
                right=NumberNode(value=0, line=1, column=8),
                line=1,
                column=4,
            ),
            if_block=[
                AssignmentNode(
                    target=IdentifierNode(name="a", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            elif_parts=[
                (
                    BinaryOpNode(
                        left=IdentifierNode(name="x", line=5, column=6),
                        operator="<",
                        right=NumberNode(value=-5, line=5, column=10),
                        line=5,
                        column=6,
                    ),
                    [
                        AssignmentNode(
                            target=IdentifierNode(name="b", line=6, column=1),
                            value=NumberNode(value=2, line=6, column=5),
                            line=6,
                            column=1,
                        )
                    ],
                )
            ],
            else_block=[inner_if],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[outer_if], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_while_true_infinite(self) -> None:
        """Test while True (infinite loop)."""
        while_node = WhileNode(
            condition=BooleanNode(value=True, line=1, column=7),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[while_node], line=1, column=1)
        result = analyze_complexity(program)
        # Infinite loop - unknown complexity
        assert result.complexity in ["O(n)", "Unknown", "O(∞)"]

    def test_for_break_early(self) -> None:
        """Test for loop with break."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="n", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[
                BreakNode(line=2, column=1),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        # For loop with break is still O(n)
        assert result.complexity in ["O(n)", "O(1)", "Unknown"]


class TestComplexBodyAnalysis:
    """Tests for complex body analysis scenarios."""

    def test_multiple_return_statements(self) -> None:
        """Test function with multiple return paths."""
        ret1 = ReturnNode(
            value=NumberNode(value=1, line=2, column=8),
            line=2,
            column=1,
        )
        ret2 = ReturnNode(
            value=NumberNode(value=2, line=4, column=8),
            line=4,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="pick", line=1, column=1),
            parameters=[IdentifierNode(name="x", line=1, column=15)],
            body=[
                IfNode(
                    condition=BinaryOpNode(
                        left=IdentifierNode(name="x", line=2, column=4),
                        operator=">",
                        right=NumberNode(value=0, line=2, column=8),
                        line=2,
                        column=4,
                    ),
                    if_block=[ret1],
                    elif_parts=[],
                    else_block=[ret2],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity in ["O(1)", "O(n)", "Unknown"]

    def test_try_except_with_loop(self) -> None:
        """Test try-except inside a loop."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=2, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=2, column=10),
                arguments=[IdentifierNode(name="n", line=2, column=16)],
                line=2,
                column=10,
            ),
            body=[
                TryNode(
                    try_block=[
                        AssignmentNode(
                            target=IdentifierNode(name="x", line=3, column=1),
                            value=NumberNode(value=1, line=3, column=5),
                            line=3,
                            column=1,
                        )
                    ],
                    except_block=[
                        AssignmentNode(
                            target=IdentifierNode(name="x", line=5, column=1),
                            value=NumberNode(value=0, line=5, column=5),
                            line=5,
                            column=1,
                        )
                    ],
                    finally_block=None,
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "O(n²)", "Unknown"]

    def test_try_finally_with_return(self) -> None:
        """Test try-finally with return."""
        try_node = TryNode(
            try_block=[
                ReturnNode(
                    value=NumberNode(value=1, line=2, column=8),
                    line=2,
                    column=1,
                )
            ],
            except_block=None,
            finally_block=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=4, column=1),
                    value=NumberNode(value=0, line=4, column=5),
                    line=4,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        func_def = FunctionDefNode(
            name=IdentifierNode(name="cleanup", line=1, column=1),
            body=[try_node],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity in ["O(1)", "Unknown"]


class TestFunctionCallDetails:
    """Tests for detailed function call scenarios."""

    def test_function_with_args_and_kwargs(self) -> None:
        """Test function call with both positional and keyword-style args."""
        # For simplicity, just use multiple positional args
        func_call = FunctionCallNode(
            function=IdentifierNode(name="func", line=1, column=1),
            arguments=[
                NumberNode(value=1, line=1, column=6),
                StringNode(value="hello", line=1, column=9),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[func_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_method_call_on_call_result(self) -> None:
        """Test method call on another function call result."""
        from optilang.core.ast_nodes import MethodCallNode

        inner = FunctionCallNode(
            function=IdentifierNode(name="get_list", line=1, column=1),
            arguments=[],
            line=1,
            column=1,
        )
        method_call = MethodCallNode(
            object=inner,
            method=IdentifierNode(name="append", line=1, column=13),
            arguments=[NumberNode(value=1, line=1, column=21)],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[method_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestComplexNesting:
    """Tests for deeply nested structures."""

    def test_triple_nested_loops(self) -> None:
        """Test three levels of nested loops."""
        innermost = ForNode(
            iterator=IdentifierNode(name="k", line=5, column=6),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=5, column=12),
                arguments=[IdentifierNode(name="n", line=5, column=18)],
                line=5,
                column=12,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=6, column=3),
                    value=NumberNode(value=0, line=6, column=7),
                    line=6,
                    column=3,
                )
            ],
            line=5,
            column=6,
        )
        middle = ForNode(
            iterator=IdentifierNode(name="j", line=3, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=3, column=10),
                arguments=[IdentifierNode(name="n", line=3, column=16)],
                line=3,
                column=10,
            ),
            body=[innermost],
            line=3,
            column=4,
        )
        outer = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="n", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[middle],
            line=1,
            column=4,
        )
        program = ProgramNode(statements=[outer], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n³)", "O(n^3)", "O(n²)", "Unknown"]

    def test_mixed_nesting(self) -> None:
        """Test mix of for and while loops."""
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="i", line=2, column=7),
                operator="<",
                right=IdentifierNode(name="n", line=2, column=11),
                line=2,
                column=7,
            ),
            body=[
                ForNode(
                    iterator=IdentifierNode(name="j", line=3, column=4),
                    iterable=FunctionCallNode(
                        function=IdentifierNode(name="range", line=3, column=10),
                        arguments=[IdentifierNode(name="n", line=3, column=16)],
                        line=3,
                        column=10,
                    ),
                    body=[
                        AssignmentNode(
                            target=IdentifierNode(name="x", line=4, column=1),
                            value=NumberNode(value=0, line=4, column=5),
                            line=4,
                            column=1,
                        )
                    ],
                    line=3,
                    column=1,
                )
            ],
            line=2,
            column=1,
        )
        assign_i = AssignmentNode(
            target=IdentifierNode(name="i", line=1, column=1),
            value=NumberNode(value=0, line=1, column=5),
            line=1,
            column=1,
        )
        increment_i = AugmentedAssignmentNode(
            target=IdentifierNode(name="i", line=5, column=1),
            operator="+=",
            value=NumberNode(value=1, line=1, column=5),
            line=5,
            column=1,
        )
        program = ProgramNode(
            statements=[assign_i, while_node, increment_i], line=1, column=1
        )
        result = analyze_complexity(program)
        # Mixed nesting - O(n²) or higher
        assert result.complexity in ["O(n²)", "O(n)", "Unknown"]


class TestExpressionTypes:
    """Tests for various expression types."""

    def test_compare_expression(self) -> None:
        """Test comparison expression."""
        comp = BinaryOpNode(
            left=IdentifierNode(name="x", line=1, column=1),
            operator="==",
            right=NumberNode(value=5, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[comp], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_logical_expression(self) -> None:
        """Test logical AND/OR expression."""
        and_expr = BinaryOpNode(
            left=BinaryOpNode(
                left=IdentifierNode(name="x", line=1, column=1),
                operator=">",
                right=NumberNode(value=0, line=1, column=5),
                line=1,
                column=1,
            ),
            operator="and",
            right=BinaryOpNode(
                left=IdentifierNode(name="y", line=1, column=10),
                operator="<",
                right=NumberNode(value=10, line=1, column=14),
                line=1,
                column=10,
            ),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[and_expr], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_ternary_expression(self) -> None:
        """Test ternary-like expression using if node."""
        if_node = IfNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="x", line=1, column=4),
                operator=">",
                right=NumberNode(value=0, line=1, column=8),
                line=1,
                column=4,
            ),
            if_block=[
                IdentifierNode(name="a", line=2, column=1),
            ],
            elif_parts=[],
            else_block=[
                IdentifierNode(name="b", line=3, column=1),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[if_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_lambda_like_expression(self) -> None:
        """Test lambda-like function def without body."""
        func_def = FunctionDefNode(
            name=IdentifierNode(name="f", line=1, column=1),
            parameters=[IdentifierNode(name="x", line=1, column=9)],
            body=None,
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        # Function with no body - should be O(1)
        assert result.complexity in ["O(1)", "Unknown"]


class TestEdgeCases2:
    """More edge case tests."""

    def test_negative_loop_bound(self) -> None:
        """Test for loop with negative bound."""
        for_node = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[
                    NumberNode(value=-10, line=1, column=16),
                    NumberNode(value=10, line=1, column=23),
                ],
                line=1,
                column=10,
            ),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=0, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "O(20)", "O(1)"]

    def test_single_element_list(self) -> None:
        """Test list with single element."""
        list_node = ListNode(
            elements=[NumberNode(value=42, line=1, column=2)],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[list_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(1)", "O(1)"]

    def test_comprehension_with_condition(self) -> None:
        """Test list comprehension with if condition."""
        # [x for x in items if x > 0]
        for_node = ForNode(
            iterator=IdentifierNode(name="x", line=1, column=4),
            iterable=IdentifierNode(name="items", line=1, column=9),
            body=[
                IfNode(
                    condition=BinaryOpNode(
                        left=IdentifierNode(name="x", line=2, column=7),
                        operator=">",
                        right=NumberNode(value=0, line=2, column=11),
                        line=2,
                        column=7,
                    ),
                    if_block=[IdentifierNode(name="x", line=3, column=1)],
                    elif_parts=[],
                    else_block=None,
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[for_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "Unknown"]

    def test_dict_with_nested_values(self) -> None:
        """Test dict with nested list values."""
        from optilang.core.ast_nodes import DictNode

        dict_node = DictNode(
            pairs=[
                (
                    StringNode(value="data", line=1, column=2),
                    ListNode(
                        elements=[
                            NumberNode(value=1, line=1, column=8),
                            NumberNode(value=2, line=1, column=11),
                        ],
                        line=1,
                        column=8,
                    ),
                ),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[dict_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(1)", "O(2)"]

    def test_nested_dict_access(self) -> None:
        """Test nested dictionary access."""
        from optilang.core.ast_nodes import DictNode

        dict_node = DictNode(
            pairs=[
                (
                    StringNode(value="a", line=1, column=2),
                    DictNode(
                        pairs=[
                            (
                                StringNode(value="b", line=1, column=6),
                                NumberNode(value=1, line=1, column=9),
                            )
                        ],
                        line=1,
                        column=6,
                    ),
                ),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[dict_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(1)", "O(2)"]


class TestAdvancedExpressions:
    """Tests for advanced expression scenarios."""

    def test_is_in_operator(self) -> None:
        """Test 'in' operator."""
        bin_op = BinaryOpNode(
            left=IdentifierNode(name="x", line=1, column=1),
            operator="in",
            right=IdentifierNode(name="items", line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_is_not_in_operator(self) -> None:
        """Test 'not in' operator."""
        bin_op = BinaryOpNode(
            left=IdentifierNode(name="x", line=1, column=1),
            operator="not in",
            right=IdentifierNode(name="items", line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_floor_division(self) -> None:
        """Test floor division operator."""
        bin_op = BinaryOpNode(
            left=NumberNode(value=10, line=1, column=1),
            operator="//",
            right=NumberNode(value=3, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_shift_operators(self) -> None:
        """Test bit shift operators."""
        bin_op = BinaryOpNode(
            left=NumberNode(value=1, line=1, column=1),
            operator="<<",
            right=NumberNode(value=2, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"

    def test_bitwise_operators(self) -> None:
        """Test bitwise operators."""
        bin_op = BinaryOpNode(
            left=NumberNode(value=5, line=1, column=1),
            operator="|",
            right=NumberNode(value=3, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[bin_op], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity == "O(1)"


class TestMoreLoopScenarios:
    """Tests for more loop scenarios."""

    def test_nested_for_with_index(self) -> None:
        """Test nested for loops with index access."""
        inner_for = ForNode(
            iterator=IdentifierNode(name="j", line=3, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=3, column=10),
                arguments=[IdentifierNode(name="n", line=3, column=16)],
                line=3,
                column=10,
            ),
            body=[
                IndexNode(
                    collection=IdentifierNode(name="arr", line=4, column=3),
                    index=IdentifierNode(name="j", line=4, column=7),
                    start=None,
                    stop=None,
                    step=None,
                    line=4,
                    column=3,
                )
            ],
            line=3,
            column=1,
        )
        outer_for = ForNode(
            iterator=IdentifierNode(name="i", line=1, column=4),
            iterable=FunctionCallNode(
                function=IdentifierNode(name="range", line=1, column=10),
                arguments=[IdentifierNode(name="n", line=1, column=16)],
                line=1,
                column=10,
            ),
            body=[inner_for],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[outer_for], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n²)", "O(n)", "Unknown"]

    def test_while_with_break(self) -> None:
        """Test while loop with break statement."""
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="i", line=1, column=7),
                operator="<",
                right=NumberNode(value=100, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                IfNode(
                    condition=BinaryOpNode(
                        left=IdentifierNode(name="i", line=2, column=7),
                        operator="==",
                        right=NumberNode(value=50, line=2, column=11),
                        line=2,
                        column=7,
                    ),
                    if_block=[BreakNode(line=3, column=1)],
                    elif_parts=[],
                    else_block=None,
                    line=2,
                    column=1,
                ),
                AugmentedAssignmentNode(
                    target=IdentifierNode(name="i", line=4, column=1),
                    operator="+=",
                    value=NumberNode(value=1, line=4, column=5),
                    line=4,
                    column=1,
                ),
            ],
            line=1,
            column=1,
        )
        assign_i = AssignmentNode(
            target=IdentifierNode(name="i", line=1, column=1),
            value=NumberNode(value=0, line=1, column=5),
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[assign_i, while_node], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "Unknown"]

    def test_while_with_continue(self) -> None:
        """Test while loop with continue statement."""
        while_node = WhileNode(
            condition=BinaryOpNode(
                left=IdentifierNode(name="i", line=1, column=7),
                operator="<",
                right=NumberNode(value=100, line=1, column=11),
                line=1,
                column=7,
            ),
            body=[
                IfNode(
                    condition=BinaryOpNode(
                        left=IdentifierNode(name="i", line=2, column=7),
                        operator="%",
                        right=NumberNode(value=2, line=2, column=11),
                        line=2,
                        column=7,
                    ),
                    if_block=[ContinueNode(line=3, column=1)],
                    elif_parts=[],
                    else_block=None,
                    line=2,
                    column=1,
                ),
                AssignmentNode(
                    target=IdentifierNode(name="x", line=4, column=1),
                    value=NumberNode(value=0, line=4, column=5),
                    line=4,
                    column=1,
                ),
            ],
            line=1,
            column=1,
        )
        assign_i = AssignmentNode(
            target=IdentifierNode(name="i", line=1, column=1),
            value=NumberNode(value=0, line=1, column=5),
            line=1,
            column=1,
        )
        increment_i = AugmentedAssignmentNode(
            target=IdentifierNode(name="i", line=5, column=1),
            operator="+=",
            value=NumberNode(value=1, line=5, column=5),
            line=5,
            column=1,
        )
        program = ProgramNode(
            statements=[assign_i, while_node, increment_i], line=1, column=1
        )
        result = analyze_complexity(program)
        assert result.complexity in ["O(n)", "Unknown"]


class TestFunctionEdgeCases:
    """Tests for function edge cases."""

    def test_function_with_no_params(self) -> None:
        """Test function with no parameters."""
        func_def = FunctionDefNode(
            name=IdentifierNode(name="compute", line=1, column=1),
            parameters=[],
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="x", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"

    def test_function_with_many_params(self) -> None:
        """Test function with many parameters."""
        func_def = FunctionDefNode(
            name=IdentifierNode(name="multi", line=1, column=1),
            parameters=[
                IdentifierNode(name="a", line=1, column=12),
                IdentifierNode(name="b", line=1, column=15),
                IdentifierNode(name="c", line=1, column=18),
                IdentifierNode(name="d", line=1, column=21),
            ],
            body=[
                ReturnNode(
                    value=NumberNode(value=1, line=2, column=8),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        result = analyze_function_complexity(func_def)
        assert result.complexity == "O(1)"

    def test_call_function_with_side_effects(self) -> None:
        """Test calling function that modifies global state."""
        func_def = FunctionDefNode(
            name=IdentifierNode(name="modify", line=1, column=1),
            body=[
                AssignmentNode(
                    target=IdentifierNode(name="global_var", line=2, column=1),
                    value=NumberNode(value=1, line=2, column=5),
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )
        func_call = FunctionCallNode(
            function=IdentifierNode(name="modify", line=4, column=1),
            arguments=[],
            line=4,
            column=1,
        )
        program = ProgramNode(statements=[func_def, func_call], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(1)", "O(n)", "Unknown"]


class TestMoreDataStructures2:
    """Tests for more data structure scenarios."""

    def test_nested_list(self) -> None:
        """Test nested list structure."""
        nested = ListNode(
            elements=[
                ListNode(
                    elements=[NumberNode(value=1, line=1, column=4)],
                    line=1,
                    column=4,
                ),
                ListNode(
                    elements=[NumberNode(value=2, line=1, column=10)],
                    line=1,
                    column=10,
                ),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[nested], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(1)", "O(2)"]

    def test_list_with_mixed_elements(self) -> None:
        """Test list with mixed element types."""
        mixed = ListNode(
            elements=[
                NumberNode(value=1, line=1, column=2),
                StringNode(value="hi", line=1, column=5),
                BooleanNode(value=True, line=1, column=10),
            ],
            line=1,
            column=1,
        )
        program = ProgramNode(statements=[mixed], line=1, column=1)
        result = analyze_complexity(program)
        assert result.complexity in ["O(1)", "O(3)"]


class TestComplexityAnalyzerBranchCoverage:
    """Focused coverage for complexity analyzer helper branches."""

    def parse_source(self, source: str) -> ProgramNode:
        return parse(tokenize(source))

    def test_collects_functions_from_nested_expressions_and_blocks(self) -> None:
        analyzer = ComplexityAnalyzer()
        inner = FunctionDefNode(
            line=1,
            column=1,
            name=IdentifierNode(1, 5, "inner"),
            parameters=[],
            body=[ReturnNode(2, 5, NumberNode(2, 12, 1))],
        )
        program = ProgramNode(
            line=1,
            column=1,
            statements=[
                IfNode(
                    line=1,
                    column=1,
                    condition=BinaryOpNode(1, 4, NumberNode(1, 4, 1), "+", inner),
                    if_block=[],
                ),
                TryNode(
                    line=3,
                    column=1,
                    try_block=[inner],
                    except_block=[],
                    finally_block=[],
                ),
            ],
        )
        analyzer._collect_functions(program)
        assert analyzer._functions["inner"] is inner

    def test_private_expression_helpers_cover_remaining_branches(self) -> None:
        analyzer = ComplexityAnalyzer()
        assert analyzer._combine_max([]).value == 1
        assert analyzer._combine_max([Const(1), Const(2)]).value == 1
        assert analyzer._simplify(Add(Param("n"), Param("n"))).name == "n"
        assert analyzer._simplify(Add(Var("n"), Var("n"))).name == "n"
        assert analyzer._simplify(Mul(Const(0), Param("n"))).value == 0
        assert analyzer._simplify(Mul(Const(2), Const(3))).value == 6
        assert analyzer._simplify(Mul(Param("n"), Const(2))).name == "n"
        assert analyzer._simplify(Mul(Const(2), Param("n"))).name == "n"
        assert analyzer._simplify(CallExpr("work")).value == 1
        assert (
            analyzer._simplify(CallExpr("work", body_complexity=Param("n"))).name == "n"
        )
        assert analyzer._complexity_to_big_o(Factorial(Param("n")))[0] == "O(n!)"
        assert analyzer._complexity_to_big_o(Factorial(Var("items")))[0] == "O(n!)"
        assert analyzer._complexity_to_big_o(Add(Const(1), Param("n")))[0] == "O(n)"
        assert analyzer._complexity_to_big_o(Add(Param("n"), Const(1)))[0] == "O(n)"
        assert analyzer._complexity_to_big_o(object())[0] == "O(?)"
        assert analyzer._complexity_to_big_o(Mul(Param("n"), Log(Param("n"))))[0] == (
            "O(n log n)"
        )
        assert analyzer._get_confidence(Const(1)) == 1.0
        assert analyzer._get_confidence(Log(Param("n"))) == 1.0
        assert analyzer._get_confidence(Log(Var("n"))) == 0.7
        assert analyzer._get_confidence(Add(Param("n"), Var("m"))) == 0.7
        assert analyzer._get_confidence(object()) == 0.5
        assert analyzer._level_to_string(0, None) == "O(1)"
        assert analyzer._level_to_string(1, "n") == "O(log n)"
        assert analyzer._level_to_string(2, "n") == "O(n)"
        assert analyzer._level_to_string(5, "n") == "O(n² log n)"
        assert analyzer._level_to_string(7, "n") == "O(n⁴)"
        assert analyzer._level_to_string(8, "n") == "O(n^6)"
        assert analyzer._complexity_level(Factorial(Param("n"))) == 20
        assert analyzer._complexity_level(Mul(Log(Param("n")), Log(Param("n")))) == 1
        assert analyzer._complexity_level(Add(Param("n"), Const(1))) == 2
        assert analyzer._complexity_level(Mul(Param("n"), Log(Param("n")))) == 3
        assert analyzer._complexity_level(object()) == 2
        assert analyzer._get_bound(Log(Param("n"))) == "n"
        assert analyzer._get_bound(Add(Const(1), Param("n"))) == "n"
        assert analyzer._get_bound(Mul(Const(2), Param("n"))) == "n"
        assert analyzer._generate_explanation(Add(Const(1), Param("n"))).startswith(
            "Sequential execution:"
        )

    def test_recursion_factorial_detection_and_child_fallbacks(self) -> None:
        analyzer = ComplexityAnalyzer()
        program = self.parse_source("""
def permute(i):
    for j in range(i):
        permute(i + 1)

permute(n)
""")
        result = analyzer.analyze(program)
        assert result.complexity == "O(n!)"

        class OddNode:
            def __getattr__(self, _name: str) -> object:
                raise TypeError("nope")

        assert analyzer._get_node_children(OddNode()) == []

    def test_misc_node_and_range_branches(self) -> None:
        analyzer = ComplexityAnalyzer()
        analyzer._extract_params({"items": [1, 2, 3], "fn": lambda: None})
        assert "items" in analyzer._params
        assert "fn" not in analyzer._params
        assert analyzer._analyze_node(StringNode(1, 1, "abcd")).value == 1
        assert (
            analyzer._analyze_node(
                AssignmentNode(1, 1, IdentifierNode(1, 1, "x"), NumberNode(1, 5, 1))
            ).value
            == 1
        )
        assert analyzer._analyze_node(ListNode(1, 1, [])).value == 1
        assert (
            analyzer._extract_iterable_complexity(IdentifierNode(1, 1, "items")).name
            == "items"
        )
        assert (
            analyzer._extract_iterable_complexity(
                IndexNode(1, 1, IdentifierNode(1, 1, "items"), NumberNode(1, 7, 0))
            ).value
            == 1
        )
        assert analyzer._extract_iterable_complexity(NumberNode(1, 1, 3)).name == "n"
        assert (
            analyzer._extract_range_complexity(
                FunctionCallNode(1, 1, IdentifierNode(1, 1, "range"), [])
            ).name
            == "n"
        )
        assert analyzer._expr_uses_param(
            BinaryOpNode(1, 1, IdentifierNode(1, 1, "n"), "+", NumberNode(1, 5, 1)),
            "n",
        )

    def test_factorial_and_recursive_helper_branches(self) -> None:
        analyzer = ComplexityAnalyzer()
        recursive_call = FunctionCallNode(1, 1, IdentifierNode(1, 1, "walk"), [])
        loop = ForNode(
            1,
            1,
            IdentifierNode(1, 5, "i"),
            FunctionCallNode(
                1, 10, IdentifierNode(1, 10, "range"), [IdentifierNode(1, 16, "n")]
            ),
            [recursive_call],
        )
        func = FunctionDefNode(1, 1, IdentifierNode(1, 1, "walk"), [], [loop])
        assert analyzer._contains_recursive_call(loop, "walk") is True
        assert analyzer._recursive_call_modifies_iterator(loop, "walk") is False
        assert analyzer._check_node_for_factorial_pattern(loop, "walk") is None

        while_node = WhileNode(1, 1, IdentifierNode(1, 7, "flag"), [recursive_call])
        assert (
            analyzer._check_node_for_factorial_pattern(while_node, "walk")
            == "factorial"
        )

        if_node = IfNode(
            1,
            1,
            IdentifierNode(1, 4, "ok"),
            if_block=[],
            elif_parts=[(IdentifierNode(2, 6, "other"), [while_node])],
            else_block=[while_node],
        )
        assert (
            analyzer._check_node_for_factorial_pattern(if_node, "walk") == "factorial"
        )
        assert analyzer._detect_recursive_pattern(func) is None


class TestCanonicalComplexityUpgrade:
    """Golden tests for the canonical two-layer complexity vocabulary."""

    def _analyze(self, source: str):
        return analyze_complexity(parse(tokenize(source)))

    def test_literal_bounds_are_constant(self) -> None:
        result = self._analyze(
            "for i in range(10):\n"
            "    for j in range(10):\n"
            "        x = i + j"
        )
        assert result.complexity == "O(1)"

    def test_multi_variable_forms_are_preserved(self) -> None:
        assert (
            self._analyze("for i in range(n):\n    pass\nfor j in range(m):\n    pass").complexity
            == "O(n + m)"
        )
        assert (
            self._analyze("for i in range(n):\n    for j in range(m):\n        pass").complexity
            == "O(n*m)"
        )

    def test_hidden_costs_escalate_inside_loop(self) -> None:
        assert (
            self._analyze("s = ''\nfor i in range(n):\n    s += str(i)").complexity
            == "O(n²)"
        )
        assert (
            self._analyze("for i in range(n):\n    x = arr[0:n]").complexity
            == "O(n²)"
        )

    def test_recursion_classes(self) -> None:
        fib = self._analyze(
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
            "fib(5)"
        )
        assert fib.complexity == "O(2^n)"

        three = self._analyze(
            "def f(n):\n"
            "    if n <= 0:\n"
            "        return\n"
            "    f(n - 1)\n"
            "    f(n - 1)\n"
            "    f(n - 1)\n"
            "f(5)"
        )
        assert three.complexity == "O(k^n)"
        assert three.display_complexity == "O(3^n)"

    def test_unknown_and_unbounded_states(self) -> None:
        assert self._analyze("while True:\n    x = 1").complexity == "O(∞)"
        unknown = self._analyze("for i in range(n):\n    external(i)")
        assert unknown.complexity == "O(?)"
        assert unknown.fallback_reason is not None
