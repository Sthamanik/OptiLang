"""
Comprehensive tests for optilang/semantic_analyzer.py

Covers all four structural checks:
    1. 'return' outside function
    2. 'break' outside loop
    3. 'continue' outside loop
    4. Duplicate parameter names in function definition

Test categories:
    1. TestReturnOutsideFunction   — basic, line numbers, nested, valid, execute()
    2. TestBreakOutsideLoop        — basic, line numbers, nested, valid, execute()
    3. TestContinueOutsideLoop     — basic, line numbers, nested, valid, execute()
    4. TestDuplicateParameters     — basic, error message, nested, valid, execute()
    5. TestScopeDepth              — critical boundary: depth reset, leak prevention
    6. TestValidPrograms           — programs that must NOT raise SemanticError
    7. TestMultipleViolations      — first violation reported, only one raised
    8. TestPipelineOrder           — semantic errors block execution, profiling, output
    9. TestAnalyzerStateIsolation  — fresh state per instance, no cross-run leakage

Key design being tested (from semantic_analyzer.py):
    - Visitor Pattern: each node type dispatched to its own _visit_* method
    - Scope Stack: _function_depth and _loop_depth counters
    - Loop depth RESET to 0 on entering a function body — prevents break/continue
      inside a function-defined-inside-a-loop from inheriting outer loop's depth
    - Loop depth RESTORED after function body — outer loop still valid after visit
"""

from __future__ import annotations
import pytest
from optilang import execute
from optilang.lexer import tokenize
from optilang.parser import parse
from optilang.semantic_analyzer import SemanticAnalyzer
from optilang.utils.errors import SemanticError

# Helper


def analyze(source: str) -> None:
    """Tokenize, parse, and run semantic analysis on *source*."""
    tokens = tokenize(source)
    program = parse(tokens)
    SemanticAnalyzer().analyze(program)


# 1. RETURN OUTSIDE FUNCTION


class TestReturnOutsideFunction:

    # Basic violations

    def test_return_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("return 5")
        assert "return" in str(exc_info.value).lower()

    def test_return_without_value_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("return")

    def test_return_with_expression_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("return 1 + 2 * 3")

    # Line number accuracy

    def test_return_at_top_level_line_1(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("return 5")
        assert exc_info.value.line == 1

    def test_return_at_top_level_line_3(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("x = 1\ny = 2\nreturn x")
        assert exc_info.value.line == 3

    # Nested structure violations

    def test_return_inside_if_block_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if True:\n    return 1")

    def test_return_inside_else_block_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if True:\n    pass\nelse:\n    return 1")

    def test_return_inside_elif_block_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if False:\n    pass\nelif True:\n    return 1")

    def test_return_inside_for_loop_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("for i in range(10):\n    return i")

    def test_return_inside_while_loop_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("while True:\n    return 1")

    def test_return_inside_try_block_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    return 1\nexcept:\n    pass")

    def test_return_inside_except_block_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    pass\nexcept:\n    return 1")

    def test_return_inside_finally_block_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    pass\nexcept:\n    pass\nfinally:\n    return 1")

    # Scope boundary: depth resets after function exits

    def test_return_after_function_def_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f():\n" "    return 1\n" "return 2\n")

    # Valid uses

    def test_return_inside_function_is_valid(self) -> None:
        analyze("def f():\n    return 1")

    def test_return_without_value_inside_function_is_valid(self) -> None:
        analyze("def f():\n    return")

    def test_multiple_returns_inside_function_is_valid(self) -> None:
        analyze("def f(x):\n" "    if x > 0:\n" "        return x\n" "    return 0")

    def test_return_inside_nested_function_is_valid(self) -> None:
        analyze(
            "def outer():\n"
            "    def inner():\n"
            "        return 1\n"
            "    return inner()"
        )

    def test_return_inside_for_loop_inside_function_is_valid(self) -> None:
        analyze("def f(items):\n" "    for x in items:\n" "        return x")

    def test_return_inside_while_loop_inside_function_is_valid(self) -> None:
        analyze("def f():\n" "    while True:\n" "        return 1")

    def test_return_inside_if_inside_function_is_valid(self) -> None:
        analyze(
            "def f(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    else:\n"
            "        return 0"
        )

    def test_return_inside_try_inside_function_is_valid(self) -> None:
        analyze(
            "def f():\n"
            "    try:\n"
            "        return 1\n"
            "    except:\n"
            "        return 0"
        )

    # execute() integration

    def test_execute_return_at_top_level_returns_error(self) -> None:
        result = execute("return 42")
        assert len(result.errors) == 1
        assert "return" in result.errors[0].lower()

    def test_execute_return_at_top_level_produces_no_output(self) -> None:
        result = execute("return 42")
        assert result.output == ""

    def test_execute_return_at_top_level_produces_no_profiling(self) -> None:
        result = execute("return 42")
        assert result.profiling is None


# 2. BREAK OUTSIDE LOOP


class TestBreakOutsideLoop:

    # Basic violations

    def test_break_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("break")
        assert "break" in str(exc_info.value).lower()

    def test_break_inside_function_but_outside_loop_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f():\n    break")

    # Line number accuracy

    def test_break_at_top_level_line_1(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("break")
        assert exc_info.value.line == 1

    def test_break_at_top_level_line_2(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("x = 1\nbreak")
        assert exc_info.value.line == 2

    # Nested structure violations

    def test_break_inside_if_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if True:\n    break")

    def test_break_inside_else_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if True:\n    pass\nelse:\n    break")

    def test_break_inside_try_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    break\nexcept:\n    pass")

    def test_break_inside_except_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    pass\nexcept:\n    break")

    def test_break_inside_finally_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    pass\nexcept:\n    pass\nfinally:\n    break")

    def test_break_inside_function_inside_if_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f():\n" "    if True:\n" "        break")

    # Scope boundary: depth resets after loop exits

    def test_break_after_for_loop_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("for i in range(10):\n" "    pass\n" "break\n")

    def test_break_after_while_loop_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("while True:\n" "    pass\n" "break\n")

    # Valid uses

    def test_break_inside_for_loop_is_valid(self) -> None:
        analyze("for i in range(10):\n    break")

    def test_break_inside_while_loop_is_valid(self) -> None:
        analyze("while True:\n    break")

    def test_break_inside_nested_for_loops_is_valid(self) -> None:
        analyze("for i in range(5):\n" "    for j in range(5):\n" "        break")

    def test_break_inside_for_inside_while_is_valid(self) -> None:
        analyze("while True:\n" "    for i in range(10):\n" "        break")

    def test_break_inside_loop_inside_function_is_valid(self) -> None:
        analyze("def f():\n" "    for i in range(10):\n" "        break")

    def test_break_inside_if_inside_loop_is_valid(self) -> None:
        analyze("for i in range(10):\n" "    if i == 5:\n" "        break")

    def test_break_inside_try_inside_loop_is_valid(self) -> None:
        analyze(
            "for i in range(10):\n"
            "    try:\n"
            "        break\n"
            "    except:\n"
            "        pass"
        )

    # execute() integration

    def test_execute_break_at_top_level_returns_error(self) -> None:
        result = execute("break")
        assert len(result.errors) == 1
        assert "break" in result.errors[0].lower()

    def test_execute_break_produces_no_output(self) -> None:
        result = execute("break")
        assert result.output == ""

    def test_execute_break_produces_no_profiling(self) -> None:
        result = execute("break")
        assert result.profiling is None


# 3. CONTINUE OUTSIDE LOOP


class TestContinueOutsideLoop:

    # Basic violations

    def test_continue_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("continue")
        assert "continue" in str(exc_info.value).lower()

    def test_continue_inside_function_but_outside_loop_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f():\n    continue")

    # Line number accuracy

    def test_continue_at_top_level_line_1(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("continue")
        assert exc_info.value.line == 1

    def test_continue_at_top_level_line_2(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("x = 1\ncontinue")
        assert exc_info.value.line == 2

    # Nested structure violations

    def test_continue_inside_if_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if True:\n    continue")

    def test_continue_inside_else_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("if True:\n    pass\nelse:\n    continue")

    def test_continue_inside_try_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    continue\nexcept:\n    pass")

    def test_continue_inside_except_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    pass\nexcept:\n    continue")

    def test_continue_inside_finally_at_top_level_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("try:\n    pass\nexcept:\n    pass\nfinally:\n    continue")

    def test_continue_inside_function_inside_if_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f():\n" "    if True:\n" "        continue")

    # Scope boundary: depth resets after loop exits

    def test_continue_after_for_loop_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("for i in range(10):\n" "    pass\n" "continue\n")

    def test_continue_after_while_loop_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("while True:\n" "    pass\n" "continue\n")

    # Valid uses

    def test_continue_inside_for_loop_is_valid(self) -> None:
        analyze("for i in range(10):\n" "    if i == 5:\n" "        continue")

    def test_continue_inside_while_loop_is_valid(self) -> None:
        analyze("while True:\n    continue")

    def test_continue_inside_nested_loops_is_valid(self) -> None:
        analyze("for i in range(5):\n" "    for j in range(5):\n" "        continue")

    def test_continue_inside_loop_inside_function_is_valid(self) -> None:
        analyze("def f():\n" "    for i in range(10):\n" "        continue")

    def test_continue_inside_if_inside_loop_is_valid(self) -> None:
        analyze("for i in range(10):\n" "    if i == 3:\n" "        continue")

    def test_continue_inside_try_inside_loop_is_valid(self) -> None:
        analyze(
            "for i in range(10):\n"
            "    try:\n"
            "        continue\n"
            "    except:\n"
            "        pass"
        )

    # execute() integration

    def test_execute_continue_at_top_level_returns_error(self) -> None:
        result = execute("continue")
        assert len(result.errors) == 1
        assert "continue" in result.errors[0].lower()

    def test_execute_continue_produces_no_output(self) -> None:
        result = execute("continue")
        assert result.output == ""

    def test_execute_continue_produces_no_profiling(self) -> None:
        result = execute("continue")
        assert result.profiling is None


# 4. DUPLICATE PARAMETER NAMES


class TestDuplicateParameters:

    # Basic violations

    def test_duplicate_first_and_second_param_raises(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("def f(x, x):\n    return x")
        assert "duplicate" in str(exc_info.value).lower()
        assert "x" in str(exc_info.value)

    def test_duplicate_first_and_third_param_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f(a, b, a):\n    return a + b")

    def test_duplicate_second_and_third_param_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f(a, b, b):\n    return a + b")

    def test_all_three_params_same_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f(x, x, x):\n    return x")

    # Error message content

    def test_duplicate_param_mentions_param_name(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("def greet(name, name):\n    return name")
        assert "name" in str(exc_info.value)

    def test_duplicate_param_mentions_function_name(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("def greet(name, name):\n    return name")
        assert "greet" in str(exc_info.value)

    def test_duplicate_param_has_line_number(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("def f(a, b, a):\n    return a")
        assert exc_info.value.line is not None

    # Nested function violations

    def test_duplicate_in_nested_function_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze(
                "def outer(a):\n"
                "    def inner(b, b):\n"
                "        return b\n"
                "    return inner(a)"
            )

    def test_outer_unique_inner_duplicate_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze(
                "def outer(x, y):\n"
                "    def inner(a, a):\n"
                "        return a\n"
                "    return inner(x)"
            )

    def test_outer_duplicate_inner_unique_raises(self) -> None:
        with pytest.raises(SemanticError):
            analyze(
                "def outer(x, x):\n"
                "    def inner(a, b):\n"
                "        return a + b\n"
                "    return inner(x, x)"
            )

    # Valid uses

    def test_no_params_is_valid(self) -> None:
        analyze("def f():\n    return 1")

    def test_single_param_is_valid(self) -> None:
        analyze("def f(x):\n    return x")

    def test_two_unique_params_is_valid(self) -> None:
        analyze("def f(a, b):\n    return a + b")

    def test_three_unique_params_is_valid(self) -> None:
        analyze("def f(a, b, c):\n    return a + b + c")

    def test_same_param_name_in_different_functions_is_valid(self) -> None:
        """Two separate functions sharing a param name — not a duplicate."""
        analyze("def f(x):\n" "    return x\n" "def g(x):\n" "    return x * 2")

    def test_param_name_same_as_outer_variable_is_valid(self) -> None:
        """A param shadowing an outer variable is valid in PyLite."""
        analyze("x = 10\n" "def f(x):\n" "    return x")

    # execute() integration

    def test_execute_duplicate_params_returns_error(self) -> None:
        result = execute("def f(x, x):\n    return x\nf(1)")
        assert len(result.errors) == 1
        assert "duplicate" in result.errors[0].lower()

    def test_execute_duplicate_params_produces_no_output(self) -> None:
        result = execute("def f(x, x):\n    return x\nf(1)")
        assert result.output == ""

    def test_execute_duplicate_params_produces_no_profiling(self) -> None:
        result = execute("def f(x, x):\n    return x\nf(1)")
        assert result.profiling is None


# 5. SCOPE DEPTH — critical boundary cases


class TestScopeDepth:
    """
    Tests that specifically verify the Scope Stack behaviour:
      - Counters increment on entry and decrement on exit
      - One scope does not leak into another (no depth bleed)
      - _loop_depth is RESET to 0 when entering a function body
        so break/continue inside the function cannot inherit the
        depth of any outer loop at the definition site
      - _loop_depth is RESTORED after the function body so the
        outer loop remains valid
    """

    # Depth decrement verified (no bleed after exit)

    def test_loop_depth_resets_after_for_loop(self) -> None:
        with pytest.raises(SemanticError):
            analyze("for i in range(5):\n" "    pass\n" "break\n")

    def test_loop_depth_resets_after_while_loop(self) -> None:
        with pytest.raises(SemanticError):
            analyze("while True:\n" "    pass\n" "continue\n")

    def test_function_depth_resets_after_function_def(self) -> None:
        with pytest.raises(SemanticError):
            analyze("def f():\n" "    return 1\n" "return 2\n")

    # Loop inside function — valid

    def test_loop_inside_function_break_valid(self) -> None:
        analyze("def f():\n" "    for i in range(10):\n" "        break")

    # Function inside loop — loop_depth RESET

    def test_break_inside_function_inside_loop_raises(self) -> None:
        """
        THE KEY TEST for the _loop_depth reset fix.

        A function defined inside a loop resets loop_depth to 0 for its body.
        break inside that function has no enclosing loop — must raise.

        Without the reset, break would inherit loop_depth=1 from the outer
        for loop and silently pass — which is the bug we fixed.
        """
        with pytest.raises(SemanticError):
            analyze("for i in range(10):\n" "    def f():\n" "        break")

    def test_continue_inside_function_inside_loop_raises(self) -> None:
        """Same loop_depth reset check for continue."""
        with pytest.raises(SemanticError):
            analyze("while True:\n" "    def f():\n" "        continue")

    def test_function_inside_loop_return_valid(self) -> None:
        """
        A function defined inside a loop has its own function scope
        Return valid.
        """
        analyze("for i in range(5):\n" "    def f():\n" "        return i")

    # Loop depth RESTORED after function body

    def test_break_valid_in_outer_loop_after_inner_function(self) -> None:
        """
        After visiting a nested function inside a loop, _loop_depth must be
        restored to 1 so that break in the outer loop is still valid.
        """
        analyze(
            "for i in range(5):\n"
            "    def helper():\n"
            "        return i * 2\n"
            "    if i == 3:\n"
            "        break"
        )

    # Deep nesting

    def test_deeply_nested_loops_break_valid(self) -> None:
        analyze(
            "for a in range(3):\n"
            "    for b in range(3):\n"
            "        for c in range(3):\n"
            "            break"
        )

    def test_deeply_nested_functions_return_valid(self) -> None:
        analyze(
            "def f1():\n"
            "    def f2():\n"
            "        def f3():\n"
            "            return 1\n"
            "        return f3()\n"
            "    return f2()"
        )

    def test_two_sequential_loops_break_only_in_first_is_valid(self) -> None:
        analyze("for i in range(5):\n" "    break\n" "for j in range(5):\n" "    pass")

    def test_two_sequential_functions_return_only_in_first_is_valid(self) -> None:
        analyze("def f():\n" "    return 1\n" "def g():\n" "    pass")


# 6. VALID PROGRAMS — must NOT raise SemanticError


class TestValidPrograms:

    def test_empty_program(self) -> None:
        analyze("")

    def test_simple_assignment(self) -> None:
        analyze("x = 5")

    def test_augmented_assignment(self) -> None:
        analyze("x = 0\nx += 1")

    def test_function_with_return(self) -> None:
        analyze("def add(a, b):\n    return a + b\nprint(add(1, 2))")

    def test_for_loop_with_break_and_continue(self) -> None:
        analyze(
            "for i in range(10):\n"
            "    if i == 3:\n"
            "        continue\n"
            "    if i == 7:\n"
            "        break"
        )

    def test_while_loop_with_break(self) -> None:
        analyze(
            "x = 0\n"
            "while x < 10:\n"
            "    x += 1\n"
            "    if x == 5:\n"
            "        break"
        )

    def test_nested_functions_each_with_return(self) -> None:
        analyze(
            "def outer(x):\n"
            "    def inner(y):\n"
            "        return y * 2\n"
            "    return inner(x) + 1"
        )

    def test_try_except_finally(self) -> None:
        analyze(
            "try:\n" "    x = 1\n" "except:\n" "    x = 0\n" "finally:\n" "    print(x)"
        )

    def test_factorial_recursive_program(self) -> None:
        analyze(
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "print(factorial(5))"
        )

    def test_list_and_dict_literals(self) -> None:
        analyze("items = [1, 2, 3]\n" "info = {'a': 1, 'b': 2}\n" "print(items[0])")

    def test_binary_and_unary_expressions(self) -> None:
        analyze("x = -1 + 2 * 3\ny = not True")

    def test_function_call_as_expression(self) -> None:
        analyze("print(len([1, 2, 3]))")

    def test_pass_statement(self) -> None:
        analyze("def f():\n    pass")

    def test_multiple_functions_defined(self) -> None:
        analyze(
            "def add(a, b):\n"
            "    return a + b\n"
            "def mul(a, b):\n"
            "    return a * b\n"
            "print(add(2, mul(3, 4)))"
        )

    def test_execute_valid_program_no_errors(self) -> None:
        result = execute(
            "def sum_list(items):\n"
            "    total = 0\n"
            "    for x in items:\n"
            "        total += x\n"
            "    return total\n"
            "print(sum_list([1, 2, 3, 4, 5]))"
        )
        assert result.errors == []
        assert result.output == "15"


# 7. MULTIPLE VIOLATIONS — first one reported


class TestMultipleViolations:

    def test_two_top_level_returns_reports_first(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("return 1\nreturn 2")
        assert exc_info.value.line == 1

    def test_return_then_duplicate_param_reports_return(self) -> None:
        """return at line 1 is visited before the function def — reported first."""
        with pytest.raises(SemanticError) as exc_info:
            analyze("return 1\ndef f(x, x):\n    pass")
        assert "return" in str(exc_info.value).lower()

    def test_duplicate_param_then_top_level_return(self) -> None:
        """Function def visited first in DFS — duplicate param reported."""
        with pytest.raises(SemanticError) as exc_info:
            analyze("def f(x, x):\n    pass\nreturn 1")
        assert "duplicate" in str(exc_info.value).lower()

    def test_break_and_continue_outside_loop_reports_break_first(self) -> None:
        with pytest.raises(SemanticError) as exc_info:
            analyze("break\ncontinue")
        assert exc_info.value.line == 1
        assert "break" in str(exc_info.value).lower()

    def test_only_one_error_raised_even_with_many_violations(self) -> None:
        """analyze() raises exactly one SemanticError regardless of violation count."""
        raised = 0
        try:
            analyze("break\ncontinue\nreturn 1")
        except SemanticError:
            raised += 1
        assert raised == 1


# 8. PIPELINE ORDER — semantic errors block execution


class TestPipelineOrder:

    def test_semantic_error_prevents_runtime_error(self) -> None:
        """Division by zero must NOT appear — execution never started."""
        result = execute("return 1\nprint(1 / 0)")
        assert len(result.errors) == 1
        assert "return" in result.errors[0].lower()
        assert "division" not in result.errors[0].lower()

    def test_semantic_error_prevents_output(self) -> None:
        result = execute("return 1\nprint('hello')")
        assert result.output == ""

    def test_semantic_error_prevents_profiling(self) -> None:
        result = execute("return 1\nx = 1")
        assert result.profiling is None

    def test_semantic_error_prevents_symbol_table(self) -> None:
        result = execute("return 1\nx = 42")
        assert result.symbol_table == {}

    def test_valid_program_still_executes_correctly(self) -> None:
        """Confirming semantic pass does not interfere with correct programs."""
        result = execute("def double(n):\n" "    return n * 2\n" "print(double(21))")
        assert result.errors == []
        assert result.output == "42"
        assert result.profiling is not None


# 9. ANALYZER STATE ISOLATION


class TestAnalyzerStateIsolation:

    def test_fresh_instance_has_zero_function_depth(self) -> None:
        a = SemanticAnalyzer()
        assert a._function_depth == 0

    def test_fresh_instance_has_zero_loop_depth(self) -> None:
        a = SemanticAnalyzer()
        assert a._loop_depth == 0

    def test_fresh_instance_has_empty_issues(self) -> None:
        a = SemanticAnalyzer()
        assert a._issues == []

    def test_two_separate_analyses_do_not_share_state(self) -> None:
        analyze("def f():\n    return 1")  # valid — must not raise
        with pytest.raises(SemanticError):  # invalid — must raise
            analyze("return 1")

    def test_execute_calls_create_independent_analyzers(self) -> None:
        """Each execute() call creates its own fresh SemanticAnalyzer."""
        result1 = execute("def f():\n    return 1\nf()")
        result2 = execute("return 1")
        assert result1.errors == []
        assert len(result2.errors) == 1
