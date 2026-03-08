"""
Tests for the OptiLang Executor (Sprint 1 & 2)

Covers:
- Environment: define, get, assign, contains, all_values, scope chain
- execute(): arithmetic, strings, booleans, control flow, functions,
  recursion, data structures, error handling, timeout, symbol table
- Executor: profiling integration, enable/disable profiling
"""

from __future__ import annotations
import pytest
from optilang import execute
from optilang.executor import Environment
from optilang.models import ExecutionResult


#  Unit Tests: Environment
class TestEnvironment:
    """Tests for the Environment scope chain."""

    def test_define_and_get(self) -> None:
        env = Environment()
        env.define("x", 42)
        assert env.get("x") == 42

    def test_get_undefined_raises(self) -> None:
        env = Environment()
        with pytest.raises(Exception):
            env.get("undefined")

    def test_assign_existing_variable(self) -> None:
        env = Environment()
        env.define("x", 1)
        env.assign("x", 99)
        assert env.get("x") == 99

    def test_assign_creates_new_variable(self) -> None:
        env = Environment()
        env.assign("y", 10)
        assert env.get("y") == 10

    def test_contains_true(self) -> None:
        env = Environment()
        env.define("a", 1)
        assert env.contains("a") is True

    def test_contains_false(self) -> None:
        env = Environment()
        assert env.contains("missing") is False

    def test_child_scope_sees_parent(self) -> None:
        parent = Environment()
        parent.define("x", 5)
        child = Environment(parent=parent)
        assert child.get("x") == 5

    def test_child_scope_shadows_parent(self) -> None:
        parent = Environment()
        parent.define("x", 5)
        child = Environment(parent=parent)
        child.define("x", 99)
        assert child.get("x") == 99
        assert parent.get("x") == 5

    def test_assign_updates_parent_scope(self) -> None:
        parent = Environment()
        parent.define("x", 1)
        child = Environment(parent=parent)
        child.assign("x", 100)
        assert parent.get("x") == 100

    def test_all_values_includes_parent(self) -> None:
        parent = Environment()
        parent.define("a", 1)
        child = Environment(parent=parent)
        child.define("b", 2)
        values = child.all_values()
        assert "a" in values
        assert "b" in values

    def test_all_values_child_overrides_parent(self) -> None:
        parent = Environment()
        parent.define("x", 1)
        child = Environment(parent=parent)
        child.define("x", 2)
        assert child.all_values()["x"] == 2

    def test_contains_checks_parent_chain(self) -> None:
        grandparent = Environment()
        grandparent.define("z", 3)
        parent = Environment(parent=grandparent)
        child = Environment(parent=parent)
        assert child.contains("z") is True


#  Unit Tests: execute() — Basic expressions
class TestExecuteBasic:
    """Tests for basic arithmetic and variable operations."""

    def test_integer_addition(self) -> None:
        result = execute("print(1 + 2)")
        assert result.output == "3"
        assert result.errors == []

    def test_integer_subtraction(self) -> None:
        result = execute("print(10 - 4)")
        assert result.output == "6"

    def test_integer_multiplication(self) -> None:
        result = execute("print(3 * 4)")
        assert result.output == "12"

    def test_integer_division(self) -> None:
        result = execute("print(10 / 4)")
        assert result.output == "2.5"

    def test_floor_division(self) -> None:
        result = execute("print(10 // 3)")
        assert result.output == "3"

    def test_modulo(self) -> None:
        result = execute("print(10 % 3)")
        assert result.output == "1"

    def test_exponentiation(self) -> None:
        result = execute("print(2 ** 8)")
        assert result.output == "256"

    def test_variable_assignment_and_print(self) -> None:
        result = execute("x = 5\nprint(x)")
        assert result.output == "5"

    def test_multiple_variables(self) -> None:
        result = execute("x = 3\ny = 4\nprint(x + y)")
        assert result.output == "7"

    def test_augmented_add(self) -> None:
        result = execute("x = 10\nx += 5\nprint(x)")
        assert result.output == "15"

    def test_augmented_subtract(self) -> None:
        result = execute("x = 10\nx -= 3\nprint(x)")
        assert result.output == "7"

    def test_augmented_multiply(self) -> None:
        result = execute("x = 4\nx *= 3\nprint(x)")
        assert result.output == "12"

    def test_augmented_divide(self) -> None:
        result = execute("x = 10\nx /= 2\nprint(x)")
        assert result.output == "5.0"

    def test_float_arithmetic(self) -> None:
        result = execute("print(1.5 + 2.5)")
        assert result.output == "4.0"

    def test_unary_negation(self) -> None:
        result = execute("x = 5\nprint(-x)")
        assert result.output == "-5"

    def test_string_concatenation(self) -> None:
        result = execute('print("hello" + " " + "world")')
        assert result.output == "hello world"

    def test_boolean_true(self) -> None:
        result = execute("print(True)")
        assert result.output == "True"

    def test_boolean_false(self) -> None:
        result = execute("print(False)")
        assert result.output == "False"

    def test_boolean_and(self) -> None:
        result = execute("print(True and False)")
        assert result.output == "False"

    def test_boolean_or(self) -> None:
        result = execute("print(False or True)")
        assert result.output == "True"

    def test_boolean_not(self) -> None:
        result = execute("print(not True)")
        assert result.output == "False"

    def test_comparison_equal(self) -> None:
        result = execute("print(5 == 5)")
        assert result.output == "True"

    def test_comparison_not_equal(self) -> None:
        result = execute("print(5 != 3)")
        assert result.output == "True"

    def test_comparison_less_than(self) -> None:
        result = execute("print(3 < 5)")
        assert result.output == "True"

    def test_comparison_greater_than(self) -> None:
        result = execute("print(5 > 3)")
        assert result.output == "True"

    def test_multiple_print_statements(self) -> None:
        result = execute("print(1)\nprint(2)\nprint(3)")
        assert result.output == "1\n2\n3"


#  Unit Tests: execute() — Control flow
class TestExecuteControlFlow:
    """Tests for if/elif/else, while, for."""

    def test_if_true_branch(self) -> None:
        result = execute("if True:\n    print('yes')")
        assert result.output == "yes"

    def test_if_false_branch(self) -> None:
        result = execute("if False:\n    print('yes')\nelse:\n    print('no')")
        assert result.output == "no"

    def test_elif_branch(self) -> None:
        result = execute(
            "x = 5\n"
            "if x > 10:\n"
            "    print('big')\n"
            "elif x > 3:\n"
            "    print('medium')\n"
            "else:\n"
            "    print('small')"
        )
        assert result.output == "medium"

    def test_while_loop(self) -> None:
        result = execute("i = 0\n" "while i < 3:\n" "    print(i)\n" "    i += 1")
        assert result.output == "0\n1\n2"

    def test_while_break(self) -> None:
        result = execute(
            "i = 0\n"
            "while True:\n"
            "    if i == 3:\n"
            "        break\n"
            "    i += 1\n"
            "print(i)"
        )
        assert result.output == "3"

    def test_while_continue(self) -> None:
        result = execute(
            "i = 0\n"
            "total = 0\n"
            "while i < 5:\n"
            "    i += 1\n"
            "    if i == 3:\n"
            "        continue\n"
            "    total += i\n"
            "print(total)"
        )
        # 1 + 2 + 4 + 5 = 12 (skip 3)
        assert result.output == "12"

    def test_for_loop_range(self) -> None:
        result = execute(
            "total = 0\n" "for i in range(5):\n" "    total += i\n" "print(total)"
        )
        assert result.output == "10"

    def test_for_loop_break(self) -> None:
        result = execute(
            "for i in range(10):\n" "    if i == 4:\n" "        break\n" "print(i)"
        )
        assert result.output == "4"

    def test_for_loop_continue(self) -> None:
        result = execute(
            "total = 0\n"
            "for i in range(6):\n"
            "    if i % 2 == 0:\n"
            "        continue\n"
            "    total += i\n"
            "print(total)"
        )
        # 1 + 3 + 5 = 9
        assert result.output == "9"

    def test_nested_loops(self) -> None:
        result = execute(
            "total = 0\n"
            "for i in range(3):\n"
            "    for j in range(3):\n"
            "        total += 1\n"
            "print(total)"
        )
        assert result.output == "9"

    def test_pass_statement(self) -> None:
        result = execute("if True:\n    pass\nprint('ok')")
        assert result.output == "ok"


#  Unit Tests: execute() — Functions
class TestExecuteFunctions:
    """Tests for function definition, calls, return, and recursion."""

    def test_simple_function(self) -> None:
        result = execute("def greet():\n" "    print('hello')\n" "greet()")
        assert result.output == "hello"

    def test_function_with_return(self) -> None:
        result = execute("def double(x):\n" "    return x * 2\n" "print(double(5))")
        assert result.output == "10"

    def test_function_with_multiple_params(self) -> None:
        result = execute("def add(a, b):\n" "    return a + b\n" "print(add(3, 4))")
        assert result.output == "7"

    def test_function_called_multiple_times(self) -> None:
        result = execute(
            "def square(n):\n"
            "    return n * n\n"
            "print(square(2))\n"
            "print(square(3))\n"
            "print(square(4))"
        )
        assert result.output == "4\n9\n16"

    def test_recursive_factorial(self) -> None:
        result = execute(
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "print(factorial(5))"
        )
        assert result.output == "120"
        assert result.errors == []

    def test_recursive_fibonacci(self) -> None:
        result = execute(
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
            "print(fib(7))"
        )
        assert result.output == "13"

    def test_function_scope_isolation(self) -> None:
        # In OptiLang, assign() walks up the scope chain and updates
        # the nearest scope that owns the variable. So x = 99 inside
        # the function updates the global x (no "global" keyword needed).
        result = execute(
            "x = 10\n" "def change():\n" "    x = 99\n" "change()\n" "print(x)"
        )
        assert result.output == "99"

    def test_function_wrong_args_error(self) -> None:
        result = execute("def add(a, b):\n" "    return a + b\n" "add(1)")
        assert len(result.errors) == 1

    def test_nested_function_calls(self) -> None:
        result = execute(
            "def add(a, b):\n"
            "    return a + b\n"
            "def triple_add(x):\n"
            "    return add(x, add(x, x))\n"
            "print(triple_add(4))"
        )
        assert result.output == "12"


#  Unit Tests: execute() — Data structures
class TestExecuteDataStructures:
    """Tests for lists and dictionaries."""

    def test_list_creation(self) -> None:
        result = execute("x = [1, 2, 3]\nprint(x)")
        assert "1" in result.output
        assert "2" in result.output

    def test_list_indexing(self) -> None:
        result = execute("x = [10, 20, 30]\nprint(x[1])")
        assert result.output == "20"

    def test_list_negative_index(self) -> None:
        result = execute("x = [1, 2, 3]\nprint(x[-1])")
        assert result.output == "3"

    def test_list_index_out_of_range(self) -> None:
        result = execute("x = [1, 2]\nprint(x[5])")
        assert len(result.errors) == 1

    def test_dict_creation(self) -> None:
        result = execute('d = {"a": 1}\nprint(d["a"])')
        assert result.output == "1"

    def test_dict_missing_key(self) -> None:
        result = execute('d = {"a": 1}\nprint(d["b"])')
        assert len(result.errors) == 1

    def test_list_len(self) -> None:
        result = execute("x = [1, 2, 3, 4]\nprint(len(x))")
        assert result.output == "4"

    def test_nested_list(self) -> None:
        result = execute("x = [[1, 2], [3, 4]]\nprint(x[0][1])")
        assert result.output == "2"


#  Unit Tests: execute() — Errors
class TestExecuteErrors:
    """Tests for runtime and parse error handling."""

    def test_undefined_variable_error(self) -> None:
        result = execute("print(undefined)")
        assert len(result.errors) == 1

    def test_zero_division_error(self) -> None:
        result = execute("print(1 / 0)")
        assert len(result.errors) == 1

    def test_zero_floor_division_error(self) -> None:
        result = execute("print(1 // 0)")
        assert len(result.errors) == 1

    def test_zero_modulo_error(self) -> None:
        result = execute("print(1 % 0)")
        assert len(result.errors) == 1

    def test_type_error_addition(self) -> None:
        result = execute('print(1 + "a")')
        assert len(result.errors) == 1

    def test_error_does_not_crash_returns_result(self) -> None:
        result = execute("print(undefined)")
        assert isinstance(result, ExecutionResult)
        assert result.output == ""

    def test_try_except_catches_runtime_error(self) -> None:
        result = execute("try:\n" "    x = 1 / 0\n" "except:\n" "    print('caught')")
        assert result.output == "caught"
        assert result.errors == []

    def test_syntax_error_returns_error(self) -> None:
        result = execute("def (:")
        assert len(result.errors) == 1


#  Unit Tests: execute() — Timeout
class TestExecuteTimeout:
    """Tests for timeout enforcement."""

    def test_timeout_raises_error(self) -> None:
        result = execute("while True:\n    x = 1", timeout_seconds=0.1)
        assert len(result.errors) == 1
        assert (
            "timeout" in result.errors[0].lower() or "time" in result.errors[0].lower()
        )

    def test_fast_program_no_timeout(self) -> None:
        result = execute("print(1 + 1)", timeout_seconds=5.0)
        assert result.errors == []
        assert result.output == "2"

    def test_timeout_disabled_with_zero(self) -> None:
        # timeout_seconds=0 disables the timeout
        result = execute("x = 1 + 1\nprint(x)", timeout_seconds=0)
        assert result.errors == []


#  Unit Tests: execute() — Symbol table
class TestExecuteSymbolTable:
    """Tests for symbol table returned by execute()."""

    def test_symbol_table_contains_variable(self) -> None:
        result = execute("x = 42")
        assert "x" in result.symbol_table
        assert result.symbol_table["x"] == 42

    def test_symbol_table_excludes_builtins(self) -> None:
        result = execute("x = 1")
        assert "print" not in result.symbol_table
        assert "range" not in result.symbol_table

    def test_symbol_table_function_serialized(self) -> None:
        result = execute("def foo():\n    pass")
        assert "foo" in result.symbol_table
        assert "<function" in str(result.symbol_table["foo"])

    def test_symbol_table_multiple_variables(self) -> None:
        result = execute("a = 1\nb = 2\nc = 3")
        assert result.symbol_table["a"] == 1
        assert result.symbol_table["b"] == 2
        assert result.symbol_table["c"] == 3


#  Unit Tests: execute() — Profiling integration
class TestExecuteProfilingIntegration:
    """Tests that executor correctly integrates with the profiler."""

    def test_profiling_enabled_by_default(self) -> None:
        result = execute("x = 1")
        assert result.profiling is not None

    def test_profiling_disabled(self) -> None:
        result = execute("x = 1", enable_profiling=False)
        assert result.profiling is None

    def test_profiling_execution_time_positive(self) -> None:
        result = execute("x = 1 + 1")
        assert result.execution_time > 0

    def test_profiling_line_stats_populated(self) -> None:
        result = execute("x = 1\ny = 2\nz = x + y")
        assert result.profiling is not None
        assert len(result.profiling.line_stats) >= 3

    def test_profiling_tracks_loop_iterations(self) -> None:
        result = execute("for i in range(20):\n" "    x = i * 2")
        assert result.profiling is not None
        counts = [s.execution_count for s in result.profiling.line_stats.values()]
        assert max(counts) >= 20

    def test_profiling_function_stats_populated(self) -> None:
        result = execute("def add(a, b):\n" "    return a + b\n" "add(1, 2)")
        assert result.profiling is not None
        assert "add" in result.profiling.function_stats

    def test_profiling_memory_bytes_positive(self) -> None:
        result = execute("items = [1, 2, 3, 4, 5]\ntotal = 0")
        assert result.profiling is not None
        assert result.profiling.peak_memory_bytes > 0
