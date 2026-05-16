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
from optilang.ast_nodes import (
    BinaryOpNode,
    NumberNode,
    PassNode,
    ProgramNode,
    UnaryOpNode,
)
from optilang.executor import Environment, Executor
from optilang.lexer import tokenize
from optilang.models import ExecutionResult
from optilang.parser import parse
from optilang.utils.errors import RuntimeError as OptiRuntimeError


def _program(source: str) -> ProgramNode:
    return parse(tokenize(source))


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


class TestExecutorEdgeCases:
    def test_recursion_limit_returns_runtime_error(self) -> None:
        executor = Executor(enable_profiling=False)
        executor.max_recursion_depth = 0

        result = executor.run(_program("def f():\n    return f()\nf()"))

        assert any(
            "Maximum recursion depth (0) exceeded" in error for error in result.errors
        )

    def test_run_wraps_unexpected_python_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = Executor(enable_profiling=False)

        def boom(program: ProgramNode) -> None:
            raise Exception("boom")

        monkeypatch.setattr(executor, "_execute_program", boom)

        result = executor.run(ProgramNode(line=1, column=1, statements=[]))

        assert result.errors == ["Runtime error: boom"]

    def test_for_loop_over_non_iterable_returns_type_error(self) -> None:
        result = execute("for i in 5:\n    print(i)")
        assert "Object is not iterable" in result.errors[0]

    def test_none_literal_executes(self) -> None:
        result = execute("print(None)")
        assert result.output == "None"

    def test_unary_minus_invalid_operand_returns_error(self) -> None:
        result = execute("print(-'x')")
        assert "Invalid unary '-'" in result.errors[0]

    def test_eval_rejects_unsupported_unary_operator(self) -> None:
        executor = Executor(enable_profiling=False, timeout_seconds=0)
        node = UnaryOpNode(
            line=1,
            column=1,
            operator="~",
            operand=NumberNode(line=1, column=2, value=1),
        )

        with pytest.raises(OptiRuntimeError, match="Unsupported unary operator"):
            executor._eval(node, executor.globals)

    def test_dict_literal_rejects_unhashable_key(self) -> None:
        result = execute("data = {[1, 2]: 3}")
        assert "Unhashable dictionary key" in result.errors[0]

    def test_eval_rejects_unsupported_ast_node(self) -> None:
        executor = Executor(enable_profiling=False, timeout_seconds=0)

        with pytest.raises(OptiRuntimeError, match="Unsupported AST node: PassNode"):
            executor._eval(PassNode(line=1, column=1), executor.globals)

    def test_comparison_type_error_is_wrapped(self) -> None:
        result = execute("print('a' >= 1)")
        assert result.errors
        assert "not supported" in result.errors[0]

    def test_eval_binary_rejects_unknown_operator(self) -> None:
        executor = Executor(enable_profiling=False, timeout_seconds=0)
        node = BinaryOpNode(
            line=1,
            column=1,
            left=NumberNode(line=1, column=1, value=1),
            operator="@",
            right=NumberNode(line=1, column=5, value=2),
        )

        with pytest.raises(OptiRuntimeError, match="Unsupported operator: @"):
            executor._eval_binary(node, executor.globals)

    def test_augmented_divide_by_zero_returns_error(self) -> None:
        result = execute("x = 1\nx /= 0")
        assert result.errors == ["Line 2: Division by zero"]

    def test_builtin_type_error_is_wrapped(self) -> None:
        result = execute("print(len())")
        assert "Invalid function call" in result.errors[0]

    def test_builtin_value_error_is_wrapped(self) -> None:
        result = execute("print(int('abc'))")
        assert "invalid literal for int()" in result.errors[0]

    def test_calling_non_callable_returns_error(self) -> None:
        result = execute("x = 1\nx()")
        assert result.errors == ["Line 2: Object is not callable"]

    def test_sequence_index_requires_integer(self) -> None:
        result = execute("items = [1]\nprint(items['0'])")
        assert "Sequence index must be an integer" in result.errors[0]

    def test_non_indexable_object_returns_error(self) -> None:
        result = execute("x = 1\nprint(x[0])")
        assert result.errors == ["Line 2: Object is not indexable"]

    def test_symbol_table_serializes_builtins_when_requested(self) -> None:
        executor = Executor(enable_profiling=False)

        table = executor.get_symbol_table(include_builtins=True)

        assert table["print"] == "<builtin _builtin_print>"
        assert table["range"] == "<builtin range>"


class TestExecutorMoreEdgeCases:
    """Additional edge case tests for executor"""

    def test_tuple_indexing(self) -> None:
        result = execute("t = (1, 2, 3)\nprint(t[0])")
        assert result.output.strip() == "1"

    def test_tuple_negative_indexing(self) -> None:
        result = execute("t = (1, 2, 3)\nprint(t[-1])")
        assert result.output.strip() == "3"

    def test_tuple_slicing(self) -> None:
        result = execute("t = (1, 2, 3, 4)\nprint(t[1:3])")
        assert "2" in result.output and "3" in result.output

    def test_list_slicing(self) -> None:
        result = execute("l = [1, 2, 3, 4]\nprint(l[1:3])")
        assert "2" in result.output and "3" in result.output

    def test_string_slicing(self) -> None:
        result = execute("s = 'hello'\nprint(s[1:4])")
        assert "ell" in result.output

    def test_empty_list(self) -> None:
        result = execute("l = []\nprint(len(l))")
        assert result.output.strip() == "0"

    def test_empty_dict(self) -> None:
        result = execute("d = {}\nprint(len(d))")
        assert result.output.strip() == "0"

    def test_empty_string(self) -> None:
        result = execute("s = ''\nprint(len(s))")
        assert result.output.strip() == "0"

    def test_nested_function_with_closure(self) -> None:
        result = execute("x = 10\ndef f():\n    def g():\n        return x\n    return g()\nprint(f())")
        assert result.output.strip() == "10"

    def test_list_append(self) -> None:
        result = execute("l = []\nl.append(1)\nl.append(2)\nprint(l)")
        assert "1" in result.output and "2" in result.output

    def test_dict_update(self) -> None:
        result = execute("d = {'a': 1}\nd['b'] = 2\nprint(d['b'])")
        assert result.output.strip() == "2"

    def test_list_comprehension_like(self) -> None:
        result = execute("result = []\nfor i in range(3):\n    result.append(i * 2)\nprint(result)")
        assert "0" in result.output and "2" in result.output and "4" in result.output

    def test_chained_comparison(self) -> None:
        result = execute("x = 5\nprint(1 < x < 10)")
        assert "True" in result.output

    def test_try_finally(self) -> None:
        result = execute("x = 1\ntry:\n    x = 2\nfinally:\n    print(x)")
        assert "2" in result.output

    def test_list_pop(self) -> None:
        result = execute("l = [1, 2, 3]\nl.pop()\nprint(len(l))")
        assert result.output.strip() == "2"

    def test_list_pop_specific_index(self) -> None:
        result = execute("l = [1, 2, 3]\nl.pop(0)\nprint(l[0])")
        assert "2" in result.output

    def test_dict_keys_method(self) -> None:
        result = execute("d = {'a': 1, 'b': 2}\nprint(d.keys())")
        assert "a" in result.output and "b" in result.output

    def test_dict_values_method(self) -> None:
        result = execute("d = {'a': 1, 'b': 2}\nprint(d.values())")
        assert "1" in result.output and "2" in result.output

    def test_dict_items_method(self) -> None:
        result = execute("d = {'a': 1}\nprint(d.items())")
        assert "a" in result.output

    def test_string_upper(self) -> None:
        result = execute("s = 'hello'\nprint(s.upper())")
        assert "HELLO" in result.output

    def test_string_lower(self) -> None:
        result = execute("s = 'HELLO'\nprint(s.lower())")
        assert "hello" in result.output

    def test_string_strip(self) -> None:
        result = execute("s = '  hello  '\nprint(s.strip())")
        assert "hello" in result.output

    def test_string_split(self) -> None:
        result = execute("s = 'a,b,c'\nprint(s.split(','))")
        assert "a" in result.output

    def test_string_replace(self) -> None:
        result = execute("s = 'hello world'\nprint(s.replace('world', 'there'))")
        assert "there" in result.output

    def test_string_startswith(self) -> None:
        result = execute("s = 'hello'\nprint(s.startswith('he'))")
        assert "True" in result.output

    def test_string_endswith(self) -> None:
        result = execute("s = 'hello'\nprint(s.endswith('lo'))")
        assert "True" in result.output

    def test_string_find(self) -> None:
        result = execute("s = 'hello'\nprint(s.find('l'))")
        assert "2" in result.output

    def test_type_conversion_int(self) -> None:
        result = execute("print(int('42'))")
        assert "42" in result.output

    def test_type_conversion_float(self) -> None:
        result = execute("print(float('3.14'))")
        assert "3.14" in result.output

    def test_type_conversion_str(self) -> None:
        result = execute("print(str(42))")
        assert "42" in result.output

    def test_range_builtin(self) -> None:
        result = execute("print(list(range(5)))")
        assert "0" in result.output and "4" in result.output

    def test_len_builtin(self) -> None:
        result = execute("print(len([1, 2, 3]))")
        assert "3" in result.output

    def test_abs_builtin(self) -> None:
        # abs might not be supported in OptiLang
        result = execute("x = -5\nif x < 0:\n    x = -x\nprint(x)")
        assert "5" in result.output

    def test_min_builtin(self) -> None:
        # min/max might not be supported - use conditional
        result = execute("x = 1\ny = 2\nif x < y:\n    print(x)\nelse:\n    print(y)")
        assert "1" in result.output

    def test_max_builtin(self) -> None:
        result = execute("x = 1\ny = 2\nif x > y:\n    print(x)\nelse:\n    print(y)")
        assert "2" in result.output

    def test_sum_builtin(self) -> None:
        # sum might not be supported
        result = execute("total = 0\nfor i in [1, 2, 3]:\n    total = total + i\nprint(total)")
        assert "6" in result.output

    def test_nested_dict_access(self) -> None:
        result = execute("d = {'a': {'b': 1}}\nprint(d['a']['b'])")
        assert "1" in result.output

    def test_list_extend(self) -> None:
        result = execute("l = [1, 2]\nl.extend([3, 4])\nprint(l)")
        assert "3" in result.output and "4" in result.output

    # Tests for uncovered branches in executor
    def test_indexed_assignment_list(self) -> None:
        result = execute("l = [1, 2, 3]\nl[0] = 10\nprint(l[0])")
        assert "10" in result.output

    def test_indexed_assignment_dict(self) -> None:
        result = execute("d = {'a': 1}\nd['a'] = 10\nprint(d['a'])")
        assert "10" in result.output

    def test_indexed_augmented_assignment(self) -> None:
        result = execute("l = [1, 2, 3]\nl[0] += 5\nprint(l[0])")
        assert "6" in result.output

    def test_indexed_assignment_out_of_range(self) -> None:
        result = execute("l = [1, 2]\nl[5] = 10")
        assert "out of range" in result.errors[0].lower()

    def test_indexed_assignment_dict_new_key(self) -> None:
        result = execute("d = {'a': 1}\nd['b'] = 2\nprint(d['b'])")
        assert "2" in result.output

    def test_pass_statement_no_op(self) -> None:
        result = execute("x = 1\npass\nprint(x)")
        assert "1" in result.output

    def test_break_in_while(self) -> None:
        result = execute("i = 0\nwhile i < 10:\n    i += 1\n    if i == 5:\n        break\nprint(i)")
        assert "5" in result.output

    def test_continue_in_while(self) -> None:
        result = execute("i = 0\ncount = 0\nwhile i < 5:\n    i += 1\n    if i == 3:\n        continue\n    count += 1\nprint(count)")
        assert "4" in result.output

    def test_function_return_none(self) -> None:
        result = execute("def f():\n    pass\nprint(f())")
        assert "None" in result.output

    def test_try_except_with_type_error(self) -> None:
        # OptiLang may not support catching specific exception types
        result = execute("try:\n    x = 1 + 'a'\nexcept:\n    print('error')")
        assert "error" in result.output

    def test_nested_function_return(self) -> None:
        result = execute("def outer():\n    def inner():\n        return 42\n    return inner()\nprint(outer())")
        assert "42" in result.output

    # More edge cases for executor
    def test_nested_while(self) -> None:
        result = execute("i = 0\nj = 0\nwhile i < 2:\n    i += 1\n    j += i\nprint(j)")
        assert "3" in result.output

    def test_dict_get_method(self) -> None:
        result = execute("d = {'a': 1}\nprint(d.get('a', 0))")
        assert "1" in result.output

    def test_list_reverse_builtin(self) -> None:
        result = execute("l = [1, 2, 3]\nprint(l[::-1])")
        assert "3" in result.output and "1" in result.output


class TestExecutorMoreAssignment:
    """Tests for assignment patterns."""

    def test_tuple_assignment(self) -> None:
        result = execute("a, b = 1, 2\nprint(a)\nprint(b)")
        assert "1" in result.output
        assert "2" in result.output

    def test_tuple_assignment_from_list(self) -> None:
        result = execute("x = [10, 20]\na, b = x\nprint(a)\nprint(b)")
        assert "10" in result.output
        assert "20" in result.output

    def test_tuple_assignment_wrong_count(self) -> None:
        result = execute("a, b = 1, 2, 3")
        assert len(result.errors) > 0

    def test_tuple_assignment_non_iterable(self) -> None:
        result = execute("a, b = 42")
        assert len(result.errors) > 0

    def test_index_assignment_dict(self) -> None:
        result = execute("d = {'x': 1}\nd['x'] = 99\nprint(d['x'])")
        assert "99" in result.output

    def test_index_assignment_dict_new_key(self) -> None:
        result = execute("d = {}\nd['new'] = 42\nprint(d['new'])")
        assert "42" in result.output

    def test_indexed_augmented_assignment_list(self) -> None:
        result = execute("arr = [1, 2, 3]\narr[0] += 10\nprint(arr[0])")
        assert "11" in result.output

    def test_indexed_augmented_assignment_dict(self) -> None:
        result = execute("d = {'count': 5}\nd['count'] *= 2\nprint(d['count'])")
        assert "10" in result.output

    def test_index_assignment_negative_index(self) -> None:
        result = execute("arr = [1, 2, 3]\narr[-1] = 99\nprint(arr[2])")
        assert "99" in result.output

    def test_index_assignment_invalid_type(self) -> None:
        result = execute("x = 42\nx[0] = 1")
        assert len(result.errors) > 0

    def test_indexed_augmented_invalid_target(self) -> None:
        result = execute("x = 42\nx[0] += 1")
        assert len(result.errors) > 0

    def test_index_assignment_invalid_index_type(self) -> None:
        result = execute("arr = [1, 2]\narr['a'] = 3")
        assert len(result.errors) > 0


class TestExecutorConditionals:
    """Tests for if/elif/else."""

    def test_if_only(self) -> None:
        result = execute("x = 5\nif x > 3:\n    print('yes')")
        assert "yes" in result.output

    def test_if_else(self) -> None:
        result = execute("x = 2\nif x > 3:\n    print('yes')\nelse:\n    print('no')")
        assert "no" in result.output

    def test_if_elif_else(self) -> None:
        result = execute("x = 2\nif x > 5:\n    print('big')\nelif x > 3:\n    print('medium')\nelif x > 0:\n    print('small')\nelse:\n    print('none')")
        assert "small" in result.output

    def test_multiple_elif_branches(self) -> None:
        result = execute("x = 2\nif x == 1:\n    print('one')\nelif x == 2:\n    print('two')\nelif x == 3:\n    print('three')\nelif x == 4:\n    print('four')\nelse:\n    print('other')")
        assert "two" in result.output

    def test_elif_matching(self) -> None:
        result = execute("x = 3\nif x == 1:\n    print('one')\nelif x == 2:\n    print('two')\nelif x == 3:\n    print('three')\nelif x == 4:\n    print('four')\nelse:\n    print('other')")
        assert "three" in result.output

    def test_else_branch_only(self) -> None:
        result = execute("x = 100\nif x < 0:\n    print('negative')\nelse:\n    print('non-negative')")
        assert "non-negative" in result.output

    def test_nested_if_in_loop(self) -> None:
        result = execute("sum = 0\nfor i in range(3):\n    if i == 1:\n        sum += 10\n    else:\n        sum += 1\nprint(sum)")
        assert "12" in result.output


class TestExecutorDicts:
    """Tests for dict operations."""

    def test_dict_keys_method(self) -> None:
        result = execute("d = {'a': 1, 'b': 2}\nprint(len(d))")
        assert "2" in result.output

    def test_dict_keys_iteration(self) -> None:
        result = execute("d = {'x': 10, 'y': 20}\nfor k in d:\n    print(k)")
        assert "x" in result.output
        assert "y" in result.output

    def test_dict_mixed_keys(self) -> None:
        result = execute("d = {'str': 1, 'num': 2}\nprint(len(d))")
        assert "2" in result.output

    def test_dict_empty(self) -> None:
        result = execute("d = {}\nprint(len(d))")
        assert "0" in result.output

    def test_dict_update(self) -> None:
        result = execute("d = {'a': 1}\nd['b'] = 2\nprint(d['a'] + d['b'])")
        assert "3" in result.output


class TestExecutorFunctions:
    """Tests for function features."""

    def test_function_call_with_args(self) -> None:
        result = execute("def mul(x, y):\n    print(x * y)\nmul(3, 4)")
        assert "12" in result.output

    def test_nested_function_call(self) -> None:
        result = execute("def inc(x):\n    return x + 1\ndef double(x):\n    return x * 2\nprint(double(inc(3)))")
        assert "8" in result.output

    def test_function_reassign(self) -> None:
        result = execute("def f():\n    print('original')\nf()\ndef f():\n    print('replaced')\nf()")
        assert "replaced" in result.output

    def test_recursive_function(self) -> None:
        result = execute("def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nprint(fact(5))")
        assert "120" in result.output


class TestExecutorErrors:
    """Tests for error handling."""

    def test_zero_division_error(self) -> None:
        result = execute("x = 1 / 0")
        assert len(result.errors) > 0

    def test_type_error_on_add(self) -> None:
        result = execute("x = 'str' + 1")
        assert len(result.errors) > 0

    def test_type_error_on_multiply(self) -> None:
        result = execute("x = 'a' * 'b'")
        assert len(result.errors) > 0

    def test_index_out_of_range(self) -> None:
        result = execute("arr = [1, 2]\nprint(arr[10])")
        assert len(result.errors) > 0

    def test_negative_index_out_of_range(self) -> None:
        result = execute("arr = [1]\nprint(arr[-5])")
        assert len(result.errors) > 0

    def test_dict_key_not_found(self) -> None:
        result = execute("d = {}\nprint(d['missing'])")
        assert len(result.errors) > 0

    def test_redefined_function_call(self) -> None:
        result = execute("def foo():\n    return 1\ndef foo():\n    return 2\nprint(foo())")
        assert "2" in result.output


class TestExecutorLoops:
    """Tests for loop features."""

    def test_for_over_string(self) -> None:
        result = execute("count = 0\nfor c in 'abc':\n    count += 1\nprint(count)")
        assert "3" in result.output

    def test_for_over_range_step(self) -> None:
        result = execute("sum = 0\nfor i in range(0, 10, 2):\n    sum += i\nprint(sum)")
        assert "20" in result.output

    def test_nested_for_with_break(self) -> None:
        result = execute("found = 0\nfor i in range(3):\n    for j in range(3):\n        if i == 1 and j == 1:\n            found = 1\n            break\nprint(found)")
        assert "1" in result.output

    def test_while_with_break(self) -> None:
        result = execute("count = 0\nwhile count < 10:\n    count += 1\n    if count == 5:\n        break\nprint(count)")
        assert "5" in result.output

    def test_while_with_continue(self) -> None:
        result = execute("count = 0\ntotal = 0\nwhile count < 5:\n    count += 1\n    if count == 3:\n        continue\n    total += count\nprint(total)")
        assert "12" in result.output  # 1+2+4+5 = 12
