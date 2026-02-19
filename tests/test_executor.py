"""
Comprehensive test suite for OptiLang Executor (optilang/executor.py)

Tests cover:
- Arithmetic operations
- Variables and scope
- Comparisons and boolean logic
- Control flow (if/elif/else)
- While loops (break, continue, timeout)
- For loops (range, list, break, continue, nested)
- Functions (definition, calling, recursion, scope)
- Data structures (list, dict, indexing)
- Built-in functions
- Exception handling (try/except/finally)
- Runtime error types and messages
- Execution result metadata
"""

import pytest
from optilang import execute


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(source: str, timeout: float = 5.0):
    """Strip and execute source code."""
    return execute(source.strip(), timeout_seconds=timeout)


# ===========================================================================
# 1. ARITHMETIC OPERATIONS
# ===========================================================================

class TestArithmetic:

    def test_integer_addition(self):
        result = run("print(2 + 3)")
        assert result.output == "5"

    def test_integer_subtraction(self):
        result = run("print(10 - 4)")
        assert result.output == "6"

    def test_integer_multiplication(self):
        result = run("print(3 * 4)")
        assert result.output == "12"

    def test_integer_division_returns_float(self):
        result = run("print(10 / 2)")
        assert result.output == "5.0"

    def test_floor_division(self):
        result = run("print(10 // 3)")
        assert result.output == "3"

    def test_modulo(self):
        result = run("print(10 % 3)")
        assert result.output == "1"

    def test_power(self):
        result = run("print(2 ** 8)")
        assert result.output == "256"

    def test_float_arithmetic(self):
        result = run("print(1.5 + 2.5)")
        assert result.output == "4.0"

    def test_negative_number(self):
        result = run("print(-5)")
        assert result.output == "-5"

    def test_operator_precedence_multiply_before_add(self):
        result = run("print(2 + 3 * 4)")
        assert result.output == "14"

    def test_parentheses_override_precedence(self):
        result = run("print((2 + 3) * 4)")
        assert result.output == "20"

    def test_chained_operations(self):
        result = run("print(1 + 2 + 3 + 4)")
        assert result.output == "10"

    def test_division_by_zero_raises_error(self):
        result = run("print(1 / 0)")
        assert len(result.errors) == 1
        assert "zero" in result.errors[0].lower()

    def test_floor_division_by_zero_raises_error(self):
        result = run("print(1 // 0)")
        assert len(result.errors) == 1

    def test_modulo_by_zero_raises_error(self):
        result = run("print(1 % 0)")
        assert len(result.errors) == 1

    def test_string_concatenation(self):
        result = run('print("hello" + " " + "world")')
        assert result.output == "hello world"

    def test_string_repetition(self):
        result = run('print("ab" * 3)')
        assert result.output == "ababab"


# ===========================================================================
# 2. VARIABLES AND SCOPE
# ===========================================================================

class TestVariables:

    def test_variable_assignment_and_use(self):
        result = run("x = 5\nprint(x)")
        assert result.output == "5"

    def test_variable_reassignment(self):
        result = run("x = 5\nx = 10\nprint(x)")
        assert result.output == "10"

    def test_multiple_variables(self):
        result = run("x = 2\ny = 3\nprint(x + y)")
        assert result.output == "5"

    def test_augmented_add(self):
        result = run("x = 5\nx += 3\nprint(x)")
        assert result.output == "8"

    def test_augmented_subtract(self):
        result = run("x = 10\nx -= 4\nprint(x)")
        assert result.output == "6"

    def test_augmented_multiply(self):
        result = run("x = 3\nx *= 4\nprint(x)")
        assert result.output == "12"

    def test_augmented_divide(self):
        result = run("x = 10\nx /= 2\nprint(x)")
        assert result.output == "5.0"

    def test_undefined_variable_raises_name_error(self):
        result = run("print(missing)")
        assert len(result.errors) == 1
        assert "missing" in result.errors[0]

    def test_undefined_variable_has_no_output(self):
        result = run("print(missing)")
        assert result.output == ""

    def test_variable_in_symbol_table(self):
        result = run("x = 42")
        assert result.symbol_table["x"] == 42

    def test_multiple_variables_in_symbol_table(self):
        result = run("x = 1\ny = 2")
        assert result.symbol_table["x"] == 1
        assert result.symbol_table["y"] == 2

    def test_builtins_not_in_symbol_table(self):
        result = run("x = 1")
        assert "print" not in result.symbol_table
        assert "range" not in result.symbol_table


# ===========================================================================
# 3. COMPARISON AND BOOLEAN OPERATIONS
# ===========================================================================

class TestComparisons:

    @pytest.mark.parametrize("expr, expected", [
        ("1 == 1", "True"),
        ("1 == 2", "False"),
        ("1 != 2", "True"),
        ("1 != 1", "False"),
        ("1 < 2",  "True"),
        ("2 < 1",  "False"),
        ("1 <= 1", "True"),
        ("2 <= 1", "False"),
        ("2 > 1",  "True"),
        ("1 > 2",  "False"),
        ("1 >= 1", "True"),
        ("1 >= 2", "False"),
    ])
    def test_comparison_operators(self, expr, expected):
        result = run(f"print({expr})")
        assert result.output == expected

    def test_and_both_true(self):
        result = run("print(True and True)")
        assert result.output == "True"

    def test_and_one_false(self):
        result = run("print(True and False)")
        assert result.output == "False"

    def test_or_one_true(self):
        result = run("print(False or True)")
        assert result.output == "True"

    def test_or_both_false(self):
        result = run("print(False or False)")
        assert result.output == "False"

    def test_not_true(self):
        result = run("print(not True)")
        assert result.output == "False"

    def test_not_false(self):
        result = run("print(not False)")
        assert result.output == "True"

    def test_and_short_circuits(self):
        """False and undefined_var should not raise NameError."""
        result = run("print(False and undefined_var)")
        assert result.errors == []
        assert result.output == "False"

    def test_or_short_circuits(self):
        """True or undefined_var should not raise NameError."""
        result = run("print(True or undefined_var)")
        assert result.errors == []
        assert result.output == "True"


# ===========================================================================
# 4. IF / ELIF / ELSE
# ===========================================================================

class TestControlFlow:

    def test_if_true_branch_executes(self):
        result = run("if True:\n    print(1)")
        assert result.output == "1"

    def test_if_false_branch_skips(self):
        result = run("if False:\n    print(1)")
        assert result.output == ""

    def test_if_else_true(self):
        result = run("if True:\n    print(1)\nelse:\n    print(2)")
        assert result.output == "1"

    def test_if_else_false(self):
        result = run("if False:\n    print(1)\nelse:\n    print(2)")
        assert result.output == "2"

    def test_elif_selects_correct_branch(self):
        source = """
x = 2
if x == 1:
    print("one")
elif x == 2:
    print("two")
else:
    print("other")
"""
        result = run(source)
        assert result.output == "two"

    def test_multiple_elif(self):
        source = """
x = 3
if x == 1:
    print("one")
elif x == 2:
    print("two")
elif x == 3:
    print("three")
else:
    print("other")
"""
        result = run(source)
        assert result.output == "three"

    def test_nested_if(self):
        source = """
x = 5
if x > 0:
    if x > 3:
        print("big")
    else:
        print("small")
"""
        result = run(source)
        assert result.output == "big"


# ===========================================================================
# 5. WHILE LOOP
# ===========================================================================

class TestWhileLoop:

    def test_while_executes_correct_times(self):
        source = """
i = 0
while i < 3:
    i += 1
print(i)
"""
        result = run(source)
        assert result.output == "3"

    def test_while_false_condition_skips(self):
        source = """
while False:
    print("never")
print("done")
"""
        result = run(source)
        assert result.output == "done"

    def test_while_break_exits_loop(self):
        source = """
i = 0
while True:
    if i == 3:
        break
    i += 1
print(i)
"""
        result = run(source)
        assert result.output == "3"

    def test_while_continue_skips_iteration(self):
        source = """
i = 0
total = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    total += i
print(total)
"""
        result = run(source)
        assert result.output == "12"

    def test_while_timeout(self):
        result = run("while True:\n    pass", timeout=0.1)
        assert len(result.errors) == 1
        assert "timeout" in result.errors[0].lower()


# ===========================================================================
# 6. FOR LOOP
# ===========================================================================

class TestForLoop:

    def test_for_range_sum(self):
        source = """
total = 0
for i in range(5):
    total += i
print(total)
"""
        result = run(source)
        assert result.output == "10"

    def test_for_range_single_arg_prints_values(self):
        result = run("for i in range(3):\n    print(i)")
        assert result.output == "0\n1\n2"

    def test_for_range_two_args(self):
        result = run("for i in range(2, 5):\n    print(i)")
        assert result.output == "2\n3\n4"

    def test_for_over_list(self):
        result = run("for x in [10, 20, 30]:\n    print(x)")
        assert result.output == "10\n20\n30"

    def test_for_break(self):
        source = """
for i in range(10):
    if i == 3:
        break
print(i)
"""
        result = run(source)
        assert result.output == "3"

    def test_for_continue(self):
        source = """
total = 0
for i in range(5):
    if i == 2:
        continue
    total += i
print(total)
"""
        result = run(source)
        assert result.output == "8"

    def test_nested_for_loops(self):
        source = """
total = 0
for i in range(3):
    for j in range(3):
        total += 1
print(total)
"""
        result = run(source)
        assert result.output == "9"

    def test_break_in_nested_loop_only_exits_inner(self):
        source = """
count = 0
for i in range(3):
    for j in range(3):
        if j == 1:
            break
        count += 1
print(count)
"""
        result = run(source)
        assert result.output == "3"

    def test_for_iterator_available_after_loop(self):
        source = """
for i in range(5):
    pass
print(i)
"""
        result = run(source)
        assert result.output == "4"


# ===========================================================================
# 7. FUNCTIONS
# ===========================================================================

class TestFunctions:

    def test_function_no_return_returns_none(self):
        source = """
def foo():
    x = 1
result = foo()
print(result)
"""
        result = run(source)
        assert result.output == "None"

    def test_function_return_value(self):
        source = """
def add(x, y):
    return x + y
print(add(3, 4))
"""
        result = run(source)
        assert result.output == "7"

    def test_function_with_no_parameters(self):
        source = """
def greet():
    return "hello"
print(greet())
"""
        result = run(source)
        assert result.output == "hello"

    def test_recursive_factorial(self):
        source = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
print(factorial(5))
"""
        result = run(source)
        assert result.output == "120"

    def test_recursive_fibonacci(self):
        source = """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
print(fib(7))
"""
        result = run(source)
        assert result.output == "13"

    def test_function_appears_in_symbol_table(self):
        result = run("def foo():\n    pass")
        assert result.symbol_table["foo"] == "<function foo>"

    def test_function_local_scope_does_not_leak(self):
        source = """
def foo():
    local_var = 99
foo()
"""
        result = run(source)
        assert "local_var" not in result.symbol_table

    def test_function_reads_global_variable(self):
        source = """
x = 10
def foo():
    return x
print(foo())
"""
        result = run(source)
        assert result.output == "10"

    def test_too_few_arguments_raises_error(self):
        source = """
def foo(x, y):
    return x + y
foo(1)
"""
        result = run(source)
        assert len(result.errors) == 1
        assert "foo" in result.errors[0]

    def test_too_many_arguments_raises_error(self):
        source = """
def foo(x):
    return x
foo(1, 2)
"""
        result = run(source)
        assert len(result.errors) == 1

    def test_recursion_depth_limit_raises_error(self):
        source = """
def infinite(n):
    return infinite(n + 1)
infinite(0)
"""
        result = run(source)
        assert len(result.errors) == 1
        assert "recursion" in result.errors[0].lower()

    def test_nested_function_calls(self):
        source = """
def double(x):
    return x * 2
def quadruple(x):
    return double(double(x))
print(quadruple(3))
"""
        result = run(source)
        assert result.output == "12"


# ===========================================================================
# 8. DATA STRUCTURES
# ===========================================================================

class TestDataStructures:

    def test_list_index_access_first(self):
        result = run("x = [10, 20, 30]\nprint(x[0])")
        assert result.output == "10"

    def test_list_index_access_last(self):
        result = run("x = [10, 20, 30]\nprint(x[2])")
        assert result.output == "30"

    def test_list_index_out_of_range_raises_error(self):
        result = run("x = [1, 2]\nprint(x[5])")
        assert len(result.errors) == 1

    def test_for_loop_over_list(self):
        result = run("for x in [1, 2, 3]:\n    print(x)")
        assert result.output == "1\n2\n3"

    def test_dict_string_key_access(self):
        result = run('d = {"name": "alice"}\nprint(d["name"])')
        assert result.output == "alice"

    def test_dict_integer_key(self):
        result = run('d = {1: "one"}\nprint(d[1])')
        assert result.output == "one"

    def test_dict_key_not_found_raises_error(self):
        result = run('d = {"a": 1}\nprint(d["b"])')
        assert len(result.errors) == 1
        assert "b" in result.errors[0]

    def test_nested_list_access(self):
        result = run("x = [[1, 2], [3, 4]]\nprint(x[1][0])")
        assert result.output == "3"


# ===========================================================================
# 9. BUILT-IN FUNCTIONS
# ===========================================================================

class TestBuiltins:

    def test_print_single_value(self):
        result = run("print(42)")
        assert result.output == "42"

    def test_print_multiple_values(self):
        result = run("print(1, 2, 3)")
        assert result.output == "1 2 3"

    def test_print_no_arguments(self):
        result = run("print()")
        assert result.output == ""

    def test_len_on_list(self):
        result = run("print(len([1, 2, 3]))")
        assert result.output == "3"

    def test_len_on_string(self):
        result = run('print(len("hello"))')
        assert result.output == "5"

    def test_len_empty_list(self):
        result = run("print(len([]))")
        assert result.output == "0"

    def test_str_converts_int(self):
        result = run("print(str(42))")
        assert result.output == "42"

    def test_int_converts_string(self):
        result = run('print(int("42"))')
        assert result.output == "42"

    def test_int_truncates_float(self):
        result = run("print(int(3.9))")
        assert result.output == "3"

    def test_float_converts_string(self):
        result = run('print(float("3.14"))')
        assert result.output == "3.14"

    def test_bool_truthy_int(self):
        result = run("print(bool(1))")
        assert result.output == "True"

    def test_bool_falsy_zero(self):
        result = run("print(bool(0))")
        assert result.output == "False"

    def test_bool_empty_list_is_false(self):
        result = run("print(bool([]))")
        assert result.output == "False"


# ===========================================================================
# 10. EXCEPTION HANDLING
# ===========================================================================

class TestExceptionHandling:

    def test_try_except_catches_runtime_error(self):
        source = """
try:
    x = 1 / 0
except:
    print("caught")
"""
        result = run(source)
        assert result.errors == []
        assert result.output == "caught"

    def test_try_except_catches_name_error(self):
        source = """
try:
    print(undefined)
except:
    print("caught")
"""
        result = run(source)
        assert result.errors == []
        assert result.output == "caught"

    def test_finally_runs_after_try_success(self):
        source = """
try:
    x = 1
finally:
    print("finally")
"""
        result = run(source)
        assert result.output == "finally"

    def test_finally_runs_after_except(self):
        source = """
try:
    x = undefined
except:
    print("except")
finally:
    print("finally")
"""
        result = run(source)
        assert result.output == "except\nfinally"

    def test_unhandled_error_propagates(self):
        result = run("x = undefined_var")
        assert len(result.errors) == 1

    def test_error_message_contains_line_number(self):
        result = run("x = undefined_var")
        assert len(result.errors) == 1
        assert "1" in result.errors[0]

    def test_partial_output_before_error(self):
        source = """
print("before")
x = undefined_var
"""
        result = run(source)
        assert result.output == "before"
        assert len(result.errors) == 1


# ===========================================================================
# 11. RUNTIME ERROR MESSAGES
# ===========================================================================

class TestRuntimeErrors:

    def test_name_error_message_contains_variable(self):
        result = run("print(missing_name)")
        assert "missing_name" in result.errors[0]

    def test_zero_division_error_message(self):
        result = run("x = 5 / 0")
        assert len(result.errors) == 1
        assert "zero" in result.errors[0].lower()

    def test_index_error_on_list(self):
        result = run("x = [1, 2]\nprint(x[10])")
        assert len(result.errors) == 1

    def test_key_error_message_contains_key(self):
        result = run('d = {"a": 1}\nprint(d["b"])')
        assert len(result.errors) == 1
        assert "b" in result.errors[0]

    def test_type_error_non_iterable_in_for(self):
        result = run("for x in 5:\n    pass")
        assert len(result.errors) == 1

    def test_argument_error_message_contains_function_name(self):
        source = """
def foo(x, y):
    pass
foo(1)
"""
        result = run(source)
        assert len(result.errors) == 1
        assert "foo" in result.errors[0]


# ===========================================================================
# 12. EXECUTION RESULT METADATA
# ===========================================================================

class TestExecutionResult:

    def test_no_errors_on_valid_code(self):
        result = run("x = 1")
        assert result.errors == []

    def test_execution_time_is_positive(self):
        result = run("x = 1")
        assert result.execution_time > 0

    def test_execution_time_is_float(self):
        result = run("x = 1")
        assert isinstance(result.execution_time, float)

    def test_profiling_data_present_by_default(self):
        result = run("x = 1")
        assert result.profiling is not None

    def test_profiling_disabled_returns_none(self):
        result = execute("x = 1", enable_profiling=False)
        assert result.profiling is None

    def test_empty_symbol_table_on_no_assignments(self):
        result = run("print(1)")
        assert result.symbol_table == {}

    def test_output_stripped_of_trailing_newline(self):
        result = run("print(1)")
        assert not result.output.endswith("\n")

    def test_multiple_prints_newline_separated(self):
        result = run("print(1)\nprint(2)\nprint(3)")
        assert result.output == "1\n2\n3"