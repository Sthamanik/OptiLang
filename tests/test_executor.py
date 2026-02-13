from optilang import execute


def test_execute_arithmetic_and_assignment():
    result = execute(
        """
x = 2
y = 3
print(x + y)
""".strip()
    )
    print("test_execute_arithmetic_and_assignment")
    print("OUTPUT:", result.output)
    print("ERRORS:", result.errors)
    print("SYMBOL_TABLE:", result.symbol_table)

    assert result.errors == []
    assert result.output == "5"
    assert result.symbol_table == {"x": 2, "y": 3}


def test_execute_control_flow_and_augmented_assignment():
    result = execute(
        """
total = 0
for i in range(5):
    total += i

if total > 5:
    print(total)
else:
    print(0)
""".strip()
    )
    print("test_execute_control_flow_and_augmented_assignment")
    print("OUTPUT:", result.output)
    print("ERRORS:", result.errors)
    print("SYMBOL_TABLE:", result.symbol_table)

    assert result.errors == []
    assert result.output == "10"


def test_execute_user_defined_function_and_return():
    result = execute(
        """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
""".strip()
    )
    print("test_execute_user_defined_function_and_return")
    print("OUTPUT:", result.output)
    print("ERRORS:", result.errors)
    print("SYMBOL_TABLE:", result.symbol_table)

    assert result.errors == []
    assert result.output == "120"
    assert result.symbol_table["factorial"] == "<function factorial>"


def test_execute_try_except_finally():
    result = execute(
        """
try:
    x = 1 / 0
except:
    print("handled")
finally:
    print("done")
""".strip()
    )
    print("test_execute_try_except_finally")
    print("OUTPUT:", result.output)
    print("ERRORS:", result.errors)
    print("SYMBOL_TABLE:", result.symbol_table)

    assert result.errors == []
    assert result.output == "handled\ndone"


def test_execute_reports_runtime_errors():
    result = execute(
        """
print(missing_name)
""".strip()
    )
    print("test_execute_reports_runtime_errors")
    print("OUTPUT:", result.output)
    print("ERRORS:", result.errors)
    print("SYMBOL_TABLE:", result.symbol_table)

    assert result.output == ""
    assert len(result.errors) == 1
    assert "missing_name" in result.errors[0]
    assert result.symbol_table == {}
