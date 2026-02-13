"""
Runtime executor for OptiLang AST programs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .ast_nodes import (
    ASTNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    BooleanNode,
    BreakNode,
    ContinueNode,
    DictNode,
    ForNode,
    FunctionCallNode,
    FunctionDefNode,
    IdentifierNode,
    IfNode,
    IndexNode,
    ListNode,
    NullNode,
    NumberNode,
    PassNode,
    ProgramNode,
    ReturnNode,
    StringNode,
    TryNode,
    UnaryOpNode,
    WhileNode,
)
from .lexer import tokenize
from .models import ExecutionResult
from .parser import parse
from .utils.errors import (
    ArgumentError,
    IndexError as OptiIndexError,
    KeyError as OptiKeyError,
    NameError as OptiNameError,
    ParserError,
    RecursionError as OptiRecursionError,
    RuntimeError as OptiRuntimeError,
    TimeoutError,
    TypeError as OptiTypeError,
    ValueError as OptiValueError,
    ZeroDivisionError as OptiZeroDivisionError,
)
from .utils.errors import LexerError


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


@dataclass
class _ReturnSignal(Exception):
    value: Any


class Environment:
    """Hierarchical variable scope."""

    def __init__(self, parent: Optional["Environment"] = None):
        self.parent = parent
        self.values: Dict[str, Any] = {}

    def define(self, name: str, value: Any) -> None:
        self.values[name] = value

    def get(self, name: str, node: Optional[ASTNode] = None) -> Any:
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name, node)
        raise OptiNameError(name, getattr(node, "line", None))

    def assign(self, name: str, value: Any) -> None:
        if name in self.values:
            self.values[name] = value
            return
        if self.parent is not None and self.parent.contains(name):
            self.parent.assign(name, value)
            return
        self.values[name] = value

    def contains(self, name: str) -> bool:
        if name in self.values:
            return True
        if self.parent is None:
            return False
        return self.parent.contains(name)


class UserFunction:
    """Callable wrapper for user-defined function nodes."""

    def __init__(self, node: FunctionDefNode, closure: Environment):
        self.node = node
        self.closure = closure

    @property
    def name(self) -> str:
        return self.node.name.name

    def call(self, executor: "Executor", args: List[Any]) -> Any:
        expected = len(self.node.parameters)
        got = len(args)
        if expected != got:
            raise ArgumentError(self.name, expected, got, self.node.line)

        executor._call_depth += 1
        if executor._call_depth > executor.max_recursion_depth:
            executor._call_depth -= 1
            raise OptiRecursionError(executor.max_recursion_depth, self.node.line)

        frame = Environment(parent=self.closure)
        for param, arg in zip(self.node.parameters, args):
            frame.define(param.name, arg)

        try:
            executor._execute_block(self.node.body, frame)
            return None
        except _ReturnSignal as signal:
            return signal.value
        finally:
            executor._call_depth -= 1


class Executor:
    """
    AST executor for OptiLang programs.
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds
        self._start_time = 0.0
        self._output: List[str] = []
        self.max_recursion_depth = 1000
        self._call_depth = 0

        self.globals = Environment()
        self._install_builtins()
        self._builtin_names = set(self.globals.values.keys())

    def run(self, program: ProgramNode) -> ExecutionResult:
        self._start_time = time.perf_counter()
        self._output = []
        errors: List[str] = []

        try:
            self._execute_program(program)
        except OptiRuntimeError as exc:
            errors.append(str(exc))
        except Exception as exc:  # pragma: no cover - unexpected fallback
            errors.append(f"Runtime error: {exc}")

        elapsed = time.perf_counter() - self._start_time
        return ExecutionResult(
            output="".join(self._output).rstrip("\n"),
            errors=errors,
            execution_time=elapsed,
            profiling=None,
            symbol_table=self.get_symbol_table(include_builtins=False),
        )

    def _install_builtins(self) -> None:
        self.globals.define("print", self._builtin_print)
        self.globals.define("range", range)
        self.globals.define("len", len)
        self.globals.define("str", str)
        self.globals.define("int", int)
        self.globals.define("float", float)
        self.globals.define("bool", bool)
        self.globals.define("list", list)
        self.globals.define("dict", dict)

    def _builtin_print(self, *args: Any) -> None:
        self._output.append(" ".join(str(arg) for arg in args) + "\n")
        return None

    def _check_timeout(self, node: Optional[ASTNode] = None) -> None:
        if self.timeout_seconds <= 0:
            return
        elapsed = time.perf_counter() - self._start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(self.timeout_seconds, getattr(node, "line", None))

    def _execute_program(self, program: ProgramNode) -> None:
        self._execute_block(program.statements, self.globals)

    def _execute_block(self, statements: List[ASTNode], env: Environment) -> None:
        for statement in statements:
            self._check_timeout(statement)
            self._execute_statement(statement, env)

    def _execute_statement(self, node: ASTNode, env: Environment) -> None:
        if isinstance(node, AssignmentNode):
            env.assign(node.target.name, self._eval(node.value, env))
            return

        if isinstance(node, AugmentedAssignmentNode):
            target = node.target.name
            current = env.get(target, node.target)
            value = self._eval(node.value, env)
            result = self._apply_augmented(node.operator, current, value, node)
            env.assign(target, result)
            return

        if isinstance(node, IfNode):
            if self._truthy(self._eval(node.condition, env)):
                self._execute_block(node.if_block, env)
                return
            for cond, block in node.elif_parts:
                if self._truthy(self._eval(cond, env)):
                    self._execute_block(block, env)
                    return
            if node.else_block is not None:
                self._execute_block(node.else_block, env)
            return

        if isinstance(node, WhileNode):
            while self._truthy(self._eval(node.condition, env)):
                self._check_timeout(node)
                try:
                    self._execute_block(node.body, env)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return

        if isinstance(node, ForNode):
            iterable = self._eval(node.iterable, env)
            try:
                iterator = iter(iterable)
            except Exception as exc:
                raise OptiTypeError(f"Object is not iterable: {exc}", node.line)

            for value in iterator:
                self._check_timeout(node)
                env.assign(node.iterator.name, value)
                try:
                    self._execute_block(node.body, env)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return

        if isinstance(node, FunctionDefNode):
            env.define(node.name.name, UserFunction(node, env))
            return

        if isinstance(node, ReturnNode):
            value = self._eval(node.value, env) if node.value is not None else None
            raise _ReturnSignal(value)

        if isinstance(node, BreakNode):
            raise _BreakSignal()

        if isinstance(node, ContinueNode):
            raise _ContinueSignal()

        if isinstance(node, PassNode):
            return

        if isinstance(node, TryNode):
            try:
                self._execute_block(node.try_block, env)
            except OptiRuntimeError:
                if node.except_block is None:
                    raise
                self._execute_block(node.except_block, env)
            finally:
                if node.finally_block is not None:
                    self._execute_block(node.finally_block, env)
            return

        # Expression statement
        self._eval(node, env)

    def _eval(self, node: ASTNode, env: Environment) -> Any:
        self._check_timeout(node)

        if isinstance(node, NumberNode):
            return node.value
        if isinstance(node, StringNode):
            return node.value
        if isinstance(node, BooleanNode):
            return node.value
        if isinstance(node, NullNode):
            return None
        if isinstance(node, IdentifierNode):
            return env.get(node.name, node)

        if isinstance(node, BinaryOpNode):
            return self._eval_binary(node, env)

        if isinstance(node, UnaryOpNode):
            operand = self._eval(node.operand, env)
            if node.operator == "-":
                try:
                    return -operand
                except Exception as exc:
                    raise OptiTypeError(f"Invalid unary '-': {exc}", node.line)
            if node.operator == "not":
                return not self._truthy(operand)
            raise OptiRuntimeError(f"Unsupported unary operator: {node.operator}", node.line)

        if isinstance(node, FunctionCallNode):
            callee = env.get(node.function.name, node.function)
            args = [self._eval(arg, env) for arg in node.arguments]
            return self._call(callee, args, node)

        if isinstance(node, ListNode):
            return [self._eval(elem, env) for elem in node.elements]

        if isinstance(node, DictNode):
            result: Dict[Any, Any] = {}
            for key_node, value_node in node.pairs:
                key = self._eval(key_node, env)
                value = self._eval(value_node, env)
                try:
                    result[key] = value
                except Exception as exc:
                    raise OptiTypeError(f"Unhashable dictionary key: {exc}", node.line)
            return result

        if isinstance(node, IndexNode):
            collection = self._eval(node.collection, env)
            index = self._eval(node.index, env)
            return self._index(collection, index, node)

        raise OptiRuntimeError(f"Unsupported AST node: {type(node).__name__}", node.line)

    def _eval_binary(self, node: BinaryOpNode, env: Environment) -> Any:
        op = node.operator

        if op == "and":
            left = self._eval(node.left, env)
            if not self._truthy(left):
                return left
            return self._eval(node.right, env)
        if op == "or":
            left = self._eval(node.left, env)
            if self._truthy(left):
                return left
            return self._eval(node.right, env)

        left = self._eval(node.left, env)
        right = self._eval(node.right, env)

        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                if right == 0:
                    raise OptiZeroDivisionError(node.line)
                return left / right
            if op == "//":
                if right == 0:
                    raise OptiZeroDivisionError(node.line)
                return left // right
            if op == "%":
                if right == 0:
                    raise OptiZeroDivisionError(node.line)
                return left % right
            if op == "**":
                return left ** right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
        except OptiRuntimeError:
            raise
        except Exception as exc:
            raise OptiTypeError(f"Invalid operation '{op}': {exc}", node.line)

        raise OptiRuntimeError(f"Unsupported binary operator: {op}", node.line)

    def _apply_augmented(
        self, operator: str, current: Any, value: Any, node: AugmentedAssignmentNode
    ) -> Any:
        if operator == "+=":
            return current + value
        if operator == "-=":
            return current - value
        if operator == "*=":
            return current * value
        if operator == "/=":
            if value == 0:
                raise OptiZeroDivisionError(node.line)
            return current / value
        raise OptiRuntimeError(f"Unsupported augmented operator: {operator}", node.line)

    def _call(self, callee: Any, args: List[Any], node: FunctionCallNode) -> Any:
        if isinstance(callee, UserFunction):
            return callee.call(self, args)
        if callable(callee):
            try:
                return callee(*args)
            except TypeError as exc:
                raise OptiTypeError(f"Invalid function call: {exc}", node.line)
            except ValueError as exc:
                raise OptiValueError(str(exc), node.line)
        raise OptiTypeError("Object is not callable", node.line)

    def _index(self, collection: Any, index: Any, node: IndexNode) -> Any:
        if isinstance(collection, (list, str, tuple)):
            if not isinstance(index, int):
                raise OptiTypeError(
                    "Sequence index must be an integer",
                    node.line,
                    expected="int",
                    got=type(index).__name__,
                )
            try:
                return collection[index]
            except Exception:
                raise OptiIndexError(index, len(collection), node.line)

        if isinstance(collection, dict):
            try:
                return collection[index]
            except Exception:
                raise OptiKeyError(index, node.line)

        raise OptiTypeError("Object is not indexable", node.line)

    @staticmethod
    def _truthy(value: Any) -> bool:
        return bool(value)

    def _serialize_symbol_value(self, value: Any) -> Any:
        if isinstance(value, UserFunction):
            return f"<function {value.name}>"
        if callable(value):
            name = getattr(value, "__name__", type(value).__name__)
            return f"<builtin {name}>"
        return value

    def get_symbol_table(self, include_builtins: bool = False) -> Dict[str, Any]:
        table: Dict[str, Any] = {}
        for name, value in self.globals.values.items():
            if not include_builtins and name in self._builtin_names:
                continue
            table[name] = self._serialize_symbol_value(value)
        return table


def execute(source: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    """
    Parse and execute OptiLang source code.
    """
    start = time.perf_counter()

    try:
        tokens = tokenize(source)
        program = parse(tokens)
        return Executor(timeout_seconds=timeout_seconds).run(program)
    except (LexerError, ParserError, OptiRuntimeError) as exc:
        elapsed = time.perf_counter() - start
        return ExecutionResult(
            output="",
            errors=[str(exc)],
            execution_time=elapsed,
            profiling=None,
            symbol_table={},
        )
