"""
Runtime executor for OptiLang AST programs.

The public execute() function at the bottom of this module orchestrates
the full pipeline: tokenize → parse → semantic analysis → run.
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
from .profiler import Profiler, ProfilingData
from .semantic_analyzer import SemanticAnalyzer
from .utils.errors import (
    ArgumentError,
    IndexError as OptiIndexError,
    KeyError as OptiKeyError,
    NameError as OptiNameError,
    ParserError,
    RecursionError as OptiRecursionError,
    RuntimeError as OptiRuntimeError,
    SemanticError,
    TimeoutError as OptiTimeoutError,
    TypeError as OptiTypeError,
    ValueError as OptiValueError,
    ZeroDivisionError as OptiZeroDivisionError,
)
from .utils.errors import LexerError


class _BreakSignal(Exception):
    """Internal signal raised by a break statement."""


class _ContinueSignal(Exception):
    """Internal signal raised by a continue statement."""


@dataclass
class _ReturnSignal(Exception):
    """Internal signal raised by a return statement, carrying its value."""

    value: Any


class Environment:
    """Hierarchical variable scope chain."""

    def __init__(self, parent: Optional["Environment"] = None) -> None:
        """
        Initialise a new scope.

        Args:
            parent: The enclosing scope, or None for the global scope.
        """
        self.parent = parent
        self.values: Dict[str, Any] = {}

    def define(self, name: str, value: Any) -> None:
        """Bind a new name in the current scope."""
        self.values[name] = value

    def get(self, name: str, node: Optional[ASTNode] = None) -> Any:
        """
        Look up a name, searching parent scopes if necessary.

        Raises:
            OptiNameError: If the name is not found in any scope.
        """
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name, node)
        raise OptiNameError(name, getattr(node, "line", None))

    def assign(self, name: str, value: Any) -> None:
        """
        Assign to an existing name (nearest scope that owns it),
        or create it in the current scope if it does not exist anywhere.
        """
        if name in self.values:
            self.values[name] = value
            return
        if self.parent is not None and self.parent.contains(name):
            self.parent.assign(name, value)
            return
        self.values[name] = value

    def contains(self, name: str) -> bool:
        """Return True if *name* exists anywhere in this scope chain."""
        if name in self.values:
            return True
        if self.parent is None:
            return False
        return self.parent.contains(name)

    def all_values(self) -> Dict[str, Any]:
        """
        Return every variable visible from this scope (including parents).

        Used by the profiler for memory estimation.
        """
        result: Dict[str, Any] = {}
        if self.parent is not None:
            result.update(self.parent.all_values())
        result.update(self.values)
        return result


class UserFunction:
    """Callable wrapper for user-defined function nodes."""

    def __init__(self, node: FunctionDefNode, closure: Environment) -> None:
        """
        Wrap a function definition node with its enclosing scope.

        Args:
            node:    The FunctionDefNode from the AST.
            closure: The environment in which the function was defined.
        """
        self.node = node
        self.closure = closure

    @property
    def name(self) -> str:
        """The function's declared name."""
        return self.node.name.name

    def call(self, executor: "Executor", args: List[Any]) -> Any:
        """
        Execute this function with the given arguments.

        Args:
            executor: The active Executor instance.
            args:     Evaluated argument values.

        Returns:
            The function's return value, or None.

        Raises:
            ArgumentError:      Wrong number of arguments.
            OptiRecursionError: Recursion limit exceeded.
        """
        expected = len(self.node.parameters)
        got = len(args)
        if expected != got:
            raise ArgumentError(self.name, expected, got, self.node.line)

        # pylint: disable=protected-access
        executor._call_depth += 1
        if executor._call_depth > executor.max_recursion_depth:
            executor._call_depth -= 1
            raise OptiRecursionError(executor.max_recursion_depth, self.node.line)

        frame = Environment(parent=self.closure)
        for param, arg in zip(self.node.parameters, args):
            frame.define(param.name, arg)

        if executor.profiler is not None:
            executor.profiler.start_function_call(self.name)

        try:
            # pylint: disable=protected-access
            executor._execute_block(self.node.body, frame)
            return None
        except _ReturnSignal as signal:
            return signal.value
        finally:
            if executor.profiler is not None:
                executor.profiler.end_function_call(self.name)
            executor._call_depth -= 1


class Executor:
    """AST tree-walking executor for OptiLang programs."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        profiler: Optional[Profiler] = None,
        enable_profiling: bool = True,
    ) -> None:
        """
        Initialise the executor.

        Args:
            timeout_seconds:  Maximum execution time before TimeoutError.
                              Set <= 0 to disable.
            profiler:         Existing Profiler instance to use. If None
                              and enable_profiling is True, one is created.
            enable_profiling: Whether to collect profiling data.
        """
        self._loop_depth: int = 0
        self._max_loop_depth_seen: int = 0
        self.timeout_seconds = timeout_seconds
        self._start_time = 0.0
        self._output: List[str] = []
        self.max_recursion_depth = 1000
        self._call_depth = 0

        if enable_profiling:
            self.profiler: Optional[Profiler] = (
                profiler if profiler is not None else Profiler()
            )
        else:
            self.profiler = None

        self.globals = Environment()
        self._install_builtins()
        self._builtin_names = set(self.globals.values.keys())

    def run(self, program: ProgramNode) -> ExecutionResult:
        """
        Execute a parsed, semantically-validated program and return its result.

        Note: SemanticAnalyzer().analyze(program) must be called before this
        method. The public execute() function handles this automatically.

        Args:
            program: The root ProgramNode from the parser.

        Returns:
            ExecutionResult with output, errors, timing, profiling data,
            and the final symbol table.
        """
        self._start_time = time.perf_counter()
        self._output = []
        errors: List[str] = []

        if self.profiler is not None:
            self.profiler.start()

        try:
            self._execute_program(program)
        except OptiRuntimeError as exc:
            errors.append(str(exc))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(f"Runtime error: {exc}")
        finally:
            if self.profiler is not None:
                self.profiler.stop()

        elapsed = time.perf_counter() - self._start_time
        profiling_data: Optional[ProfilingData] = (
            self.profiler.get_data() if self.profiler is not None else None
        )

        return ExecutionResult(
            output="".join(self._output).rstrip("\n"),
            errors=errors,
            execution_time=elapsed,
            profiling=profiling_data,
            symbol_table=self.get_symbol_table(include_builtins=False),
        )

    def _install_builtins(self) -> None:
        """Register built-in functions and types into the global scope."""
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
        """Implementation of the built-in print() function."""
        self._output.append(" ".join(str(arg) for arg in args) + "\n")

    def _check_timeout(self, node: Optional[ASTNode] = None) -> None:
        """Raise OptiTimeoutError if the execution time limit is exceeded."""
        if self.timeout_seconds <= 0:
            return
        elapsed = time.perf_counter() - self._start_time
        if elapsed > self.timeout_seconds:
            raise OptiTimeoutError(self.timeout_seconds, getattr(node, "line", None))

    def _execute_program(self, program: ProgramNode) -> None:
        """Execute the top-level list of statements."""
        self._execute_block(program.statements, self.globals)

    def _execute_block(self, statements: List[ASTNode], env: Environment) -> None:
        """Execute a list of statements sequentially in the given scope."""
        for statement in statements:
            self._check_timeout(statement)
            self._execute_statement(statement, env)

    def _execute_statement(self, node: ASTNode, env: Environment) -> None:
        """Wrap a single statement with profiler hooks, then execute it."""
        line = getattr(node, "line", None)

        if self.profiler is not None and line is not None:
            self.profiler.start_line(line, env.all_values())

        try:
            self._exec(node, env)
        finally:
            if self.profiler is not None and line is not None:
                self.profiler.end_line(line)

    def _exec(self, node: ASTNode, env: Environment) -> None:
        """Execute a single AST statement node."""
        if isinstance(node, AssignmentNode):
            value = self._eval(node.value, env)
            env.assign(node.target.name, value)

        elif isinstance(node, AugmentedAssignmentNode):
            current = env.get(node.target.name, node.target)
            value = self._eval(node.value, env)
            new_value = self._apply_augmented_op(node.operator, current, value, node)
            env.assign(node.target.name, new_value)

        elif isinstance(node, IfNode):
            condition = self._eval(node.condition, env)
            if self._truthy(condition):
                self._execute_block(node.if_block, env)
            else:
                matched = False
                for elif_cond, elif_block in node.elif_parts:
                    if self._truthy(self._eval(elif_cond, env)):
                        self._execute_block(elif_block, env)
                        matched = True
                        break
                if not matched and node.else_block:
                    self._execute_block(node.else_block, env)

        elif isinstance(node, WhileNode):
            self._loop_depth += 1
            self._max_loop_depth_seen = max(self._max_loop_depth_seen, self._loop_depth)
            while self._truthy(self._eval(node.condition, env)):
                self._check_timeout(node)
                try:
                    self._execute_block(node.body, env)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue
            self._loop_depth -= 1

        elif isinstance(node, ForNode):
            self._loop_depth += 1
            self._max_loop_depth_seen = max(self._max_loop_depth_seen, self._loop_depth)
            iterable = self._eval(node.iterable, env)
            try:
                items = list(iterable)
            except TypeError as exc:
                raise OptiTypeError(
                    f"Object is not iterable: {exc}", node.line
                ) from exc
            for item in items:
                self._check_timeout(node)
                env.assign(node.iterator.name, item)
                try:
                    self._execute_block(node.body, env)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue
            self._loop_depth -= 1

        elif isinstance(node, FunctionDefNode):
            func = UserFunction(node, env)
            env.define(node.name.name, func)

        elif isinstance(node, ReturnNode):
            value = self._eval(node.value, env) if node.value is not None else None
            raise _ReturnSignal(value)

        elif isinstance(node, BreakNode):
            raise _BreakSignal()

        elif isinstance(node, ContinueNode):
            raise _ContinueSignal()

        elif isinstance(node, PassNode):
            pass

        elif isinstance(node, TryNode):
            try:
                self._execute_block(node.try_block, env)
            except OptiRuntimeError:
                if node.except_block is not None:
                    self._execute_block(node.except_block, env)
            finally:
                if node.finally_block is not None:
                    self._execute_block(node.finally_block, env)

        else:
            # Expression statement (e.g. a bare function call)
            self._eval(node, env)

    def _eval(self, node: ASTNode, env: Environment) -> Any:
        """
        Evaluate an expression node and return its value.

        Args:
            node: Any expression ASTNode.
            env:  The current scope.

        Returns:
            The Python value the expression evaluates to.
        """
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
                except TypeError as exc:
                    raise OptiTypeError(f"Invalid unary '-': {exc}", node.line) from exc
            if node.operator == "not":
                return not self._truthy(operand)
            raise OptiRuntimeError(
                f"Unsupported unary operator: {node.operator}", node.line
            )

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
                except TypeError as exc:
                    raise OptiTypeError(
                        f"Unhashable dictionary key: {exc}", node.line
                    ) from exc
            return result

        if isinstance(node, IndexNode):
            collection = self._eval(node.collection, env)
            index = self._eval(node.index, env)
            return self._index(collection, index, node)

        raise OptiRuntimeError(
            f"Unsupported AST node: {type(node).__name__}", node.line
        )

    def _eval_binary(self, node: BinaryOpNode, env: Environment) -> Any:
        """
        Evaluate a binary operation node.

        Short-circuits for 'and' / 'or'; evaluates both sides otherwise.
        """
        op = node.operator

        # Short-circuit operators
        if op == "and":
            left = self._eval(node.left, env)
            return left if not self._truthy(left) else self._eval(node.right, env)
        if op == "or":
            left = self._eval(node.left, env)
            return left if self._truthy(left) else self._eval(node.right, env)

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
                return left**right
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
        except (TypeError, ValueError) as exc:
            raise OptiTypeError(str(exc), node.line) from exc

        raise OptiRuntimeError(f"Unsupported operator: {op}", node.line)

    def _apply_augmented_op(
        self, operator: str, current: Any, value: Any, node: ASTNode
    ) -> Any:
        """Apply an augmented assignment operator and return the new value."""
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
        """Invoke a callable (user function or built-in)."""
        if isinstance(callee, UserFunction):
            return callee.call(self, args)
        if callable(callee):
            try:
                return callee(*args)
            except TypeError as exc:
                raise OptiTypeError(f"Invalid function call: {exc}", node.line) from exc
            except ValueError as exc:
                raise OptiValueError(str(exc), node.line) from exc
        raise OptiTypeError("Object is not callable", node.line)

    def _index(self, collection: Any, index: Any, node: IndexNode) -> Any:
        """Perform an index/key lookup on a collection."""
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
            except IndexError as exc:
                raise OptiIndexError(index, len(collection), node.line) from exc

        if isinstance(collection, dict):
            try:
                return collection[index]
            except KeyError as exc:
                raise OptiKeyError(index, node.line) from exc

        raise OptiTypeError("Object is not indexable", node.line)

    @staticmethod
    def _truthy(value: Any) -> bool:
        """Return the boolean truth value of *value*."""
        return bool(value)

    def _serialize_symbol_value(self, value: Any) -> Any:
        """Convert a runtime value to a JSON-friendly representation."""
        if isinstance(value, UserFunction):
            return f"<function {value.name}>"
        if callable(value):
            fn_name = getattr(value, "__name__", type(value).__name__)
            return f"<builtin {fn_name}>"
        return value

    def get_symbol_table(self, include_builtins: bool = False) -> Dict[str, Any]:
        """
        Return the global symbol table as a serializable dictionary.

        Args:
            include_builtins: If False (default), built-in names are excluded.
        """
        table: Dict[str, Any] = {}
        for name, value in self.globals.values.items():
            if not include_builtins and name in self._builtin_names:
                continue
            table[name] = self._serialize_symbol_value(value)
        return table


# Public API


def execute(
    source: str,
    timeout_seconds: float = 5.0,
    enable_profiling: bool = True,
) -> ExecutionResult:
    """
    Tokenize, parse, semantically analyse, and execute OptiLang source code.

    Full pipeline:
        tokenize(source)
            → parse(tokens)
                → SemanticAnalyzer().analyze(program)
                    → Executor(...).run(program)

    If any phase raises an error, execution is aborted immediately and the
    error message is returned inside result.errors. The later phases
    (profiling, optimizer, scorer) are never reached in that case.
    """
    start = time.perf_counter()

    try:
        tokens = tokenize(source)
        program = parse(tokens)
        SemanticAnalyzer().analyze(program)
        return Executor(
            timeout_seconds=timeout_seconds,
            enable_profiling=enable_profiling,
        ).run(program)

    except (LexerError, ParserError, SemanticError, OptiRuntimeError) as exc:
        elapsed = time.perf_counter() - start
        return ExecutionResult(
            output="",
            errors=[str(exc)],
            execution_time=elapsed,
            profiling=None,
            symbol_table={},
        )
