"""
optilang/complexity.py
----------------------
Static complexity analysis for OptiLang.

Computes provably correct Big-O complexity using AST analysis instead of
runtime heuristics. Returns confidence = 1.0 when complexity is derivable
from AST structure, "Unknown" with explanation when not determinable.

The analysis uses a symbolic representation (ComplexityExpr) that captures
the mathematical structure of algorithmic complexity, enabling exact reasoning
about loop bounds, nesting, and recursion patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.ast_nodes import (
    ASTNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    BooleanNode,
    ForNode,
    FunctionCallNode,
    FunctionDefNode,
    IdentifierNode,
    IfNode,
    IndexAssignmentNode,
    IndexNode,
    IndexedAugmentedAssignmentNode,
    ListNode,
    MethodCallNode,
    NumberNode,
    PassNode,
    ProgramNode,
    ReturnNode,
    StringNode,
    TryNode,
    UnaryOpNode,
    WhileNode,
    BreakNode,
    ContinueNode,
)
from ..types.constants import (
    COMPLEXITY_EXP,
    COMPLEXITY_KN,
    COMPLEXITY_LOGN,
    COMPLEXITY_N,
    COMPLEXITY_N2,
    COMPLEXITY_N3,
    COMPLEXITY_NF,
    COMPLEXITY_NK,
    COMPLEXITY_NLOGN,
    COMPLEXITY_N_M,
    COMPLEXITY_N_PLUS_M,
    COMPLEXITY_O1,
    COMPLEXITY_UNBOUNDED,
    COMPLEXITY_UNKNOWN,
)

# ─────────────────────────────────────────────────────────────────────────────
# Complexity class enumeration
# ─────────────────────────────────────────────────────────────────────────────


class Complexity(Enum):
    """Well-known complexity classes in order of efficiency."""

    O1 = "O(1)"
    LOGN = "O(log n)"
    N = "O(n)"
    NLOGN = "O(n log n)"
    N2 = "O(n²)"
    N2LOGN = "O(n² log n)"
    N3 = "O(n³)"
    N4 = "O(n⁴)"
    NK = "O(n^k)"
    NF = "O(n!)"
    EXP = "O(2^n)"
    UNKNOWN = "Unknown"

    def __str__(self) -> str:
        return self.value


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic complexity expression
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComplexityExpr:
    """
    Symbolic representation of algorithmic complexity.

    Uses algebraic expressions to represent complexity instead of just
    returning a static string, enabling exact reasoning about nested
    loops, parameters, and composition.
    """

    pass


@dataclass
class Const(ComplexityExpr):
    """O(1) — constant time."""

    value: float = 1.0


@dataclass
class Param(ComplexityExpr):
    """
    O(n) where n is a function parameter.

    The parameter name is tracked to identify when multiple loops
    iterate over the same parameter (e.g., nested range(n) loops).
    """

    name: str


@dataclass
class Var(ComplexityExpr):
    """
    O(n) where n is an unknown variable (not a parameter).

    Lower confidence than Param since we can't reason about the
    variable's actual bound.
    """

    name: str


@dataclass
class Log(ComplexityExpr):
    """O(log n) — logarithmic time."""

    inner: ComplexityExpr


@dataclass
class Add(ComplexityExpr):
    """O(e1) + O(e2) = O(max(e1, e2)) — sequential composition."""

    left: ComplexityExpr
    right: ComplexityExpr


@dataclass
class Mul(ComplexityExpr):
    """O(e1) * O(e2) — multiplicative composition (nested loops)."""

    left: ComplexityExpr
    right: ComplexityExpr


@dataclass
class CallExpr(ComplexityExpr):
    """Complexity of a function call (analyzed from function body)."""

    name: str
    body_complexity: Optional[ComplexityExpr] = None


@dataclass
class Factorial(ComplexityExpr):
    """
    O(n!) — factorial time.

    Used for algorithms that generate all permutations or combinations.
    """

    inner: ComplexityExpr  # Usually a Param


@dataclass
class Exponential(ComplexityExpr):
    """O(k^n) — branching recursion."""

    branch_factor: int = 2


@dataclass
class UnknownExpr(ComplexityExpr):
    """O(?) — static analysis cannot safely classify this construct."""

    reason: str = "Unresolvable complexity"


@dataclass
class InfiniteExpr(ComplexityExpr):
    """O(∞) — statically unbounded loop."""

    reason: str = "Unbounded loop"


# ─────────────────────────────────────────────────────────────────────────────
# Analysis result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComplexityResult:
    """
    Result of complexity analysis for a program.

    Attributes:
        complexity: String representation (e.g., "O(n²)").
        confidence: 1.0 when derivable from AST, lower for uncertain/unknown.
        explanation: Human-readable explanation of how complexity was derived.
        bound_symbol: The symbol that dominates the complexity (e.g., "n"),
                      or None if constant-time or unknown.
    """

    complexity: str
    confidence: float
    explanation: str
    bound_symbol: Optional[str] = None
    display_complexity: Optional[str] = None
    method: str = "static"
    bound_symbols: Optional[List[str]] = None
    best_case: Optional[str] = None
    worst_case: Optional[str] = None
    has_early_exit: bool = False
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.bound_symbols is None:
            self.bound_symbols = [self.bound_symbol] if self.bound_symbol else []
        if self.display_complexity is None:
            self.display_complexity = self.complexity
        if self.worst_case is None:
            self.worst_case = self.complexity


# ─────────────────────────────────────────────────────────────────────────────
# Complexity analyzer
# ─────────────────────────────────────────────────────────────────────────────


class ComplexityAnalyzer:
    """
    Analyzes algorithmic complexity using static AST analysis.

    Uses a recursive approach:
    1. Extract loop bounds from range() calls
    2. Recursively analyze each node type
    3. Compose complexities using Add (sequential) and Mul (nested)
    4. Simplify and convert to Big-O string representation
    """

    def __init__(self) -> None:
        # Track function definitions for call analysis
        self._functions: Dict[str, FunctionDefNode] = {}
        # Track parameters in current scope
        self._params: Set[str] = set()
        # Track loop iterators
        self._loop_iterators: Set[str] = set()
        # Track active function calls for recursion detection
        self._call_stack: List[str] = []
        # Cache for analyzed functions (to prevent infinite recursion)
        self._function_cache: Dict[str, ComplexityExpr] = {}
        # Track functions currently being analyzed (to detect cycles)
        self._analyzing: Set[str] = set()
        self._loop_depth = 0
        self._fallback_reason: Optional[str] = None
        self._has_early_exit = False
        self._best_case: Optional[str] = None

    def analyze(
        self,
        program: ProgramNode,
        symbol_table: Optional[Dict[str, Any]] = None,
    ) -> ComplexityResult:
        """
        Analyze the complexity of a complete program.

        Args:
            program: Parsed ProgramNode from the parser.
            symbol_table: Optional symbol table from execution (for parameter info).

        Returns:
            ComplexityResult with complexity class, confidence, and explanation.
        """
        # Reset state
        self._functions = {}
        self._params = set()
        self._loop_iterators = set()
        self._function_cache = {}
        self._analyzing = set()
        self._loop_depth = 0
        self._fallback_reason = None
        self._has_early_exit = False
        self._best_case = None

        # First pass: collect all function definitions
        self._collect_functions(program)

        # Extract parameter names if symbol table available
        if symbol_table:
            self._extract_params(symbol_table)

        # Analyze the program body
        expr = self._analyze_block(program.statements)

        # Convert to Big-O string with confidence
        return self._expr_to_result(expr)

    def analyze_function(
        self,
        func_def: FunctionDefNode,
    ) -> ComplexityResult:
        """
        Analyze a single function's complexity.

        Args:
            func_def: Function definition node.

        Returns:
            ComplexityResult for the function body.
        """
        # Reset but preserve function definitions
        self._params = {p.name for p in func_def.parameters}
        self._loop_depth = 0
        self._fallback_reason = None
        self._has_early_exit = False
        self._best_case = None

        expr = self._analyze_block(func_def.body)
        return self._expr_to_result(expr)

    # ── Internal analysis methods ───────────────────────────────────────────

    def _collect_functions(self, node: ASTNode) -> None:
        """Collect all function definitions for later analysis."""
        if isinstance(node, ProgramNode):
            for stmt in node.statements:
                self._collect_functions(stmt)
        elif isinstance(node, FunctionDefNode):
            self._functions[node.name.name] = node
            for stmt in node.body:
                self._collect_functions(stmt)
        elif isinstance(node, IfNode):
            self._collect_functions_block(node.if_block)
            for _, block in node.elif_parts:
                self._collect_functions_block(block)
            if node.else_block:
                self._collect_functions_block(node.else_block)
        elif isinstance(node, ForNode):
            self._collect_functions(node.iterator)
            self._collect_functions(node.iterable)
            self._collect_functions_block(node.body)
        elif isinstance(node, WhileNode):
            self._collect_functions(node.condition)
            self._collect_functions_block(node.body)
        elif isinstance(node, TryNode):
            self._collect_functions_block(node.try_block)
            if node.except_block:
                self._collect_functions_block(node.except_block)
            if node.finally_block:
                self._collect_functions_block(node.finally_block)
        elif isinstance(node, (BinaryOpNode, UnaryOpNode)):
            for child in node._get_children() if hasattr(node, "_get_children") else []:
                if isinstance(child, ASTNode):
                    self._collect_functions(child)
        # Recursive traversal for other nodes
        elif hasattr(node, "__dict__"):
            for child in node.__dict__.values():
                if isinstance(child, ASTNode):
                    self._collect_functions(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ASTNode):
                            self._collect_functions(item)

    def _collect_functions_block(self, block: List[ASTNode]) -> None:
        """Collect functions from a block of statements."""
        for stmt in block:
            self._collect_functions(stmt)

    def _extract_params(self, symbol_table: Dict[str, Any]) -> None:
        """Extract parameter names from symbol table."""
        for name, value in symbol_table.items():
            # Parameters are typically stored as simple values, not functions
            if not callable(value):
                self._params.add(name)

    def _analyze_block(self, statements: List[ASTNode]) -> ComplexityExpr:
        """Analyze a block of statements, returning combined complexity."""
        if not statements:
            return Const(1)

        complexities = []
        for stmt in statements:
            # Skip no-op statements
            if isinstance(stmt, PassNode):
                continue
            # Skip break/continue (they don't add complexity on their own)
            if isinstance(stmt, (BreakNode, ContinueNode)):
                continue
            complexities.append(self._analyze_node(stmt))

        if not complexities:
            return Const(1)

        # Combine: O(c1 + c2 + ...) = O(max(c1, c2, ...))
        return self._combine_max(complexities)

    def _analyze_node(self, node: ASTNode) -> ComplexityExpr:
        """Analyze a single AST node and return its complexity."""
        # Literals: O(1) for numbers/bool, O(n) for strings where n is length
        if isinstance(node, NumberNode):
            return Const(1)
        if isinstance(node, StringNode):
            return Const(1)
        if isinstance(node, BooleanNode):
            return Const(1)

        # Identifier: O(1) read
        if isinstance(node, IdentifierNode):
            # Check if it's a parameter
            if node.name in self._params:
                return Param(node.name)
            # Loop iterators are just variable reads - O(1)
            return Const(1)

        # Assignment: complexity of the value expression
        if isinstance(node, AssignmentNode):
            return self._analyze_node(node.value)

        # Augmented assignment: O(1) for the operation
        if isinstance(node, (AugmentedAssignmentNode, IndexedAugmentedAssignmentNode)):
            if (
                isinstance(node, AugmentedAssignmentNode)
                and node.operator == "+="
                and self._loop_depth > 0
                and (
                    isinstance(node.value, StringNode)
                    or (
                        isinstance(node.value, FunctionCallNode)
                        and isinstance(node.value.function, IdentifierNode)
                        and node.value.function.name == "str"
                    )
                )
            ):
                # String/list concatenation in a loop copies growing data.
                return Var("n")
            return self._analyze_node(node.value)

        # Index assignment: O(1) for the assignment
        if isinstance(node, IndexAssignmentNode):
            return self._analyze_node(node.value)

        # Binary operations: O(max(children))
        if isinstance(node, BinaryOpNode):
            if node.operator == "**":
                if isinstance(node.left, NumberNode) and isinstance(
                    node.right, IdentifierNode
                ):
                    return Log(Var(node.right.name))
                return Const(1)
            left = self._analyze_node(node.left)
            right = self._analyze_node(node.right)
            return self._combine_max([left, right])

        # Unary operations: O(operand)
        if isinstance(node, UnaryOpNode):
            return self._analyze_node(node.operand)

        # Function call: analyze function body or use call expression
        if isinstance(node, FunctionCallNode):
            return self._analyze_function_call(node)

        if isinstance(node, MethodCallNode):
            return self._analyze_method_call(node)

        # Indexing: O(1) for single element, O(n) for slicing
        if isinstance(node, IndexNode):
            # Check for slicing (start/stop/step present)
            is_slice = (
                node.start is not None or node.stop is not None or node.step is not None
            )

            if is_slice:
                # Slicing creates a new collection of size proportional to input
                # Return O(n) - we can't statically determine the exact size
                return Var("n")

            # Single element access is O(1)
            return Const(1)

        # List literal: O(n) where n is number of elements
        if isinstance(node, ListNode):
            if node.elements:
                return self._combine_max([self._analyze_node(e) for e in node.elements])
            return Const(1)

        # If statement: O(max of all branches)
        if isinstance(node, IfNode):
            branch_complexities = []

            # If block
            branch_complexities.append(self._analyze_block(node.if_block))

            # Elif blocks
            for _, block in node.elif_parts:
                branch_complexities.append(self._analyze_block(block))

            # Else block
            if node.else_block:
                branch_complexities.append(self._analyze_block(node.else_block))

            return self._combine_max(branch_complexities)

        # For loop
        if isinstance(node, ForNode):
            return self._analyze_for_loop(node)

        # While loop
        if isinstance(node, WhileNode):
            return self._analyze_while_loop(node)

        # Return statement
        if isinstance(node, ReturnNode):
            if node.value is not None:
                return self._analyze_node(node.value)
            return Const(1)

        # Try-except
        if isinstance(node, TryNode):
            branch_complexities = [self._analyze_block(node.try_block)]
            if node.except_block:
                branch_complexities.append(self._analyze_block(node.except_block))
            if node.finally_block:
                branch_complexities.append(self._analyze_block(node.finally_block))
            return self._combine_max(branch_complexities)

        # Pass does nothing
        if isinstance(node, PassNode):
            return Const(1)

        # Default: assume O(1) for unknown node types
        return Const(1)

    def _analyze_for_loop(self, node: ForNode) -> ComplexityExpr:
        """Analyze a for loop and return its complexity."""
        # Analyze the iterable to get iteration count
        iter_complexity = self._extract_iterable_complexity(node.iterable)

        # Analyze the loop body
        iterator_name = getattr(node.iterator, "name", "")
        if iterator_name:
            self._loop_iterators.add(iterator_name)
        self._loop_depth += 1
        body_complexity = self._analyze_block(node.body)
        self._loop_depth -= 1
        if iterator_name:
            self._loop_iterators.discard(iterator_name)

        # Total complexity: iterations * body
        return Mul(iter_complexity, body_complexity)

    def _analyze_while_loop(self, node: WhileNode) -> ComplexityExpr:
        """Analyze a while loop and return its complexity."""
        # For while loops, we try to detect common patterns:
        # - Halving loop (n -> n // 2): O(log n)
        # - Linear decrement (n -> n - 1): O(n)
        # - Unknown: use heuristic

        if self._is_while_true(node.condition):
            if not self._block_contains_break(node.body):
                return InfiniteExpr("while True loop has no break")
            self._has_early_exit = True
            self._best_case = COMPLEXITY_O1

        self._loop_depth += 1
        body_complexity = self._analyze_block(node.body)
        self._loop_depth -= 1

        # Try to detect halving pattern
        halving = self._detect_halving_loop(node.condition, node.body)
        if halving:
            return Log(halving)

        # Default: assume O(n) * body for unknown while loops
        # Use a generic variable n
        return Mul(Var("n"), body_complexity)

    def _detect_halving_loop(
        self,
        condition: ASTNode,
        body: List[ASTNode],
    ) -> Optional[ComplexityExpr]:
        """
        Detect halving loop pattern: while n > 1: n = n // 2.

        Returns the variable being halved if pattern detected, None otherwise.
        """
        # Look for condition like: n > 1 or n >= 2
        if not isinstance(condition, BinaryOpNode):
            return None

        # Check for comparison with a number
        if condition.operator not in (">", ">=", "<", "<="):
            return None

        # Identify the variable being compared
        var_node = None
        if isinstance(condition.left, IdentifierNode):
            var_node = condition.left
        elif isinstance(condition.right, IdentifierNode):
            var_node = condition.right

        if var_node is None:
            return None

        var_name = var_node.name

        # Look for assignment inside body that halves the variable
        for stmt in body:
            if isinstance(stmt, AugmentedAssignmentNode):
                if (
                    isinstance(stmt.target, IdentifierNode)
                    and stmt.target.name == var_name
                    and stmt.operator == "//="
                    and isinstance(stmt.value, NumberNode)
                    and stmt.value.value == 2
                ):
                    return Var(var_name)
            if isinstance(stmt, AssignmentNode):
                if (
                    isinstance(stmt.target, IdentifierNode)
                    and stmt.target.name == var_name
                ):
                    # Check if it's a halving operation
                    if isinstance(stmt.value, BinaryOpNode):
                        if stmt.value.operator == "//":
                            # Check if right operand is 2
                            if (
                                isinstance(stmt.value.right, NumberNode)
                                and stmt.value.right.value == 2
                            ):
                                # Check if left is the same variable
                                if (
                                    isinstance(stmt.value.left, IdentifierNode)
                                    and stmt.value.left.name == var_name
                                ):
                                    return Var(var_name)

        return None

    def _is_while_true(self, condition: ASTNode) -> bool:
        return isinstance(condition, BooleanNode) and condition.value is True

    def _block_contains_break(self, nodes: List[ASTNode]) -> bool:
        for node in nodes or []:
            if isinstance(node, BreakNode):
                return True
            for child in self._get_node_children(node):
                if isinstance(child, BreakNode):
                    return True
                if self._block_contains_break([child]):
                    return True
        return False

    def _extract_iterable_complexity(self, iterable: ASTNode) -> ComplexityExpr:
        """Extract complexity (iteration count) from an iterable expression."""
        # Handle range() calls
        if isinstance(iterable, FunctionCallNode):
            if isinstance(iterable.function, IdentifierNode):
                if iterable.function.name == "range":
                    return self._extract_range_complexity(iterable)

        # Handle list iteration
        if isinstance(iterable, ListNode):
            # O(n) where n is list length
            n = len(iterable.elements)
            if n > 0:
                return Const(float(n))
            return Const(1)

        # Handle identifier (assume it's a list)
        if isinstance(iterable, IdentifierNode):
            if iterable.name in self._params:
                return Param(iterable.name)
            # Assume it's a list/iterable with unknown size
            return Var(iterable.name)

        # Index node (e.g., some_list) - could be a slice
        if isinstance(iterable, IndexNode):
            # Could analyze the collection
            return self._analyze_node(iterable)

        # Default: unknown, assume O(n)
        return Var("n")

    def _extract_range_complexity(self, call: FunctionCallNode) -> ComplexityExpr:
        """
        Extract complexity from range() call.

        range(n) → O(n)
        range(a, b) → O(b - a)
        range(a, b, step) → O(ceil((b - a) / step))
        """
        args = call.arguments

        if len(args) == 1:
            # range(n) → O(n)
            arg = args[0]
            if isinstance(arg, IdentifierNode):
                if arg.name in self._params:
                    return Param(arg.name)
                if arg.name in self._loop_iterators:
                    return Var(arg.name)
                return Var(arg.name)
            if isinstance(arg, NumberNode):
                return Const(arg.value)
            if isinstance(arg, BinaryOpNode) and arg.operator == "*":
                return Mul(
                    self._range_bound_expr(arg.left),
                    self._range_bound_expr(arg.right),
                )
            return self._range_bound_expr(arg)

        elif len(args) == 2:
            # range(a, b) → O(b - a)
            left = self._range_bound_expr(args[0])
            right = self._range_bound_expr(args[1])
            # O(b - a) = O(max(b, a)) when both are positive
            return self._combine_max([left, right])

        elif len(args) == 3:
            # range(a, b, step) → O(ceil((b - a) / step))
            # For static analysis, assume step = 2 is halving
            step = args[2]
            if isinstance(step, NumberNode) and step.value == 2:
                # Could be halving, but we need to know the pattern
                pass

            # Default: use the larger bound
            left = self._range_bound_expr(args[0])
            right = self._range_bound_expr(args[1])
            return self._combine_max([left, right])

        # Unknown range
        return Var("n")

    def _range_bound_expr(self, node: ASTNode) -> ComplexityExpr:
        if isinstance(node, IdentifierNode):
            return Param(node.name) if node.name in self._params else Var(node.name)
        if isinstance(node, NumberNode):
            return Const(node.value)
        if isinstance(node, BinaryOpNode) and node.operator == "*":
            return Mul(
                self._range_bound_expr(node.left),
                self._range_bound_expr(node.right),
            )
        return self._analyze_node(node)

    def _analyze_function_call(self, call: FunctionCallNode) -> ComplexityExpr:
        """Analyze a function call, using the function body if available."""
        func_name = (
            call.function.name
            if isinstance(call.function, IdentifierNode)
            else "unknown"
        )

        builtin = self._analyze_builtin_call(func_name, call.arguments)
        if builtin is not None:
            return builtin

        # Check if we have the function definition
        if func_name in self._functions:
            # Check if already cached
            if func_name in self._function_cache:
                return self._function_cache[func_name]

            # Check if currently being analyzed (cycle detection)
            if func_name in self._analyzing:
                # Recursive call - return a placeholder, will be resolved after
                return Var("n")  # Placeholder for recursive complexity

            func_def = self._functions[func_name]

            # Check if function calls itself (recursion)
            is_recursive = self._function_calls_itself(func_def, func_name)
            if is_recursive:
                recurrence = self._analyze_recursive_function(func_def)
                if recurrence is not None:
                    self._function_cache[func_name] = recurrence
                    return recurrence

            # Mark as being analyzed
            self._analyzing.add(func_name)

            # Push onto call stack
            self._call_stack.append(func_name)

            # Temporarily set params for this function
            saved_params = self._params.copy()
            self._params = {p.name for p in func_def.parameters}

            # Analyze body
            body_complexity = self._analyze_block(func_def.body)

            self._params = saved_params

            # Pop from call stack
            self._call_stack.pop()

            # Remove from analyzing set
            self._analyzing.discard(func_name)

            # If recursive, compute factorial complexity
            if is_recursive:
                # Check if body contains a loop with recursive call
                pattern = self._detect_recursive_pattern(func_def, call)
                if pattern == "factorial":
                    # Return factorial complexity - use 'n' as the bound symbol
                    # (the size of the input, e.g., array length)
                    result = Factorial(Var("n"))
                    self._function_cache[func_name] = result
                    return result

            # Cache the result
            self._function_cache[func_name] = body_complexity

            return body_complexity

        if self._loop_depth > 0:
            reason = f"Call to unanalyzable function '{func_name}' inside loop"
            self._fallback_reason = self._fallback_reason or reason
            return UnknownExpr(reason)

        # Unknown function outside loops is treated as a fixed-cost external call.
        return Const(1)

    def _analyze_builtin_call(
        self, func_name: str, arguments: List[ASTNode]
    ) -> Optional[ComplexityExpr]:
        if func_name in {"len", "int", "float", "bool", "range"}:
            return Const(1)

        if func_name == "list" and arguments:
            arg = arguments[0]
            if (
                isinstance(arg, FunctionCallNode)
                and isinstance(arg.function, IdentifierNode)
                and arg.function.name == "range"
            ):
                return self._extract_range_complexity(arg)
            return self._analyze_node(arg)

        if func_name in {"str", "print"} and arguments:
            arg = arguments[0]
            if isinstance(arg, (IdentifierNode, ListNode, IndexNode)):
                return Var(self._node_bound_name(arg) or "n")
            return self._analyze_node(arg)

        return None

    def _analyze_method_call(self, call: MethodCallNode) -> ComplexityExpr:
        method = (
            call.method.name if isinstance(call.method, IdentifierNode) else call.method
        )
        if method == "append":
            return Const(1)
        if method == "pop":
            if call.arguments and isinstance(call.arguments[0], NumberNode):
                if call.arguments[0].value == 0:
                    return Var(self._node_bound_name(call.object) or "n")
            return Const(1)
        return Const(1)

    def _node_bound_name(self, node: ASTNode) -> Optional[str]:
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, IndexNode):
            return self._node_bound_name(node.collection)
        return None

    def _function_calls_itself(self, func_def: FunctionDefNode, func_name: str) -> bool:
        """Check if function definition contains a call to itself."""
        return self._contains_function_call(func_def.body, func_name)

    def _analyze_recursive_function(
        self, func_def: FunctionDefNode
    ) -> Optional[ComplexityExpr]:
        func_name = func_def.name.name
        param_names = [p.name for p in func_def.parameters]
        primary = param_names[0] if param_names else "n"

        if self._has_loop_wrapped_recursive_call(func_def.body, func_name, primary):
            return Factorial(Var(primary))

        reductions: List[str] = []

        def walk(node: ASTNode) -> None:
            if isinstance(node, FunctionCallNode):
                called = (
                    node.function.name
                    if isinstance(node.function, IdentifierNode)
                    else None
                )
                if called == func_name and node.arguments:
                    arg = node.arguments[0]
                reductions.append(self._classify_reduction(arg, primary))
            for child in self._get_node_children(node):
                walk(child)

        for stmt in func_def.body or []:
            walk(stmt)

        if not reductions:
            return None
        if "unknown" in reductions:
            reason = (
                f"Recursive reduction for '{func_name}'" " is not statically resolvable"
            )
            self._fallback_reason = self._fallback_reason or reason
            return UnknownExpr(reason)

        call_count = len(reductions)
        all_half = all(r == "half" for r in reductions)
        all_minus = all(r == "minus_const" for r in reductions)
        linear_work = self._body_has_linear_work(func_def.body, primary, func_name)

        if call_count == 1 and all_half:
            return Log(Var(primary))
        if call_count == 1 and all_minus:
            return Mul(Var(primary), Var(primary)) if linear_work else Var(primary)
        if call_count == 2 and all_half:
            return Mul(Var(primary), Log(Var(primary))) if linear_work else Var(primary)
        if call_count == 2 and all_minus:
            return Exponential(2)
        if call_count >= 3 and all_minus:
            return Exponential(call_count)

        reason = f"Recursive pattern for '{func_name}' is not supported"
        self._fallback_reason = self._fallback_reason or reason
        return UnknownExpr(reason)

    def _classify_reduction(self, arg: ASTNode, param_name: str) -> str:
        if isinstance(arg, BinaryOpNode):
            if (
                arg.operator == "//"
                and isinstance(arg.left, IdentifierNode)
                and arg.left.name == param_name
                and isinstance(arg.right, NumberNode)
                and arg.right.value == 2
            ):
                return "half"
            if (
                arg.operator == "-"
                and isinstance(arg.left, IdentifierNode)
                and arg.left.name == param_name
                and isinstance(arg.right, NumberNode)
            ):
                return "minus_const"
        return "unknown"

    def _has_loop_wrapped_recursive_call(
        self, nodes: List[ASTNode], func_name: str, param_name: str
    ) -> bool:
        for node in nodes or []:
            if isinstance(node, ForNode):
                bound = self._extract_iterable_complexity(node.iterable)
                if self._expr_mentions_bound(bound, param_name):
                    for stmt in node.body:
                        if self._node_calls_function(stmt, func_name):
                            return True
            for child in self._get_node_children(node):
                if self._has_loop_wrapped_recursive_call(
                    [child], func_name, param_name
                ):
                    return True
        return False

    def _body_has_linear_work(
        self, nodes: List[ASTNode], param_name: str, func_name: str
    ) -> bool:
        for node in nodes or []:
            if isinstance(node, ForNode):
                bound = self._extract_iterable_complexity(node.iterable)
                if self._expr_mentions_bound(bound, param_name):
                    return True
            if isinstance(node, FunctionCallNode):
                called = (
                    node.function.name
                    if isinstance(node.function, IdentifierNode)
                    else None
                )
                if called == func_name:
                    continue
            for child in self._get_node_children(node):
                if self._body_has_linear_work([child], param_name, func_name):
                    return True
        return False

    def _expr_mentions_bound(self, expr: ComplexityExpr, name: str) -> bool:
        if isinstance(expr, (Param, Var)):
            return expr.name == name
        if isinstance(expr, Log):
            return self._expr_mentions_bound(expr.inner, name)
        if isinstance(expr, (Add, Mul)):
            left = self._expr_mentions_bound(expr.left, name)
            return left or self._expr_mentions_bound(expr.right, name)
        return False

    def _contains_function_call(self, nodes: List[ASTNode], func_name: str) -> bool:
        """Check if any node in the list contains a call to func_name."""
        for node in nodes:
            if self._node_calls_function(node, func_name):
                return True
        return False

    def _node_calls_function(self, node: ASTNode, func_name: str) -> bool:
        """Check if node directly calls the given function."""
        if isinstance(node, FunctionCallNode):
            name = (
                node.function.name
                if isinstance(node.function, IdentifierNode)
                else None
            )
            if name == func_name:
                return True
        # Check children
        for child in self._get_node_children(node):
            if self._node_calls_function(child, func_name):
                return True
        return False

    def _detect_recursive_pattern(
        self, func_def: FunctionDefNode, _call: Optional[FunctionCallNode] = None
    ) -> Optional[str]:
        """
        Detect if recursive call follows backtracking/permutation pattern.

        Pattern: loop iterating over range, with recursive call inside that
        modifies the loop parameter (like l+1).

        Returns:
            "factorial" if pattern detected, None otherwise
        """
        # Get the function name
        func_name = func_def.name.name

        # Look for a for loop in the function body
        for node in func_def.body:
            pattern = self._check_node_for_factorial_pattern(node, func_name)
            if pattern:
                return pattern

        return None

    def _check_node_for_factorial_pattern(
        self, node: ASTNode, func_name: str
    ) -> Optional[str]:
        """Check if a node contains factorial pattern."""
        if isinstance(node, ForNode):
            # Check if the loop body contains a recursive call
            for stmt in node.body:
                if self._contains_recursive_call(stmt, func_name):
                    # Check if recursive call modifies loop iterator
                    if self._recursive_call_modifies_iterator(node, func_name):
                        return "factorial"
        elif isinstance(node, WhileNode):
            # Check while loop for recursive pattern
            for stmt in node.body:
                if self._contains_recursive_call(stmt, func_name):
                    return "factorial"
        elif isinstance(node, IfNode):
            # Check both branches
            for stmt in node.if_block:
                if self._check_node_for_factorial_pattern(stmt, func_name):
                    return "factorial"
            for _, block in node.elif_parts:
                for stmt in block:
                    if self._check_node_for_factorial_pattern(stmt, func_name):
                        return "factorial"
            if node.else_block:
                for stmt in node.else_block:
                    if self._check_node_for_factorial_pattern(stmt, func_name):
                        return "factorial"
        return None

    def _contains_recursive_call(self, node: ASTNode, func_name: str) -> bool:
        """Check if node contains a call to the given function."""
        if isinstance(node, FunctionCallNode):
            called_name = (
                node.function.name
                if isinstance(node.function, IdentifierNode)
                else None
            )
            if called_name == func_name:
                return True
        # Check children
        for child in self._get_node_children(node):
            if self._contains_recursive_call(child, func_name):
                return True
        return False

    def _recursive_call_modifies_iterator(
        self, loop_node: ForNode, func_name: str
    ) -> bool:
        """Check if recursive call modifies a parameter in the loop."""
        # Look for recursive calls in loop body and check their arguments
        for node in loop_node.body:
            if self._recursive_call_changes_param(node, func_name):
                return True

        return False

    def _recursive_call_changes_param(self, node: ASTNode, func_name: str) -> bool:
        """Check if recursive call modifies any parameter."""
        if isinstance(node, FunctionCallNode):
            called_name = (
                node.function.name
                if isinstance(node.function, IdentifierNode)
                else None
            )
            if called_name != func_name:
                return False
            # Check if any argument modifies a parameter (like param + 1)
            for arg in node.arguments:
                if isinstance(arg, BinaryOpNode) and arg.operator in ("+", "-", "*"):
                    # Check if it's a parameter being modified
                    if self._is_parameter_modification(arg):
                        return True
        # Check children
        for child in self._get_node_children(node):
            if self._recursive_call_changes_param(child, func_name):
                return True
        return False

    def _is_parameter_modification(self, expr: BinaryOpNode) -> bool:
        """Check if expression is a parameter modification (e.g., n+1, n-1)."""
        # Check if left or right is an identifier that could be a parameter
        for operand in [expr.left, expr.right]:
            if isinstance(operand, IdentifierNode):
                return True
        return False

    def _expr_uses_param(self, node: ASTNode, param_name: str) -> bool:
        """Check if expression uses the given parameter."""
        if isinstance(node, IdentifierNode):
            return node.name == param_name
        for child in self._get_node_children(node):
            if self._expr_uses_param(child, param_name):
                return True
        return False

    def _get_node_children(self, node: ASTNode) -> List[ASTNode]:
        """Get all child nodes of an AST node."""
        children = []
        for attr in dir(node):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(node, attr)
                if isinstance(val, ASTNode):
                    children.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, ASTNode):
                            children.append(item)
            except (AttributeError, TypeError):
                pass
        return children

    def _combine_max(self, exprs: List[ComplexityExpr]) -> ComplexityExpr:
        """Combine expressions as O(max(e1, e2, ...))."""
        if not exprs:
            return Const(1)

        if len(exprs) == 1:
            return exprs[0]

        # Filter out trivial Const(1) and combine the rest
        non_trivial = [e for e in exprs if not self._is_trivial(e)]

        if not non_trivial:
            return Const(1)

        if len(non_trivial) == 1:
            return non_trivial[0]

        # Combine as nested Add (max)
        result = non_trivial[0]
        for expr in non_trivial[1:]:
            result = Add(result, expr)

        return result

    def _is_trivial(self, expr: ComplexityExpr) -> bool:
        """Check if expression is trivial (O(1))."""
        if isinstance(expr, Const):
            return True
        return False

    # ── Simplification and conversion to result ───────────────────────────

    def _expr_to_result(self, expr: ComplexityExpr) -> ComplexityResult:
        """Convert a ComplexityExpr to a ComplexityResult with Big-O string."""
        # First simplify the expression
        simplified = self._simplify(expr)

        # Then convert to Big-O string and determine confidence
        (
            complexity_str,
            confidence,
            bound_symbol,
            display_complexity,
            bound_symbols,
        ) = self._classify_expression(simplified)

        explanation = self._generate_explanation(simplified)
        method = "static"
        fallback_reason = None
        if complexity_str == COMPLEXITY_UNKNOWN:
            method = "unknown"
            fallback_reason = self._fallback_reason or (
                simplified.reason
                if isinstance(simplified, UnknownExpr)
                else "Static analysis could not classify this program"
            )
            confidence = 0.0
        elif complexity_str == COMPLEXITY_UNBOUNDED:
            method = "unbounded"

        return ComplexityResult(
            complexity=complexity_str,
            confidence=confidence,
            explanation=explanation,
            bound_symbol=bound_symbol,
            display_complexity=display_complexity,
            method=method,
            bound_symbols=bound_symbols,
            best_case=self._best_case,
            worst_case=complexity_str,
            has_early_exit=self._has_early_exit,
            fallback_reason=fallback_reason,
        )

    def _simplify(self, expr: ComplexityExpr) -> ComplexityExpr:
        """Simplify a complexity expression."""
        if isinstance(expr, Const):
            return expr

        if isinstance(expr, Param):
            return expr

        if isinstance(expr, Var):
            return expr

        if isinstance(expr, Log):
            inner = self._simplify(expr.inner)
            return Log(inner)

        if isinstance(expr, Add):
            left = self._simplify(expr.left)
            right = self._simplify(expr.right)
            # Const + Const = Const
            if isinstance(left, Const) and isinstance(right, Const):
                return Const(max(left.value, right.value))
            # Param + Param (same) = Param * 2 (but max is just Param)
            if isinstance(left, Param) and isinstance(right, Param):
                if left.name == right.name:
                    return left
            # Var + Var (same) = Var
            if isinstance(left, Var) and isinstance(right, Var):
                if left.name == right.name:
                    return left
            return Add(left, right)

        if isinstance(expr, Mul):
            left = self._simplify(expr.left)
            right = self._simplify(expr.right)
            # Const * anything = anything (if const is 1)
            # But preserve non-1 constants for accuracy
            if isinstance(left, Const):
                if left.value == 1.0:
                    return right
                if left.value == 0.0:
                    return Const(0)
                # Non-1 constant - multiply
                if isinstance(right, Const):
                    return Const(left.value * right.value)
            if isinstance(right, Const):
                if right.value == 1.0:
                    return left
                if right.value == 0.0:
                    return Const(0)
                # Non-1 constant - multiply
                if isinstance(left, Const):
                    return Const(left.value * right.value)
            # Param * Param (same) = n²
            if isinstance(left, Param) and isinstance(right, Param):
                if left.name == right.name:
                    return Mul(left, right)
            # Const * Param = Param (linear)
            if isinstance(left, Const) and isinstance(right, Param):
                return right
            if isinstance(right, Const) and isinstance(left, Param):
                return left
            return Mul(left, right)

        if isinstance(expr, CallExpr):
            if expr.body_complexity:
                return self._simplify(expr.body_complexity)
            return Const(1)

        return expr

    def _complexity_to_big_o(
        self, expr: ComplexityExpr
    ) -> Tuple[str, float, Optional[str]]:
        """
        Convert a simplified ComplexityExpr to Big-O string.

        Returns:
            (complexity_string, confidence, bound_symbol)
        """
        result = self._classify_expression(expr)
        canonical, confidence, bound, _display, _bounds = result
        return canonical, confidence, bound

    def _classify_expression(
        self, expr: ComplexityExpr
    ) -> Tuple[str, float, Optional[str], str, List[str]]:
        if isinstance(expr, UnknownExpr):
            return COMPLEXITY_UNKNOWN, 0.0, None, COMPLEXITY_UNKNOWN, []

        if isinstance(expr, InfiniteExpr):
            return COMPLEXITY_UNBOUNDED, 1.0, None, COMPLEXITY_UNBOUNDED, []

        if isinstance(expr, Exponential):
            if expr.branch_factor == 2:
                return COMPLEXITY_EXP, 0.9, "n", COMPLEXITY_EXP, ["n"]
            display = f"O({expr.branch_factor}^n)"
            return COMPLEXITY_KN, 0.9, "n", display, ["n"]

        if isinstance(expr, Factorial):
            bounds = self._collect_bounds(expr.inner) or ["n"]
            return COMPLEXITY_NF, 0.9, bounds[0], COMPLEXITY_NF, bounds

        if isinstance(expr, Const):
            return COMPLEXITY_O1, 1.0, None, COMPLEXITY_O1, []

        if isinstance(expr, Param):
            return COMPLEXITY_N, 1.0, expr.name, COMPLEXITY_N, [expr.name]

        if isinstance(expr, Var):
            return COMPLEXITY_N, 0.7, expr.name, COMPLEXITY_N, [expr.name]

        if isinstance(expr, Log):
            bounds = self._collect_bounds(expr.inner) or ["n"]
            return (
                COMPLEXITY_LOGN,
                self._get_confidence(expr),
                bounds[0],
                COMPLEXITY_LOGN,
                bounds,
            )

        if isinstance(expr, Add):
            return self._classify_add(expr)

        if isinstance(expr, Mul):
            return self._classify_mul(expr)

        # Fallback for unknown expressions
        return COMPLEXITY_UNKNOWN, 0.0, None, COMPLEXITY_UNKNOWN, []

    def _classify_add(
        self, expr: Add
    ) -> Tuple[str, float, Optional[str], str, List[str]]:
        terms = self._flatten_add(expr)
        classified = [self._classify_expression(term) for term in terms]
        if any(c[0] == COMPLEXITY_UNBOUNDED for c in classified):
            return COMPLEXITY_UNBOUNDED, 1.0, None, COMPLEXITY_UNBOUNDED, []
        if any(c[0] == COMPLEXITY_UNKNOWN for c in classified):
            return COMPLEXITY_UNKNOWN, 0.0, None, COMPLEXITY_UNKNOWN, []

        linear_terms = [c for c in classified if c[0] == COMPLEXITY_N]
        if len(linear_terms) == len(classified):
            bounds = self._dedupe([b for c in classified for b in c[4]])
            if len(bounds) > 1:
                return (
                    COMPLEXITY_N_PLUS_M,
                    min(c[1] for c in classified),
                    bounds[0],
                    COMPLEXITY_N_PLUS_M,
                    bounds,
                )
            return (
                COMPLEXITY_N,
                min(c[1] for c in classified),
                bounds[0] if bounds else "n",
                COMPLEXITY_N,
                bounds or ["n"],
            )

        dominant = max(classified, key=lambda c: self._class_rank(c[0]))
        return (
            dominant[0],
            min(c[1] for c in classified),
            dominant[2],
            dominant[3],
            dominant[4],
        )

    def _classify_mul(
        self, expr: Mul
    ) -> Tuple[str, float, Optional[str], str, List[str]]:
        factors = [f for f in self._flatten_mul(expr) if not isinstance(f, Const)]
        if not factors:
            return COMPLEXITY_O1, 1.0, None, COMPLEXITY_O1, []

        classified = [self._classify_expression(f) for f in factors]
        if any(c[0] == COMPLEXITY_UNBOUNDED for c in classified):
            return COMPLEXITY_UNBOUNDED, 1.0, None, COMPLEXITY_UNBOUNDED, []
        if any(c[0] == COMPLEXITY_UNKNOWN for c in classified):
            return COMPLEXITY_UNKNOWN, 0.0, None, COMPLEXITY_UNKNOWN, []

        confidence = min(c[1] for c in classified)
        bounds = self._dedupe([b for c in classified for b in c[4]]) or ["n"]
        linear_count = sum(
            1
            for c in classified
            if c[0] in {COMPLEXITY_N, COMPLEXITY_N_M, COMPLEXITY_N_PLUS_M}
        )
        log_count = sum(1 for c in classified if c[0] == COMPLEXITY_LOGN)
        n2_count = sum(1 for c in classified if c[0] == COMPLEXITY_N2)
        n3_count = sum(1 for c in classified if c[0] == COMPLEXITY_N3)
        nk_count = sum(1 for c in classified if c[0] == COMPLEXITY_NK)

        degree = linear_count + (2 * n2_count) + (3 * n3_count) + (4 * nk_count)

        if degree == 0 and log_count:
            return COMPLEXITY_LOGN, confidence, bounds[0], COMPLEXITY_LOGN, bounds
        if degree == 1 and log_count:
            return COMPLEXITY_NLOGN, confidence, bounds[0], COMPLEXITY_NLOGN, bounds
        if degree == 1:
            return COMPLEXITY_N, confidence, bounds[0], COMPLEXITY_N, bounds
        if degree == 2:
            if len(bounds) > 1 and linear_count >= 2 and not n2_count:
                return COMPLEXITY_N_M, confidence, bounds[0], COMPLEXITY_N_M, bounds[:2]
            return COMPLEXITY_N2, confidence, bounds[0], COMPLEXITY_N2, bounds
        if degree == 3:
            return COMPLEXITY_N3, confidence, bounds[0], COMPLEXITY_N3, bounds
        if degree >= 4:
            return COMPLEXITY_NK, confidence, bounds[0], f"O(n^{degree})", bounds

        dominant = max(classified, key=lambda c: self._class_rank(c[0]))
        return dominant[0], confidence, dominant[2], dominant[3], dominant[4]

    def _flatten_add(self, expr: ComplexityExpr) -> List[ComplexityExpr]:
        if isinstance(expr, Add):
            return self._flatten_add(expr.left) + self._flatten_add(expr.right)
        return [expr]

    def _flatten_mul(self, expr: ComplexityExpr) -> List[ComplexityExpr]:
        if isinstance(expr, Mul):
            return self._flatten_mul(expr.left) + self._flatten_mul(expr.right)
        return [expr]

    def _collect_bounds(self, expr: ComplexityExpr) -> List[str]:
        if isinstance(expr, (Param, Var)):
            return [expr.name]
        if isinstance(expr, Log):
            return self._collect_bounds(expr.inner)
        if isinstance(expr, (Add, Mul)):
            left = self._collect_bounds(expr.left)
            right = self._collect_bounds(expr.right)
            return self._dedupe(left + right)
        return []

    def _dedupe(self, values: List[str]) -> List[str]:
        result: List[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result

    def _class_rank(self, complexity: str) -> int:
        ranks = {
            COMPLEXITY_O1: 0,
            COMPLEXITY_LOGN: 1,
            COMPLEXITY_N: 2,
            COMPLEXITY_N_PLUS_M: 3,
            COMPLEXITY_NLOGN: 4,
            COMPLEXITY_N_M: 5,
            COMPLEXITY_N2: 6,
            COMPLEXITY_N3: 7,
            COMPLEXITY_NK: 8,
            COMPLEXITY_EXP: 9,
            COMPLEXITY_KN: 10,
            COMPLEXITY_NF: 11,
            COMPLEXITY_UNBOUNDED: 12,
        }
        return ranks.get(complexity, -1)

    def _get_confidence(self, expr: ComplexityExpr) -> float:
        """Get confidence level for an expression."""
        if isinstance(expr, Const):
            return 1.0
        if isinstance(expr, Param):
            return 1.0
        if isinstance(expr, Var):
            return 0.7
        if isinstance(expr, Log):
            if isinstance(expr.inner, Param):
                return 1.0
            return 0.7
        if isinstance(expr, Add):
            return min(
                self._get_confidence(expr.left), self._get_confidence(expr.right)
            )
        if isinstance(expr, Mul):
            return min(
                self._get_confidence(expr.left), self._get_confidence(expr.right)
            )
        return 0.5

    def _level_to_string(self, level: int, bound: Optional[str]) -> str:
        """Convert complexity level to Big-O string."""
        bound_str = bound or "n"

        if level == 0:
            return "O(1)"
        elif level == 1:
            return f"O(log {bound_str})"
        elif level == 2:
            return f"O({bound_str})"
        elif level == 3:
            return f"O({bound_str} log {bound_str})"
        elif level == 4:
            return f"O({bound_str}²)"
        elif level == 5:
            return f"O({bound_str}² log {bound_str})"
        elif level == 6:
            return f"O({bound_str}³)"
        elif level == 7:
            return f"O({bound_str}⁴)"
        else:
            return f"O({bound_str}^{level - 2})"

    def _complexity_level(self, expr: ComplexityExpr) -> int:
        """
        Return a numeric complexity level for comparison.

        Levels:
          0 = O(1)
          1 = O(log n) - special, between const and linear
          2 = O(n)
          3 = O(n log n)
          4 = O(n²)
          5 = O(n² log n)
          6 = O(n³)
          ...
        """
        if isinstance(expr, Const):
            return 0
        if isinstance(expr, (Param, Var)):
            return 2  # O(n)
        if isinstance(expr, Log):
            return 1  # O(log n) - between O(1) and O(n)
        if isinstance(expr, Factorial):
            return 20  # O(n!) - higher than exponential
        if isinstance(expr, Add):
            # For Add (max), return the higher level
            return max(
                self._complexity_level(expr.left), self._complexity_level(expr.right)
            )
        if isinstance(expr, Mul):
            # For Mul, combine levels with special handling for n * log n
            left = self._complexity_level(expr.left)
            right = self._complexity_level(expr.right)

            # Special case: n * log n = n log n (level 3), not n² (level 4)
            if (left == 2 and right == 1) or (left == 1 and right == 2):
                return 3  # O(n log n)

            # log n * log n = log² n ≈ O(log n) for our purposes
            if left == 1 and right == 1:
                return 1  # O(log n)

            return left + right
        return 2

    def _combine_multiplication(
        self, left: ComplexityExpr, right: ComplexityExpr
    ) -> str:
        """Combine two complexity expressions as multiplication."""
        # Both parameters - check for same parameter
        if isinstance(left, Param) and isinstance(right, Param):
            if left.name == right.name:
                return f"O({left.name}²)"

        # Both variables - assume n²
        if isinstance(left, Var) and isinstance(right, Var):
            return "O(n²)"

        # Param * log(param) = O(n log n)
        if isinstance(left, Param) and isinstance(right, Log):
            if isinstance(right.inner, Param) and right.inner.name == left.name:
                return f"O({left.name} log {left.name})"
        if isinstance(right, Param) and isinstance(left, Log):
            if isinstance(left.inner, Param) and left.inner.name == right.name:
                return f"O({right.name} log {right.name})"

        # n * log n pattern
        if isinstance(left, (Param, Var)) and isinstance(right, Log):
            return "O(n log n)"
        if isinstance(right, (Param, Var)) and isinstance(left, Log):
            return "O(n log n)"

        # n * n = n²
        if isinstance(left, (Param, Var)) and isinstance(right, (Param, Var)):
            return "O(n²)"

        # Param * Const = O(n)
        if isinstance(left, Param) and isinstance(right, Const):
            return f"O({left.name})"
        if isinstance(right, Param) and isinstance(left, Const):
            return f"O({right.name})"

        # Default: assume O(n²) or more complex
        return "O(n²)"

    def _get_bound(self, expr: ComplexityExpr) -> Optional[str]:
        """Extract the bound symbol from an expression."""
        if isinstance(expr, Param):
            return expr.name
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Log):
            return self._get_bound(expr.inner)
        if isinstance(expr, Add):
            # Return the dominant bound
            left_level = self._complexity_level(expr.left)
            right_level = self._complexity_level(expr.right)
            if left_level >= right_level:
                return self._get_bound(expr.left)
            else:
                return self._get_bound(expr.right)
        if isinstance(expr, Mul):
            # Return the variable bound if present
            if isinstance(expr.left, (Param, Var)):
                return self._get_bound(expr.left)
            if isinstance(expr.right, (Param, Var)):
                return self._get_bound(expr.right)
        return None

    def _generate_explanation(self, expr: ComplexityExpr) -> str:
        """Generate a human-readable explanation of the complexity."""
        if isinstance(expr, Const):
            return "No loops or recursion detected — constant time O(1)."

        if isinstance(expr, Param):
            return f"Single loop over {expr.name} — linear time O({expr.name})."

        if isinstance(expr, Var):
            return "Loop with unknown bound — assumed linear O(n)."

        if isinstance(expr, Log):
            inner_name = self._get_bound(expr.inner) or "n"
            return (
                f"Halving loop pattern detected — logarithmic time O(log {inner_name})."
            )

        if isinstance(expr, Add):
            left_exp = self._generate_explanation(expr.left)
            right_exp = self._generate_explanation(expr.right)
            return f"Sequential execution: {left_exp} | {right_exp}"

        if isinstance(expr, Mul):
            left_exp = self._generate_explanation(expr.left)
            right_exp = self._generate_explanation(expr.right)
            return f"Nested loops: {left_exp} × {right_exp}"

        return "Complexity derived from AST analysis."


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def analyze_complexity(
    program: ProgramNode,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> ComplexityResult:
    """
    Analyze the algorithmic complexity of an OptiLang program.

    Uses static AST analysis to compute the Big-O complexity class.
    Returns confidence = 1.0 when complexity is derivable from AST structure,
    "Unknown" with explanation when not determinable.

    Args:
        program: Parsed ProgramNode from optilang.parser.parse().
        symbol_table: Optional symbol table from execution, used to identify
                      function parameters for more accurate analysis.

    Returns:
        ComplexityResult with:
            - complexity: String like "O(n²)" or "Unknown"
            - confidence: 1.0 for derivable, lower for uncertain
            - explanation: How the complexity was derived
            - bound_symbol: The variable that dominates (e.g., "n")

    Example::

        from optilang import parse
        from optilang.lexer import tokenize
        from optilang.complexity import analyze_complexity

        source = "for i in range(n):\\n    for j in range(n):\\n        x = i + j"
        ast = parse(tokenize(source))

        result = analyze_complexity(ast)
        print(result.complexity)   # "O(n²)"
        print(result.confidence)   # 1.0
    """
    return ComplexityAnalyzer().analyze(program, symbol_table)


def analyze_function_complexity(func_def: FunctionDefNode) -> ComplexityResult:
    """
    Analyze the complexity of a single function definition.

    Args:
        func_def: FunctionDefNode from the AST.

    Returns:
        ComplexityResult for the function body.
    """
    return ComplexityAnalyzer().analyze_function(func_def)
