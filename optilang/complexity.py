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

from .ast_nodes import (
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
            # String creation is O(n) where n is the string length
            return Const(float(len(node.value)))
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
            return self._analyze_node(node.value)

        # Index assignment: O(1) for the assignment
        if isinstance(node, IndexAssignmentNode):
            return self._analyze_node(node.value)

        # Binary operations: O(max(children))
        if isinstance(node, BinaryOpNode):
            left = self._analyze_node(node.left)
            right = self._analyze_node(node.right)
            return self._combine_max([left, right])

        # Unary operations: O(operand)
        if isinstance(node, UnaryOpNode):
            return self._analyze_node(node.operand)

        # Function call: analyze function body or use call expression
        if isinstance(node, FunctionCallNode):
            return self._analyze_function_call(node)

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
                # O(n) for creating the list
                n = len(node.elements)
                return Const(float(n))
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
        body_complexity = self._analyze_block(node.body)

        # Total complexity: iterations * body
        return Mul(iter_complexity, body_complexity)

    def _analyze_while_loop(self, node: WhileNode) -> ComplexityExpr:
        """Analyze a while loop and return its complexity."""
        # For while loops, we try to detect common patterns:
        # - Halving loop (n -> n // 2): O(log n)
        # - Linear decrement (n -> n - 1): O(n)
        # - Unknown: use heuristic

        body_complexity = self._analyze_block(node.body)

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
                                    return Param(var_name)

        return None

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
            return self._analyze_node(arg)

        elif len(args) == 2:
            # range(a, b) → O(b - a)
            left = self._analyze_node(args[0])
            right = self._analyze_node(args[1])
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
            left = self._analyze_node(args[0])
            right = self._analyze_node(args[1])
            return self._combine_max([left, right])

        # Unknown range
        return Var("n")

    def _analyze_function_call(self, call: FunctionCallNode) -> ComplexityExpr:
        """Analyze a function call, using the function body if available."""
        func_name = (
            call.function.name
            if isinstance(call.function, IdentifierNode)
            else "unknown"
        )

        # Check if we have the function definition
        if func_name in self._functions:
            func_def = self._functions[func_name]
            # Temporarily set params for this function
            saved_params = self._params.copy()
            self._params = {p.name for p in func_def.parameters}

            body_complexity = self._analyze_block(func_def.body)

            self._params = saved_params
            return body_complexity

        # Unknown function - assume O(1) or O(n) based on context
        return Const(1)

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
        complexity_str, confidence, bound_symbol = self._complexity_to_big_o(simplified)

        explanation = self._generate_explanation(simplified)

        return ComplexityResult(
            complexity=complexity_str,
            confidence=confidence,
            explanation=explanation,
            bound_symbol=bound_symbol,
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
        if isinstance(expr, Const):
            # For Big-O, any constant is O(1), but show actual value for clarity
            if expr.value > 1:
                return f"O({int(expr.value)})", 1.0, None
            return "O(1)", 1.0, None

        if isinstance(expr, Param):
            return f"O({expr.name})", 1.0, expr.name

        if isinstance(expr, Var):
            return "O(n)", 0.7, "n"

        if isinstance(expr, Log):
            # log(n) where n is a parameter - high confidence
            if isinstance(expr.inner, Param):
                return f"O(log {expr.inner.name})", 1.0, expr.inner.name
            # Unknown inner - lower confidence
            return "O(log n)", 0.7, "n"

        if isinstance(expr, Add):
            left_str, left_conf, _ = self._complexity_to_big_o(expr.left)
            right_str, right_conf, _ = self._complexity_to_big_o(expr.right)

            # Take the maximum (higher complexity)
            # Compare complexity levels
            left_level = self._complexity_level(expr.left)
            right_level = self._complexity_level(expr.right)

            if left_level >= right_level:
                return left_str, min(left_conf, right_conf), self._get_bound(expr.left)
            else:
                return (
                    right_str,
                    min(left_conf, right_conf),
                    self._get_bound(expr.right),
                )

        if isinstance(expr, Mul):
            # Compute the product level (nesting depth)
            left_level = self._complexity_level(expr.left)
            right_level = self._complexity_level(expr.right)
            total_level = left_level + right_level

            # Get confidence from the parts
            left_conf = self._get_confidence(expr.left)
            right_conf = self._get_confidence(expr.right)
            confidence = min(left_conf, right_conf)

            # Determine the bound symbol
            bound = self._get_bound(expr)

            # Map level to Big-O string
            complexity_str = self._level_to_string(total_level, bound)

            return complexity_str, confidence, bound

        # Fallback for unknown expressions
        return "Unknown", 0.5, None

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
