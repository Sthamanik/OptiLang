"""
Semantic Analysis phase for OptiLang.

Algorithm: Visitor Pattern + Scope Stack (single DFS pass over the AST)

Checks performed (all zero false-positive risk):
    1. 'return' outside a function definition
    2. 'break' outside a loop (for / while)
    3. 'continue' outside a loop (for / while)
    4. Duplicate parameter names in a function definition
       e.g.  def f(x, x):  →  SemanticError

Design decisions:
    - Unused variable detection and dead code detection are intentionally
      NOT implemented here. They are optimization suggestions (not errors)
      and belong in the Optimizer (unused_vars.py, dead_code.py) which runs
      after execution with full profiling data available.
    - These purely structural checks have zero false-positive risk because
      they rely only on the shape of the AST, never on runtime values.

Algorithm details:
    Visitor Pattern:
        _visit() dispatches each AST node to a dedicated _visit_<NodeType>()
        method by name. Unknown nodes fall back to _generic_visit() (no-op).
        This means adding new AST nodes never breaks the analyzer.

    Scope Stack:
        Two integer counters act as implicit depth stacks:
          _function_depth  increments on FunctionDefNode entry, decrements on exit
          _loop_depth      increments on ForNode/WhileNode entry, decrements on exit

        IMPORTANT: _loop_depth is reset to 0 when entering a function body.
        This prevents a function defined inside a loop from inheriting the
        outer loop's depth, which would cause break/continue inside the
        function to be wrongly accepted.

        Example of what the reset prevents:
            for i in range(10):   # loop_depth = 1
                def f():
                    break         # must ERROR — f has no enclosing loop
                                  # without reset, loop_depth=1 would pass it
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..core.ast_nodes import (
    ASTNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    BreakNode,
    ContinueNode,
    DictNode,
    ForNode,
    FunctionCallNode,
    FunctionDefNode,
    IfNode,
    IndexNode,
    ListNode,
    ProgramNode,
    ReturnNode,
    TryNode,
    UnaryOpNode,
    WhileNode,
)
from ..utils.errors import SemanticError


# Internal error record (collected before raising)
@dataclass
class _SemanticIssue:
    """A single semantic violation found during the AST walk."""

    message: str
    line: Optional[int] = None


# Main analyzer


class SemanticAnalyzer:
    """
    Single-pass AST visitor that enforces structural semantic rules.

    Attributes:
        _function_depth: How many nested function definitions we are
                         currently inside. > 0 means a return is legal.
        _loop_depth:     How many nested loops (for / while) we are
                         currently inside. > 0 means break/continue is legal.
                         Reset to 0 on entering a function body so that
                         break/continue inside a function do not inherit
                         a loop depth from an enclosing outer loop.
        _issues:         All violations collected during the walk.
                         Only the first one is raised at the end of analyze().
    """

    def __init__(self) -> None:
        self._function_depth: int = 0
        self._loop_depth: int = 0
        self._issues: List[_SemanticIssue] = []

    # Public API

    def analyze(self, program: ProgramNode) -> None:
        """
        Walk the entire AST and raise SemanticError on the first violation.

        Args:
            program: Root ProgramNode returned by the parser.

        Raises:
            SemanticError: On the first structural violation found.
        """
        self._visit(program)

        if self._issues:
            first = self._issues[0]
            raise SemanticError(first.message, first.line)

    # Dispatcher

    def _visit(self, node: ASTNode) -> None:
        """
        Dispatch to the correct visit method based on node type.

        Falls back to _generic_visit for nodes that need no checking.
        """
        method_name = f"_visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self._generic_visit)
        visitor(node)

    def _visit_children(self, nodes: List[ASTNode]) -> None:
        """Visit a list of AST nodes in order."""
        for node in nodes:
            self._visit(node)

    # Entry point

    def _visit_ProgramNode(self, node: ProgramNode) -> None:
        """Entry point — visit all top-level statements."""
        self._visit_children(node.statements)

    # Function definition

    def _visit_FunctionDefNode(self, node: FunctionDefNode) -> None:
        """
        Check for duplicate parameter names, then walk the function body
        with function_depth incremented so return statements inside are legal.

        Also resets _loop_depth to 0 before entering the function body so
        that break/continue inside this function cannot inherit a loop depth
        from any enclosing outer loop at the call site.

        Example violations:
            def greet(name, name):   # duplicate parameter 'name'
                return name

            for i in range(10):
                def f():
                    break            # no loop inside f — must ERROR
        """
        # Check duplicate parameters
        seen_params: set = set()
        for param in node.parameters:
            if param.name in seen_params:
                self._issues.append(
                    _SemanticIssue(
                        message=(
                            f"Duplicate parameter name '{param.name}' "
                            f"in function '{node.name.name}'"
                        ),
                        line=param.line,
                    )
                )
            seen_params.add(param.name)

        # Walk body with isolated scope
        self._function_depth += 1
        saved_loop_depth = self._loop_depth  # save outer loop context
        self._loop_depth = 0  # function body starts with no enclosing loop
        self._visit_children(node.body)
        self._loop_depth = saved_loop_depth  # restore outer loop context
        self._function_depth -= 1

    # Return

    def _visit_ReturnNode(self, node: ReturnNode) -> None:
        """
        Ensure 'return' is used inside a function body.

        Example violation:
            x = 5
            return x    # return at top level — function_depth is 0
        """
        if self._function_depth == 0:
            self._issues.append(
                _SemanticIssue(
                    message="'return' outside function",
                    line=node.line,
                )
            )
        # Visit return value expression even on violation — collect all issues
        if node.value is not None:
            self._visit(node.value)

    # Break

    def _visit_BreakNode(self, node: BreakNode) -> None:
        """
        Ensure 'break' is used inside a loop body.

        Example violations:
            break                  # top level

            def f():
                break              # inside function but no loop
        """
        if self._loop_depth == 0:
            self._issues.append(
                _SemanticIssue(
                    message="'break' outside loop",
                    line=node.line,
                )
            )

    # Continue

    def _visit_ContinueNode(self, node: ContinueNode) -> None:
        """
        Ensure 'continue' is used inside a loop body.

        Example violations:
            continue               # top level

            def f():
                continue           # inside function but no loop
        """
        if self._loop_depth == 0:
            self._issues.append(
                _SemanticIssue(
                    message="'continue' outside loop",
                    line=node.line,
                )
            )

    # Loops

    def _visit_ForNode(self, node: ForNode) -> None:
        """
        Increment loop depth and walk the for-loop body.

        Visits iterable and iterator before incrementing depth — they are
        expressions evaluated outside the loop body context.

        The iterator variable is visited for completeness so that when
        unused-variable detection is added to the optimizer later, the
        for-loop variable is already wired into the AST walk.
        """
        self._visit(node.iterable)  # evaluated outside loop context
        self._visit(node.iterator)  # IdentifierNode — no checks now, future-proof

        self._loop_depth += 1
        self._visit_children(node.body)
        self._loop_depth -= 1

    def _visit_WhileNode(self, node: WhileNode) -> None:
        """
        Increment loop depth and walk the while-loop body.

        Visits condition before incrementing depth — it is evaluated
        outside the loop body context.
        """
        self._visit(node.condition)  # evaluated outside loop context

        self._loop_depth += 1
        self._visit_children(node.body)
        self._loop_depth -= 1

    # Control flow

    def _visit_IfNode(self, node: IfNode) -> None:
        """Visit condition, if-block, all elif parts, and else block."""
        self._visit(node.condition)
        self._visit_children(node.if_block)

        for elif_condition, elif_block in node.elif_parts:
            self._visit(elif_condition)
            self._visit_children(elif_block)

        if node.else_block:
            self._visit_children(node.else_block)

    # Exception handling

    def _visit_TryNode(self, node: TryNode) -> None:
        """Visit try block, except block, and finally block."""
        self._visit_children(node.try_block)

        if node.except_block:
            self._visit_children(node.except_block)

        if node.finally_block:
            self._visit_children(node.finally_block)

    # Assignments

    def _visit_AssignmentNode(self, node: AssignmentNode) -> None:
        """Visit the right-hand side value expression."""
        self._visit(node.value)

    def _visit_AugmentedAssignmentNode(self, node: AugmentedAssignmentNode) -> None:
        """Visit the right-hand side value expression."""
        self._visit(node.value)

    # Expressions

    def _visit_BinaryOpNode(self, node: BinaryOpNode) -> None:
        """Visit both operands."""
        self._visit(node.left)
        self._visit(node.right)

    def _visit_UnaryOpNode(self, node: UnaryOpNode) -> None:
        """Visit the operand."""
        self._visit(node.operand)

    def _visit_FunctionCallNode(self, node: FunctionCallNode) -> None:
        """Visit all argument expressions."""
        self._visit_children(node.arguments)

    def _visit_IndexNode(self, node: IndexNode) -> None:
        """Visit both the collection and the index expressions."""
        self._visit(node.collection)
        if node.index is not None:
            self._visit(node.index)

    def _visit_ListNode(self, node: ListNode) -> None:
        """Visit all element expressions."""
        self._visit_children(node.elements)

    def _visit_DictNode(self, node: DictNode) -> None:
        """Visit all key and value expressions."""
        for key_node, value_node in node.pairs:
            self._visit(key_node)
            self._visit(value_node)

    # Fallback

    def _generic_visit(self, node: ASTNode) -> None:
        """
        No-op fallback for leaf nodes that need no semantic checking.

        Handles: NumberNode, StringNode, BooleanNode, NullNode,
                 IdentifierNode, PassNode, and any future nodes added
                 to ast_nodes.py that require no semantic checking.
        """
