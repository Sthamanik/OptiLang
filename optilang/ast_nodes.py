"""
AST Node Definitions for OptiLang Parser
Each node represents a syntactic construct in the language
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Base AST Node
@dataclass
class ASTNode:
    """Base class for all AST nodes"""

    line: int
    column: int


# Literals
@dataclass
class NumberNode(ASTNode):
    """Represents a numeric literal"""

    value: float


@dataclass
class StringNode(ASTNode):
    """Represents a string literal"""

    value: str


@dataclass
class BooleanNode(ASTNode):
    """Represents True/False literals"""

    value: bool


@dataclass
class NullNode(ASTNode):
    """Represents a null or None literal"""

    pass


# Identifiers
@dataclass
class IdentifierNode(ASTNode):
    """Node representing an identifier"""

    name: str


# Expressions
# Binary Operations
@dataclass
class BinaryOpNode(ASTNode):
    """Represents a binary operations:
    +, -, *, /, %, **, //, ==, !=, <, >, <=, >=, and, or
    """

    left: ASTNode
    operator: str
    right: ASTNode


# Unary Operations
@dataclass
class UnaryOpNode(ASTNode):
    """Represents unary operations: -, not"""

    operator: str
    operand: ASTNode


# Assignments
@dataclass
class AssignmentNode(ASTNode):
    """Represents variable assignment: x = 5"""

    target: IdentifierNode
    value: ASTNode


# Augmented Assignments
@dataclass
class AugmentedAssignmentNode(ASTNode):
    """Represents augmented assignment: x += 5"""

    target: IdentifierNode
    operator: str  # +=, -=, *=, etc.
    value: ASTNode


# Control FLow
@dataclass
class IfNode(ASTNode):
    """Represents an if statement"""

    condition: ASTNode
    if_block: List[ASTNode] = field(default_factory=list)
    elif_parts: List[Tuple[ASTNode, List[ASTNode]]] = field(
        default_factory=list
    )  # List of (condition, block)
    else_block: Optional[List[ASTNode]] = None


@dataclass
class WhileNode(ASTNode):
    """Represents a while loop"""

    condition: ASTNode
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class ForNode(ASTNode):
    """Represents a for loop"""

    iterator: IdentifierNode
    iterable: ASTNode
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class BreakNode(ASTNode):
    """Represents a break statement"""

    pass


@dataclass
class ContinueNode(ASTNode):
    """Represents continue statement"""

    pass


@dataclass
class PassNode(ASTNode):
    """Represents pass statement"""

    pass


# Functions
@dataclass
class FunctionDefNode(ASTNode):
    """Represents a function definition"""

    name: IdentifierNode
    parameters: List[IdentifierNode] = field(default_factory=list)
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class FunctionCallNode(ASTNode):
    """Represents a function call"""

    function: IdentifierNode
    arguments: List[ASTNode] = field(default_factory=list)


@dataclass
class ReturnNode(ASTNode):
    """Represents a return statement"""

    value: Optional[ASTNode] = None


# Data Structures
@dataclass
class ListNode(ASTNode):
    """Represents a list literal"""

    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class DictNode(ASTNode):
    """Represents a dictionary literal"""

    pairs: List[Tuple[ASTNode, ASTNode]] = field(default_factory=list)


@dataclass
class IndexNode(ASTNode):
    """Represents indexing into a list[0] or dictionary[key]"""

    collection: ASTNode
    index: ASTNode


# Exception Handling
@dataclass
class TryNode(ASTNode):
    """Represents try-except block"""

    try_block: List[ASTNode] = field(default_factory=list)
    except_block: Optional[List[ASTNode]] = None
    finally_block: Optional[List[ASTNode]] = None


# Program
@dataclass
class ProgramNode(ASTNode):
    """Root node representing entire program"""

    statements: List[ASTNode] = field(default_factory=list)
