"""
OptiLang Parser - Converts tokens to Abstract Syntax Tree (AST)

This parser implements a recursive descent parser for the PyLite language.
It builds an AST from the token stream produced by the lexer.

Grammar (Simplified):
    program         → statements EOF
    statements      → statement*
    statement       → assignment | aug_assignment | if_stmt | while_stmt | for_stmt |
                      function_def | return_stmt | break_stmt | continue_stmt |
                      pass_stmt | try_stmt | expression_stmt

    assignment      → IDENTIFIER ASSIGN expression
    aug_assignment  → IDENTIFIER (PLUS_ASSIGN | MINUS_ASSIGN | MULTIPLY_ASSIGN |
                      DIVIDE_ASSIGN) expression

    if_stmt         → IF expression COLON block
                      (ELIF expression COLON block)* (ELSE COLON block)?
    while_stmt      → WHILE expression COLON block
    for_stmt        → FOR IDENTIFIER IN expression COLON block

    function_def    → DEF IDENTIFIER LPAREN params? RPAREN COLON block
    params          → IDENTIFIER (COMMA IDENTIFIER)*

    return_stmt     → RETURN expression?
    break_stmt      → BREAK
    continue_stmt   → CONTINUE
    pass_stmt       → PASS

    try_stmt        → TRY COLON block EXCEPT COLON block (FINALLY COLON block)?

    block           → INDENT statements DEDENT | statement
    expression_stmt → expression

    # Expression hierarchy (lowest to highest precedence)
    expression      → logical_or
    logical_or      → logical_and (OR logical_and)*
    logical_and     → equality (AND equality)*
    equality        → comparison ((EQ | NE) comparison)*
    comparison      → term ((LT | LE | GT | GE) term)*
    term            → factor ((PLUS | MINUS) factor)*
    factor          → unary ((MULTIPLY | DIVIDE | MODULO | FLOOR_DIVIDE) unary)*
    unary           → (MINUS | NOT) unary | power
    power           → primary (POWER unary)?
    primary         → NUMBER | STRING | TRUE | FALSE | NONE |
                      IDENTIFIER | function_call | list_literal | dict_literal |
                      index_access | LPAREN expression RPAREN

    function_call   → IDENTIFIER LPAREN arguments? RPAREN
    arguments       → expression (COMMA expression)*

    list_literal    → LBRACKET (expression (COMMA expression)*)? RBRACKET
    dict_literal    → LBRACE (dict_pair (COMMA dict_pair)*)? RBRACE
    dict_pair       → expression COLON expression

    index_access    → primary LBRACKET expression RBRACKET
"""

from typing import List, Optional
from .token import Token, TokenType
from .ast_nodes import (
    ASTNode,
    ProgramNode,
    NumberNode,
    StringNode,
    BooleanNode,
    NullNode,
    IdentifierNode,
    BinaryOpNode,
    UnaryOpNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    IfNode,
    WhileNode,
    ForNode,
    BreakNode,
    ContinueNode,
    PassNode,
    FunctionDefNode,
    FunctionCallNode,
    ReturnNode,
    ListNode,
    DictNode,
    IndexNode,
    TryNode,
)
from .utils.errors import ParserError


class Parser:
    """
    Recursive Descent Parser for PyLite

    The parser maintains a current position in the token stream and uses
    lookahead to make parsing decisions. It builds an AST by recursively
    calling parsing methods that correspond to grammar rules.
    """

    def __init__(self, tokens: List[Token]):
        """
        Initialize parser with token stream

        Args:
            tokens: List of tokens from the lexer
        """
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None

    # ===== Token Management =====

    def peek(self, offset: int = 0) -> Optional[Token]:
        """
        Look ahead at a token without consuming it

        Args:
            offset: How many tokens ahead to look (0 = current)

        Returns:
            Token at position or None if out of bounds
        """
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def advance(self) -> Token:
        """
        Consume current token and move to next

        Returns:
            The consumed token

        Raises:
            ParserError: If trying to advance past EOF
        """
        if self.current_token is None:
            raise ParserError("Cannot advance past end of file")

        old_token = self.current_token
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
        return old_token

    def expect(self, token_type: str) -> Token:
        """
        Consume token if it matches expected type, otherwise raise error

        Args:
            token_type: Expected token type

        Returns:
            The consumed token

        Raises:
            ParserError: If token doesn't match expected type
        """
        if self.current_token is None:
            raise ParserError(f"Expected {token_type} but reached end of file")

        if self.current_token.type != token_type:
            raise ParserError(
                f"Expected {token_type} but got {self.current_token.type}",
                self.current_token,
            )

        return self.advance()

    def match(self, *token_types: str) -> bool:
        """
        Check if current token matches any of the given types

        Args:
            token_types: Token types to check against

        Returns:
            True if current token matches any type
        """
        if self.current_token is None:
            return False
        return self.current_token.type in token_types

    def skip_newlines(self) -> None:
        """Skip any newline tokens"""
        while self.match(TokenType.NEWLINE):
            self.advance()

    # ===== Main Parser Entry Point =====

    def parse(self) -> ProgramNode:
        """
        Parse the entire program

        Returns:
            Root AST node (ProgramNode)

        Raises:
            ParserError: If parsing fails
        """
        try:
            statements = self.parse_statements()
            self.expect(TokenType.EOF)

            # Create program node with line 1, col 1
            return ProgramNode(line=1, column=1, statements=statements)
        except ParserError:
            raise
        except Exception as e:
            raise ParserError(f"Unexpected error during parsing: {str(e)}")

    # ===== Statement Parsing =====

    def parse_statements(self) -> List[ASTNode]:
        """
        Parse a sequence of statements

        Returns:
            List of statement AST nodes
        """
        statements = []

        # Skip leading newlines
        self.skip_newlines()

        while not self.match(TokenType.EOF, TokenType.DEDENT):
            # Skip empty lines
            if self.match(TokenType.NEWLINE):
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

            # Expect newline or EOF after statement
            if not self.match(TokenType.EOF, TokenType.DEDENT):
                if self.match(TokenType.NEWLINE):
                    self.advance()
                    self.skip_newlines()

        return statements

    def parse_statement(self) -> Optional[ASTNode]:
        """
        Parse a single statement

        Returns:
            Statement AST node or None
        """
        # Skip newlines
        self.skip_newlines()

        # Control flow statements
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_statement()
        elif self.match(TokenType.FOR):
            return self.parse_for_statement()

        # Function definition
        elif self.match(TokenType.DEF):
            return self.parse_function_def()

        # Jump statements
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        elif self.match(TokenType.BREAK):
            return self.parse_break_statement()
        elif self.match(TokenType.CONTINUE):
            return self.parse_continue_statement()
        elif self.match(TokenType.PASS):
            return self.parse_pass_statement()

        # Exception handling
        elif self.match(TokenType.TRY):
            return self.parse_try_statement()

        # Assignment or expression
        else:
            # Check for assignment (including augmented)
            if self.match(TokenType.IDENTIFIER):
                # Look ahead to check for assignment
                next_token = self.peek(1)
                if next_token and next_token.type == TokenType.ASSIGN:
                    return self.parse_assignment()
                elif next_token and next_token.type in (
                    TokenType.PLUS_ASSIGN,
                    TokenType.MINUS_ASSIGN,
                    TokenType.MULTIPLY_ASSIGN,
                    TokenType.DIVIDE_ASSIGN,
                ):
                    return self.parse_augmented_assignment()

            # Otherwise, it's an expression statement
            return self.parse_expression()

    def parse_block(self) -> List[ASTNode]:
        """
        Parse a block of statements (indented)

        Returns:
            List of statement nodes
        """
        statements = []

        # Skip newlines before block
        self.skip_newlines()

        # Block must start with INDENT
        self.expect(TokenType.INDENT)

        # Parse statements in block
        statements = self.parse_statements()

        # Block must end with DEDENT
        self.expect(TokenType.DEDENT)

        return statements

    # ===== Assignment Statements =====

    def parse_assignment(self) -> AssignmentNode:
        """
        Parse variable assignment: x = expression

        Returns:
            AssignmentNode
        """
        # Get identifier
        id_token = self.expect(TokenType.IDENTIFIER)
        target = IdentifierNode(
            line=id_token.line, column=id_token.column, name=id_token.value
        )

        # Consume '='
        self.expect(TokenType.ASSIGN)

        # Parse value expression
        value = self.parse_expression()

        return AssignmentNode(
            line=id_token.line, column=id_token.column, target=target, value=value
        )

    def parse_augmented_assignment(self) -> AugmentedAssignmentNode:
        """
        Parse augmented assignment: x += expression

        Returns:
            AugmentedAssignmentNode
        """
        # Get identifier
        id_token = self.expect(TokenType.IDENTIFIER)
        target = IdentifierNode(
            line=id_token.line, column=id_token.column, name=id_token.value
        )

        # Get operator (+=, -=, etc.)
        if self.current_token is None:
            raise ParserError(
                "Expected augmented assignment operator but reached end of file"
            )

        if not self.match(
            TokenType.PLUS_ASSIGN,
            TokenType.MINUS_ASSIGN,
            TokenType.MULTIPLY_ASSIGN,
            TokenType.DIVIDE_ASSIGN,
        ):
            raise ParserError(
                "Expected augmented assignment operator", self.current_token
            )

        op_token = self.current_token
        self.advance()

        # Parse value expression
        value = self.parse_expression()

        return AugmentedAssignmentNode(
            line=id_token.line,
            column=id_token.column,
            target=target,
            operator=op_token.value,
            value=value,
        )

    # ===== Control Flow Statements =====

    def parse_if_statement(self) -> IfNode:
        """
        Parse if statement with optional elif and else

        Returns:
            IfNode
        """
        if_token = self.expect(TokenType.IF)

        # Parse condition
        condition = self.parse_expression()

        # Expect colon
        self.expect(TokenType.COLON)

        # Parse if block
        if_block = self.parse_block()

        # Parse elif parts
        elif_parts = []
        while self.match(TokenType.ELIF):
            self.advance()
            elif_condition = self.parse_expression()
            self.expect(TokenType.COLON)
            elif_block = self.parse_block()
            elif_parts.append((elif_condition, elif_block))

        # Parse else part
        else_block = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.expect(TokenType.COLON)
            else_block = self.parse_block()

        return IfNode(
            line=if_token.line,
            column=if_token.column,
            condition=condition,
            if_block=if_block,
            elif_parts=elif_parts,
            else_block=else_block,
        )

    def parse_while_statement(self) -> WhileNode:
        """
        Parse while loop

        Returns:
            WhileNode
        """
        while_token = self.expect(TokenType.WHILE)

        # Parse condition
        condition = self.parse_expression()

        # Expect colon
        self.expect(TokenType.COLON)

        # Parse body
        body = self.parse_block()

        return WhileNode(
            line=while_token.line,
            column=while_token.column,
            condition=condition,
            body=body,
        )

    def parse_for_statement(self) -> ForNode:
        """
        Parse for loop: for i in iterable:

        Returns:
            ForNode
        """
        for_token = self.expect(TokenType.FOR)

        # Parse iterator variable
        iter_token = self.expect(TokenType.IDENTIFIER)
        iterator = IdentifierNode(
            line=iter_token.line, column=iter_token.column, name=iter_token.value
        )

        # Expect 'in'
        self.expect(TokenType.IN)

        # Parse iterable expression
        iterable = self.parse_expression()

        # Expect colon
        self.expect(TokenType.COLON)

        # Parse body
        body = self.parse_block()

        return ForNode(
            line=for_token.line,
            column=for_token.column,
            iterator=iterator,
            iterable=iterable,
            body=body,
        )

    # ===== Function Statements =====

    def parse_function_def(self) -> FunctionDefNode:
        """
        Parse function definition

        Returns:
            FunctionDefNode
        """
        def_token = self.expect(TokenType.DEF)

        # Parse function name
        name_token = self.expect(TokenType.IDENTIFIER)
        name = IdentifierNode(
            line=name_token.line, column=name_token.column, name=name_token.value
        )

        # Parse parameters
        self.expect(TokenType.LPAREN)

        parameters = []
        if not self.match(TokenType.RPAREN):
            # Parse parameter list
            param_token = self.expect(TokenType.IDENTIFIER)
            parameters.append(
                IdentifierNode(
                    line=param_token.line,
                    column=param_token.column,
                    name=param_token.value,
                )
            )

            while self.match(TokenType.COMMA):
                self.advance()
                param_token = self.expect(TokenType.IDENTIFIER)
                parameters.append(
                    IdentifierNode(
                        line=param_token.line,
                        column=param_token.column,
                        name=param_token.value,
                    )
                )

        self.expect(TokenType.RPAREN)

        # Expect colon
        self.expect(TokenType.COLON)

        # Parse body
        body = self.parse_block()

        return FunctionDefNode(
            line=def_token.line,
            column=def_token.column,
            name=name,
            parameters=parameters,
            body=body,
        )

    def parse_return_statement(self) -> ReturnNode:
        """
        Parse return statement

        Returns:
            ReturnNode
        """
        return_token = self.expect(TokenType.RETURN)

        # Check if there's a return value
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            value = self.parse_expression()

        return ReturnNode(
            line=return_token.line, column=return_token.column, value=value
        )

    # ===== Jump Statements =====

    def parse_break_statement(self) -> BreakNode:
        """Parse break statement"""
        break_token = self.expect(TokenType.BREAK)
        return BreakNode(line=break_token.line, column=break_token.column)

    def parse_continue_statement(self) -> ContinueNode:
        """Parse continue statement"""
        continue_token = self.expect(TokenType.CONTINUE)
        return ContinueNode(line=continue_token.line, column=continue_token.column)

    def parse_pass_statement(self) -> PassNode:
        """Parse pass statement"""
        pass_token = self.expect(TokenType.PASS)
        return PassNode(line=pass_token.line, column=pass_token.column)

    # ===== Exception Handling =====

    def parse_try_statement(self) -> TryNode:
        """
        Parse try-except-finally statement

        Returns:
            TryNode
        """
        try_token = self.expect(TokenType.TRY)
        self.expect(TokenType.COLON)

        # Parse try block
        try_block = self.parse_block()

        # Parse except block (required)
        except_block = None
        if self.match(TokenType.EXCEPT):
            self.advance()
            self.expect(TokenType.COLON)
            except_block = self.parse_block()

        # Parse finally block (optional)
        finally_block = None
        if self.match(TokenType.FINALLY):
            self.advance()
            self.expect(TokenType.COLON)
            finally_block = self.parse_block()

        return TryNode(
            line=try_token.line,
            column=try_token.column,
            try_block=try_block,
            except_block=except_block,
            finally_block=finally_block,
        )

    # ===== Expression Parsing =====

    def parse_expression(self) -> ASTNode:
        """Parse expression (entry point for expression hierarchy)"""
        return self.parse_logical_or()

    def parse_logical_or(self) -> ASTNode:
        """Parse logical OR expression"""
        left = self.parse_logical_and()

        while self.match(TokenType.OR):
            op_token = self.advance()
            right = self.parse_logical_and()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator="or",
                right=right,
            )

        return left

    def parse_logical_and(self) -> ASTNode:
        """Parse logical AND expression"""
        left = self.parse_equality()

        while self.match(TokenType.AND):
            op_token = self.advance()
            right = self.parse_equality()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator="and",
                right=right,
            )

        return left

    def parse_equality(self) -> ASTNode:
        """Parse equality comparison (==, !=)"""
        left = self.parse_comparison()

        while self.match(TokenType.EQ, TokenType.NE):
            op_token = self.advance()
            right = self.parse_comparison()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator=op_token.value,
                right=right,
            )

        return left

    def parse_comparison(self) -> ASTNode:
        """Parse comparison (<, <=, >, >=)"""
        left = self.parse_term()

        while self.match(TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            op_token = self.advance()
            right = self.parse_term()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator=op_token.value,
                right=right,
            )

        return left

    def parse_term(self) -> ASTNode:
        """Parse addition and subtraction"""
        left = self.parse_factor()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.advance()
            right = self.parse_factor()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator=op_token.value,
                right=right,
            )

        return left

    def parse_factor(self) -> ASTNode:
        """Parse multiplication, division, modulo"""
        left = self.parse_unary()

        while self.match(
            TokenType.MULTIPLY,
            TokenType.DIVIDE,
            TokenType.MODULO,
            TokenType.FLOOR_DIVIDE,
        ):
            op_token = self.advance()
            right = self.parse_unary()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator=op_token.value,
                right=right,
            )

        return left

    def parse_unary(self) -> ASTNode:
        """Parse unary operations (-, not)"""
        if self.match(TokenType.MINUS, TokenType.NOT):
            op_token = self.current_token
            if op_token is None:
                raise ParserError("Expected unary operator but got None")

            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(
                line=op_token.line,
                column=op_token.column,
                operator=op_token.value if op_token.type == TokenType.MINUS else "not",
                operand=operand,
            )

        return self.parse_power()

    def parse_power(self) -> ASTNode:
        """Parse power operation (**)"""
        left = self.parse_primary()

        if self.match(TokenType.POWER):
            op_token = self.advance()
            # Right associative
            right = self.parse_unary()
            left = BinaryOpNode(
                line=op_token.line,
                column=op_token.column,
                left=left,
                operator="**",
                right=right,
            )

        return left

    def parse_primary(self) -> ASTNode:
        """
        Parse primary expressions (literals, identifiers, function calls, etc.)

        Returns:
            Primary expression node
        """
        node: ASTNode
        # Number literal
        if self.match(TokenType.NUMBER):
            token = self.advance()
            node = NumberNode(line=token.line, column=token.column, value=token.value)
            # Check for indexing
            return self.parse_postfix(node)

        # String literal
        elif self.match(TokenType.STRING):
            token = self.advance()
            node = StringNode(line=token.line, column=token.column, value=token.value)
            return self.parse_postfix(node)

        # Boolean literals
        elif self.match(TokenType.TRUE):
            token = self.advance()
            node = BooleanNode(line=token.line, column=token.column, value=True)
            return self.parse_postfix(node)

        elif self.match(TokenType.FALSE):
            token = self.advance()
            node = BooleanNode(line=token.line, column=token.column, value=False)
            return self.parse_postfix(node)

        # None literal
        elif self.match(TokenType.NONE):
            token = self.advance()
            node = NullNode(line=token.line, column=token.column)
            return self.parse_postfix(node)

        # Identifier or function call
        elif self.match(TokenType.IDENTIFIER):
            id_token = self.advance()

            # Check for function call
            if self.match(TokenType.LPAREN):
                return self.parse_function_call(id_token)

            # Otherwise, it's just an identifier
            node = IdentifierNode(
                line=id_token.line, column=id_token.column, name=id_token.value
            )
            return self.parse_postfix(node)

        # List literal
        elif self.match(TokenType.LBRACKET):
            return self.parse_list_literal()

        # Dictionary literal
        elif self.match(TokenType.LBRACE):
            return self.parse_dict_literal()

        # Parenthesized expression
        elif self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return self.parse_postfix(expr)

        else:
            raise ParserError(
                f"""Unexpected token in expression:
                    {self.current_token.type if self.current_token else 'None'}""",
                self.current_token,
            )

    def parse_postfix(self, node: ASTNode) -> ASTNode:
        """
        Parse postfix operations (indexing)

        Args:
            node: Base node to apply postfix operations to

        Returns:
            Node with postfix operations applied
        """
        while self.match(TokenType.LBRACKET):
            bracket_token = self.advance()
            index = self.parse_expression()
            self.expect(TokenType.RBRACKET)

            node = IndexNode(
                line=bracket_token.line,
                column=bracket_token.column,
                collection=node,
                index=index,
            )

        return node

    def parse_function_call(self, id_token: Token) -> FunctionCallNode:
        """
        Parse function call

        Args:
            id_token: Identifier token for function name

        Returns:
            FunctionCallNode
        """
        function = IdentifierNode(
            line=id_token.line, column=id_token.column, name=id_token.value
        )

        self.expect(TokenType.LPAREN)

        # Parse arguments
        arguments = []
        if not self.match(TokenType.RPAREN):
            arguments.append(self.parse_expression())

            while self.match(TokenType.COMMA):
                self.advance()
                arguments.append(self.parse_expression())

        self.expect(TokenType.RPAREN)

        return FunctionCallNode(
            line=id_token.line,
            column=id_token.column,
            function=function,
            arguments=arguments,
        )

    def parse_list_literal(self) -> ListNode:
        """
        Parse list literal: [1, 2, 3]

        Returns:
            ListNode
        """
        bracket_token = self.expect(TokenType.LBRACKET)

        elements = []
        if not self.match(TokenType.RBRACKET):
            elements.append(self.parse_expression())

            while self.match(TokenType.COMMA):
                self.advance()
                # Allow trailing comma
                if self.match(TokenType.RBRACKET):
                    break
                elements.append(self.parse_expression())

        self.expect(TokenType.RBRACKET)

        return ListNode(
            line=bracket_token.line, column=bracket_token.column, elements=elements
        )

    def parse_dict_literal(self) -> DictNode:
        """
        Parse dictionary literal: {"key": "value"}

        Returns:
            DictNode
        """
        brace_token = self.expect(TokenType.LBRACE)

        pairs = []
        if not self.match(TokenType.RBRACE):
            # Parse first pair
            key = self.parse_expression()
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((key, value))

            while self.match(TokenType.COMMA):
                self.advance()
                # Allow trailing comma
                if self.match(TokenType.RBRACE):
                    break
                key = self.parse_expression()
                self.expect(TokenType.COLON)
                value = self.parse_expression()
                pairs.append((key, value))

        self.expect(TokenType.RBRACE)

        return DictNode(line=brace_token.line, column=brace_token.column, pairs=pairs)


def parse(tokens: List[Token]) -> ProgramNode:
    """
    Convenience function to parse tokens into AST

    Args:
        tokens: List of tokens from lexer

    Returns:
        Root AST node (ProgramNode)

    Raises:
        ParserError: If parsing fails
    """
    parser = Parser(tokens)
    return parser.parse()
