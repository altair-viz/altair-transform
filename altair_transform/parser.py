"""
Simple parser based on ply: 
"""
import sys

if sys.version_info[0] >= 3:
    raw_input = input

import ply.lex as lex
import ply.yacc as yacc
import os


class ParserBase(object):
    """
    Base class for a lexer/parser that has the rules defined as methods
    """
    tokens = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.names = {}
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[
                1] + "_" + self.__class__.__name__
        except:
            modname = "parser" + "_" + self.__class__.__name__
        self.debugfile = modname + ".dbg"
        self.tabmodule = modname + "_" + "parsetab"

        # Build the lexer and parser
        lex.lex(module=self, debug=self.debug)
        yacc.yacc(module=self,
                  debug=self.debug,
                  debugfile=self.debugfile,
                  tabmodule=self.tabmodule)

    def parse(self, expression, names=None):
        original_names = self.names
        self.names = names or {}
        try:
            result = yacc.parse(expression)
        finally:
            self.names = original_names
        return result


class Parser(ParserBase):

    tokens = (
        'NAME', 'FLOAT', 'INTEGER',
        'PLUS', 'MINUS', 'EXP', 'TIMES', 'DIVIDE', 'PERIOD',
        'LPAREN', 'RPAREN',
    )

    # Tokens

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_EXP = r'\*\*'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_PERIOD = r'\.'
    t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

    def t_FLOAT(self, t):
        r'(\d+(\.\d*)?|\.\d+)'
        t.value = float(t.value)
        return t

    def t_INTEGER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    t_ignore = " \t"

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self, t):
        raise ValueError("Illegal character '%s'" % t.value[0])

    # Parsing rules

    precedence = (
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('left', 'EXP'),
        ('right', 'UMINUS', 'UPLUS'),
    )

    def p_statement_expr(self, p):
        'statement : expression'
        p[0] = p[1]

    def p_expression_binop(self, p):
        """
        expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression EXP expression
        """
        # print [repr(p[i]) for i in range(0,4)]
        if p[2] == '+':
            p[0] = p[1] + p[3]
        elif p[2] == '-':
            p[0] = p[1] - p[3]
        elif p[2] == '*':
            p[0] = p[1] * p[3]
        elif p[2] == '/':
            p[0] = p[1] / p[3]
        elif p[2] == '**':
            p[0] = p[1] ** p[3]

    def p_expression_uminus(self, p):
        'expression : MINUS expression %prec UMINUS'
        p[0] = -p[2]

    def p_expression_uplus(self, p):
        'expression : PLUS expression %prec UPLUS'
        p[0] = +p[2]

    def p_expression_group(self, p):
        'expression : group'
        p[0] = p[1]

    def p_number(self, p):
        """
        number : INTEGER
               | FLOAT
        """
        p[0] = p[1]

    def p_expression_number(self, p):
        'expression : number'
        p[0] = p[1]

    def p_expression_name(self, p):
        'expression : NAME'
        try:
            p[0] = self.names[p[1]]
        except LookupError:
            print("Undefined name '%s'" % p[1])
            p[0] = 0

    def p_expression_attr(self, p):
        """
        expression : group PERIOD NAME
                   | NAME PERIOD NAME
        """
        try:
            p[1] = self.names[p[1]]
        except LookupError:
            print("Undefined name '%s'" % p[1])
            p[0] = 0
        p[0] = getattr(p[1], p[3])

    def p_group_paren(self, p):
        'group : LPAREN expression RPAREN'
        p[0] = p[2]

    def p_error(self, p):
        if p:
            raise ValueError("Syntax error at '%s'" % p.value)
        else:
            raise ValueError("Syntax error at EOF")

parser = Parser()