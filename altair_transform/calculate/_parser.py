"""
Simple parser based on ply: 
"""
import sys
import os

import ply.lex as lex
import ply.yacc as yacc


# TODO: 
# - square brackets,
# - list literals
# - object literals
# - JS operators (inequalities, ternary)


def _decode_escapes(s):
    """Decode string escape sequences"""
    if sys.version_info[0] == 2:
        return s.decode("string-escape")
    else:
        return bytes(s, "utf-8").decode("unicode_escape")


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
        'NAME', 'FLOAT', 'INTEGER', 'STRING',
        'PLUS', 'MINUS', 'EXP', 'TIMES', 'DIVIDE', 'PERIOD',
        'LBRACKET', 'RBRACKET', 'LPAREN', 'RPAREN', 'COMMA',
    )

    # Tokens

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_EXP = r'\*\*'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_PERIOD = r'\.'
    t_COMMA = r','
    t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

    def t_FLOAT(self, t):
        r'(\d+(\.\d*)?|\.\d+)'
        t.value = float(t.value)
        return t

    def t_INTEGER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_STRING(self, t):
        r'''(?P<openquote>["'])((\\{2})*|(.*?[^\\](\\{2})*))(?P=openquote)'''
        t.value = _decode_escapes(t.value[1:-1])
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

    def p_expression_term(self, p):
        'expression : term'
        p[0] = p[1]

    def p_term(self, p):
        """
        term : atom
             | attraccess
             | functioncall
             | indexing
        """
        p[0] = p[1]

    def p_atom(self, p):
        """
        atom : INTEGER
             | FLOAT
             | STRING
             | global
             | list
             | group
        """
        p[0] = p[1]

    def p_global(self, p):
        'global : NAME'
        p[0] = self.names[p[1]]

    def p_list(self, p):
        """
        list : LBRACKET RBRACKET
             | LBRACKET arglist RBRACKET
        """
        if len(p) == 3:
            p[0] = []
        elif len(p) == 4:
            p[0] = list(p[2])
        else:
            raise NotImplementedError()

    def p_group(self, p):
        'group : LPAREN expression RPAREN'
        p[0] = p[2]

    def p_attraccess(self, p):
        'attraccess : atom PERIOD NAME'
        p[0] = getattr(p[1], p[3])

    def p_indexing(self, p):
        'indexing : atom LBRACKET expression RBRACKET'
        p[0] = getattr(p[1], p[3])

    def p_functioncall(self, p):
        """
        functioncall : atom LPAREN RPAREN
                     | atom LPAREN arglist RPAREN
        """
        if len(p) == 4:
            p[0] = p[1]()
        elif len(p) == 5:
            p[0] = p[1](*p[3])
        else:
            raise NotImplementedError()

    def p_arglist(self, p):
        """
        arglist : arglist COMMA expression
                | expression
        """
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_error(self, p):
        if p:
            raise ValueError("Syntax error at '%s'" % p.value)
        else:
            raise ValueError("Syntax error at EOF")