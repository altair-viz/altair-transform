"""
Simple parser based on ply: 
"""
import sys
import os
import operator

import ply.lex as lex
import ply.yacc as yacc


# TODO: 
# - JS operators (inequalities, ternary)

OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "%": operator.mod
}


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

    def __init__(self, names=None, **kw):
        self.debug = kw.get('debug', 0)
        self.names = names or {}
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

    def parse(self, expression):
        return yacc.parse(expression)


class Parser(ParserBase):

    tokens = (
        'NAME', 'FLOAT', 'INTEGER', 'STRING', 'BINARY', 'OCTAL', 'HEX',
        'PLUS', 'MINUS', 'EXP', 'TIMES', 'DIVIDE', 'MODULO',
        'PERIOD', 'COMMA', 'COLON',
        'LPAREN', 'RPAREN',
        'LBRACKET', 'RBRACKET',
        'LBRACE', 'RBRACE'
    )

    # Tokens

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_EXP = r'\*\*'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MODULO = r'%'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_PERIOD = r'\.'
    t_COMMA = r','
    t_COLON = r'\:'
    t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

    def t_FLOAT(self, t):
        r'((\d+\.(\d*)?|\.\d+)([eE]\d+)?|[1-9]\d*[eE]\d+)'
        t.value = float(t.value)
        return t

    def t_BINARY(self, t):
        r'0[bB][01]+'
        t.value = int(t.value, 2)
        return t

    def t_OCTAL(self, t):
        r'0[oO]?[0-7]+'
        t.value = int(t.value, 8)
        return t

    def t_HEX(self, t):
        r'0[xX][0-9A-Fa-f]+'
        t.value = int(t.value, 16)
        return t

    def t_INTEGER(self, t):
        r'(0|[1-9]\d*)'
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
        ('left', 'TIMES', 'DIVIDE', 'MODULO'),
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
                   | expression MODULO expression
        """
        op = OPERATORS[p[2]]
        p[0] = op(p[1], p[3])

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

    def p_number(self, p):
        """
        number : HEX
               | OCTAL
               | BINARY
               | INTEGER
               | FLOAT
        """
        p[0] = p[1]

    def p_atom(self, p):
        """
        atom : number
             | STRING
             | global
             | list
             | object
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

    def p_object(self, p):
        """
        object : LBRACE RBRACE
               | LBRACE objectarglist RBRACE
        """
        if len(p) == 3:
            p[0] = {}
        elif len(p) == 4:
            p[0] = dict(p[2])

    def p_objectarglist(self, p):
        """
        objectarglist : objectarglist COMMA objectarg
                      | objectarg
        """
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_objectarg(self, p):
        """
        objectarg : objectkey COLON expression
                  | NAME
        """
        if len(p) == 2:
            p[0] = (p[1], self.names[p[1]])
        elif len(p) == 4:
            p[0] = (p[1], p[3])

    def p_objectkey(self, p):
        """
        objectkey : NAME
                  | STRING
                  | INTEGER
                  | FLOAT
        """ 
        p[0] = p[1]

    def p_group(self, p):
        'group : LPAREN expression RPAREN'
        p[0] = p[2]

    def p_attraccess(self, p):
        'attraccess : atom PERIOD NAME'
        p[0] = getattr(p[1], p[3])

    def p_indexing(self, p):
        'indexing : atom LBRACKET expression RBRACKET'
        obj = p[1]
        ind = p[3]
        if isinstance(obj, list) and isinstance(ind, int):
            p[0] = obj[ind]
        else:
            p[0] = getattr(obj, ind)

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