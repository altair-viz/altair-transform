
# _parser_Parser_parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'leftPLUSMINUSleftTIMESDIVIDEleftEXPrightUMINUSUPLUSCOLON COMMA DIVIDE EXP FLOAT INTEGER LBRACE LBRACKET LPAREN MINUS NAME PERIOD PLUS RBRACE RBRACKET RPAREN STRING TIMESstatement : expression\n        expression : expression PLUS expression\n                   | expression MINUS expression\n                   | expression TIMES expression\n                   | expression DIVIDE expression\n                   | expression EXP expression\n        expression : MINUS expression %prec UMINUSexpression : PLUS expression %prec UPLUSexpression : term\n        term : atom\n             | attraccess\n             | functioncall\n             | indexing\n        \n        atom : INTEGER\n             | FLOAT\n             | STRING\n             | global\n             | list\n             | object\n             | group\n        global : NAME\n        list : LBRACKET RBRACKET\n             | LBRACKET arglist RBRACKET\n        \n        object : LBRACE RBRACE\n               | LBRACE objectarglist RBRACE\n        \n        objectarglist : objectarglist COMMA objectarg\n                      | objectarg\n        \n        objectarg : objectkey COLON expression\n                  | NAME\n        \n        objectkey : NAME\n                  | STRING\n                  | INTEGER\n                  | FLOAT\n        group : LPAREN expression RPARENattraccess : atom PERIOD NAMEindexing : atom LBRACKET expression RBRACKET\n        functioncall : atom LPAREN RPAREN\n                     | atom LPAREN arglist RPAREN\n        \n        arglist : arglist COMMA expression\n                | expression\n        '
    
_lr_action_items = {'MINUS':([0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,29,30,31,32,34,35,43,44,45,46,47,48,49,51,52,53,54,55,57,58,59,60,62,],[4,22,4,4,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,4,4,4,4,4,4,4,-8,-7,4,4,22,-22,22,-24,-2,-3,-4,-5,-6,-35,-37,22,-34,-23,4,-25,4,-38,-36,22,22,]),'PLUS':([0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,29,30,31,32,34,35,43,44,45,46,47,48,49,51,52,53,54,55,57,58,59,60,62,],[3,21,3,3,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,3,3,3,3,3,3,3,-8,-7,3,3,21,-22,21,-24,-2,-3,-4,-5,-6,-35,-37,21,-34,-23,3,-25,3,-38,-36,21,21,]),'INTEGER':([0,3,4,18,19,20,21,22,23,24,25,29,30,54,56,57,],[10,10,10,10,10,41,10,10,10,10,10,10,10,10,41,10,]),'FLOAT':([0,3,4,18,19,20,21,22,23,24,25,29,30,54,56,57,],[11,11,11,11,11,42,11,11,11,11,11,11,11,11,42,11,]),'STRING':([0,3,4,18,19,20,21,22,23,24,25,29,30,54,56,57,],[12,12,12,12,12,40,12,12,12,12,12,12,12,12,40,12,]),'NAME':([0,3,4,18,19,20,21,22,23,24,25,28,29,30,54,56,57,],[17,17,17,17,17,39,17,17,17,17,17,48,17,17,17,39,17,]),'LBRACKET':([0,3,4,6,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,29,30,32,35,52,53,54,55,57,],[19,19,19,30,-14,-15,-16,-17,-18,-19,-20,-21,19,19,19,19,19,19,19,19,19,-22,-24,-34,-23,19,-25,19,]),'LBRACE':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[20,20,20,20,20,20,20,20,20,20,20,20,20,20,]),'LPAREN':([0,3,4,6,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,29,30,32,35,52,53,54,55,57,],[18,18,18,29,-14,-15,-16,-17,-18,-19,-20,-21,18,18,18,18,18,18,18,18,18,-22,-24,-34,-23,18,-25,18,]),'$end':([1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,32,35,43,44,45,46,47,48,49,52,53,55,58,59,],[0,-1,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-8,-7,-22,-24,-2,-3,-4,-5,-6,-35,-37,-34,-23,-25,-38,-36,]),'TIMES':([2,5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,31,32,34,35,43,44,45,46,47,48,49,51,52,53,55,58,59,60,62,],[23,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-8,-7,23,-22,23,-24,23,23,-4,-5,-6,-35,-37,23,-34,-23,-25,-38,-36,23,23,]),'DIVIDE':([2,5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,31,32,34,35,43,44,45,46,47,48,49,51,52,53,55,58,59,60,62,],[24,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-8,-7,24,-22,24,-24,24,24,-4,-5,-6,-35,-37,24,-34,-23,-25,-38,-36,24,24,]),'EXP':([2,5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,31,32,34,35,43,44,45,46,47,48,49,51,52,53,55,58,59,60,62,],[25,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-8,-7,25,-22,25,-24,25,25,25,25,-6,-35,-37,25,-34,-23,-25,-38,-36,25,25,]),'RPAREN':([5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,29,31,32,34,35,43,44,45,46,47,48,49,50,52,53,55,58,59,60,],[-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-8,-7,49,52,-22,-40,-24,-2,-3,-4,-5,-6,-35,-37,58,-34,-23,-25,-38,-36,-39,]),'RBRACKET':([5,6,7,8,9,10,11,12,13,14,15,16,17,19,26,27,32,33,34,35,43,44,45,46,47,48,49,51,52,53,55,58,59,60,],[-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,32,-8,-7,-22,53,-40,-24,-2,-3,-4,-5,-6,-35,-37,59,-34,-23,-25,-38,-36,-39,]),'COMMA':([5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,32,33,34,35,36,37,39,43,44,45,46,47,48,49,50,52,53,55,58,59,60,61,62,],[-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-8,-7,-22,54,-40,-24,56,-27,-29,-2,-3,-4,-5,-6,-35,-37,54,-34,-23,-25,-38,-36,-39,-26,-28,]),'RBRACE':([5,6,7,8,9,10,11,12,13,14,15,16,17,20,26,27,32,35,36,37,39,43,44,45,46,47,48,49,52,53,55,58,59,61,62,],[-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,35,-8,-7,-22,-24,55,-27,-29,-2,-3,-4,-5,-6,-35,-37,-34,-23,-25,-38,-36,-26,-28,]),'PERIOD':([6,10,11,12,13,14,15,16,17,32,35,52,53,55,],[28,-14,-15,-16,-17,-18,-19,-20,-21,-22,-24,-34,-23,-25,]),'COLON':([38,39,40,41,42,],[57,-30,-31,-32,-33,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'statement':([0,],[1,]),'expression':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[2,26,27,31,34,43,44,45,46,47,34,51,60,62,]),'term':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[5,5,5,5,5,5,5,5,5,5,5,5,5,5,]),'atom':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[6,6,6,6,6,6,6,6,6,6,6,6,6,6,]),'attraccess':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,]),'functioncall':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,]),'indexing':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,]),'global':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[13,13,13,13,13,13,13,13,13,13,13,13,13,13,]),'list':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[14,14,14,14,14,14,14,14,14,14,14,14,14,14,]),'object':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[15,15,15,15,15,15,15,15,15,15,15,15,15,15,]),'group':([0,3,4,18,19,21,22,23,24,25,29,30,54,57,],[16,16,16,16,16,16,16,16,16,16,16,16,16,16,]),'arglist':([19,29,],[33,50,]),'objectarglist':([20,],[36,]),'objectarg':([20,56,],[37,61,]),'objectkey':([20,56,],[38,38,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> statement","S'",1,None,None,None),
  ('statement -> expression','statement',1,'p_statement_expr','_parser.py',124),
  ('expression -> expression PLUS expression','expression',3,'p_expression_binop','_parser.py',129),
  ('expression -> expression MINUS expression','expression',3,'p_expression_binop','_parser.py',130),
  ('expression -> expression TIMES expression','expression',3,'p_expression_binop','_parser.py',131),
  ('expression -> expression DIVIDE expression','expression',3,'p_expression_binop','_parser.py',132),
  ('expression -> expression EXP expression','expression',3,'p_expression_binop','_parser.py',133),
  ('expression -> MINUS expression','expression',2,'p_expression_uminus','_parser.py',148),
  ('expression -> PLUS expression','expression',2,'p_expression_uplus','_parser.py',152),
  ('expression -> term','expression',1,'p_expression_term','_parser.py',156),
  ('term -> atom','term',1,'p_term','_parser.py',161),
  ('term -> attraccess','term',1,'p_term','_parser.py',162),
  ('term -> functioncall','term',1,'p_term','_parser.py',163),
  ('term -> indexing','term',1,'p_term','_parser.py',164),
  ('atom -> INTEGER','atom',1,'p_atom','_parser.py',170),
  ('atom -> FLOAT','atom',1,'p_atom','_parser.py',171),
  ('atom -> STRING','atom',1,'p_atom','_parser.py',172),
  ('atom -> global','atom',1,'p_atom','_parser.py',173),
  ('atom -> list','atom',1,'p_atom','_parser.py',174),
  ('atom -> object','atom',1,'p_atom','_parser.py',175),
  ('atom -> group','atom',1,'p_atom','_parser.py',176),
  ('global -> NAME','global',1,'p_global','_parser.py',181),
  ('list -> LBRACKET RBRACKET','list',2,'p_list','_parser.py',186),
  ('list -> LBRACKET arglist RBRACKET','list',3,'p_list','_parser.py',187),
  ('object -> LBRACE RBRACE','object',2,'p_object','_parser.py',198),
  ('object -> LBRACE objectarglist RBRACE','object',3,'p_object','_parser.py',199),
  ('objectarglist -> objectarglist COMMA objectarg','objectarglist',3,'p_objectarglist','_parser.py',208),
  ('objectarglist -> objectarg','objectarglist',1,'p_objectarglist','_parser.py',209),
  ('objectarg -> objectkey COLON expression','objectarg',3,'p_objectarg','_parser.py',218),
  ('objectarg -> NAME','objectarg',1,'p_objectarg','_parser.py',219),
  ('objectkey -> NAME','objectkey',1,'p_objectkey','_parser.py',228),
  ('objectkey -> STRING','objectkey',1,'p_objectkey','_parser.py',229),
  ('objectkey -> INTEGER','objectkey',1,'p_objectkey','_parser.py',230),
  ('objectkey -> FLOAT','objectkey',1,'p_objectkey','_parser.py',231),
  ('group -> LPAREN expression RPAREN','group',3,'p_group','_parser.py',236),
  ('attraccess -> atom PERIOD NAME','attraccess',3,'p_attraccess','_parser.py',240),
  ('indexing -> atom LBRACKET expression RBRACKET','indexing',4,'p_indexing','_parser.py',244),
  ('functioncall -> atom LPAREN RPAREN','functioncall',3,'p_functioncall','_parser.py',249),
  ('functioncall -> atom LPAREN arglist RPAREN','functioncall',4,'p_functioncall','_parser.py',250),
  ('arglist -> arglist COMMA expression','arglist',3,'p_arglist','_parser.py',261),
  ('arglist -> expression','arglist',1,'p_arglist','_parser.py',262),
]
