'''
A shallow embedding to compile a subset of Python to Lisp.
'''

import argparse
import sys
from StringIO import StringIO
from functools import partial

import inspect
import ast
import meta
import lispify as lpfy

class Embedding(object):
  '''
  This only operates on ast.Expr, so assignments, return
  expressions, etc. should be excluded. Therefore you
  basically write a 'script' by not assigning any names,
  and interleave this with writing python normally
  (although you cannot extract values back
  out of the "Lisp" part of the code as this requires
  running it in the target Lisp implementation).
  To accomplish this we do some tricky things with
  introspection and execution frames.
  '''
  @staticmethod
  def toSource(x):
    return meta.dump_python_source(x).strip()

  @classmethod
  def toList(cls, x, frame):
    recur = partial(cls.toList, frame=frame)
    if isinstance(x, ast.Name):
      try:
        value = frame.f_locals[x.id]
      except KeyError:
        value = x.id
      return lpfy.Atom(value)
    # Python-style function calls are evaluated
    elif isinstance(x, ast.Call):
      # ew
      if x.func.id == 'Quote':
        return "'" + recur(x.args)
      else:
        normalized = repr(eval(cls.toSource(x)))
        recur_with = ast.parse(normalized).body[0].value
        return recur(recur_with)
    elif isinstance(x, ast.Num):
      return x.n
    elif isinstance(x, ast.Str):
      return x.s
    elif isinstance(x, ast.List):
      return map(recur, x.elts)
    elif isinstance(x, ast.UnaryOp):
      return [lpfy.Atom(cls.toSource(x.op)),
                recur(x.operand)]
    elif isinstance(x, ast.BinOp):
      return [lpfy.Atom(cls.toSource(x.op)),
                recur(x.left),
                recur(x.right)]
    else:
      return cls.toSource(x)

  @classmethod
  def parse_node(cls, node, frame):
    parsed = cls.toList(node.value, frame=frame)
    return (lpfy.lispify(parsed) + '\n')

  @classmethod
  def parse_ast(cls, t):
    # This is done very naively
    result = ''
    for __x in t:
      frame = inspect.currentframe()
      if isinstance(__x, ast.Expr):
        result += cls.parse_node(__x, frame=frame)
      else:
        exec(meta.dump_python_source(__x))
    return result

def parse_func(func):
  source = inspect.getsource(func)
  func_ast = ast.parse(source).body[0]
  embedding = Embedding()
  return embedding.parse_ast(func_ast.body)

def parse(source):
  embedding = Embedding()
  return embedding.parse_ast(ast.parse(source))

def example():
  z = 7*5
  print z
  [y] * z
  [lispfunc, "x", str(3), Quote(q)]

if __name__ == '__main__':
  sys.stdout.write(parse_func(example))
