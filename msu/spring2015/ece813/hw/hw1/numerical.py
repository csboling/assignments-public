import math
import numpy
from scipy.optimize import fsolve

def Problem3pt2():
  tol   = 0.001
  phi_T = (0.65 / math.log((2.5e16 * 5e15) / 1.5e10))
  I_S   = 1e-17 * 12
  V_S   = 3.3
  R_S   = 2000

  def lhs(v):
    return (v + I_S*(math.exp(v / phi_T) - 1)*R_S + V_S)
  V_D_guess = I_S*R_S - V_S

  V_D = fsolve(lhs, V_D_guess)
  print V_D

Problem3pt2()
