import pandas as pd
import matplotlib.pyplot as plt

import os
from itertools import cycle

def mkdir(path):
  p = os.path.normpath(path)
  os.makedirs(p, exist_ok = True)
  return p

def index_vars(names):
  vars = {}
  for s in names:
    var, index = s.split()
    if var not in vars:
      vars[var] = {}
    vars[var][index] = s
  return vars

def index_dframe(df, indep='X', *args, **kwargs):
  vars  = index_vars(df.columns)
  dependents = {}
  for k, v in vars.items():
    for dep in v:
      if dep != indep:
        dependents[k] = pd.Series(df[v[dep]].values, 
                                  index=df[v[indep]].values)
  data = pd.DataFrame(dependents)
  return data

def cycle_plot(trace_list):
  styles = cycle(['-', '--', '-.', ':'])
  for trace, style in zip(trace_list, styles):
    plt.plot(trace['x'], trace['y'], style,
             label=trace['label'])
