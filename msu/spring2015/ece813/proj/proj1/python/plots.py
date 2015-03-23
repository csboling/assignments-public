import pandas as pd
import matplotlib.pyplot as plt

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
  print(data)
  return data

def transistor_sizing(data_fname):
  df = index_dframe(pd.read_csv(data_fname))
  df.plot()

if __name__ == '__main__':
  analyses = {
    transistor_sizing : "../data/transition_times.csv",
  }
  for k, v in analyses.items():
    k(v)
    plt.show()
