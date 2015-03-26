import matplotlib.pyplot as plt

import os
pathjoin = os.path.join

from utils import mkdir
import analyses
  
if __name__ == '__main__':
  plotdir = mkdir('../plots')
  plt.rc('text', usetex=True)
  parameters = dict()
  for analysis, kwargs in analyses.analyses.items():
    print()
    print(analysis.__name__)
    results = analysis(
                parameters=parameters,
                fig=plt.figure(), plotdir=plotdir,
                **kwargs
              )
    print(parameters, results)
    parameters.update(results)
    print(parameters)
    plt.show()
