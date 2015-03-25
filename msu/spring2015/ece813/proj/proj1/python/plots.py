import matplotlib.pyplot as plt

import os
pathjoin = os.path.join

from utils import mkdir
import analyses
  
if __name__ == '__main__':
  plotdir = mkdir('../plots')

  plt.rc('text', usetex=True)
  for analysis, kwargs in analyses.analyses.items():
    analysis(fig=plt.figure(), plotdir=plotdir, **kwargs)
    plt.show()
