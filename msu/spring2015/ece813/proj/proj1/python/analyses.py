import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
pathjoin = os.path.join

import mathops
import utils

gridsize = 10e-9
datadir = utils.mkdir('../data')

def beta_optimization(ifname, fig, ofname=None, plotdir=None):
  df = utils.index_dframe(pd.read_csv(ifname)) / 1e-12
  df.index.name = 'beta'
  trace_names = {'t_HL' : r'$t_{pHL}$', 't_LH' : r'$t_{pLH}$'}

  utils.cycle_plot([dict(
                      x     = df.index.values,
                      y     = df[k].values,
                      label = trace_names[k]
                    ) for k in df])
  symmetric_pt = mathops.intersection(df['t_HL'], df['t_LH'])
  plt.plot(*symmetric_pt, marker='x')
  plt.annotate( r'$\beta$' 
               + '= {:.2f}'.format(symmetric_pt[0])
               + '\n for symmetric VTC',
               xy = symmetric_pt,
               xytext = (-10, -50), textcoords = 'offset points',
               arrowprops = dict(arrowstyle = '->',
                                 connectionstyle = 'arc3,rad=0'))

  plt.title('Inverter propagation delay (rising and falling) versus relative PMOS size')
  plt.xlabel(r'$\beta$')
  plt.ylabel('Propagation delay (ps)')
  plt.legend(loc='best')

  if ofname != None:
    df.to_csv(ofname)
  if plotdir != None:
    plt.savefig(pathjoin(plotdir, 'Figure1.png'))

analyses = {
  beta_optimization : dict(
                        ifname = pathjoin(datadir, 'beta_rawdata.csv'), 
                        ofname = pathjoin(datadir, 'tp_vs_beta.csv'),
                      ),
}
