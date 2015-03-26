import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from collections import OrderedDict

import os
pathjoin = os.path.join
import sys

import mathops
import utils

gridsize = 10e-9
datadir = utils.mkdir('../data')

def beta_optimization(parameters, ifname, fig, ofname=None, plotdir=None):
  df = utils.index_dframe(pd.read_csv(ifname)) / 1e-12
  df.index.name = 'beta'
  trace_names = {'t_HL' : r'$t_{pHL}$', 't_LH' : r'$t_{pLH}$'}

  utils.cycle_plot([dict(
                      x     = df.index.values,
                      y     = df[k].values,
                      label = trace_names[k]
                    ) for k in df])
  symmetric_pt = mathops.intersection(df.t_HL, df.t_LH)
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

  return dict()

def gamma_estimation(parameters, ifname, fig, ofname=None, plotdir=None):
  df = utils.index_dframe(pd.read_csv(ifname)) / 1e-12
  df.index_name = 'fanout'
  trace_names = { 't_pLH' : r'$t_{pLH}$' }

  x, y = df.index.values, df.t_pLH.values

  slope, intercept, r, p, stderr = linregress(x, y)

  t_p0, gamma = intercept, intercept/slope
  x_range = np.linspace(0, df.index[-1], 100)

  print('t_p vs gamma:', df)
  print(( 'slope: {:.3f}, '
        + 'intercept: {:.3f}, '
        + 'R: {:.3f}').format(slope, intercept, r))
  plt.plot(x, y, 
          marker='o', linestyle=' ', label='data points')
  plt.plot(x_range, slope*x_range + intercept,
           label = 'fit line')
  plt.xlabel('Fanout')
  plt.ylabel(r'$t_{pLH}$' + ' (ps)')
  plt.legend(loc='best')
  plt.title('Propagation delay versus fanout')

  plt.annotate(r'$t_{p0}$' + ' = {:.3f} ps'.format(t_p0), 
               xy = (0, intercept),
               xytext = (20, 2), textcoords = 'offset points',
               arrowprops = dict(arrowstyle = '->'))
  plt.annotate(r'$\gamma$' + ' = {:.3f}'.format(gamma),
               xy = (df.index[0], df.t_pLH[df.index[0]]),
               xytext = (10, -10), textcoords = 'offset points')

  if ofname != None:
    df.to_csv(ofname)
  if plotdir != None:
    plt.savefig(pathjoin(plotdir, 'Figure3.png'))

  return dict(t_p0 = t_p0, gamma = gamma)

def Cg1_estimation(parameters, ifname, fig, ofname=None, plotdir=None):
  df = utils.index_dframe(pd.read_csv(ifname)) / 1e-12
  zero_pt = pd.DataFrame([parameters['t_p0']], 
                         index=[0], columns=df.columns)
  df = pd.concat([zero_pt, df])
  df.index_name = 'Cext'
  trace_names = { 't_pLH' : r'$t_{pLH}$' }


  x, y = df.index.values, df['t_pLH'].values

  slope, intercept, r, p, stderr = linregress(x, y)

  C_g1 = parameters['t_p0']/(parameters['gamma']*slope)
  x_range = np.linspace(0, df.index[-1], 100)

  print('t_p vs C_ext:', df)
  print(( 'slope: {:.3e}, '
        + 'intercept: {:.3f}, '
        + 'R: {:.3f}').format(slope, intercept, r))
  plt.plot(x, y, 
          marker='o', linestyle=' ', label='data points')
  plt.plot(x_range, slope*x_range + intercept,
           label = 'fit line')
  plt.xlabel(r'$C_{ext}$')
  plt.ylabel(r'$t_{pLH}$' + ' (ps)')
  plt.legend(loc='best')
  plt.title('Propagation delay versus $C_{ext}$')

  plt.annotate(r'$t_{p0}$' + ' = {:.3f} ps (sanity check)'.format(parameters['t_p0']), 
               xy = (0, parameters['t_p0']),
               xytext = (20, 2), textcoords = 'offset points',
               arrowprops = dict(arrowstyle = '->'))
  plt.annotate(r'$C_{g1}$' + ' = {:.3e} F'.format(C_g1),
               xy = (df.index[1], df.t_pLH[df.index[1]]),
               xytext = (10, -10), textcoords = 'offset points')

  if ofname != None:
    df.to_csv(ofname)
  if plotdir != None:
    plt.savefig(pathjoin(plotdir, 'Figure4.png'))

  return dict(C_g1 = C_g1)

def theoretical_opt(parameters, load_cap, *args, **kwargs):
  def optimal(N):
    f = F**(1/N)
    tp = N*parameters['t_p0']*(1 + f / parameters['gamma'])
    return dict(f = f, tp = tp)

  F = load_cap / parameters['C_g1']
  N_est = np.log(F) / np.log(4)
  Nvalues = range(1, int(np.round(N_est*1.5)))
  df = pd.DataFrame(list(map(optimal, Nvalues)), index=Nvalues)
  print(df)
  df.plot()
  #  print(',\t'.join(['N = {}', 
  #                    'f = {:.4f}', 
  #                    'tp = {:.4f} ps']).format(N, f, tp))
  return dict(N = N, f = f, tp = tp)

analyses = OrderedDict()
# analyses[beta_optimization] = dict(
#                                 ifname = pathjoin(datadir, 'beta_rawdata.csv'), 
#                                 ofname = pathjoin(datadir, 'tp_vs_beta.csv'),
#                              )
analyses[gamma_estimation]  = dict(
                               ifname = pathjoin(datadir, 'tp_vs_fanout_rawdata.csv'),
                              )
analyses[Cg1_estimation]    = dict(
                                ifname = pathjoin(datadir, 'tp_vs_Cext_rawdata.csv'),
                              )
analyses[theoretical_opt]   = dict(
                                load_cap = 25e-12,
                              )
