"""
Simplified optimisation functions uses for tutorials.
"""

import numpy as np
from scipy.optimize import fmin

def fopt(f,x0,var=None,args=[],min=False,**kwargs):
  """
  Optimisation function using scipy's fmin.
  
  This uses a simple wrapper to allow maximising as well as minimising functions, as well
   as allowing for a fixed set of parameters. inferno.opt or inferno.optimise has more
   options.
  
  f - function to be optimised, which can be called as f(x0,*args)
  x0 - array of starting points
  var - array with the same length as x0 indicating variable parameters. Parameters are
    variable if >0, so boolean True/False, integers, or even error arrays will work
  args - additional arguments passed to fmin (see scipy.optimize.fmin)
  min - if True minimises rather than maximises the function f wrt x0
  kwargs - additional keyword args are passed to fmin
  
  """

  if var is None: var = np.ones(x0.size)
  var_ind = var>0
  x = np.copy(x0)
  
  #define wrapper function to re-construct the full parameter array from subset
  def wrapper(p,*args):
    x[var_ind] = p # get full parameter array
    if min: return f(x,*args) #call the function and return
    else: return -f(x,*args) #or the negative of the func to maximise the function instead
  
  #call scipy's fmin on the wrapper function
  x[var_ind] = fmin(wrapper,x0[var_ind],args=args,**kwargs)  
  return x 
