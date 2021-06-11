"""
Simple functions/decorators to produce some likelihoods from input functions
"""
##
import numpy as np

def logLikelihood_iid(func,x,y=None,e=None,beta=False,*args,**kwargs):
  """
  Simple function to create log likelihood from a function of form
  func(p,x,*args,**kwargs)
  
  """
  
  if y is None:
    raise ValueError("Must provide y values")
  if e is None:
    e = np.ones(y.size)
  
  const = 0.5*np.sum(np.log(2*np.pi*e**2))
  prec = 1/e**2
  N = y.size
  
  def log_likelihood(p):
    """
    log of iid normal dist
    """
    
    r = y-func(p,x,*args,**kwargs)
    
    return - 0.5 * ( r**2 * prec ).sum() - const #- N*np.log(beta)

  def log_likelihood_beta(p):
    """
    log of iid normal dist with noise scaling factor beta
    beta = noise if e not provided
    """
    
    r = y-func(p[:-1],x,*args,**kwargs)
    
    return - 0.5 * ( r**2 * prec / p[-1]**2 ).sum() - const - N*np.log(p[-1])
  
  if beta==False: return log_likelihood
  else: return log_likelihood_beta

#add aliases
logL_iid = logLikelihood_iid

# could add a decorator wrapper, but I don't see the use case