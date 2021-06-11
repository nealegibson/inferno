"""
Simple functions/decorators to produce some commonly used priors
"""
##
import numpy as np

def logPrior(bounds=None,lower=None,upper=None,p=None,e=None):
  """
  Simple function to create log prior from inputs, returns the logP function.
  
  logPosterior = inferno.logPrior(logLikelihood,[(-1,1),(-2,2),None],p=[0,0,0],e=[1,1,1])
  logPrior = inferno.logPrior([(-1,1),(-2,2),None],p=[0,0,0],e=[1,1,1])
  
  """
  
  if bounds is not None:
    lower = np.array([t[0] if type(t) is tuple else -np.inf for t in bounds])
    upper = np.array([t[1] if type(t) is tuple else np.inf for t in bounds])
  else:
    if lower is None and upper is None:
      raise ValueError("must provide bounds or lower/upper lims")
  
  if p is not None and e is None:
    raise ValueError("must provide both p/e for normal prior")
  elif e is not None and p is None:
    raise ValueError("must provide both p/e for normal prior")
  elif p is None and e is None:
    def log_normal(x):
      return 0.
  else:
    p,e = np.array(p),np.array(e)
    prec = 1/e**2
    const = 0.5*np.sum(np.log(2*np.pi*e**2))
    def log_normal(x):
      return -0.5*np.sum(prec*(x-p)**2) - const
  
  def log_prior(x,*args,**kwargs):
    """
    logPrior with simple bounds
    """
    if np.any(x<lower): return -np.inf
    elif np.any(x>upper): return -np.inf
    else: return log_normal(x)
  
  return log_prior
  
# def addlogPrior(*args,**kwargs):
#   """
#   Returns decorator that applies above prior to function, eg:
#   
#   @addlogPrior(bounds=blah) #...etc
#   def logLikelihood(p,...):
#     ...
#     return ...
#   
#   """
#   
#   log_prior = logPrior(*args,**kwargs)
#   
#   def decorator(func):
#     """
#     Decorator that takes in func and adds log_prior
#     """
#     
#     def log_posterior(x,*args,**kwargs):
#       """
#       logPosterior with added prior
#       """
#       return log_prior(x) + func(x,*args,**kwargs)
#     
#     return log_posterior
#   return decorator

def addlogPrior(*args,**kwargs):
  """
  If first argument is function, adds above logPrior to that function.
  Otherwise returns decorator that applies above prior to function, eg:
  
  @addlogPrior(bounds=blah) #...etc
  def logLikelihood(p,...):
    ...
    return ...
  
  """
  #check if first argument is a callable function, and if so remove it from args tuple
  #and set func to args[0]
  if len(args)>0 and callable(args[0]):
    func = args[0]
    args = args[1:]
  else: func=None #otherwise set func to None
  
  #create the logPrior function
  log_prior = logPrior(*args,**kwargs)
  
  #define posterior to return
  def log_posterior(x,*args,**kwargs):
    """
    logPosterior with added prior
    """
    #first get logPrior and check is not -np.inf
    lp = log_prior(x)
    if lp == -np.inf: return -np.inf
    else: return lp + func(x,*args,**kwargs)
  
  #or otherwise a decorator with same nested posterior
  def decorator(func):
    """
    Decorator that takes in func and adds log_prior
    """
    def log_posterior(x,*args,**kwargs):
      """
      logPosterior with added prior
      """
      #first get logPrior and check is not -np.inf
      lp = log_prior(x)
      if lp == -np.inf: return -np.inf
      else: return lp + func(x,*args,**kwargs)
            
    return log_posterior
  
  if func is None: return decorator
  else: return log_posterior

def addlogPriorClassMethod(*args,**kwargs):
  """
  Same as above, but for use within class methods
  Not properly tested, but appears to work if placed before call within class.
  Outside the class, the normal version can be used.
  Presumably because a call to func(x,...) will automatically revert to func(self,x,...) anyway
  """
  
  #check if first argument is a callable function, and if so remove it from args tuple
  #and set func to args[0]
  if len(args)>0 and callable(args[0]):
    func = args[0]
    args = args[1:]
  else: func=None #otherwise set func to None
  
  #create the logPrior function
  log_prior = logPrior(*args,**kwargs)
  
  #define posterior to return
  def log_posterior(self,x,*args,**kwargs):
    """
    logPosterior with added prior
    """
    #first get logPrior and check is not -np.inf
    lp = log_prior(x)
    if lp == -np.inf: return -np.inf
    else: return lp + func(self,x,*args,**kwargs)
  
  #or otherwise a decorator with same nested posterior
  def decorator(func):
    """
    Decorator that takes in func and adds log_prior
    """
    def log_posterior(self,x,*args,**kwargs):
      """
      logPosterior with added prior
      """
      #first get logPrior and check is not -np.inf
      lp = log_prior(x)
      if lp == -np.inf: return -np.inf
      else: return lp + func(self,x,*args,**kwargs)
            
    return log_posterior
  
  if func is None: return decorator
  else: return log_posterior


  