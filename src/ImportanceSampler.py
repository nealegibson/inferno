
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time
import multiprocessing
import sys
from tqdm.auto import tqdm

import scipy.stats
#from . import MCMC_Class

#method that can be added to main mcmc class directly
def imsamp(self,n_samples,m=None,K=None,scaling=2,verbose=True):
  """
  Simple importance sampler using a normal distribution and current covariance matrix
  
  """
  
  #get mean and convariance matrix of distribution
  if m is None: m = self.p
  if K is None: K = self.cov
  
  #scale the covariance matrix (square to scale the distribution in parameter space)
  K = K*scaling**2
      
  #get the samples to use from a multivariate normal distribution
  X = np.random.multivariate_normal(m,K,n_samples)
  #evaluate proposal probability at each sample 
  var_index = self.e>0
  Ks = K.compress(var_index,axis=0).compress(var_index,axis=1) #compress the covariance to exclude constants
  logP_prop = scipy.stats.multivariate_normal.logpdf(X[:,var_index],m[var_index],Ks)
  
  #evaluate the samples at target distribution
  if self.parallel: #open the pool if running in parallel
    self.pool = multiprocessing.Pool(self.n_p)
    self.map_func = self.pool.map    
  logP_target = self.map(X) #compute logP for each
  #close the pool if running in parallel
  if self.parallel: self.pool.close()
  
  #finally call importance sampler
  log_z,p,pe = ImSampler(X,logP_prop,logP_target)
  
  if verbose:
    print('-' * 100)
    print('Importance sampled marginalised distributions:')
    print (" par = mean +- stdev")
    for i in range(p.size):
      print (" p[{}] = {:.7f} +- {:.7f}".format(i,p[i],pe[i]))
    print('log E = {}'.format(log_z))
    print('-' * 100)
   
  return log_z,p,pe

#extend the MCMC_Class to include importance sampler
#preferred to add directly to the class definition rather than overwrite base class
# class mcmc_imsamp(MCMC_Class.mcmc):
#   """new docstring"""
#   imsamp = imsamp

def ImSampler(X,logP_proposal,logP_target):
  """
  Simple importance sampler from pre-computed distributions
  See chapter 11 in PRML book for simple description
  
  X - samples from proposal distribution
  logProposalProb - corresponding log of proposal distribution
    (this should be a proper prior, i.e. int to 1)
  logP - log Posterior of target distribution at samples
  
  """

  #get difference between probabilities to each numerical calculation
  arg_max = logP_target.argmax()
  z_diff = logP_target[arg_max] - logP_proposal[arg_max]
  #z_diff = 0 #to test if required
  
  #get weights - subtracting z_diff is equivalent to dividing the posterior by a constant
  weights = np.exp(logP_target - logP_proposal - z_diff) # (equivalent to r_l for Z_q = 1)
  
  #get integral of the sample weights - equivalent to expectation of 1/evidence
  z = np.mean(weights) # (equivalent to Z_p, 11.21)
  
  #get log evidence after correcting for z_diff subtraction
  log_z = np.log(z) + z_diff # can just add z_diff back to logE - equivalent to multiplying evidence by const
  
  #get normalised weights
  w = weights / weights.sum() # (equivalent to w_l, these are not sensitive to z_diff as normalises out, 11.23)
  
  #finally compute the means
  means = np.dot(w,X)
  stdev = np.dot(w,(X-means)**2)
  
  return log_z,means,stdev
