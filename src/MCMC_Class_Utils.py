
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time
import multiprocessing
import sys
import dill

##########################################################################################

def load_mcmc(dillfile):
  """
  Reload an mcmc object from dill file generated from mcmc.save()
  This tries to reset the globals required for easy parallelisation. Function
   dependencies may need to be imported.
  No guarantee this will work on different platforms.
  
  """
  mcmc = dill.load(open(dillfile,'rb'))
  #run reset_logP to reset the globals used for parallelisation etc
  mcmc.reset_logP()
  return mcmc
  
##########################################################################################

def computeGR(X,conv=0,N=None):
  """
  Compute the Gelman and Rubin statistic and errors for all variable parameters
   after optionally reshaping chains.
  Convergence is given in terms of original chain length
  """
  
  assert X.ndim==3, "X must be 3D L x N x pars"
  
  #reshape chains if N is given
  if N is not None:
    N_input = X.shape[1]
    X = X.reshape(-1,N,X.shape[-1])
    conv = conv * N_input // N #rescale convergence to match new shape
      
  L = X.shape[0]-conv #length of individual chain
  assert conv>=0, ValueError("conv must be >=0")
  assert L>0, ValueError("conv set is too high for chain of length {}".format(X.shape[0]))
  X = X[conv:] #filter out steps before convergence
  
  #compute only for variables
  p = np.where(X.std(axis=(0,1))>0)[0]
  grs = np.zeros(X.shape[-1])
  
  for i in p:
  
    #get mean and variance for the individual chains
    mean = X[...,i].mean(axis=0)
    var = X[...,i].var(axis=0)
        
    #and calculate the GR stat
    W = var.mean(dtype=np.float64) #mean of the variances
    B = mean.var(dtype=np.float64) #variance of the means
    grs[i] = np.sqrt((((L-1.)/L)*W + B) / W) #GR stat

  return grs
  
##########################################################################################

def analyse_chains(X,conv=0,logP_max=None,n_gr=None):
  """
  Simple function to get results from a chain or saved chain file
  """
  
  if type(X) is str:
    #check extension and load in different formats
    if X.split('.')[-1] == 'dill':
      mcmc = dill.load(open(X,'rb'))
      X = mcmc.chains
      logP_max = mcmc.log_prob.max()
      note = "chain loaded from saved class: {}".format(X)      
    elif X.split('.')[-1] == 'npy':
      X = np.load(X)
      note = "chain loaded from numpy file: {}".format(X)      
    elif X.split('.')[-1] == 'npz':
      f = np.load(X)
      X = f['X']
      logP_max = f['log_prob'].max()
      note = "chain loaded from .npz archive: {}".format(X)      
    else: #default if no extension provided
      raise ValueError("if X is str, must have dill, npy or npz extension")
  
  X_flat = X.reshape(-1,X.shape[-1])
  X_flat = X_flat[conv:] # get rid of points before convergence
  assert X_flat.shape[0]>0, ValueError("conv set is too high for chain of length {}".format(X.shape[0]))
  
  means = means = X_flat.mean(axis=0)
  medians = medians = np.median(X_flat,axis=0)
  stdevs = stdevs = X_flat.std(axis=0)
  #get assymetric errors
  lower_lims,medians,upper_lims = np.percentile(X_flat,[15.9,50,100-15.9],axis=0)
  pos_errs = upper_lims - medians
  neg_errs = medians - lower_lims

  if X.ndim==2: grs=np.zeros(means.size)
  else: grs = computeGR(X,conv=conv,N=n_gr)
    
  print('-' * 100)
  print ("MCMC Marginalised distributions:")
  print (" par = mean +- stdev [med +err -err]: GR")
  for i in range(means.size):
    print (" p[{}] = {:.7f} +- {:.7f} [{:.7f} +{:.7f} -{:.7f}]: GR = {:.4f}".format(i,means[i],stdevs[i],medians[i],pos_errs[i],neg_errs[i],grs[i]))
 
  if logP_max is not None:
    #calculate evidence approximations
    K = np.cov(X_flat-means,rowvar=0) #get covariance - better to mean subtract first to avoid rounding errors
  
    #compress K where 
    var_par = np.diag(K)>0
    Ks = K.compress(var_par,axis=0)
    Ks = Ks.compress(var_par,axis=1)
    D = np.diag(Ks).size #no of dimensions
    sign,logdetK = np.linalg.slogdet( 2*np.pi*Ks ) # get log determinant
    logE = logP_max + 0.5 * logdetK #get evidence approximation based on Gaussian assumption
    print ("Gaussian Evidence approx:")
    print (" log ML =", logP_max)
    print (" log E =", logE)
    logE_BIC = logP_max
    print (" log E (BIC) = log ML - D/2.*np.log(N) =", logP_max, "- {}/2.*np.log(N)".format(D))
    logE_AIC = logP_max - D
    print (" log E (AIC) = log ML - D =", logE_AIC, "(D = {})".format(D))
  print('-' * 100)

  return means,stdevs

##########################################################################################
