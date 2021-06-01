
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

import inferno

#create test posterior - multivariate normal - removing zeros on diag of covariance
mu = np.ones(10)
K = np.diag(np.ones(mu.size)**2)
#add some covaraince
K[0,1] = K[1,0] = 0.8
K[2,3] = K[3,2] = -0.99
K[5,8] = K[8,5] = 0.7
def logP(x,mu,K):
  """
  Simple normal distribution.
  could be sped up by computing chofactor, logdetK outside the distribution, but may
    as well slow down the example.
  """
  #get residuals from mean
  r = x - mu  
  #reduce the covariance matrix
  var_par = np.diag(K)>0
  Ks = K.compress(var_par,axis=0)
  Ks = Ks.compress(var_par,axis=1)
  r = r.compress(var_par)
  #compute the multivariate Gaussian
  choFactor = LA.cho_factor(Ks,check_finite=False)
  logdetK = (2*np.log(np.diag(choFactor[0])).sum())
  return -0.5 * np.dot(r,LA.cho_solve(choFactor,r,check_finite=False)) - 0.5 * logdetK - (r.size/2.) * np.log(2*np.pi)

#define starting values for chain
p = np.arange(10) # pars
e = np.ones(10)*0.2 # errors
p[2] = p[5] = 1. # fix some parameters
e[2] = e[5] = 0. # fix some parameters

#test posterior works ok
print("logP =",logP(p,mu,K))
print("logP =",logP(p+e,mu,K))
print("logP =",logP(p-e,mu,K))

#optimise the function first?
# p = inferno.opt(logP,p,[mu,K],fixed=np.isclose(e,0))
# p = inferno.DE(logP,p,[mu,K],epar=e)

#define the mcmc object with logP + args + optional pars
mcmc = inferno.mcmc(logP, args=[mu,K],N=20)
mcmc = inferno.mcmc_imsamp(logP, args=[mu,K],N=20)
# mcmc = inferno.mcmc(logP, args=[mu,K],mode='MH') # 2 chains by default
# mcmc = inferno.mcmc(logP, args=[mu,K],mode='Gibbs',N=5,parallel=0,gibbs_ind=[1,2,0,1,1,0,1,1,1,1])
# mcmc = inferno.mcmc(logP, args=[mu,K],mode='MH',parallel=1)

#then set up the chain with initial conditions - most parameters can be set here too
mcmc.setup(p=p,e=e)
# mcmc.setup(p=p,K=K,dist='norm')
# mcmc.setup(p=p,e=e,dist='uniform')
# mcmc.setup(X=np.random.multivariate_normal(p,np.diag(e**2),mcmc.N),cull=False) #give samples directly
# mcmc.setup(np.random.multivariate_normal(p,np.diag(e**2),mcmc.N),p=p,e=e,cull=True) #provide p and e if culling is on
# mcmc.setup(p=p,e=e,parallel=True,thin=10) #provide p and e if culling is on

#then run the chain(s)
mcmc.burn(1000) #perform burnin of length 2000
pars,errors = mcmc.chain(1000) #run main chain of length 2000

#create some nice plots of the chain
axes = inferno.chainPlot(mcmc.chains)
ax = inferno.samplePlot(mcmc.chains_reshaped(2))
