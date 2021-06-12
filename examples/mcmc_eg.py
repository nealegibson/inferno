
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

import inferno

#create test posterior - multivariate normal after removing any zeros on diag of covariance
n_pars = 5
@inferno.addlogPrior(bounds=[(-2,2),]*n_pars) # can add simple priors using decorator
def logP(x,mu,K):
  """
  Simple normal distribution.
  could be sped up by computing chofactor, logdetK first, but will do for example.
  """
  r = x - mu  #get residuals from mean

  #reduce the covariance matrix and residuals if any zeros in covariance diagonal
  var_par = np.diag(K)>0
  Ks = K.compress(var_par,axis=0).compress(var_par,axis=1)
  r = r.compress(var_par)
  
  #finally compute the multivariate Gaussian and return
  choFactor = LA.cho_factor(Ks,check_finite=False)
  logdetK = (2*np.log(np.diag(choFactor[0])).sum())
  return -0.5 * np.dot(r,LA.cho_solve(choFactor,r,check_finite=False)) - 0.5 * logdetK - (r.size/2.) * np.log(2*np.pi)

#define parameters of posterior distribution
mu = np.ones(n_pars)
mu[0] = 1.2
K = np.diag(np.ones(mu.size)**2)
#add some covaraince

K[0,1] = K[1,0] = 0.8
K[2,3] = K[3,2] = -0.99
#K[5,8] = K[8,5] = 0.7

#define starting/guess parameters
p = np.ones(n_pars) # pars
e = np.ones(n_pars)*0.2 # errors
#p[0] = 1.2 # change some parameters
#e[2] = 0. # fix some parameters

#test posterior works ok
# print("logP =",logP(p,mu,K))
# print("logP =",logP(p+e,mu,K))
# print("logP =",logP(p-e,mu,K))

#optimise the function first?
# p = inferno.opt(logP,p,[mu,K],fixed=np.isclose(e,0))
# p = inferno.DE(logP,p,[mu,K],epar=e)

#define the mcmc object with logP + args + optional pars
#mcmc = inferno.mcmc(logP, args=[mu,K],N=20,mode='DEMC',filename='test.pkl')
mcmc = inferno.mcmc(logP, args=[mu,K],mode='MH') # 2 chains by default
mcmc = inferno.mcmc(logP, args=[mu,K],mode='Gibbs',N=4,parallel=0,gibbs_ind=[1,2,2,1,1])
# #mcmc = inferno.mcmc(logP, args=[mu,K],mode='MH',N=20,parallel=1)
# mcmc = inferno.mcmc(logP, args=[mu,K],mode='AffInv',N=100,parallel=0)
# mcmc = inferno.mcmc(logP, args=[mu,K],mode='DEMC',N=100,parallel=0)

#use built in wrappers for optimisers/slicers to refine initial conditions
# p = mcmc.opt(p=p,e=e)
# opt_par,pos_err,neg_err = mcmc.error1D(par_index,p=p,e=e)
# p,e = mcmc.errors1D(p=p,e=e)

#then set up the chain with initial conditions - most parameters can be set/reset here too
mcmc.setup(p=p,e=e)
# mcmc.setup(p=p,K=K,dist='norm')
# mcmc.setup(p=p,e=e,dist='uniform')
# mcmc.setup(X=np.random.multivariate_normal(p,np.diag(e**2),mcmc.N),cull=False) #give samples directly
# mcmc.setup(np.random.multivariate_normal(p,np.diag(e**2),mcmc.N),p=p,e=e,cull=True) #provide p and e if culling is on
# mcmc.setup(p=p,e=e,parallel=True,thin=10) #provide p and e if culling is on
# pars,errors = mcmc.setup(p=p,e=e,burn=1000,chain=5000) #run the burnin+chains immediately

#then run the chain(s)
mcmc.burn(1000) #perform burnin of length 2000
pars,errors = mcmc.chain(20000,verbose=True) #run main chain of length 2000
# pars,errors = mcmc.chain(5000,verbose=True) #extend the chain again

#create some plots of the chain/distributions
f,*axes =inferno.chainPlot(mcmc.chains)
f,ax = inferno.samplePlot(mcmc.chains_reshaped(2))
#and known distribution
#_ = inferno.correlationEllipses(mu=mu[e>0],K=K[e>0][:,e>0],ax=ax)
