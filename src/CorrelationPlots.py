"""
Updated correlation plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import scipy.ndimage as ndimage

###############################################################################

def correlationAxes(N,inv=False,labels=None,left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.03,hspace=0.03):
  """
  Returns axes for correlation plots
  """
  
  
#  if fig is None: plt.figure()
  
  ax = {}
  if labels is None:
    labels = [r'$\theta_{{{}}}$'.format(i) for i in range(N)]
  
  #create normal axes
  if not inv:
    for i in range(N): #loop over the parameter indexes supplied
      for q in range(i+1):
        ax['{}{}'.format(i,q)] = plt.gcf().add_subplot(N,N,i*N+q+1,xticks=[],yticks=[])
        #add labels...
        if i == (N-1): ax['{}{}'.format(i,q)].set_xlabel(labels[q])
      ax['{}{}'.format(i,0)].set_ylabel(labels[i])
        
  #or inverse axes
  else:
    print('Need to refine labels + positions for inverse axes')
    for i in range(N):
      for q in range(i+1):
        ax['{}{}'.format(i,q)] = plt.gcf().add_subplot(N,N,(N-i)*N-q,xticks=[],yticks=[])
        if i == (N-1):
          ax['{}{}'.format(i,q)].set_xlabel(labels[q])
          ax['{}{}'.format(i,q)].xaxis.set_label_position('top') 
          
      ax['{}{}'.format(i,0)].set_ylabel(labels[i])
      ax['{}{}'.format(i,0)].yaxis.set_label_position('right') 
  
  plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)
  
  return ax

def correlationAxesPadded(N,inv=False,labels=None,left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.03,hspace=0.03):
  """
  Returns axes for correlation plots
  
  row_pad - add some more space for each row
  col_pad - add some more space for each column
  row_add - offset by rows
  col_add - offset by cols
  
  """
  
  print("not yet supported/tested!")
  
  ax = {}
  if labels is None:
    labels = [r'$\theta_{{{}}}$'.format(i) for i in range(N)]
  
  #create normal axes
  if not inv:
    for i in range(N): #loop over the parameter indexes supplied
      for q in range(i+1):
        ax['{}{}'.format(i,q)] = plt.gcf().add_subplot(N,N,i*N+q+1,xticks=[],yticks=[])
        #add labels...
        if i == (N-1): ax['{}{}'.format(i,q)].set_xlabel(labels[q])
      ax['{}{}'.format(i,0)].set_ylabel(labels[i])
        
  #or inverse axes
  else:
    print('Need to refine labels + positions for inverse axes')
    for i in range(N):
      for q in range(i+1):
        ax['{}{}'.format(i,q)] = plt.gcf().add_subplot(N,N,(N-i)*N-q,xticks=[],yticks=[])
        if i == (N-1):
          ax['{}{}'.format(i,q)].set_xlabel(labels[q])
          ax['{}{}'.format(i,q)].xaxis.set_label_position('top') 
          
      ax['{}{}'.format(i,0)].set_ylabel(labels[i])
      ax['{}{}'.format(i,0)].yaxis.set_label_position('right') 
  
  plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)
  
  return ax
  
###############################################################################

def correlationHist(X,ax=False,inv=False,**kwargs):
  
  #get no of dimensions
  N = X.shape[1]
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)
  
  #make a plot of the histograms accross the diagonal
  for i in range(N):
    ax['{}{}'.format(i,i)].hist(X[:,i],20,histtype='step',density=1,**kwargs)
  
  return ax

###############################################################################

def correlationNormalMarg(p,pe,X=None,Nsig=5,Nsamp=500,ax=False,inv=False,**kwargs):
  
  #get no of dimensions
  N = p.size
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)
  
  #make a plot of the histograms accross the diagonal
  for i in range(N):
    x = np.linspace(p[i]-pe[i]*Nsig,p[i]+pe[i]*Nsig,Nsamp)
    y = np.exp(-0.5*(x-p[i])**2/pe[i]**2) / np.sqrt(2*np.pi) / pe[i]
    ax['{}{}'.format(i,i)].plot(x,y,**kwargs)
  
  #also reset xranges for histograms if X is provided
  if X is not None:
    for i in range(N):
      ax['{}{}'.format(i,i)].set_xlim(X[:,i][:].min(),X[:,i][:].max())
  
  return ax
  
###############################################################################

def correlationScatterPlot(X,ax=False,fmt='.',samples=100,inv=False,alpha=0.6,zorder=3,**kwargs):
  
  #get no of dimensions
  S,N = X.shape
  ind = np.random.randint(0,S,samples)
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)

  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):
      ax['{}{}'.format(i,q)].plot(X[:,q][ind],X[:,i][ind],fmt,alpha=alpha,zorder=zorder,**kwargs)
  
  return ax

###############################################################################

def correlationCrossHairs(x,ax=False,inv=False,alpha=0.6,zorder=3,lw=1.5,ls='--',color='0.2',**kwargs):
  """
  Plot cross-hairs for a known point, e.g. to test simulations
  
  """
  
  #get no of dimensions
  N = x.size
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)

  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):
      #print(i,q,N)
      ax['{}{}'.format(i,q)].axhline(x[i],color=color,lw=lw,ls=ls,alpha=alpha,zorder=zorder,**kwargs)
      ax['{}{}'.format(i,q)].axvline(x[q],color=color,lw=lw,ls=ls,alpha=alpha,zorder=zorder,**kwargs)
      #ax['{}{}'.format(i,q)].plot(X[:,q][ind],X[:,i][ind],'.',alpha=alpha,zorder=zorder,**kwargs)
  for i in range(N):
    ax['{}{}'.format(i,i)].axvline(x[i],color=color,lw=lw,ls=ls,alpha=alpha,zorder=zorder,**kwargs)
  
  return ax
  
###############################################################################

def correlationEllipses(X=None,mu=None,K=None,ax=False,inv=False,alpha=0.6,zorder=3,color='b',**kwargs):
    
  #get covariance matrix of X if not given
  if K is None:
    if X is None:
      raise ValueError("must provide either X or K!")
    #get no of dimensions
    N = X.shape[1]
    #get mean and covariance
    K = np.cov(X.T)
    mu = X.mean(axis=0)
    #check if parameter is fixed
    for i in range(N):
      if np.all(X[:,i] == X[:,i][::-1]): #check if array equals its reverse
        K[i,i] = 0.
  
  else: #use K
    N = K.shape[0]
    if mu is None:
      raise ValueError("Must provide mean mu as well as cov K")
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)
  
  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):

      if K[i,i] == 0. or K[q,q] == 0.:
        continue
      #first get 2D covariance and mean:
      m = [mu[q],mu[i]]      
      K_t = np.diag([K[q][q],K[i][i]]) #note that the axes are swapped - q is the x-axis!
      K_t[1,0],K_t[0,1] = K[i][q],K[i][q]
      
      #get eigen decomposition
      w,v = np.linalg.eig(K_t) #first get eigen decomposition
      #define ellipse for 1,2 and 3 sigma
      angle = np.arctan(v[:,0][1]/v[:,0][0]) * 180./np.pi #get angle from principle component

      e = [Ellipse(m,2*np.sqrt(w[0])*np.sqrt(n),2*np.sqrt(w[1])*np.sqrt(n),\
          angle,lw=1,fill=True,alpha=0.2,ec='k',zorder=2,color=color) for n in [2.295817,6.1801,11.83]]

      for n in range(3): ax['{}{}'.format(i,q)].add_patch(e[n])
      ax['{}{}'.format(i,q)].plot()

  return ax
      
###############################################################################

def correlationContours(X,ax=False,inv=False,Nz=5,Nm=5,Ng=2,alpha=0.8,zorder=3,filled=False,colors='k',lw=1,**kwargs):
  
  #get no of dimensions
  S,N = X.shape
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)
  
  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):
      
      #set means and ranges for both axes
      mq,sq = X[:,q][:].mean(),X[:,q][:].std()
      mi,si = X[:,i][:].mean(),X[:,i][:].std()
      rq = np.linspace(mq-10.*sq,mq+10.*sq,50)
      ri = np.linspace(mi-10.*si,mi+10.*si,50)
      
      if np.isclose(sq,0.0) or np.isclose(si,0.0):
        continue
      
      #get 2D histogram
      H,a,b = np.histogram2d(X[:,q][:],X[:,i][:],bins=(rq,ri),density=1)
      a_mid = a[:-1] + (a[1]-a[0])/2. #convert to midpoints (hist returns limits)
      b_mid = b[:-1] + (b[1]-b[0])/2.
      
      #find the values for the contours via cumulative distribution
      H = H[:] / H.max() #normalise the histogram
      fl = H.flatten() #flatten
      fl.sort() #and sort
      qsum = np.cumsum(fl) #get cumulative distribution
      qsum /= qsum.max() #and normalise
      #finally, get closest values in array to 1,2,3 sigma limits
      ind1 = np.abs(qsum - 0.3173).argmin()
      ind2 = np.abs(qsum - 0.0455).argmin()
      ind3 = np.abs(qsum - 0.0027).argmin()
      s1,s2,s3 = fl[ind1],fl[ind2],fl[ind3]
      
      #smooth image for contour plot
      a_mid = ndimage.zoom(a_mid,Nz)
      b_mid = ndimage.zoom(b_mid,Nz)
      H = ndimage.zoom(H,Nz)
      #apply median filter and gaussian filter to smooth
      H = ndimage.median_filter(H, size=Nm*Nz)
      H = ndimage.gaussian_filter(H, sigma=Ng)
      
      #only plot a subset of the contour plot
      filt = np.where( (a_mid > mq-5.*sq) * (a_mid < mq+5.*sq) )
            
#      return a_mid,b_mid,H,filt
      ax['{}{}'.format(i,q)].contour(a_mid[filt],b_mid[filt],H[filt].T[filt],origin='lower',levels=(s3,s2,s1),colors=colors,zorder=30,alpha=alpha,linewidths=lw)
      
      ax['{}{}'.format(i,q)].set_xlim(X[:,q][:].min(),X[:,q][:].max())
      ax['{}{}'.format(i,q)].set_ylim(X[:,i][:].min(),X[:,i][:].max())
  
  #also set xranges for histograms
  for i in range(N):
    ax['{}{}'.format(i,i)].set_xlim(X[:,i][:].min(),X[:,i][:].max())
  
  #return the axes
  return ax

def correlationFilledContours(X,ax=False,inv=False,Nz=5,Nm=5,Ng=2,alpha=0.2,zorder=3,filled=False,colors=['b','b','b'],lw=1,**kwargs):
  
  #get no of dimensions
  S,N = X.shape
  
  #create axes if not provided
  if ax==False:  
    ax = correlationAxes(N,inv=inv)
  
  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):
      
      #set means and ranges for both axes
      mq,sq = X[:,q][:].mean(),X[:,q][:].std()
      mi,si = X[:,i][:].mean(),X[:,i][:].std()
      rq = np.linspace(mq-10.*sq,mq+10.*sq,50)
      ri = np.linspace(mi-10.*si,mi+10.*si,50)
      
      if np.isclose(sq,0.0) or np.isclose(si,0.0):
        continue
      
      #get 2D histogram
      H,a,b = np.histogram2d(X[:,q][:],X[:,i][:],bins=(rq,ri),density=1)
      a_mid = a[:-1] + (a[1]-a[0])/2. #convert to midpoints (hist returns limits)
      b_mid = b[:-1] + (b[1]-b[0])/2.
      
      #find the values for the contours via cumulative distribution
      H = H[:] / H.max() #normalise the histogram
      fl = H.flatten() #flatten
      fl.sort() #and sort
      qsum = np.cumsum(fl) #get cumulative distribution
      qsum /= qsum.max() #and normalise
      #finally, get closest values in array to 1,2,3 sigma limits
      ind1 = np.abs(qsum - 0.3173).argmin()
      ind2 = np.abs(qsum - 0.0455).argmin()
      ind3 = np.abs(qsum - 0.0027).argmin()
      s1,s2,s3 = fl[ind1],fl[ind2],fl[ind3]
      
      #smooth image for contour plot
      a_mid = ndimage.zoom(a_mid,Nz)
      b_mid = ndimage.zoom(b_mid,Nz)
      H = ndimage.zoom(H,Nz)
      #apply median filter and gaussian filter to smooth
      H = ndimage.median_filter(H, size=Nm*Nz)
      H = ndimage.gaussian_filter(H, sigma=Ng)
      
      #only plot a subset of the contour plot
      filt = np.where( (a_mid > mq-5.*sq) * (a_mid < mq+5.*sq) )
            
#      return a_mid,b_mid,H,filt
      #ax['{}{}'.format(i,q)].contourf(a_mid[filt],b_mid[filt],H[filt].T[filt],origin='lower',levels=(s3,s2,s1,1),colors=colors,zorder=30)
      ax['{}{}'.format(i,q)].contourf(a_mid[filt],b_mid[filt],H[filt].T[filt],origin='lower',levels=(s3,1),colors=colors,zorder=30,alpha=alpha)
      ax['{}{}'.format(i,q)].contourf(a_mid[filt],b_mid[filt],H[filt].T[filt],origin='lower',levels=(s2,1),colors=colors,zorder=30,alpha=alpha)
      ax['{}{}'.format(i,q)].contourf(a_mid[filt],b_mid[filt],H[filt].T[filt],origin='lower',levels=(s1,1),colors=colors,zorder=30,alpha=alpha)
      
      ax['{}{}'.format(i,q)].set_xlim(X[:,q][:].min(),X[:,q][:].max())
      ax['{}{}'.format(i,q)].set_ylim(X[:,i][:].min(),X[:,i][:].max())
  
  #also set xranges for histograms
  for i in range(N):
    ax['{}{}'.format(i,i)].set_xlim(X[:,i][:].min(),X[:,i][:].max())
  
  #return the axes
  return ax
  
###############################################################################

def samplePlot(X,conv=None,fig=None,hist=True,scatter=True,filt=True,x=None,inv=False,labels=None,samples=500,contour=False,contourf=False,ret_filt=False):
  """
  Convenience function for plotting MCMC samples from a file or array
    using defaults for above functions
  Converts X from various formats into N x chains x pars
  If from a file removes the first array, assuming likelihoods
  
  conv is assumed to be in 'N'
  
  """
  
  #first convert X into arrays if given as string(s)
  if (type(X) is list or type(X) is tuple):
    if type(X[0]) is str: #list of files
      X = [np.load(file)[:,1:] for file in X]
  elif type(X) is str:
    X = np.load(X)[:,1:]
    
  #check/convert X into right array format
  if (type(X) is list or type(X) is tuple):
    if type(X[0]) is np.ndarray and X[0].ndim == 2:
      S = np.hstack([x[conv:,np.newaxis,:] for x in X]) # join 2D arrays along new 'middle' axes  
  elif X.ndim==2: S = X[conv:,np.newaxis,:]
  elif X.ndim==3: S = X[conv:]
  else: raise ValueError("X not in correct format. Should be filename or list of filenames, multiple 2D arrays, or 3D array")
      
  #define labels is not done already (as need to reverse for inv)
  if labels is None: labels = [r'$\theta_{{{}}}$'.format(i) for i in range(S.shape[-1])]

  if filt: #filter out the fixed parameters
    filt = ~np.isclose(np.std(S,axis=(0,1)),0)
  else:
    filt = np.ones(S.shape[-1])>0
  
  if inv:
    S = S[...,::-1] #reverse parameter array for inverse
    labels = labels[::-1] #and labels
    filt = filt[::-1]
    if x is not None: x = x[::-1]
  
  #first get the axes
  if fig is None: fig = plt.figure()
  ax = correlationAxes(filt.sum(),inv=inv,labels=np.array(labels)[filt])

  #do histograms
  if hist:
    for i in range(S.shape[1]):
      correlationHist(S[:,i,filt],ax=ax)
  
  #do scatter plot
  if scatter:
    for i in range(S.shape[1]):
      correlationScatterPlot(S[:,i,filt],ax=ax,samples=samples)
  
  #plot crosshairs
  if x is not None:
    correlationCrossHairs(x[filt],ax=ax)

  #contour plot(s)
  if contour:
    correlationContours(S[...,filt].reshape(-1,filt.sum()),ax=ax)
  if contourf:
    correlationFilledContours(S[...,filt].reshape(-1,filt.sum()),filled=True,ax=ax)
   
  if ret_filt: return fig,ax,filt
  else: return fig,ax
  
###############################################################################
###############################################################################

