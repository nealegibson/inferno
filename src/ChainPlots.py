"""
Updated chain plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import scipy.ndimage as ndimage

###############################################################################
###############################################################################

def chainAxes(N,span=1,offset=0,width=5,labels=None,sharex=True,xticks=True,left=0.07,bottom=0.1,right=0.93,top=0.93,wspace=0.03,hspace=0.03):
  """
  span is fraction  
  """
  
  ax = {}
      
  #set axes - define bottom first and all others shrae its x-axis
#   ax_bottom = plt.subplot(N,width,(width*(N-1)+1+offset,width*(N-1)+span+offset))
#   if sharex: ax = [plt.subplot(N,width,(i*width+1+offset,i*width+span+offset),sharex=ax_bottom) for i in range(0,N-1)] + [ax_bottom]
#   else: ax = [plt.subplot(N,width,(i*width+1+offset,i*width+span+offset)) for i in range(0,N-1)] + [ax_bottom]
  
  ax = [plt.subplot(N,width,(i*width+1+offset,i*width+span+offset)) for i in range(0,N)] 
  if sharex: #share the x-axes of all plots with the bottom one 
    for a in ax[:-1]: a.sharex(ax[-1])
  
  #set/hide labels
  for a in ax[:-1]: plt.setp(a.get_xticklabels(), visible=False) #hide axes
  #if not xticks: plt.setp(ax[-1].get_xticklabels(), visible=False) #hide last one
  for a in ax: plt.setp(a.get_yticklabels(), visible=False) #hide axes
  for a in ax: a.set_yticks([])
  if not xticks:
    for a in ax: a.set_xticks([])
  
  #set ylabels
  if labels is not None:
    for a,label in zip(ax,labels): a.set_ylabel(label) 
  
  plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)
      
  return ax
  
def chainPlot(X,fmt='-',alpha=1,fig=None,conv=None,filt=True,x=None,labels=None,lw=0.5):
  """
  Plot the 1D chains.
  Assumes X is chainlength x Nchains x pars
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
    
  #first get the axes if not provided
  if fig is None: #create new plot by default if all axes are None
    plt.figure()
  ax = chainAxes(filt.sum(),3,labels=np.array(labels)[filt]) 
  ax_acorr = chainAxes(filt.sum(),1,3,) 
  ax_hist = chainAxes(filt.sum(),1,4,sharex=False,xticks=False) 
  
  Sfilt = S[...,filt] #get rid of fixed parameters
  Sfilt_meansub = Sfilt - Sfilt.mean(axis=-1)[...,np.newaxis] #mean subtract chains (for acorr)
  for q in range(Sfilt.shape[2]):
    for i in range(Sfilt.shape[1]): #loop over chains
      ax[q].plot(Sfilt[:,i,q],fmt,lw=lw,alpha=alpha)
      ax_hist[q].hist(Sfilt[:,i,q],density=True,histtype='step',orientation='horizontal',lw=lw)
      ax_acorr[q].plot(np.arange(Sfilt_meansub.shape[0])-Sfilt_meansub.shape[0]//2,np.correlate(Sfilt_meansub[:,i,q],Sfilt_meansub[:,i,q],mode='same'),fmt,lw=lw,alpha=alpha)
      ax_acorr[q].axhline(0,ls='--',lw=0.5,color='0.5')
      ax_acorr[q].axvline(0,ls='--',lw=0.5,color='0.5')
  ax[-1].set_xlabel('N')        
  ax[-1].set_xlim(0,Sfilt.shape[0])
  ax_acorr[-1].set_xlabel('lag')        
  ax_acorr[-1].set_xticks([0,Sfilt_meansub.shape[0]//2])
      
  return fig,ax,ax_acorr,ax_hist

###############################################################################
###############################################################################

