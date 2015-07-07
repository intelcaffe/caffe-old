#!/usr/bin/python

"""Extract accuracy and loss from log files
   usage: file [file ...]
   Written by Lior Shani.
"""

import sys
import re
import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def smooth(x,window_len=5,window='hanning'):
    """smooth the data using a window with requested size.
    
    based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
      t=linspace(-2,2,0.1)
      x=sin(t)+randn(len(t))*0.1
      y=smooth(x)
    
    see also: 
      numpy.hanning, numpy.hamming, numpy.bartlett, 
      numpy.blackman, numpy.convolve
      scipy.signal.lfilter
 
     NOTE: length(output) != length(input), 
     to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    #return y
    return y[(window_len/2-1):-(window_len/2)]

#--------------------------------------------------------------------
def extract_train_accuracy(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), train loss =', openFile)
  #print iteration
  accuracy = re.findall(r'Train net output #0: accuracy = (\d*.\d*)', openFile) 
  x=[];
  for i in range(len(accuracy)):
    x.append(float(accuracy[i]))
  y= np.asarray(x)
  z = smooth(y,13)
  #print accuracy
  out = []
  for i in range(len(accuracy)):
    #out.append((iteration[i],accuracy[i]))
    out.append((iteration[i],z[i]))
  return out

#--------------------------------------------------------------------
def extract_accuracy(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  #print iteration
  accuracy = re.findall(r'Test net #0 output #0: accuracy = (\d*.\d*)', openFile) 
  #print accuracy
  out = []
  for i in range(len(iteration)):
    out.append((iteration[i],accuracy[i]))
  return out 

#--------------------------------------------------------------------
def extract_test_loss(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  #print iteration
  loss = re.findall(r'Test net #0 output #1: loss = (\d*.\d*)', openFile) 
  #print loss
  out = []
  for i in range(len(iteration)):
    out.append((iteration[i],loss[i]))
  #print out
  return out 
  
#--------------------------------------------------------------------

def extract_train_loss(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), train ', openFile)
  loss = re.findall(r'Train net output #1: loss = (\d*.\d*)', openFile) 
  x=[];
  for i in range(len(loss)):
    x.append(float(loss[i]))
  y= np.asarray(x)
  z = smooth(y,13)
  out = []
  for i in range(len(loss)):
     out.append((iteration[i],z[i]))
  
  return out 

#--------------------------------------------------------------------

def extract_iteration(filename):
  f=open(filename,'r')
  print filename
  openFile = f.read()
  iteration = re.findall(r"Iteration (\d*), Testing net \(#0\)", openFile)
  return iteration

#----------------------------------------------------------------------
def main():
  # command-line parsing
  args = sys.argv[1:]
  if not args:
    print 'usage: [file ...]'
    sys.exit(1)

  #build legend names from file names 
  filenames = []
  legendnames = []
  for name in args:
    if name.endswith('.log'):
      filenames.append(os.path.split(os.path.splitext(name)[0])[1])
  #    legendnames.append(filenames[-1])
  
  colors = iter( ['b','g','r','c','m','k'] )
  # ---- plot accuracy -------------------------------------------  
  #colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  plt.figure(1)
  for filename in args:
    testName=os.path.split(os.path.splitext(filename)[0])[1]
    accuracy = extract_train_accuracy(filename)
    c=next(colors);
    plt.plot(*zip(*accuracy),color=c,linestyle ='-')
    legendnames.append(testName + " train")
    print filename + ": Train accuracy is = " + str(accuracy[len(accuracy)-1][1])
    
    accuracy = extract_accuracy(filename)
    plt.plot(*zip(*accuracy), color=c, linestyle ='--')
    legendnames.append(testName + " test")
    print filename + ": Test accuracy is = " + str(accuracy[len(accuracy)-1][1])
  
  plt.title('Train & Test Accuracy')
  plt.xlabel('Iteration')
  plt.xlim(0,30000)
  plt.ylabel('Accuracy')
  plt.ylim(0,1)
  plt.yticks(np.arange(0, 1.05, 0.05))
  plt.legend(legendnames,loc='lower right')
  plt.grid()
  #plt.show() 

  # --------------------------------------
  #plot loss   
	#
  #colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  plt.figure(2)
  colors = iter( ['b','g','r','c','m','k'] )
  for filename in args:
    c=next(colors);
    train_loss = extract_train_loss(filename)
    [iteration,loss] =  zip(*train_loss)
    loss_new = []
    for i in range(len(loss)):
      loss_new.append(np.log(float(loss[i])))
    plt.plot(iteration,loss_new,color=c,linestyle ='-')
    
    test_loss = extract_test_loss(filename)
    [iteration,loss] =  zip(*test_loss)
    loss_new = []
    for i in range(len(loss)):
      loss_new.append(np.log(float(loss[i])))
    plt.plot(iteration,loss_new,color=c,linestyle ='--')
  
  plt.title('Train and test Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Log Loss')
  plt.xlim(0,30000)
  #plt.legend(legendnames,loc='upper right')
  plt.grid()
  plt.show() 
  
if __name__ == '__main__':
  main()
