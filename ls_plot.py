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
import scipy as sp

def smooth(x,window_len=5,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
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
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
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
    return y

#--------------------------------------------------------------------
def extract_all_iterations(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  return iteration 

#--------------------------------------------------------------------
def extract_TestAccuracy(filename, netId):
  f=open(filename,'r')
  openFile = f.read()

  itregex = r"Iteration (\d*), Testing net \(#" + re.escape(str(netId)) + r"\)"
  #print "itregex =" + itregex
  #iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  iteration = re.findall(itregex, openFile)

  acregex = r"Test net #" + re.escape(str(netId)) + r" output #0: accuracy = (\d*.\d*)"
  #print "acregex ="+ acregex
  #accuracy = re.findall(r'Test net #0 output #0: accuracy = (\d*.\d*)', openFile)
  accuracy = re.findall(acregex, openFile)

  if debug :
    print iteration
  if debug :
    print accuracy 
  out = []
  for i in range(len(iteration)):
    out.append((iteration[i],accuracy[i]))
  return out 

#--------------------------------------------------------------------
def extract_TestLoss(filename, netId):
  f=open(filename,'r')
  openFile = f.read()

  itregex = r"Iteration (\d*), Testing net \(#" + re.escape(str(netId)) + r"\)"
  #print "itregex =" + itregex
  #iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  iteration = re.findall(itregex, openFile)

  regex = r"Test net #" + re.escape(str(netId)) + r" output #1: loss = (\d*.\d*)"
  #accuracy = re.findall(r'Test net #0 output #0: loss = (\d*.\d*)', openFile)
  loss = re.findall(regex, openFile)
  if debug :
    print iteration
  if debug :
    print loss 
  out = []
  for i in range(len(iteration)):
    out.append( (iteration[i], np.log(float(loss[i])) ))
  return out 
#----------------------------------------------------------------------
def extract_TestLossAfter(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'LINE SEARCH: iteration (\d*)', openFile)
  loss = re.findall(r'test loss after = (\d*.\d*)', openFile) 
  #if debug :
  print iteration
  #if debug :
  print loss 
  out = []
  for i in range(len(loss)):
    out.append((iteration[i], np.log(float(loss[i]))))
  return out
#----------------------------------------------------------------------

def extract_LsLossAfter(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'LINE SEARCH: iteration (\d*)', openFile)
  loss = re.findall(r'ls loss after = (\d*.\d*)', openFile) 
  #if debug :
  print iteration
  #if debug :
  print loss 
  out = []
  for i in range(len(loss)):
    out.append((iteration[i], np.log(float(loss[i]))))
  return out
#----------------------------------------------------------------------

def extract_TrainLoss(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), train loss ', openFile)
  loss = re.findall(r'train loss = (\d*.\d*)', openFile) 
  #if debug :
  #print iteration
  #if debug :
  #print loss 
  out = []
  for i in range(len(loss)):
    out.append((iteration[i], np.log(float(loss[i]))))
  return out

#----------------------------------------------------------------------
# plot Test accuracy  
def plot_TestAccuracy():
  global figure
  colors = iter(cm.rainbow(np.linspace(0, 1, 2*len(args))))
  plt.figure(figure)
  legendnames = []
  for filename in args:
    fname =  os.path.splitext(os.path.split(filename)[1])[0]
    testAccuracy = extract_TestAccuracy(filename, 0)
    #print testAccuracy
    #plt.scatter(*zip(*testAccuracy),color='blue')
    plt.plot(*zip(*testAccuracy),color='blue')
    #legendnames.append(fname + "_TestAccuracy")  
    legendnames.append("Test Accuracy") 
     
    lsAccuracy = extract_TestAccuracy(filename, 1)
    plt.plot(*zip(*lsAccuracy),color='red')
    #legendnames.append(fname + "_LsTrainAccuracy")
    legendnames.append("Train Accuracy") 
  
  plt.title('Accuracy', fontsize=14)
  plt.xlabel('Iteration', fontsize=14)
  plt.ylabel('Accuracy', fontsize=14)
  plt.legend(legendnames,loc='lower right')
  plt.xlabel('Iteration',fontsize=14)
  plt.xlim(0,30500)
  plt.ylim(0,1)
  #print filename + ": Final TestAccuracy is: " + testAccuracy[len(testAccuracy)-1][1]

#----------------------------------------------------------------------

def plot_TestLoss():
  global figure
  figure = figure + 1
  colors = iter(cm.rainbow(np.linspace(0, 1, 4*len(args))))
  plt.figure(figure)
  legendnames = []
  for filename in args:
    fname =  os.path.splitext(os.path.split(filename)[1])[0]
    testLoss = extract_TestLoss(filename, 0)
    #print testLoss
    plt.scatter(*zip(*testLoss),color=next(colors))
    #legendnames.append(fname + "_TestLoss") 
    legendnames.append("Test Loss") 
     
    lsLoss = extract_TestLoss(filename, 1)
    plt.scatter(*zip(*lsLoss),color=next(colors))
    legendnames.append("LS Loss")
    
   # testLossAfter = extract_TestLossAfter(filename)
   # plt.scatter(*zip(*testLossAfter),color=next(colors))
   # legendnames.append("Test Loss After")

   # lsLossAfter = extract_LsLossAfter(filename)
   # plt.scatter(*zip(*lsLossAfter),color=next(colors))
   # legendnames.append("LS Loss After")

  plt.title('Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Log Loss')
  plt.legend(legendnames,loc='upper right')

# ---------------------------------------------------------------------------   
def plot_TrainLoss():
  global figure
  figure = figure + 1
  colors = iter(cm.rainbow(np.linspace(0, 1, 3*len(args))))
  plt.figure(figure)
  legendnames = []
  for filename in args:
    fname =  os.path.splitext(os.path.split(filename)[1])[0]
    trainLoss = extract_TrainLoss(filename)
    it,loss = zip(*trainLoss) 
    plt.scatter(it,loss,color=next(colors))
    #plt.scatter(*zip(*trainLoss),color=next(colors))
    #legendnames.append(fname + "_FullTrainLoss")
    legendnames.append("TrainLoss")

  plt.title('Train Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Log Loss')
  plt.legend(legendnames,loc='upper right')
  #plt.grid(True)
  #plt.xticks(np.arange(int(it[0]), int(it[-1]), 500.0))

#--------------------------------------------------------------------

def extract_ls_iteration(filename):
  f=open(filename,'r')
  #print filename
  openFile = f.read()
  iteration = re.findall(r"LINE SEARCH: iteration (\d*)", openFile)
  return iteration

#-------------------------------------------------------------------
def extract_ls_a(filename):
  f=open(filename,'r')
  openFile = f.read()
  a = re.findall(r" a = (\-?\d*.\d*)", openFile)
  return a

#--------------------------------------------------------------------
def extract_ls_b(filename):
  f=open(filename,'r')
  openFile = f.read()
  b= re.findall(r" b = (\-?\d*.\d*)", openFile)
  #print b
  return b
#----------------------------------------------------------------------
def extract_ls_c(filename):
  f=open(filename,'r')
  openFile = f.read()
  c = re.findall(r" c = (\-?\d*.\d*)", openFile)
  return c
#-------------------------------------------------------------------- 
def extract_ls_r2(filename):
  f=open(filename,'r')
  openFile = f.read()
  r2 = re.findall(r" Rsquare = (\d*.\d*)", openFile)
  out = []
  for i in range(len(r2)):
    out.append(float(r2[i]))
  return out
#--------------------------------------------------------------------
def extract_alpha(filename):
  f=open(filename,'r')
  openFile = f.read()
  alpha = re.findall(r" alpha_opt = (\-?\d*.\d*)", openFile)
  return alpha
#--------------------------------------------------------------------
def extract_newlr(filename):
  f=open(filename,'r')
  openFile = f.read()
  lr = re.findall(r" new_lr = (\-?\d*.\d*)", openFile)
  return lr
#--------------------------------------------------------------------  
def extract_newmoment(filename):
  f=open(filename,'r')
  openFile = f.read()
  moment = re.findall(r" new_moment = (\d*.\d*)", openFile)
  return moment
#--------------------------------------------------------------------
def plot_R2():
  global figure 
  filename = args[0]
  ls_iteration = extract_ls_iteration(filename)
  #print ls_iteration
  #print len(ls_iteration)
  r2 = extract_ls_r2(filename)
  #print len(r2)
  #r2_arr = np.asarray(r2)
  #print r2_arr
  #r2_smooth=smooth(r2_arr, 9)
  #print len(r2_smooth)
  #ind1=(len(r2_smooth)-len(r2) )/2
  #r2_out = r2_smooth[ind1: ind1+len(r2)]   
  #print len(r2_out) 
  figure = figure +1
  plt.figure(figure)
  plt.plot(ls_iteration,r2, color='black')
  plt.title('R^2')
  plt.xlabel('Iteration',fontsize=14)
  plt.ylabel('R^2',fontsize=14)
  plt.xlim(0,30500)
  #plt.legend(['R^2'],loc='upper right', fontsize=14)
  
#--------------------------------------------------------------------

def plot_a_b_c():
  global figure 
  figure = figure +1
  #plt.figure(figure)
  
  filename = args[0]
  ls_iteration = extract_ls_iteration(filename)
  #print ls_iteration
  #print len(ls_iteration)
  
  a = extract_ls_a(filename)
  b = extract_ls_b(filename)
  c = extract_ls_c(filename)

  #print a
  #print len(a)
  
  f, (ax1,ax2, ax3) = plt.subplots(3, sharex = True, figsize=(8, 18))
  ax1.plot(ls_iteration,a,color='black')
  ax1.set_title('a', fontsize=14)
  ax1.set_ylim(-0.2,1)
  ax1.spines['bottom'].set_position(('data', 0))
  
  ax2.plot(ls_iteration, b,color='red')
  ax2.set_ylim(-1,0.2)
  ax2.set_title('b', fontsize=14)
  ax2.spines['bottom'].set_position(('data', 0))
  
  ax3.plot(ls_iteration,c,color='blue')
  ax3.set_title('c', fontsize=14)
  ax3.set_ylim(-0.2,1)
  ax3.spines['bottom'].set_position(('data', 0))
  
  plt.xlabel('Iteration',fontsize=14)
  plt.xlim(0,30500)

#---------------------------------------------------------------------   
def plot_alpha():
  global figure 

  filename = args[0]
  ls_iteration = extract_ls_iteration(filename)
  alpha = extract_alpha(filename)
    
  figure = figure +1

  plt.figure(figure)
  plt.plot(ls_iteration,alpha,color='black')
  plt.title('alpha', fontsize=14)
  plt.xlabel('Iteration', fontsize=14)
  plt.xlim(0,30500)
  plt.gca().set_ylim(bottom=0)
  
#---------------------------------------------------------------------
def plot_ls_lr():
  global figure 
  figure = figure +1
  
  filename = args[0]
  ls_iteration = extract_ls_iteration(filename)
  newlr = extract_newlr(filename)

  plt.figure(figure)
  plt.step(ls_iteration,newlr,color='black')
  plt.title('Automatic learning rate control', fontsize=14)
  plt.xlabel('Iteration', fontsize=14) 
  plt.xlim(0,30500)
  plt.ylim(0,0.01)
#---------------------------------------------------------------------
def plot_ls_moment():
  global figure 
    
  filename = args[0]
  ls_iteration = extract_ls_iteration(filename)
  moment = extract_newmoment(filename)
  
  figure = figure +1
  plt.figure(figure)
  plt.scatter(ls_iteration,moment, color='black')
  plt.title('Automatic moment Control', fontsize=14)
  plt.xlabel('Iteration', fontsize=14) 
  plt.xlim(0,30500)
   
#---------------------------------------------------------------------
def plot_lr():   
  global figure
  figure = figure +1
  colors = iter(cm.rainbow(np.linspace(0, 1, 10)))
  plt.figure(figure)
  filename = args[0]
  lr_iteration = extract_lr(filename)
  [iteration,lr] =  zip(*lr_iteration)
  plt.scatter(iteration,lr,color=next(colors))
  plt.title('lr')
  plt.xlabel('Iteration')
  plt.ylabel('lr')
  plt.legend(['lr'],loc='upper right')
#---------------------------------------------------------------------

#---------------------------------------------


def extract_accuracy(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net ', openFile)
  accuracy = re.findall(r'Test net output #0: accuracy = (\d*.\d*', openFile) 
  out = []
  for i in range(len(iteration)):
    out.append((iteration[i],accuracy[i]))
  return out 

#--------------------------------------------------------------------
def extract_loss(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net ', openFile)
  loss = re.findall(r'Testnew_lr net output #1: loss = (\d*.\d*)', openFile) 
  out = []
  for i in range(len(loss)):
    out.append((iteration[i],loss[i]))
  return out
 

#----------------------------------------------------------------------

def extract_lr(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), lr =', openFile)
  lr = re.findall(r' lr = (\d*.\d*(e-\d*)?)', openFile) 
  #print iteration
  #print lr
  out = []
  for i in range(len(iteration)):
    out.append((iteration[i],lr[i][0]))
  return out 
  
# ------- global variables ---------------------------------------------

args = sys.argv[1:]
figure = 0
debug = 0
# ------ main ---------------------------------------------------------
def main():
  # command-line parsing
  global  args
  global figure

  if not args:
    print 'usage: [file ...]'
    sys.exit(1)

  #------------------------------------------------
  plot_TestAccuracy()
  #plot_TestLoss()
  plot_R2()
  plot_a_b_c()
  plot_alpha()
  plot_ls_lr()
  plt.show()
   
  #build legend names from file names 
  filenames = []
  legendnames = []
  for name in args:
    if name.endswith('.log'):
      filenames.append(os.path.splitext(name)[0])
      legendnames.append(filenames[-1])

    #-------------------------------------------------
  #plot accuracy  
  #colors = iter(cm.rainbow(np.linspace(0, 1, 2*len(args))))
  #plt.figure(1)plot show axes
  #for filename in args:
  #  testAccuracy = extract_TestAccuracy(filename, 0)
    #print testAccuracy
  #  plt.scatter(*zip(*testAccuracy),color=next(colors))
    #print filename + ": Final TestAccuracy is: " + testAccuracy[len(testAccuracy)-1][1]
  #  lsAccuracy = extract_TestAccuracy(filename, 1)
    #print lsAccuracy
  #  plt.scatter(*zip(*testAccuracy),color=next(colors))
  #plt.title('Accuracy')
  #plt.xlabel('Iteration')
  #plt.ylabel('Accuracy')
  #plt.legend(legendnames,loc='lower right')
 
  #plot loss   
  #colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  #plt.figure(2)
  #for filename in args:
  #  loss = extract_loss(filename)
  #  [iteration,loss] =  zip(*loss)
  #  loss_new = []
  #  for i in range(len(loss)):
  #    loss_new.append(np.log(float(loss[i])))
  #  plt.scatter(iteration,loss_new,color=next(colors))
  #plt.title('Log Loss')
  #plt.xlabel('Iteration')
  #plt.ylabel('Log Loss')
  #plt.legend(legendnames,loc='upper right')

  #plt.show()   
  


if __name__ == '__main__':
  main()
