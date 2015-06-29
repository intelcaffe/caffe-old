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
    plt.scatter(*zip(*testAccuracy),color=next(colors))
    #legendnames.append(fname + "_TestAccuracy")  
    legendnames.append("Test Accuracy") 
     
    lsAccuracy = extract_TestAccuracy(filename, 1)
    plt.scatter(*zip(*lsAccuracy),color=next(colors))
    #legendnames.append(fname + "_LsTrainAccuracy")
    legendnames.append("LS Accuracy") 
  
  plt.title('Accuracy')
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy')
  plt.legend(legendnames,loc='lower right')
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
  return r2
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
def plot_a_b_c():
  #plot a b c   
  global figure
  figure = figure +1
  colors = iter(cm.rainbow(np.linspace(0, 1, 10)))
  plt.figure(figure)
  filename = args[0]
  ls_iteration = extract_ls_iteration(filename)
  #print ls_iteration
  #print len(ls_iteration)
  a = extract_ls_a(filename)
  #print a
  #print len(a)
  plt.scatter(ls_iteration,a,color= next(colors))
  plt.title('a')
  plt.xlabel('Iteration')
  plt.ylabel('a')
  plt.legend(['a'],loc='upper right')
  #----------------
  figure = figure +1
  b = extract_ls_b(filename)
  #print b
  plt.figure(figure)
  plt.scatter(ls_iteration, b,color=next(colors))
  plt.title('b')
  plt.xlabel('Iteration')
  plt.ylabel('b')
  plt.legend(['b'],loc='upper right')
  #----------------
  figure = figure + 1
  c = extract_ls_c(filename)
  #print c
  plt.figure(figure)
  plt.scatter(ls_iteration,c,color=next(colors))
  plt.title('c')
  plt.xlabel('Iteration')
  plt.ylabel('c')
  plt.legend(['c'],loc='upper right')
  #----------------
  figure = figure +1
  r2 = extract_ls_r2(filename)
  #print r2
  plt.figure(figure)
  plt.scatter(ls_iteration,r2,color=next(colors))
  plt.title('R^2')
  plt.xlabel('Iteration')
  plt.ylabel('R^2')
  plt.legend(['R^2'],loc='upper right')
  # ----------------------
  figure = figure +1
  alpha = extract_alpha(filename)
  #print r2
  plt.figure(figure)
  plt.scatter(ls_iteration,alpha,color=next(colors))
  plt.title('alpha')
  plt.xlabel('Iteration')
  plt.ylabel('alpha')
  plt.legend(['alpha'],loc='upper right')

  figure = figure +1
  newlr = extract_newlr(filename)
  plt.figure(figure)
  plt.scatter(ls_iteration,newlr)
  plt.title('LS_baselr')
  plt.xlabel('Iteration')
  plt.ylabel('baselr')
  plt.legend(['baselr'],loc='upper right')
  
  figure = figure +1
  new_moment = extract_newmoment(filename)
  plt.figure(figure)
  plt.scatter(ls_iteration,new_moment)
  plt.title('Automatic Moment Control')
  plt.xlabel('Iteration')
  plt.ylabel('moment')
  plt.legend(['moment'],loc='upper right')
  
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
  plot_TestLoss()
  plot_a_b_c()
  plot_lr()
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
  colors = iter(cm.rainbow(np.linspace(0, 1, 2*len(args))))
  plt.figure(1)
  for filename in args:
    testAccuracy = extract_TestAccuracy(filename, 0)
    #print testAccuracy
    plt.scatter(*zip(*testAccuracy),color=next(colors))
    #print filename + ": Final TestAccuracy is: " + testAccuracy[len(testAccuracy)-1][1]
    lsAccuracy = extract_TestAccuracy(filename, 1)
    #print lsAccuracy
    plt.scatter(*zip(*testAccuracy),color=next(colors))
  plt.title('Accuracy')
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy')
  plt.legend(legendnames,loc='lower right')
 
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
