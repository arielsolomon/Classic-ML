# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
from run_classifiers_p import classify, highest_score
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import fnmatch
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")

def main(X, y):

  root = 'your project root dir'
  X = 'list of input files that is under root'
  y = []
  for file in X:

  #appending the correct binary label to your data

  if fnmatch.fnmatch(file, '*,0.*') or fnmatch.fnmatch(file, '*,N.*') or fnmatch.fnmatch(file, '*,en.*'):
          y.append(0)
      else:
          y.append(1)
  y = np.array(y).reshape(-1,1)
    
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
     
    #creating a list of loaded arrays for train and test
    
  X_train1 = np.array([np.load(root+'Database/numpys/'+arr) for arr in X_train]).reshape(len(X_train), 100*255)
  X_test1 = np.array([np.load(root+'Database/numpys/'+arr) for arr in X_test]).reshape(len(X_test), 100*255)
  st = time.time()
  aucs, scores, names, preds_prob, models, df = classify(X_train1, X_test1, y_train, y_test)

# a bunch of stats output from classification

  score_dict = {}
  auc_dict = {}
  predicted_prob_dict = dict(zip(names, preds_prob))
  score_dic = highest_score(names, scores, score_dict)
  auc_dic =   highest_score(names, aucs, auc_dict)  
  sorted_scores = sorted(score_dic.items(), key=lambda kv: kv[1],reverse=True)
  sorted_auc = sorted(auc_dic.items(), key=lambda kv: kv[1],reverse=True)
  high_accuracy_model = sorted_scores[0]
  highest_auc_model = sorted_auc[0]
  print('highest accuracy model is:', high_accuracy_model[0], 'at', "%.1f" % int(high_accuracy_model[1][0]),'%')
  print('highest auc model is:', highest_auc_model[0], 'at', "%.1f" % int(100*highest_auc_model[1][0]),'%')
  print ('end:', float((time.time()-st))/60)
  Xt = TSNE(n_components=2).fit_transform(X_test1)
  plt.scatter(Xt[:, 0], Xt[:, 1], c=y_test.astype(np.int32),
                                  alpha=0.2, cmap=plt.cm.viridis)
  return aucs, scores, names, preds_prob, models, df
if __name__ == '__main__':

  _,_,_,_,_,df = main(X, y)
  df = df.sort_values(by=['TPR', 'FPR'], ascending=False)
  df.to_csv('full_scan_classification.csv', index=False)

  ###fig definitions
  width=8
  height=5
  rows = 2
  cols = 3
  axes=[]
  fig=plt.figure()
  x_axis = np.linspace(0,255, num=255)
  for a in range(rows*cols):
      choice = np.random.choice(len(X))
      choice2 = np.random.choice(100)
      b = np.load(os.path.join(root, 'Database/numpys/', X[choice]))
      print(b.shape)
      axes.append( fig.add_subplot(rows, cols, a+1) )
      subplot_title = ("Subplot"+str(X[choice].split(',')[2]))
      axes[-1].set_title(subplot_title)  
      plt.scatter(x_axis, b[choice2,:],marker=".", pickradius=1, linewidths=0.5)
  fig.tight_layout()    
  plt.show()



  plt.figure()
  ax_n = plt.axes(projection='3d')
  ax_n.set_title('All')
  for i, arr in enumerate(test[0]):
      ax_n.plot(np.arange(255), np.repeat(i, 255), arr)
