#imports:
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
from run_classifiers_p import classify, highest_score, plot_tsne, plot_data
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import fnmatch
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

def main(X, y):

  root = 'your project root dir/'
  data_path = root+'data' 
  classification_path = root+'cls_res/' # tabulated classification results path
  models_path = root+'models # trained models path
  stats_img_path = root+'imgs' # images statistics path
  
  X = 'list the input files, e.g. '
  y = []
  for file in X:

  #appending the correct binary label to your data

  if fnmatch.fnmatch(file, '*,0.*') or fnmatch.fnmatch(file, '*,N.*') or fnmatch.fnmatch(file, '*,en.*'):
          y.append(0)
      else:
          y.append(1)
  y = np.array(y).reshape(-1,1)
    
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
     
    #creating a list of loaded arrays for train and test - adjuct to your project/array size
    
  X_train1 = np.array([np.load(rdata_path'+arr) for arr in X_train]).reshape(len(X_train), 100*255)
  X_test1 = np.array([np.load(data_path+arr) for arr in X_test]).reshape(len(X_test), 100*255)
                               
  # can you see any clustering (indication for possible classification):
  plot_tsne(X_test1)
                               
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
  
  # printing not a must but usful to see when running which is the leading classifier
  
  print('highest accuracy model is:', high_accuracy_model[0], 'at', "%.1f" % int(high_accuracy_model[1][0]),'%')
  print('highest auc model is:', highest_auc_model[0], 'at', "%.1f" % int(100*highest_auc_model[1][0]),'%')
  print ('end:', float((time.time()-st))/60)

  return aucs, scores, names, preds_prob, models, df
if __name__ == '__main__':

  _,_,_,_,_,df = main(X, y)
  df = df.sort_values(by=['TPR', 'FPR'], ascending=False)
  df.to_csv(root+classification_path+'clasification_res.csv', index=False)

  # some more info: 3d data plot for different data features
  plt.figure()
  ax_n = plt.axes(projection='3d')
  ax_n.set_title('All')
  for i, arr in enumerate(test[0]):
      ax_n.plot(np.arange(255), np.repeat(i, 255), arr)
