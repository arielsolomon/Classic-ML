from datetime import datetime
#from Results import Results
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier,  RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import warnings
import time
from sklearn.metrics import roc_curve, auc, recall_score, make_scorer, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings("ignore")

# datetime object containing current date and time
now = str(datetime.now())[:-10]
root = 'project root folder/'
model_path = root+'models/'
res_path = root+'cls_results/'
img_path = root+'images'


def add_values_in_dict(sample_dict, key, list_of_values):
    """Append multiple values to a key in the given dictionary"""
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict
    
def Roc(y_true, y_pred, name):
    #ROC_AUC plots and optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    sensitivity = 1 - fpr
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : " , "%.2f" %roc_auc)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_threshold=optimal_threshold
    pred=[]
    for i in range(len(y_pred)):
    
        if y_pred[i]>=optimal_threshold:
            pred.append(1)
        else:
            pred.append(0)
    Conf = sns.heatmap(confusion_matrix(y_true,np.asarray(pred)), annot=True, cmap='BuPu', cbar=False,  fmt='g')
    plt.title(name+' text_features '+now)
    plt.savefig(root+'cls_results/images/'+'text_feature_heatmap'+name+' '+now+'.png')
    plt.show()

    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name+' text_features'+now)
    plt.legend()
    plt.savefig(root+'images/'+'text_feature_Roc'+name+' '+now+'.png')
    plt.show()

def fp_10(y_true, y_pred, fp=0.1):
    """
    Calculates with a given fp rate

    Parameters
    ----------
    fp : Ffloat
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmin(np.abs(fpr -fp))
    score=0
    optimal_threshold = thresholds[optimal_idx]
    pred=[]
    for i in range(len(y_pred)):
    
        if y_pred[i]>=optimal_threshold:
            pred.append(1)
        else:
            pred.append(0)
        if pred[i]==y_true[i]:
            score+=1
    conf = sns.heatmap(confusion_matrix(y_true,np.asarray(pred)), annot=True, cmap='BuPu', cbar=False,  fmt='g')
    print(confusion_matrix(y_true,np.asarray(pred)))
    return optimal_threshold,fpr[optimal_idx],tpr[optimal_idx],score/len(y_true)
        

def classify(X, y, X_train, X_test, y_train, y_test):
    names = ["KNearest Neighbors", "Linear SVM", "RBF SVM", 
             "Gaussian Process","Decision Tree", "Random Forest"
             , "Neural Net", "AdaBoost", "XgBoost"
              ,"Naive Bayes"]#, "QDA", "LDA"]
    class_weights = compute_class_weight('balanced', np.unique(y[:,0]), y[:,0]) 
    classifiers = [
        KNeighborsClassifier(n_neighbors=5),
        SVC(kernel="linear", C=0.025, probability=True, class_weight='balanced'),
        SVC(gamma=2, C=1, probability=True, class_weight='balanced'),
        GaussianProcessClassifier(multi_class = 'one_vs_one'),
        DecisionTreeClassifier(class_weight='balanced',max_depth=5),
        RandomForestClassifier(max_depth=3,n_estimators=100,class_weight='balanced'),
        MLPClassifier(alpha=1, max_iter=1000,early_stopping=False),
        AdaBoostClassifier(learning_rate = 0.01),
        GradientBoostingClassifier(n_estimators=100, min_samples_split=3, min_samples_leaf=1, max_depth=7, random_state=65446),
        GaussianNB()]
        #QuadraticDiscriminantAnalysis(), 
        #LinearDiscriminantAnalysis()]
    
    scores =[]
    aucs = []
    fprs = []
    tprs = []
    models = []
    data = []
    columns = ['Classifier', 'AUC', 'TPR', 'FPR']
    df = pd.DataFrame(data, columns = columns)  
    i = 1
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    res_cv = []
    scoring = make_scorer(roc_auc_score)
    # scoring = make_scorer(recall_score, pos_label=0)
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        names = names
        clf.fit(X_train, y_train)
        cv_results = cross_validate(clf, X, y, scoring = scoring, cv=cv)
        file_name = model_root+name+' text_features '+now+'.sav'
        pickle.dump(clf, open(file_name, 'wb'))
        score = 100*clf.score(X_test, y_test)
        preds = clf.predict(X_test)
        preds_prob = clf.predict_proba(X_test)        #metrics.plot_roc_curve (clf.predict, X_test, y_test)
        fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)
        sensitivity = recall_score(y_test, preds)
        specificity = 1-fpr[1]
        Roc(y_test, preds_prob[:,1], name)
        # fp_10(y_test, preds_prob[:,1])
        df = df.append({'Classifier': name, 'AUC': "%.2f" % auc(fpr, tpr), 'TPR': "%.2f" % tpr[1],
                        'FPR':"%.2f" % fpr[1],'Specificity': "%.2f" % specificity,
                        'Sensitivity': "%.2f" % sensitivity },  ignore_index=True)
        # print(name , 'AUC = ', "%.2f" % metrics.auc(fpr, tpr))
        # print('FPR = ', "%.2f" % fpr[1])
        # print('TPR = ', "%.2f" % tpr[1])
        
        aucs.append(auc(fpr, tpr))
        scores.append(score)
        fprs.append(fpr)
        tprs.append(tpr)
        res_cv.append(cv_results)
        print(name , 'AUC = ', "%.2f" % auc(fpr, tpr), ', ', 'TPR = ', "%.2f" % tpr[1],
              ', ', 'FPR = ', "%.2f" % fpr[1], 'Specificity = ', "%.2f" % specificity)


    return aucs, scores, names, preds_prob, df, res_cv

def highest_score(names, params, dictionary):
    for name, param in zip(names, params):
        dictionary.setdefault(name, []).append(param)
        
         
  

    return dictionary



