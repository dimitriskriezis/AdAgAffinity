import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pandas as pd



"""
labels is a binary numpy array where an element is 1 if that 
input was a true pair, otherwise the value is 0.

similarity is a numpy array of cosine similarities

returns TPR, FPR, and ROC AUC
"""
def eval(labels,similarity):
    labels = np.array(labels)
    similarity = np.array(similarity)
    # labels = labels*-1 + 1
    # similarity = abs(similarity)
    auc = roc_auc_score(labels,similarity)
    fpr, tpr, thresh = metrics.roc_curve(labels,similarity)
    return fpr,tpr,auc


def roc_plot(fpr,tpr,auc):
    fig,ax = plt.subplots()
    ax.plot(fpr,tpr)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('AUC: '+str(auc))
    plt.show()



if __name__ =='__main__':
    samples = 100
    fpr,tpr,auc = eval(np.rint(np.random.rand(samples,)),(np.random.rand(samples,)*2)-1)
    roc_plot(fpr,tpr,auc)

    

