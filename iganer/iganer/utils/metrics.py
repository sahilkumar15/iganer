import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
def auc(s,l):
    s,l=np.asarray(s),np.asarray(l)
    if len(np.unique(l))<2: return float("nan")
    return float(roc_auc_score(l,s))
def eer(s,l):
    s,l=np.asarray(s),np.asarray(l)
    if len(np.unique(l))<2: return float("nan")
    fpr,tpr,_=roc_curve(l,s); fnr=1-tpr; i=int(np.nanargmin(np.abs(fnr-fpr)))
    return float((fpr[i]+fnr[i])/2)
