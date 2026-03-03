import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    auc = 0
    for i in range(len(fpr) - 1):
        auc += (tpr[i] + tpr[i+1]) * (fpr[i+1] - fpr[i]) / 2
    
    return auc