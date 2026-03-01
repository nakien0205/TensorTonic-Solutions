import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    fpr = [0]
    tpr = [0]

    TP = 0
    FN = 0
    FP = y_true.count(1)
    TN = y_true.count(0)

    zipped_pairs = zip(y_score, y_true)

    sorted_pairs = sorted(zipped_pairs, key=lambda pair: pair[0], reverse=True)
    result = [item[1] for item in sorted_pairs]

    y_score.sort(reverse=True)
    thresholds = [np.inf]

    for i in range(len(y_score)):
        if result[i] == 1:
            TP += 1
        else:
            FN += 1

        if i + 1 == len(y_true) or y_score[i] != y_score[i+1]:
            fpr.append(FN / FP)
            tpr.append(TP / TN)
            thresholds.append(y_score[i])
    
    return (fpr, tpr, thresholds)