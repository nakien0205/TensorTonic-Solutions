import numpy as np

def roc_curve(y_true, y_score):
    # 1. Sort by score descending
    zipped = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    sorted_scores = [x[0] for x in zipped]
    sorted_y_true = [x[1] for x in zipped]

    total_pos = sum(y_true) # More efficient than .count(1) for large lists
    total_neg = len(y_true) - total_pos

    # 2. Initialize
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [np.inf]
    tp = 0
    fp = 0

    # 3. Calculate with handling for ties
    for i in range(len(sorted_y_true)):
        if sorted_y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        
        # KEY FIX: Only add a point if this is the LAST occurrence of this score
        # OR if it's the very last element in the list.
        if i + 1 == len(sorted_y_true) or sorted_scores[i] != sorted_scores[i+1]:
            fpr.append(fp / total_neg)
            tpr.append(tp / total_pos)
            thresholds.append(sorted_scores[i])

    return np.array(fpr), np.array(tpr), np.array(thresholds)