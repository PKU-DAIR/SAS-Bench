import itertools
import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def composite_distance(true_tuple, pred_tuple, weights):
    """
    Computes the weighted difference (e.g., Manhattan distance) between composite score tuples.

    Args:
        true_tuple: Ground truth scores (total_score, step1_score, step2_score)
        pred_tuple: Predicted scores (total_score, step1_score, step2_score)
        weights: Weights for each component [w_total, w_step1, w_step2], e.g., [0.5, 0.25, 0.25]

    Returns:
        Weighted composite difference between the tuples
    """
    return np.sum(weights * np.abs(np.array(true_tuple) - np.array(pred_tuple)))

def compute_QWK(X, Y, grading_scale=1):
    X = [int(x * grading_scale) for x in X]
    Y = [int(y * grading_scale) for y in Y]
    X = np.array(X)
    Y = np.array(Y)

    qwk = cohen_kappa_score(Y, X, weights='quadratic')
    return qwk

def adjusted_qwk(true_tuples, pred_tuples, weights, max_scores):
    """
    Computes weighted overall and step-wise consistent scores between ground truth and predicted evaluation tuples.

    Args:
        true_tuples: List of ground truth score tuples [(total_score, step1, ..., step10), ...]
        pred_tuples: List of predicted score tuples [(total_score, step1, ..., step10), ...]
        weights: Weight coefficients for each dimension [w_total, w_step1, ..., w_step10] 
                (should satisfy sum(weights) = 1)
        max_scores: Maximum possible scores for normalization [max_total, max_step1, ..., max_step10]
    """
    # Validate each score entry is of tuple type
    true_tuples = [tuple(t) for t in true_tuples]
    pred_tuples = [tuple(p) for p in pred_tuples]

    # Construct unified scoring composition levels (with deduplication)
    unique_tuples = sorted(list(set(true_tuples + pred_tuples)))
    n_levels = len(unique_tuples)
    tuple_to_idx = {t: i for i, t in enumerate(unique_tuples)}

    # Build observation matrix O (actual score frequencies)
    O = np.zeros((n_levels, n_levels))
    for t, p in zip(true_tuples, pred_tuples):
        i = tuple_to_idx[t]
        j = tuple_to_idx[p]
        O[i, j] += 1

    # Build expectation matrix E (theoretical score probabilities) 
    row_sums = O.sum(axis=1)
    col_sums = O.sum(axis=0)
    E = np.outer(row_sums, col_sums) / np.sum(O)

    # Construct weight matrix W (normalized squared differences)
    max_scores = np.array(max_scores)
    weights = np.array(weights)

    W = np.zeros((n_levels, n_levels))
    Ln = 1e-10
    for i, ti in enumerate(unique_tuples):
        for j, tj in enumerate(unique_tuples):
            ti = np.array(ti)
            tj = np.array(tj)
            diff = (ti - tj) / (max_scores + Ln)
            if i > len(weights) - 1:
                W[i, j] = np.sum(0)
            else:
                W[i, j] = np.sum(weights[i] * diff ** 2)

    # Compute adjusted Composite Consistency Score (CCS)
    ccs = 1 - np.sum(O * W) / np.sum(E * W)

    X = [item[0] for item in pred_tuples]
    Y = [item[0] for item in true_tuples]
    qwk = compute_QWK(X, Y)
    
    return ccs, qwk
