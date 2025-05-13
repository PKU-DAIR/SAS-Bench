import torch
import numpy as np
from scipy.stats import spearmanr

def compute_ecs(pred_scores, ori_scores, total_scores, pred_errors, gold_errors, max_error=5):
    """
    Computes the consistency score between predicted and ground truth evaluations.

    Args:
        pred_scores: List of predicted scores for each sample
        ori_scores: List of original ground truth scores for each sample
        total_scores: Maximum possible total score for normalization
        pred_errors: Predicted error frequencies as 2D list [[freq1, freq2,...], ...] 
                    where each sublist represents error type frequencies for a sample
        gold_errors: Ground truth error frequencies with same structure as pred_errors
        max_error: Maximum possible number of errors for normalization
    """

    all_norm_scores = []
    Ln = 1e-10
    for score, max_s in zip(ori_scores, total_scores):
        norm_score = score / (max_s + Ln)
        all_norm_scores.append(norm_score)
    range1, range2 = 0, 0
    sorted_scores = sorted(all_norm_scores)
    range1 = sorted_scores[int(len(sorted_scores) * 0.33)]
    range2 = sorted_scores[int(len(sorted_scores) * 0.67)]

    ori_range_error_matrix = torch.zeros((3, max_error))
    pred_range_error_matrix = torch.zeros((3, max_error))
    for i in range(len(pred_errors)):
        pred, gold, max_s, p_errors, g_errors = pred_scores[i], ori_scores[i], total_scores[i], pred_errors[i], gold_errors[i]
        pred = pred / (max_s + Ln)
        gold = gold / (max_s + Ln)
        if pred <= range1:
            for j, freq in enumerate(p_errors):
                pred_range_error_matrix[0][j] += freq
        elif pred < range2:
            for j, freq in enumerate(p_errors):
                pred_range_error_matrix[1][j] += freq
        else:
            for j, freq in enumerate(p_errors):
                pred_range_error_matrix[2][j] += freq
        if gold <= range1:
            for j, freq in enumerate(g_errors):
                ori_range_error_matrix[0][j] += freq
        elif gold < range2:
            for j, freq in enumerate(g_errors):
                ori_range_error_matrix[1][j] += freq
        else:
            for j, freq in enumerate(g_errors):
                ori_range_error_matrix[2][j] += freq
    
    spearmans = []
    for i in range(pred_range_error_matrix.shape[0]):
        spearman_score = spearmanr(pred_range_error_matrix[i].tolist(), ori_range_error_matrix[i].tolist())
        spearmans.append(0 if str(spearman_score.correlation) == 'nan' else spearman_score.correlation)
    
    return np.mean(spearmans), spearmans
