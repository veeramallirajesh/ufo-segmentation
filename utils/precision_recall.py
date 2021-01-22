# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
# precision-recall curve and f1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

precision_avg = []
recall_avg = []


def calc_precision_recall(pred, gt):
    """precision = TP / TP + FP
    Recall = TP / TP + FN"""

    TP = np.sum(pred * gt)
    FP = np.sum(pred * (1 - gt))
    FN = np.sum((1 - pred) * gt)
    TN = np.sum((1 - pred) * (1 - gt))

    return TP, FP, FN, TN


def precision_recall(gt_path, pred_path, is_npy=False):
    # List of ground truth masks
    gt_list = sorted(os.listdir(gt_path))
    pred_list = sorted(os.listdir(pred_path))
    # Containers for true positives and false positive rates
    precision_scores = []
    recall_scores = []
    prob_thresh = np.linspace(0, 1, num=101)
    for thresh in prob_thresh:
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives
        for gt_seg in gt_list:
            if is_npy:
                # Prediction and ground-truth npy files output
                gt = np.load(os.path.join(gt_path, gt_seg))
                pred = np.load(os.path.join(pred_path, gt_seg))
            else:
                # code to open PIL Images and convert them to numpy arrays
                pass

            pred = np.where(pred >= thresh, 1, 0)
            tp, fp, fn, tn = calc_precision_recall(pred, gt)
            TP = TP + tp
            FP = FP + fp
            FN = FN + fn
            TN = TN + tn
        try:
            precision = TP / (TP + FP)
        except:
            precision = 1
        try:
            recall = TP / (TP + FN)
        except:
            recall = 1
        precision_scores.append(precision)
        recall_scores.append(recall)
        print(
            "At Threshold {:.2f}, Precision: {:.4f}, Recall: {:.4f}".format(
                thresh, precision, recall
            )
        )
    return precision_scores, recall_scores, prob_thresh


def plot_pr_curve(p_scores, r_scores, prob_thresh):
    print("plotting Precision-Recall curve")
    assert isinstance(r_scores, list) and isinstance(
        p_scores, list
    ), "All scores should be of type list"

    # Plot PR curve
    fig, ax = plt.subplots(figsize=(6, 6))
    # no_skill = len(testy[testy == 1]) / len(testy)
    ax.plot([0.2, 1], [0, 0], linestyle="--", label="No Skill")
    ax.plot(r_scores, p_scores, marker=".", label="Model-Output")
    # for idx in range(0, len(p_scores), 2):
    #     plt.text(r_scores[idx], p_scores[idx], "{:.2f}".format(prob_thresh[idx]))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="center left")
    plt.title("PR Curve", loc="center")
    plt.show()


if __name__ == "__main__":
    gt_path = "/home/rveeramalli/ufo-segmentation/data_v1/eval/test_gt"
    # path to ONNX output
    pred_path = "/home/rveeramalli/ufo-segmentation/data_v1/eval/pred"
    print("Generating Precision-Recall Scores for Model Output")
    p_scores, r_scores, prob_thresh = precision_recall(gt_path, pred_path, is_npy=True)
    plot_pr_curve(p_scores, r_scores, prob_thresh)
