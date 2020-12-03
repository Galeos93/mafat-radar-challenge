import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def top_1_acc(output, target):
    return top_k_acc(output, target, k=1)


def top_3_acc(output, target):
    return top_k_acc(output, target, k=3)


def top_k_acc(output, target, k):
    pred = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def auc_score_metric(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        output = roc_auc_score(target.cpu(), pred.cpu())
    return output


def average_classification_error_rate(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        fpr, tpr, _ = roc_curve(target.cpu(), pred.cpu())
        fnr = 1 - tpr
        fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        fnr_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        output = np.mean([fpr_eer, fnr_eer])
    return output
