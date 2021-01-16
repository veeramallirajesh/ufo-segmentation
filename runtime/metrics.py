# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import torch
from torch.nn.functional import one_hot

from runtime.utils import iou_pytorch, move_dim, multiclass_iou_score


def threshold(ypred, yhat, thresh):
    ypred = ypred > thresh
    yhat = yhat > thresh

    return ypred, yhat


def threshold_softmax(ypred, class_dim=1):
    return ypred.argmax(class_dim)


class AccuracyMetric:
    def __init__(self, num_classes=None, thresh=0.5):
        self.name = "accuracy"
        self.short_name = "acc."
        self.thresh = thresh
        self.num_classes = num_classes

    def __call__(self, ypred, yhat, *args, **kwargs):
        if self.num_classes is not None:
            ypred = threshold_softmax(ypred)
            return ((ypred == yhat).byte().sum().item()) / ypred.nelement()

        ypred, yhat = threshold(ypred, yhat, self.thresh)
        correct_el = (ypred == yhat).byte()
        correct = correct_el.sum().item()

        return correct / ypred.nelement()


class IOUMetric:
    def __init__(self, thresh=0.75): #TODO thresh was set to 0.5
        self.name = "iou"
        self.short_name = "iou"
        self.thresh = thresh

    def __call__(self, ypred, yhat, *args, **kwargs):
        ypred, yhat = threshold(ypred, yhat, self.thresh)
        return iou_pytorch(ypred.byte(), yhat.byte()).item()


class IOUMultiClassMetric:
    def __init__(self, nclasses, thresh=0.5):
        self.name = "iou"
        self.short_name = "iou"
        self.thresh = thresh
        self.nclasses = nclasses

    def __call__(self, ypred, yhat, *args, **kwargs):
        return multiclass_iou_score(ypred, yhat, self.nclasses).item()


class AreaCoveredMetric:
    def __init__(self, class_dim=None, thresh=0.5):
        self.name = "area covered"
        self.short_name = "area"
        self.thresh = thresh
        self.class_dim = class_dim

    def __call__(self, ypred, yhat, *args, **kwargs):
        if self.class_dim is not None:
            ypred = threshold_softmax(ypred)
            return torch.clamp(ypred, 0, 1).sum().item() / ypred.nelement()
        else:
            ypred, yhat = threshold(ypred, yhat, self.thresh)
            return ypred.byte().sum().item() / ypred.nelement()


class MSEMetric:
    def __init__(self):
        self.name = "mean squared error"
        self.short_name = "mse"

    def __call__(self, ypred, yhat, *args, **kwargs):
        return torch.mean((ypred - yhat) ** 2)


class MAEMetric:
    def __init__(self):
        self.name = "mean absolute error"
        self.short_name = "mae"

    def __call__(self, ypred, yhat, *args, **kwargs):
        return torch.mean(torch.abs(ypred - yhat))


class FalseNegativeRateMetric:
    def __init__(self, thresh=0.5, num_classes=None, smooth=1e-6):
        self.name = "mean false negative rate"
        self.short_name = "fn rate"
        self.thresh = thresh
        self.smooth = smooth
        self.num_classes = num_classes

    def __call__(self, ypred, yhat, *args, **kwargs):
        if self.num_classes is not None:
            # attention this assumes that if we detect anything its sufficient, can be the wrong class however
            ypred = torch.clamp(threshold_softmax(ypred), 0, 1)
            yhat = torch.clamp(yhat, 0, 1)

            positives = yhat.sum().item()
            false_negatives = (yhat - (ypred * yhat)).sum().item()
        else:
            ypred, yhat = threshold(ypred, yhat, self.thresh)
            positives = yhat.float().sum().item()
            false_negatives = (yhat ^ (ypred & yhat)).float().sum().item()

        return false_negatives / (positives + self.smooth)


class TruePositiveMetric:
    def __init__(self, thresh=0.5, num_classes=None, smooth=1e-6):
        self.name = "mean total positives"
        self.short_name = "tot. pos"
        self.thresh = thresh
        self.smooth = smooth
        self.num_classes = num_classes

    def __call__(self, ypred, yhat, *args, **kwargs):
        if self.num_classes is not None:
            yhat = torch.clamp(yhat, 0, 1)
            positives = yhat.sum()
        else:
            ypred, yhat = threshold(ypred, yhat, self.thresh)
            positives = yhat.float().sum()

        return positives.item()


class FalseNegativesMetric:
    def __init__(self, thresh=0.5, num_classes=None, smooth=1e-6):
        self.name = "mean total false negatives"
        self.short_name = "tot. fn"
        self.thresh = thresh
        self.smooth = smooth
        self.num_classes = num_classes

    def __call__(self, ypred, yhat, *args, **kwargs):
        if self.num_classes is not None:
            ypred = torch.clamp(threshold_softmax(ypred), 0, 1)
            yhat = torch.clamp(yhat, 0, 1)

            false_negatives = (yhat - (ypred * yhat)).sum()
        else:
            ypred, yhat = threshold(ypred, yhat, self.thresh)
            false_negatives = (yhat ^ (ypred & yhat)).float().sum()

        return false_negatives.item()
