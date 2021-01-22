# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import torch
from imageio import get_writer
import numpy as np
import os
import matplotlib.pyplot as plt


def validate_batch(model_trainer, sample, metrics):
    with torch.no_grad():
        model_trainer.model.eval()  # just make sure we are in the right mode
        x, yhat = model_trainer.get_data_from_sample(sample, require_org=False)
        # ypred = model_trainer.model(x).squeeze(1)
        ypred = model_trainer.model(x)
        loss = model_trainer.loss_fn(ypred, yhat).item()

        m_results = {}
        for metric in metrics:
            m_results[metric.name] = metric(ypred, yhat)

        # return averages and number of samples
        return m_results, loss, x.shape[0]


def validate_model(model_trainer, data_val, metrics):
    acc_metric = {m.name: 0 for m in metrics}
    acc_loss = 0
    n_samples = 0

    for sample in data_val:
        m_results, loss, n = validate_batch(model_trainer, sample, metrics)
        acc_loss += loss * n

        # accumulate statistics
        # we should weight according to batch size, because the last batch might be different
        # does it really work as expected? (issue with n. positives)
        for m in metrics:
            acc_metric[m.name] += m_results[m.name] * n
        n_samples += n

    for m in metrics:
        acc_metric[m.name] = acc_metric[m.name] / n_samples
        print(f"{m.name}: {acc_metric[m.name]}")

    print(f"val loss: {acc_loss / n_samples}")
    return acc_loss / n_samples, acc_metric


def evaluate_segmentation(cfg, model_trainer, data_loader, split, fps=6):
    result_vid = get_writer("predictions.mp4")
    indices = split.get_indices("test")
    path = cfg["data"]["eval_path"]
    acc_detected_objects = []

    def thresh(arr):
        arr[arr >= 0.75] = 1
        arr[arr <= 0.75] = 0

    for i, idx in enumerate(indices):
        with torch.no_grad():
            print("processing frame {}".format(idx))
            sample = data_loader[idx]
            x, y = model_trainer.get_data_from_single_example(sample, require_org=False)
            model_trainer.model.eval()  # just make sure we are in the right mode
            ypred = model_trainer.model(x).squeeze().cpu().numpy()
            thresh(ypred)
            frame = ypred.squeeze()

            # save test grounf truth images and predictions as npy files
            # np.save(os.path.join(path, "test_gt", str(idx) + ".npy"), y.numpy().squeeze()) # GT test image
            # np.save(os.path.join(path, "pred", str(idx) + ".npy"), ypred) # corresponding predicted image

            frame = frame.astype(np.uint8)
            frame = np.where(frame == 1, 255, 0).astype(np.uint8)
            result_vid.append_data(frame)
            # gt = y.numpy().squeeze()
            # gt = gt.astype(np.uint8)
            # gt = np.where(gt==1, 255, 0).astype(np.uint8)
            # result_vid.append_data(gt)
            # acc_detected_objects.append(correct / count)

        # print(f"\rvalidating {(i / len(indices)) * 100:.2f}%. fpr: {1 - mean(acc_detected_objects)}", end='')
    print("video created")

    result_vid.close()
    print("\ncreated video. ")
