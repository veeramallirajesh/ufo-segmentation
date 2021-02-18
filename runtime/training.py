# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import datetime
import time
from pathlib import Path
import os
import sys

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from runtime.evaluation import validate_model
from runtime.utils import iou_loss


class ModelTrainer:
    def __init__(
        self,
        model_path="saved_models",
        use_gpu=False,
        loss_fn=iou_loss,
    ):
        self.use_gpu = use_gpu
        self.model = None
        self.optimizer = None
        self.tracker_client = None
        self.model_path = None

        self.loss_fn = loss_fn  # BCELoss()
        self.metrics = []
        self.models_dir = Path(model_path)

    def set_model(self, model, save_path=None):
        self.model = model.cuda() if self.use_gpu else model
        self.optimizer = Adam(model.parameters())
        self.model_path = self._make_model_path() if save_path is None else save_path

    def _make_model_path(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        model_dir = now.strftime("%d-%m-%Y %H:%M:%S")
        new_dir = self.models_dir / model_dir
        new_dir.mkdir()
        return new_dir

    def add_metric(self, metric):
        self.metrics.append(metric)

    def get_data_from_sample(self, sample):
        if self.use_gpu:
            # return sample["image"].cuda(), sample[self.label_key].cuda()
            return sample[0].cuda(), sample[1].cuda()
        else:
            return sample[0], sample[1]

    def get_data_from_single_example(self, sample):
        img, mask = sample[0].unsqueeze(0), sample[1].unsqueeze(0)
        return img, mask

    def make_train_step(self):
        def train_step(x, y):
            self.model.train()  # just make sure we are in the right mode now
            ypred = self.model(x)

            # encourage the model to find non zero regions
            # correction = ypred.sum() / ypred.nelement()
            # loss -= torch.log(correction)

            loss = self.loss_fn(ypred.squeeze(), y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return train_step

    def train_model(
        self, data_train, data_val, n_epochs=1, break_early=-1, save_every_n_epochs=1
    ):
        train_step = self.make_train_step()
        tensor_board_writer = SummaryWriter()
        acc_loss = 0

        # if self.tracker_client:
        #     self.tracker_client.on_train_begin(
        #         n_epochs, data_train.batch_size, len(data_train)
        #     )

        current_batch = 0
        for epoch in range(n_epochs):
            start = time.time()
            for sample in data_train:
                current_batch += 1
                # break here to get results faster during debugging
                # if -1 < break_early < current_batch:
                #     print(f"----------- debugging stopped after {break_early} batches. validating")
                #     validate_model(self, data_val, self.metrics)
                #     return None

                x, yhat = self.get_data_from_sample(sample)
                loss = train_step(x, yhat)
                acc_loss += loss
                print(f"{epoch + 1}-{current_batch}: {loss}")

                # if self.tracker_client:
                #     self.tracker_client.on_train_batch_end(current_batch, epoch)

            end = time.time()
            print(f"----------- epoch {epoch + 1} done in {end-start:.2f}s. validating")
            current_batch = 0
            loss_val, m_val = validate_model(self, data_val, self.metrics)
            # print(f"After epochs:{epoch + 1}:validation loss is:{loss_val}, validation acc:{m_val}")

            for m in self.metrics:
                tensor_board_writer.add_scalar(m.name, m_val[m.name], epoch)

            tensor_board_writer.add_scalar(
                "Loss/Train", acc_loss / len(data_train), epoch + 1
            )
            tensor_board_writer.add_scalar("Loss/Val", loss_val, epoch + 1)
            acc_loss = 0

            if (epoch + 1) % save_every_n_epochs == 0:
                self.save(epoch + 1)

        # self.tracker_client.on_train_end()
        print("done.")
        return self.save()

    def save(self, epoch=None):
        print("saving model")
        model_key = f"model_epoch_{epoch}.pt" if epoch else "model.pt"
        torch.save(self.model, self.model_path / model_key)
        return self.model

    def load(self, path=None):
        if not path:  # load latest model
            candidates = {}
            for subdir in self.models_dir.glob("*/"):
                candidates[
                    datetime.datetime.strptime(subdir.name, "%d-%m-%Y %H:%M:%S")
                ] = subdir

            if len(candidates) == 0:
                raise RuntimeError("no models so far")

            latest = sorted(candidates.keys())[-1]  # Loads latest model file
            print(f"loading lastest entry from {latest}")

            path = candidates[latest] / "model.pt"

            if not os.path.exists(path):
                print(
                    "The model is not completely trained. Please select appropriate trained model."
                )
                sys.exit()

        loaded_model = (
            torch.load(path) if self.use_gpu else torch.load(path, map_location="cpu")
        )
        self.set_model(loaded_model, path)
