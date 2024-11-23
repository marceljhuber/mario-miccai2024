# encoding: utf-8

"""
PyTorch Trainer base class.
"""

from ..RetFound.feature_extractor import RetFoundFeatureExtractor
from evaluator import Evaluator
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import os


class BaseTrainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: Optional[nn.Module],
                 optimizer: Optional[optim.Optimizer],
                 feature_extractor_oct: Optional[RetFoundFeatureExtractor],
                 feature_extractor_fun: Optional[RetFoundFeatureExtractor],
                 scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau],
                 tracker,
                 train_loader: Optional[DataLoader],
                 valid_loader: Optional[DataLoader],
                 ) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.feature_extractor_oct = feature_extractor_oct
        self.feature_extractor_fun = feature_extractor_fun
        self.scheduler = scheduler
        self.tracker = tracker

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def state_dict(self):
        """ Current state of learning. """
        return {
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    @property
    def device(self):
        """ Device of the (first) model parameters. """
        return next(self.model.parameters()).device

    def forward(self,
                oct_t: torch.Tensor,
                oct_t1: torch.Tensor,
                fundus: Optional[torch.Tensor],
                y: torch.Tensor,
                df: Optional[dict],
                metric: callable,
                evaluator: Optional[Evaluator] = None,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def data_iterator(self, data_loader: DataLoader,
                      metric: callable,
                      epoch: int,
                      train: bool = True,
                      evaluator: Optional[Evaluator] = None) -> list[float]:
        """
        Implements train or validation loop.

        :param epoch: int epoch
        :param data_loader: DataLoader loading the dataset
        :param metric: callable to compute loss
        :param train: bool true for training, false for validation, testing
        :param evaluator: Evaluator used for evaluating the validation predictions
        :return: list[float] containing the loss values
        """

        # Set model mode
        if train:
            self.model.train()
        else:
            self.model.eval()

        errs = []
        with torch.set_grad_enabled(train):
            for data in data_loader:

                # Get errors from prediction
                err, _ = self.forward(*data, metric=metric, evaluator=evaluator)
                errs.append(err.item())

                # Update parameters if training
                if train:
                    self.optimizer.zero_grad()
                    err.backward()
                    self.optimizer.step()

        return errs

    def __call__(self, save_path: str, save_every: int = 1, num_epochs: int = 10) -> None:
        """
        Execute the model training and validation loop.

        :param save_path: str path to save the model
        :param save_every: int frequency when the model is being saved
        :param num_epochs: int number of iterations over train data
        """

        for epoch in range(1, num_epochs + 1):

            # train
            train_evaluator = Evaluator()

            train_errs = self.data_iterator(
                self.train_loader,
                metric=self.criterion,
                train=True,
                evaluator=train_evaluator,
                epoch=epoch,
            )
            train_loss_avg = sum(train_errs)/len(train_errs)

            # validate
            valid_evaluator = Evaluator()

            valid_errs = self.data_iterator(
                self.valid_loader,
                metric=self.criterion,
                train=False,
                evaluator=valid_evaluator,
                epoch=epoch,
            )
            valid_loss_avg = sum(valid_errs) / len(valid_errs)

            wandb_log = {
                'train/loss': train_loss_avg,
                'train/f1_score': train_evaluator.f1_score,
                'train/matthew_cc': train_evaluator.matthew_cc,
                'train/specificity': train_evaluator.specificity,

                'valid/loss': valid_loss_avg,
                'valid/f1_score': valid_evaluator.f1_score,
                'valid/matthew_cc': valid_evaluator.matthew_cc,
                'valid/specificity': valid_evaluator.specificity,

                'params/learning_rate': self.optimizer.param_groups[-1]['lr'],
            }

            print(f'{train_loss_avg=} | {valid_loss_avg=}')

            # Log the training
            if self.tracker:
                self.tracker.log(wandb_log)

            self.scheduler.step(metrics=valid_loss_avg)

            if save_every > 0 and epoch % save_every == 0:
                torch.save(
                    self.state_dict(),
                    os.path.join(save_path, f'epoch_{epoch}.pth'),
                )
