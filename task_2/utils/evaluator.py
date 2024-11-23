# encoding: utf-8

"""
Implements an evaluator class that scores predictions and
ground truths and returns various metrics as properties.
"""

from sklearn.metrics import matthews_corrcoef, f1_score
from evaluation import specificity


class Evaluator:
    """
    Implements an evaluator class that scores predictions and
    ground truths and returns various metrics as properties.
    """
    def __init__(self) -> None:

        self.labels: list = []
        self.preds: list = []

    def add(self, labels: list[int], preds: list[int]) -> None:
        """
        Add labels and predictions.

        :param labels: list of integers
        :param preds: list of integers
        """

        labels = [labels] if not isinstance(labels, list) else labels
        preds = [preds] if not isinstance(preds, list) else preds

        if len(labels) != len(preds):
            raise Warning(f'Labels and predictions must have the same length, but '
                          f'received {len(labels)} labels and {len(preds)} predictions.')

        self.labels.extend(labels)
        self.preds.extend(preds)

    def reset(self) -> None:
        """ Deletes stored predictions and labels. """
        self.labels = []
        self.preds = []

    @property
    def f1_score(self) -> float:
        return f1_score(self.labels, self.preds, average='micro', zero_division=1)

    @property
    def matthew_cc(self) -> float:
        return matthews_corrcoef(self.labels, self.preds)

    @property
    def specificity(self) -> float:
        return specificity(self.labels, self.preds)

