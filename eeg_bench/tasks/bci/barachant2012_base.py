from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Barachant2012MDataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class barachant2012_base(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Right Hand vs Feet MI",
            classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            datasets=[
                Barachant2012MDataset,
            ],
            subjects_split={
                Barachant2012MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6],
                    Split.TEST: [7, 8],
                }
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
