from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Faller2012MDataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class faller2012_base(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Right Hand vs Feet MI",
            classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            datasets=[
                Faller2012MDataset,
            ],
            subjects_split={
                Faller2012MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 12)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
