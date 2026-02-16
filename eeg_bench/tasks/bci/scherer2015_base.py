from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Scherer2015MDataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class scherer2015(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Right Hand vs Feet MI",
            classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            datasets=[
                Scherer2015MDataset,
                #Kaya2018Dataset,
            ],
            subjects_split={
                Scherer2015MDataset: {
                    Split.TRAIN: list(range(1, 7)),
                    Split.TEST: list(range(7, 9)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
