from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Weibo2014MDataset,
    Schirrmeister2017MDataset,
    PhysionetMIMDataset,
    BCICompIV2aMDataset,
    Barachant2012MDataset,
    Zhou2016MDataset,
    Faller2012MDataset,
    Scherer2015MDataset,
    Kaya2018Dataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class PhysionetMIBaseTask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand vs Feet vs Hands MI",
            classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI, BCIClasses.BOTH_HANDS_MI],
            datasets=[
                PhysionetMIMDataset,
            ],
            subjects_split={
                PhysionetMIMDataset: {
                    Split.TRAIN: list(range(1, 88)),
                    Split.TEST: list(range(89, 92)) + list(range(93, 100)) + list(range(101, 110)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
