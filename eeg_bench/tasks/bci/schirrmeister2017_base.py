from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Schirrmeister2017MDataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split


class schirrmeister2017_base(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand vs Feet MI",
            classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            datasets=[
                Schirrmeister2017MDataset,
            ],
            subjects_split={
                Schirrmeister2017MDataset: {
                    Split.TRAIN: list(range(1, 12)),
                    Split.TEST: list(range(12, 15)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
