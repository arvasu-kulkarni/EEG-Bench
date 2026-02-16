from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Cho2017MDataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split


class cho2017(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand MI",
            classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI],
            datasets=[
                Cho2017MDataset,
            ],
            subjects_split={
                Cho2017MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
