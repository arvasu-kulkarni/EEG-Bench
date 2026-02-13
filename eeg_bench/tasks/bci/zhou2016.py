from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Zhou2016MDataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split


class zhou2016(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand MI",
            classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI],
            datasets=[
                Zhou2016MDataset,
            ],
            subjects_split={
                Zhou2016MDataset: {
                    Split.TRAIN: [],
                    Split.TEST: list(range(1, 5)),
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
