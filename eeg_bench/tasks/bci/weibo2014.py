from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Weibo2014MDataset
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split


class weibo2014(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand MI",
            classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI],
            datasets=[
                Weibo2014MDataset
            ],
            subjects_split={
                Weibo2014MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9, 10],
                }
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
