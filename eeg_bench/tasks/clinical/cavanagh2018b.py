from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    Cavanagh2018bDataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class cavanagh2018b(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="parkinsons_clinical",
            clinical_classes = [ClinicalClasses.PARKINSONS],
            datasets = [
                Cavanagh2018bDataset,
            ],
            subjects_split={
                Cavanagh2018bDataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 31, 32, 33, 34, 37, 38, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                    Split.TEST: [20, 21, 22, 23, 24, 25, 26, 27, 30, 35, 36, 39, 40, 42, 45, 55],
                },
            }
        )

    def get_data(
        self, split: Split
    ):
        data = [
            dataset(
                target_class=self.clinical_classes[0],
                subjects=self.subjects_split[dataset][split],
            ).get_data(split)
            for dataset in self.datasets
        ]

        X, y, meta = map(list, zip(*data))
        for m in meta:
            m["task_name"] = self.name
        return X, y, meta

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
