from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    Cavanagh2018aDataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class cavanagh2018a(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="parkinsons_clinical",
            clinical_classes = [ClinicalClasses.PARKINSONS],
            datasets = [
                Cavanagh2018aDataset
            ],
            subjects_split={
                Cavanagh2018aDataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 28, 29, 30, 31, 34, 35, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                    Split.TEST: [17, 18, 19, 20, 21, 22, 23, 24, 27, 32, 33, 36, 37, 39, 42, 52],
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
