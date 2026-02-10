import argparse
import logging
from tqdm import tqdm
from eeg_bench.enums.split import Split
from eeg_bench.models.bci.eegembed_model import EEGEmbedModel
from eeg_bench.models.bci.jepa_model import JEPAModel
from eeg_bench.tasks.clinical import (
    AbnormalClinicalTask,
    SchizophreniaClinicalTask,
    MTBIClinicalTask,
    OCDClinicalTask,
    EpilepsyClinicalTask,
    ParkinsonsClinicalTask,
    SeizureClinicalTask,
    ArtifactBinaryClinicalTask,
    ArtifactMulticlassClinicalTask,
    SleepStagesClinicalTask,
)
from eeg_bench.tasks.bci import (
    LeftHandvRightHandMITask,
    RightHandvFeetMITask,
    LeftHandvRightHandvFeetvTongueMITask,
    FiveFingersMITask,
    bcicompiv2a,
    bcicompiv2b,
    weibo2014,
    cho2017,
    bcicompiv2a_4class,
    schirrmeister2017,
    physionetmi
)
from eeg_bench.models.clinical import (
    BrainfeaturesLDAModel as BrainfeaturesLDA,
    BrainfeaturesSVMModel as BrainfeaturesSVM,
    LaBraMModel as LaBraMClinical,
    BENDRModel as BENDRClinical,
    NeuroGPTModel as NeuroGPTClinical,
)
from eeg_bench.models.bci import (
    CSPLDAModel as CSPLDA,
    CSPSVMModel as CSPSVM,
    LaBraMModel as LaBraMBci,
    BENDRModel as BENDRBci,
    NeuroGPTModel as NeuroGPTBci,
    ReveBaseModel as ReveBaseBci,

)
from eeg_bench.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from eeg_bench.utils.utils import set_seed, save_results, get_multilabel_tasks
from eeg_bench.models.clinical.LaBraM.utils_2 import make_multilabels

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


BCI_MODEL_TASK_EMBED_DIMS = {
    "revebase": {
        "bci42a": 45056,
        "bci42b": 6144,
        "weibo2014": 122880,
        "cho2017": 98304,
        "bci42a_4class": 45056,
        "schirrmeister2017": 262144,
        "physionetmi": 96768,
    },
    "eegembed": {
        "bci42a": 45056,
        "bci42b": 6144,
        "weibo2014": 122880,
        "cho2017": 98304,
        "bci42a_4class": 45056,
        "schirrmeister2017": 147456,
        "physionetmi": 98304,
    },
    "jepa": {
        "bci42a": 256,
        "bci42b": 256,
        "weibo2014": 256,
        "cho2017": 256,
        "bci42a_4class": 256,
        "schirrmeister2017": 256,
        "physionetmi": 256,
    },
}


def _get_model_kwargs(model_key, task_key, num_classes=None):
    kwargs = {}
    if model_key in BCI_MODEL_TASK_EMBED_DIMS:
        task_dims = BCI_MODEL_TASK_EMBED_DIMS[model_key]
        if task_key in task_dims:
            kwargs["embedding_dim"] = task_dims[task_key]
    if num_classes is not None and model_key in {"revebase", "eegembed", "jepa"}:
        kwargs["num_classes"] = num_classes
    return kwargs


def benchmark(tasks, models, seed, reps, task_key=None):
    for task in tasks:
        logger.info(f"Running benchmark for task {task}")
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)

        metrics = task.get_metrics()
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        y_trues = []
        y_trains = []
        is_multilabel_task = task.name in get_multilabel_tasks()
        num_classes = len(task.classes) if hasattr(task, "classes") else None
        for model_key, model_class in models:
            print(f'running {model_key} on {task.name}')
            for i in range(reps):
                print(f"Repetition {i+1}/{reps} with seed {seed + i}")
                set_seed(seed + i)  # set seed for reproducibility
                model_kwargs = _get_model_kwargs(model_key, task_key, num_classes=num_classes)
                if is_multilabel_task:
                    num_classes = len(task.clinical_classes) + 1
                    model = model_class(
                        num_classes=num_classes,
                        num_labels_per_chunk=task.num_labels_per_chunk,
                        **model_kwargs,
                    )
                    this_y_train = make_multilabels(X_train, y_train, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                    this_y_test = make_multilabels(X_test, y_test, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                else:
                    model = model_class(**model_kwargs)
                    this_y_train = y_train
                    this_y_test = y_test

                model.fit(X_train, this_y_train, meta_train)
                y_pred = []
                for x, m in zip(X_test, meta_test):
                    y_pred.append(model.predict([x], [m]))

                models_names.append(str(model))
                results.append(y_pred)
                y_trues.append(this_y_test)
                y_trains.append(this_y_train)

        save_results(y_trains, y_trues, models_names, results, dataset_names, task.name)
        print_classification_results(
            y_trains, y_trues, models_names, results, dataset_names, task.name, metrics
        )
        generate_classification_plots(y_trains, y_trues, models_names, results, dataset_names, task.name, metrics)


def main():

    parser = argparse.ArgumentParser(
        description="Run EEG-Bench for a specific task and model."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run. Options: parkinsons, schizophrenia, mtbi, ocd, epilepsy, abnormal, sleep_stages, seizure, binary_artifact, multiclass_artifact, left_right, right_feet, left_right_feet_tongue, 5_fingers, bci42a"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model(s) to use. Comma-separated. Options: lda, svm, labram, bendr, neurogpt, revebase"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for reproducibility (default: 100)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of repetitions with different seeds for variability assessment"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all combinations of tasks and models"
    )
    args = parser.parse_args()

    # Mapping command-line strings to task classes
    tasks_map = {
        "parkinsons": ParkinsonsClinicalTask,
        "schizophrenia": SchizophreniaClinicalTask,
        "mtbi": MTBIClinicalTask,
        "ocd": OCDClinicalTask,
        "epilepsy": EpilepsyClinicalTask,
        "abnormal": AbnormalClinicalTask,
        "left_right": LeftHandvRightHandMITask,
        "right_feet": RightHandvFeetMITask,
        "left_right_feet_tongue": LeftHandvRightHandvFeetvTongueMITask,
        "5_fingers": FiveFingersMITask,
        "sleep_stages": SleepStagesClinicalTask,
        "seizure": SeizureClinicalTask,
        "binary_artifact": ArtifactBinaryClinicalTask,
        "multiclass_artifact": ArtifactMulticlassClinicalTask,
        "bci42a": bcicompiv2a,
        "bci42b": bcicompiv2b,
        "weibo2014": weibo2014,
        "cho2017": cho2017,
        "bci42a_4class": bcicompiv2a_4class,
        "schirrmeister2017": schirrmeister2017,
        "physionetmi": physionetmi
    }

    # Mapping command-line strings to model classes
    clinical_models_map = {
        "lda": BrainfeaturesLDA,
        "svm": BrainfeaturesSVM,
        "labram": LaBraMClinical,
        "bendr": BENDRClinical,
        "neurogpt": NeuroGPTClinical,
    }
    bci_models_map = {
        "lda": CSPLDA,
        "svm": CSPSVM,
        "labram": LaBraMBci,
        "bendr": BENDRBci,
        "neurogpt": NeuroGPTBci,
        "revebase": ReveBaseBci,
        "eegembed": EEGEmbedModel,
        "jepa": JEPAModel
    }

    if args.all:
        logger.info("Running all task/model combinations...")
        for task_key, task_cls in tasks_map.items():
            if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
                models_map = clinical_models_map
            else:
                models_map = bci_models_map

            task_instance = task_cls()
            model_classes = list(models_map.items())
            benchmark([task_instance], model_classes, args.seed, args.reps, task_key=task_key)

    else:
        if not args.task or not args.model:
            parser.error("Both --task and --model must be specified unless --all is used.")
        
        task_key = args.task.lower()
        model_keys = [m.strip().lower() for m in args.model.split(",") if m.strip()]
        if not model_keys:
            parser.error("At least one model must be specified.")

        if task_key not in tasks_map:
            parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())}")
        task_instance = tasks_map[task_key]()
        
        if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
            models_map = clinical_models_map
        elif task_key in ["left_right", "right_feet", "left_right_feet_tongue", "5_fingers", "bci42a", "bci42b", "weibo2014", "cho2017", "bci42a_4class", "schirrmeister2017", "physionetmi"]:
            models_map = bci_models_map
        else:
            models_map = {}
            parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())}")
        
        invalid_models = [model_key for model_key in model_keys if model_key not in models_map]
        if invalid_models:
            parser.error(
                "Invalid model(s) specified: "
                f"{', '.join(invalid_models)}. Choose from: {', '.join(models_map.keys())}"
            )

        model_instances = [(model_key, models_map[model_key]) for model_key in model_keys]
        benchmark([task_instance], model_instances, args.seed, args.reps, task_key=task_key)

if __name__ == "__main__":
    main()
