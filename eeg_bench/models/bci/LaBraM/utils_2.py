import numpy as np
from typing import List
import logging

def n_unique_labels(task_name: str) -> int:
    """
    Get the number of unique labels for the given task.
    Args:
        task_name (str): The name of the task.
    Returns:
        int: The number of unique labels.
    """
    if task_name == "Left Hand vs Right Hand MI":
        return 2
    elif task_name == "Right Hand vs Feet MI":
        return 2
    elif task_name == "Left Hand vs Right Hand vs Feet MI":
        return 3
    elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
        return 4
    elif task_name == "Left Hand vs Right Hand vs Feet vs Hands MI":
        return 4
    elif task_name == "Five Fingers MI":
        return 5
    else:
        # for all clinical tasks, we assume binary classification
        return 2

def map_label(label, task_name: str) -> int:
    """
    Map the label to a numerical value.
    Args:
        label (str): The label to map.
        task_name (str): The name of the task.
    Returns:
        int: The mapped numerical value.
    """
    if label is None:
        raise ValueError("Label cannot be None")

    if isinstance(label, (int, np.integer, float, np.floating)):
        label_int = int(label)
        if task_name == "Left Hand vs Right Hand MI":
            if label_int in (0, 1):
                return label_int
            if label_int in (1, 2):
                return label_int - 1
        elif task_name == "Right Hand vs Feet MI":
            if label_int in (0, 1):
                return label_int
            if label_int in (2, 3):
                return label_int - 2
        elif task_name == "Left Hand vs Right Hand vs Feet MI":
            if label_int in (0, 1, 2):
                return label_int
            if label_int in (1, 2, 3):
                return label_int - 1
            if label_int in (1, 2, 4):
                return {1: 0, 2: 1, 4: 2}[label_int]
        elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
            if label_int in (0, 1, 2, 3):
                return label_int
            if label_int in (1, 2, 3, 4):
                return label_int - 1
        elif task_name == "Left Hand vs Right Hand vs Feet vs Hands MI":
            if label_int in (0, 1, 2, 3):
                return label_int
            if label_int in (1, 2, 3, 4):
                return label_int - 1
        elif task_name == "Five Fingers MI":
            if label_int in (0, 1, 2, 3, 4):
                return label_int
            if label_int in (1, 2, 3, 4, 5):
                return label_int - 1
        raise ValueError("Invalid numeric label: ", label)

    if task_name == "Left Hand vs Right Hand MI":
        if label == "left_hand":
            return 0
        if label == "right_hand":
            return 1
    elif task_name == "Right Hand vs Feet MI":
        if label == "right_hand":
            return 0
        if label == "feet":
            return 1
    elif task_name == "Left Hand vs Right Hand vs Feet MI":
        if label == "left_hand":
            return 0
        if label == "right_hand":
            return 1
        if label == "feet":
            return 2
    elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
        if label == "left_hand":
            return 0
        if label == "right_hand":
            return 1
        if label == "feet":
            return 2
        if label == "tongue":
            return 3
    elif task_name == "Left Hand vs Right Hand vs Feet vs Hands MI":
        if label == "left_hand":
            return 0
        if label == "right_hand":
            return 1
        if label == "feet":
            return 2
        if label == "hands":
            return 3
    elif task_name == "Five Fingers MI":
        if label == "thumb":
            return 0
        if label == "index finger":
            return 1
        if label == "middle finger":
            return 2
        if label == "ring finger":
            return 3
        if label == "little finger":
            return 4
    elif task_name in ["parkinsons_clinical", "schizophrenia_clinical", "mtbi_clinical", "ocd_clinical", "epilepsy_clinical", "abnormal_clinical", "sleep_stages_clinical", "seizure_clinical", "binary_artifact_clinical", "multiclass_artifact_clinical", "cavanagh2018a"]:
        if label == "parkinsons":
            return 1
    else:
        raise ValueError("Invalid label: ", label)
        
def reverse_map_label(label: int, task_name: str) -> str:
    """
    Reverse map the numerical label to its string representation.
    Args:
        label (int): The numerical label to reverse map.
        task_name (str): The name of the task.
    Returns:
        str: The string representation of the label.
    """
    if task_name == "Left Hand vs Right Hand MI":
        return "left_hand" if label == 0 else "right_hand"
    elif task_name == "Right Hand vs Feet MI":
        return "right_hand" if label == 0 else "feet"
    elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
        return ["left_hand", "right_hand", "feet", "tongue"][label]
    elif task_name == "Left Hand vs Right Hand vs Feet vs Hands MI":
        return ["left_hand", "right_hand", "feet", "hands"][label]
    elif task_name == "Left Hand vs Right Hand vs Feet MI":
        return ["left_hand", "right_hand", "feet"][label]
    elif task_name == "Five Fingers MI":
        return ["thumb", "index finger", "middle finger", "ring finger", "little finger"][label]
    else:
        raise ValueError("Invalid label: ", label)

def calc_class_weights(labels: List[np.ndarray], task_name: str) -> List[float]:
    """
    Calculate class weights for the given labels.
    Args:
        labels (List[np.ndarray]): List of numpy arrays containing the labels.
    Returns:
        List[float]: List of weights for each class.
    """
    # Flatten the list of labels
    all_labels = np.concatenate(labels)

    # Map labels to integers
    all_labels = np.array([map_label(label, task_name) for label in all_labels], dtype=np.int64)
    
    # Count the occurrences of each class
    class_counts = np.bincount(all_labels)
    
    # Calculate the total number of samples
    total_samples = len(all_labels)
    
    # Calculate class weights for each class (0 weight if class count is 0)
    n_classes = len(class_counts)
    class_weights = [np.float32(total_samples / (n_classes * count)) if count > 0 else np.float32(0.0) for count in class_counts]
    
    return class_weights
