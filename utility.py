import os
import glob
import random

import numpy as np
import torch
from torch import Tensor
from typing import Callable, List, Dict, Tuple

def euclidean_distance(v1: Tensor, v2: Tensor) -> float:
    """Compute the Euclidean distance between two tensors."""
    return (v1 - v2).norm().item()


def pairwise_distances(vectors: List[Tensor]) -> Dict[int, Dict[int, float]]:
    """Compute pairwise Euclidean distances between all vectors."""
    n = len(vectors)
    distances = {i: {i: 0.0} for i in range(n)}
    for i in range(n):
        for j in range(i):
            d = euclidean_distance(vectors[i], vectors[j])
            distances[i][j] = distances[j][i] = d
    return distances


def flatten(lst: list) -> list:
    """
    Flat list of list.
    For instance, [[0, 1], [2, 3], [4]] becomes [0, 1, 2, 3, 4]
    """
    return [item for sublist in lst for item in sublist]

def save(data: dict, name: str, experiment_id: int, experiment_folder:str) -> None:
    """
    Saves the dictionary to a file in the experiment_folder directory.

    Creates the experiment_folder directory if it doesn't exist and saves the provided
    data to a file named based on the experiment_id and name.
    """
    os.makedirs(experiment_folder, exist_ok=True)
    run_name = str(experiment_id) + name
    torch.save(data, experiment_folder + '/' + run_name + '.pt')

def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility in Python, NumPy, and PyTorch.
    """
    # Fixing the seed for Python
    random.seed(seed)
    # Fixing the seed for NumPy
    np.random.seed(seed)
    # Fixing the seed for PyTorch (CPU)
    torch.manual_seed(seed)
    # Fixing the seed for PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring the reproducibility of deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable heuristics-based optimization
    torch.backends.cudnn.benchmark = False

class Statistics:
    """
    Stores and manages statistics during training or evaluation.
    """
    def __init__(self):
        self.data = {}  # Dictionary to store data

    def append(self, key, value):
        """
        Appends a value to the list corresponding to the given key.
        
        setdefault(key, []): It checks if the key already exists in the dictionary self.data:
            - If the key exists, it returns the corresponding value (which is expected to be a list)
            - If the key does not exist, it creates a new entry and initializes with an empty list ([])
        """
        self.data.setdefault(key, []).append(value)
    
    def __getitem__(self, key):
        """
        Returns:
        List of values associated with the key.
        """
        return self.data[key]