from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ExperimentParams:
    random_seed: Optional[int] = 0
    multiply_by_inputs: bool = True
    noise_scale: float = 0.1
    noise_ratio: float = 1.0


@dataclass
class DatasetString:
    dataset: str


@dataclass
class ModelParams:
    input_size: int
    tot_units: int
    input_scaling: float
    inter_scaling: float
    spectral_radius: float
    leaky: float
    connectivity_recurrent: int
    connectivity_input: int
    connectivity_inter: int


def instance_wise_weighted_average(importances, hidden_states):
    normalized_importances = np.abs(importances) / np.sum(
        np.abs(importances), axis=1, keepdims=True
    )
    weighted_average = (normalized_importances * hidden_states).sum(
        axis=1, keepdims=False
    )
    return weighted_average, normalized_importances


def add_noise(data, noise_ratio=0.1, noise_scale=0.1):
    n, m, _ = data.shape
    m_noise = int(m // noise_ratio - m)
    total_length = m + m_noise
    modified_data = np.full((n, total_length, 1), np.nan)
    mask = np.zeros((n, total_length, 1))

    max_start_index = total_length - m
    start_indices = np.random.randint(0, max_start_index + 1, size=n)

    for i in range(n):
        ts = data[i, :, 0]
        start_index = start_indices[i]
        mask[i, start_index : start_index + m, 0] = 1
        std_dev = ts.std() * noise_scale
        mean = ts.mean()
        modified_ts = np.random.normal(mean, std_dev, (1, total_length, 1))
        modified_ts[0, start_index : start_index + m, 0] = ts
        modified_data[i, :, :] = modified_ts

    return modified_data, mask
