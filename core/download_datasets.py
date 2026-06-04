from __future__ import annotations

"""
Utility functions to download the real-world datasets used in FactoryNet.

We rely on `kagglehub` which will cache datasets under `~/.cache/kagglehub`.
Each helper returns the local path where the dataset is stored so that
wrappers can load and convert them to the internal `TelemetryData` format.
"""

from typing import Dict

import kagglehub


KAGGLE_DATASETS: Dict[str, str] = {
    "shivamb_machine_predictive_maintenance": "shivamb/machine-predictive-maintenance-classification",
    "ai4i_2020": "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
    "nasa_bearing": "vinayak123tyagi/bearing-dataset",
}


def download_dataset(key: str) -> str:
    """
    Download a single Kaggle dataset (if not already cached) and
    return the local path.

    Parameters
    ----------
    key:
        One of:
        - "shivamb_machine_predictive_maintenance"
        - "ai4i_2020"
        - "nasa_bearing"
    """
    if key not in KAGGLE_DATASETS:
        raise KeyError(f"Unknown dataset key: {key!r}. "
                       f"Valid keys: {sorted(KAGGLE_DATASETS)}")

    dataset_id = KAGGLE_DATASETS[key]
    path = kagglehub.dataset_download(dataset_id)
    return path


def download_all() -> Dict[str, str]:
    """
    Download all configured Kaggle datasets and return a mapping
    from dataset key to local path.
    """
    return {key: download_dataset(key) for key in KAGGLE_DATASETS}


if __name__ == "__main__":
    paths = download_all()
    for key, path in paths.items():
        print(f"{key}: {path}")
