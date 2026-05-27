from __future__ import annotations

"""
Build a single Hugging Face Dataset that combines:

- Synthetic telemetry from `TimeseriesGenerator`
- Real-world datasets:
    - shivamb/machine-predictive-maintenance-classification
    - stephanmatzka/predictive-maintenance-dataset-ai4i-2020
    - vinayak123tyagi/bearing-dataset (NASA bearings)

All records are converted to the common `TelemetryData` schema and stored with
the shared `telemetry_features()` definition from `telemetry_dataset.py`.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional


from telemetry_dataset import TelemetryData, build_telemetry_dataset  
from telemetry_data_generator import TimeseriesGenerator  
from real_timeseries_wrappers import (
    iter_ai4i_2020,
    iter_nasa_bearing,
    iter_shivamb_predictive_maintenance,
)


def iter_combined_telemetry(
    total_synthetic: int,
    type_proportions: Dict[str, float],
    include_shivamb: bool = True,
    include_ai4i: bool = True,
    include_nasa: bool = True,
    shivamb_root: Optional[str | Path] = None,
    ai4i_root: Optional[str | Path] = None,
    nasa_root: Optional[str | Path] = None,
    time_duration: float = 10.0,
    frequency: float = 100.0,
    seed: int = 42,
) -> Iterable[TelemetryData]:
    """
    Streaming generator that yields TelemetryData objects from:

    1. Synthetic telemetry (first)
    2. Real-world Kaggle datasets (shivamb, ai4i, NASA bearings)
    """
    # Synthetic series
    gen = TimeseriesGenerator(
        n_timeseries=1,
        time_duration=time_duration,
        frequency=frequency,
        seed=seed,
    )
    for ts in gen.iter_telemetry(total=total_synthetic, type_proportions=type_proportions):
        # `ts` already has standardized statistics and metadata
        yield ts

    #  Real datasets (already standardized via TelemetryData)
    if include_shivamb:
        for ts in iter_shivamb_predictive_maintenance(root=shivamb_root):
            yield ts

    if include_ai4i:
        for ts in iter_ai4i_2020(root=ai4i_root):
            yield ts

    if include_nasa:
        for ts in iter_nasa_bearing(root=nasa_root):
            yield ts


def build_combined_telemetry_dataset(
    total_synthetic: int,
    type_proportions: Dict[str, float],
    dataset_path: str = "./dataset/combined",
    include_shivamb: bool = True,
    include_ai4i: bool = True,
    include_nasa: bool = True,
    shivamb_root: Optional[str | Path] = None,
    ai4i_root: Optional[str | Path] = None,
    nasa_root: Optional[str | Path] = None,
    time_duration: float = 10.0,
    frequency: float = 100.0,
    seed: int = 42,
) -> None:
    """
    Build and save a combined HF Dataset (Arrow on disk) with synthetic + real data.

    The statistics field is already standardized for both synthetic and real
    series (mean, std, min, max) before conversion to HF rows.
    """
    gen_kwargs = {
        "total_synthetic": total_synthetic,
        "type_proportions": type_proportions,
        "include_shivamb": include_shivamb,
        "include_ai4i": include_ai4i,
        "include_nasa": include_nasa,
        "shivamb_root": shivamb_root,
        "ai4i_root": ai4i_root,
        "nasa_root": nasa_root,
        "time_duration": time_duration,
        "frequency": frequency,
        "seed": seed,
    }

    # `total=None` -> unknown global length; HF + tqdm still work,
    # and all rows share the same telemetry_features() schema.
    build_telemetry_dataset(
        generation_func=iter_combined_telemetry,
        gen_kwargs=gen_kwargs,
        total=None,
        dataset_path=dataset_path,
    )


if __name__ == "__main__":
    # Example default: 50k synthetic series, evenly split across types
    default_type_proportions = {
        "industrial": 20,
        "step": 20,
        "sine": 20,
        "random_walk": 20,
        "stock": 20,
    }
    total_synthetic = 50_000

    build_combined_telemetry_dataset(
        total_synthetic=total_synthetic,
        type_proportions=default_type_proportions,
        dataset_path="./dataset/combined",
    )

