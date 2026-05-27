from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional


from telemetry_data_generator import TimeseriesGenerator

from datasets import Dataset, Features, Sequence as HFSequence, Value


@dataclass
class TelemetryData:
    """Single univariate telemetry time series with attached information."""

    id: str
    time: np.ndarray
    timeseries: np.ndarray
    metadata: Dict[str, Any]
    statistics: Dict[str, float]

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float32)
        self.timeseries = np.asarray(self.timeseries, dtype=np.float32)

        if self.time.ndim != 1:
            raise ValueError("time must be a 1D array")
        if self.timeseries.ndim != 1:
            raise ValueError("timeseries must be a 1D array")
        if self.time.shape[0] != self.timeseries.shape[0]:
            raise ValueError("time and timeseries must have the same length")

    def to_hf_row(self) -> Dict[str, Any]:
        """Convert to a row suitable for Hugging Face Datasets."""
        domain = self.metadata.get("domain", "")
        subtype = self.metadata.get("subtype", "")

        stats = {
            "mean": float(self.statistics.get("mean", float(np.mean(self.timeseries)))),
            "std": float(self.statistics.get("std", float(np.std(self.timeseries)))),
            "min": float(self.statistics.get("min", float(np.min(self.timeseries)))),
            "max": float(self.statistics.get("max", float(np.max(self.timeseries)))),
        }

        return {
            "id": self.id,
            "timestamps": self.time,
            "values": self.timeseries,
            "domain": domain,
            "subtype": subtype,
            "statistics": stats,
        }

# Features Definition
def telemetry_features() -> Features:
    return Features(
        {
            "id": Value("string"),
            "timestamps": HFSequence(Value("float32")),
            "values": HFSequence(Value("float32")),
            "domain": Value("string"),
            "subtype": Value("string"),
            "statistics": {
                "mean": Value("float32"),
                "std": Value("float32"),
                "min": Value("float32"),
                "max": Value("float32"),
            },
        }
    )

# Generator-based Dataset Builder 
def build_telemetry_dataset(
    generation_func: Callable[..., Iterable[TelemetryData]],
    gen_kwargs: Dict[str, Any],
    total: Optional[int] = None,
    dataset_path: str = './telemetry_dataset'
) -> None:
    """
    Builds a dataset by passing the generator function and args to HF.
    Crucial: We pass the FUNCTION, not the ITERATOR, to avoid pickling errors.
    """

    # This is the adapter that HF will call.
    # It receives the kwargs we pass to from_generator below.
    def hf_generator_wrapper(**kwargs):
        # Initialize the generator now (inside the safe context)
        iterator = generation_func(**kwargs)
        
        # Wrap with tqdm for visibility
        tqdm_iter = tqdm(
            iterator, 
            total=total, 
            desc="Streaming to Arrow", 
            unit=" series"
        )
        
        for record in tqdm_iter:
            yield record.to_hf_row()

    print("Initializing HF Dataset generation...")
    
    # We pass the wrapper and the arguments separately.
    # This allows HF to hash the arguments (dict) and the function (callable),
    # which are both picklable.
    ds = Dataset.from_generator(
        generator=hf_generator_wrapper,
        features=telemetry_features(),
        gen_kwargs=gen_kwargs  # <--- Arguments go here
    )

    print(f"Generation complete. Saving to disk at {dataset_path}...")

    ds.save_to_disk(
        dataset_path, 
        max_shard_size="500MB"
    )
    print("Done.")


def load_dataset_to_hub(dataset_path:str, repo_id:str) -> None:
    """
    Load a dataset from disk and push it to the Hugging Face Hub.
    """
    ds = Dataset.load_from_disk(dataset_path)
    ds.push_to_hub(repo_id)
    print(f"Dataset pushed to Hub at {repo_id}.")


if __name__ == "__main__":

    # Setup the generator instance
    gen = TimeseriesGenerator(
            n_timeseries=1,     
            time_duration=10.0, 
            frequency=100.0,    
            seed=42,
    )
    
    total_count = 50000

    # Define the arguments dictionary
    # These are the arguments that usually go into gen.iter_telemetry()
    generation_arguments = {
        "total": total_count,
        "type_proportions": {
            "industrial": 20,
            "step": 20,
            "sine": 20,
            "random_walk": 20,
            "stock": 20,
        },
    }

    # Pass the bound method and the dict
    # Note: We pass 'gen.iter_telemetry' without ()
    build_telemetry_dataset(
        generation_func=gen.iter_telemetry, 
        gen_kwargs=generation_arguments,
        total=total_count,
        dataset_path='/home/gpiatelli/Xelerit/FactoryNet/dataset'
    )
    
    # Push to Hub
    repo_id = "Forgis/FactoryNet"
    load_dataset_to_hub(
        dataset_path='/home/gpiatelli/Xelerit/FactoryNet/dataset',
        repo_id=repo_id
    )
