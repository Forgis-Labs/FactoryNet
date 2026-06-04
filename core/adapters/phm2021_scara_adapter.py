"""PHM 2021 SCARA Robot Adapter.

Adapter for the PHM Europe 2021 Data Challenge dataset featuring
fault detection on a 4-axis SCARA robot fuse feeder system.

Dataset: https://github.com/PHME-Datachallenge/Data-Challenge-2021
PHM Society: https://data.phmsociety.org/2021-phm-conference-data-challenge/
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base_adapter import (
    BaseDatasetAdapter,
    DatasetMetadata,
    FaultType,
    RawEpisode,
    SensorChannel,
    SeverityLevel,
)
from .registry import register_adapter

logger = logging.getLogger(__name__)

# PHM 2021 SCARA dataset constants
PHM2021_SAMPLING_RATE = 10  # Approximate - varies by signal

# Fault types in the challenge
PHM2021_FAULT_LABELS = {
    0: ("healthy", FaultType.NORMAL, SeverityLevel.HEALTHY),
    1: ("robot_arm_mode", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    2: ("pressure_leakage", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    3: ("conveyor_speed", FaultType.UNKNOWN, SeverityLevel.MINOR),
    4: ("combined_1_2", FaultType.UNKNOWN, SeverityLevel.SEVERE),
    5: ("combined_1_3", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    6: ("combined_2_3", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    7: ("combined_1_2_3", FaultType.UNKNOWN, SeverityLevel.SEVERE),
    8: ("intermittent_1", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    9: ("intermittent_2", FaultType.UNKNOWN, SeverityLevel.MODERATE),
}

# Signal categories from dataset documentation
PHM2021_SIGNALS = {
    "machine_health": [
        "Pressure", "Vacuum", "FuseHeatSlope",
    ],
    "environment": [
        "Temperature", "Humidity",
    ],
    "system": [
        "CPUTemperature", "ProcessMemoryConsumption",
    ],
    "robot": [
        "RobotPosition", "RobotVelocity", "RobotAcceleration",
        "MotorCurrent", "MotorTorque",
    ],
}


@register_adapter("phm2021_scara", aliases=["phm2021", "scara_robot"])
class PHM2021SCARAAdapter(BaseDatasetAdapter):
    """Adapter for PHM 2021 SCARA Robot Fault Detection Dataset.

    The PHM Europe 2021 Data Challenge dataset features a 4-axis SCARA
    robot fuse feeder system with various fault conditions for
    classification.

    System components:
    - 4-axis SCARA robot fuse feeder
    - Electrically powered feeding movement
    - Pneumatically powered barrier
    - Thermal camera (382×288 pixel)
    - Visual camera (1280×1024 pixel)

    Fault types:
    - Healthy operation (label 0)
    - Robot arm operating mode modification (label 1)
    - Pressure leakage in pneumatic system (label 2)
    - Conveyor belt speed alteration (label 3)
    - Various combinations (labels 4-7)
    - Intermittent faults (labels 8-9)

    Dataset characteristics:
    - Robot: 4-axis SCARA
    - Signals: 50 time series
    - Experiment duration: 1-3 hours each
    - Total experiments: 70

    Example:
        adapter = PHM2021SCARAAdapter(data_dir="./data/phm2021")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: {episode.fault_type.value}")
    """

    METADATA = DatasetMetadata(
        name="phm2021_scara",
        full_name="PHM Europe 2021 SCARA Robot Data Challenge",
        description="4-axis SCARA robot fault detection dataset from PHM 2021 challenge",
        source_url="https://data.phmsociety.org/2021-phm-conference-data-challenge/",
        citation=(
            "Biggio, L., Russi, M., Bigdeli, S., Kastanis, I., Giordano, D., "
            "& Gagar, D. (2021). PHME 2021 Data Challenge. Swiss Centre for "
            "Electronics and Microtechnology (CSEM)."
        ),
        license="CC BY-NC-SA 4.0",
        num_samples=70,  # Number of experiments
        file_format=".csv",
        sampling_rate_hz=10,  # Varies by signal
        machine_type="industrial_robot",
        machine_synset="M.rob.ind.scara.phm2021",
        download_size_mb=500,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        segment_duration_s: Optional[float] = None,
    ):
        """Initialize PHM 2021 SCARA adapter.

        Args:
            data_dir: Directory containing dataset files
            cache_dir: Optional cache directory
            segment_duration_s: If set, segment long experiments into episodes
        """
        super().__init__(data_dir, cache_dir)
        self.segment_duration_s = segment_duration_s

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover data files in the dataset."""
        files = []

        # Check for CSV files
        files.extend(list(self.data_dir.rglob("*.csv")))

        # Check for Parquet files
        files.extend(list(self.data_dir.rglob("*.parquet")))

        # Filter metadata files
        files = [f for f in files if "label" not in f.name.lower() and "readme" not in f.name.lower()]

        logger.info(f"Discovered {len(files)} data files in {self.data_dir}")
        return sorted(files)

    def _load_labels(self) -> Dict[str, int]:
        """Load experiment labels from labels file."""
        labels = {}

        # Try common label file locations
        label_files = [
            self.data_dir / "labels.csv",
            self.data_dir / "train_labels.csv",
            self.data_dir / "y_train.csv",
        ]

        for label_file in label_files:
            if label_file.exists():
                try:
                    data = np.loadtxt(str(label_file), delimiter=",", skiprows=1, dtype=str)
                    if data.ndim == 1:
                        labels["0"] = int(float(data[-1]))
                    else:
                        for row in data:
                            exp_id = str(row[0])
                            label = int(float(row[-1]))
                            labels[exp_id] = label
                    logger.info(f"Loaded {len(labels)} labels from {label_file}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load labels from {label_file}: {e}")

        return labels

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a data file into RawEpisode objects.

        Args:
            file_path: Path to data file

        Yields:
            RawEpisode objects
        """
        # Load labels
        labels = self._load_labels()

        # Extract experiment ID from filename
        exp_id = self._extract_experiment_id(file_path)
        label = labels.get(exp_id, labels.get(file_path.stem, 0))

        # Load data
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            data, columns = self._load_csv(file_path)
        elif suffix == ".parquet":
            data, columns = self._load_parquet(file_path)
        else:
            logger.warning(f"Unsupported format: {suffix}")
            return

        if data is None or len(data) == 0:
            return

        # Create episodes
        if self.segment_duration_s:
            yield from self._create_segmented_episodes(
                data, columns, label, file_path, exp_id
            )
        else:
            yield self._create_episode(data, columns, label, file_path, exp_id)

    def _extract_experiment_id(self, file_path: Path) -> str:
        """Extract experiment ID from filename."""
        stem = file_path.stem

        # Try various patterns
        patterns = [
            r"exp[_-]?(\d+)",
            r"experiment[_-]?(\d+)",
            r"run[_-]?(\d+)",
            r"(\d+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, stem, re.IGNORECASE)
            if match:
                return match.group(1)

        return stem

    def _load_csv(self, file_path: Path) -> Tuple[Optional[np.ndarray], List[str]]:
        """Load CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
            data = df.values
            return data, columns
        except ImportError:
            # Fall back to numpy
            data = np.loadtxt(str(file_path), delimiter=",", skiprows=1)
            # Generate generic column names
            columns = [f"signal_{i}" for i in range(data.shape[1] if data.ndim == 2 else 1)]
            return data, columns
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, []

    def _load_parquet(self, file_path: Path) -> Tuple[Optional[np.ndarray], List[str]]:
        """Load Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            return df.values, df.columns.tolist()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, []

    def _create_episode(
        self,
        data: np.ndarray,
        columns: List[str],
        label: int,
        file_path: Path,
        exp_id: str,
    ) -> RawEpisode:
        """Create a single RawEpisode from data."""
        channels = []

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Detect time column
        time_col_idx = None
        for idx, col in enumerate(columns):
            if col.lower() in ("time", "timestamp", "t"):
                time_col_idx = idx
                break

        # Create channels for each signal
        for col_idx, col_name in enumerate(columns):
            if col_idx == time_col_idx:
                continue

            col_data = data[:, col_idx].astype(np.float64)

            # Skip if all NaN
            if np.all(np.isnan(col_data)):
                continue

            # Determine unit based on signal name
            unit = self._infer_unit(col_name)

            # Determine category
            category = self._categorize_signal(col_name)

            channels.append(SensorChannel(
                channel_id=f"{exp_id}_{col_name}",
                channel_type=col_name,
                unit=unit,
                data=col_data.tolist(),
                sampling_rate_hz=PHM2021_SAMPLING_RATE,
            ))

        # Get fault info
        fault_info = PHM2021_FAULT_LABELS.get(
            label, ("unknown", FaultType.UNKNOWN, SeverityLevel.MINOR)
        )
        fault_name, fault_type, severity = fault_info

        return RawEpisode(
            raw_id=f"phm2021_{exp_id}",
            source_dataset="phm2021_scara",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location="scara_fuse_feeder",
            severity=severity,
            raw_metadata={
                "experiment_id": exp_id,
                "label": label,
                "fault_name": fault_name,
                "robot_type": "4-axis SCARA",
                "num_signals": len(channels),
                "duration_s": len(data) / PHM2021_SAMPLING_RATE,
            },
        )

    def _create_segmented_episodes(
        self,
        data: np.ndarray,
        columns: List[str],
        label: int,
        file_path: Path,
        exp_id: str,
    ) -> Iterable[RawEpisode]:
        """Create segmented episodes from long experiment."""
        samples_per_segment = int(self.segment_duration_s * PHM2021_SAMPLING_RATE)
        total_samples = len(data)

        segment_idx = 0
        for start in range(0, total_samples - samples_per_segment + 1, samples_per_segment):
            end = start + samples_per_segment
            segment_data = data[start:end]

            episode = self._create_episode(
                segment_data, columns, label, file_path, f"{exp_id}_seg{segment_idx}"
            )
            episode.raw_id = f"phm2021_{exp_id}_seg{segment_idx:04d}"
            episode.raw_metadata["segment_index"] = segment_idx
            episode.raw_metadata["segment_start_s"] = start / PHM2021_SAMPLING_RATE

            yield episode
            segment_idx += 1

    def _infer_unit(self, signal_name: str) -> str:
        """Infer unit from signal name."""
        name_lower = signal_name.lower()

        if "pressure" in name_lower or "vacuum" in name_lower:
            return "bar"
        elif "temperature" in name_lower:
            return "°C"
        elif "humidity" in name_lower:
            return "%"
        elif "current" in name_lower:
            return "A"
        elif "velocity" in name_lower:
            return "m/s"
        elif "position" in name_lower:
            return "m"
        elif "memory" in name_lower:
            return "MB"
        elif "torque" in name_lower:
            return "Nm"
        else:
            return ""

    def _categorize_signal(self, signal_name: str) -> str:
        """Categorize signal by type."""
        name_lower = signal_name.lower()

        for category, signals in PHM2021_SIGNALS.items():
            for sig in signals:
                if sig.lower() in name_lower:
                    return category

        return "other"

    def get_download_urls(self) -> List[str]:
        """Return download URLs."""
        return [
            "https://github.com/PHME-Datachallenge/Data-Challenge-2021",
            "https://data.phmsociety.org/2021-phm-conference-data-challenge/",
        ]
