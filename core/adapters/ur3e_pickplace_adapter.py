"""UR3e Pick and Place Anomaly Detection Adapter.

Adapter for the Industrial Robotic Arm Pick and Place dataset from Kaggle,
featuring velocity-based anomaly detection on a UR3e collaborative robot.

Dataset: https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-anomaly-detection
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

# Dataset constants
UR3E_PP_SAMPLING_RATE = 125  # Typical UR robot sampling rate

# Anomaly labels based on joint velocity modifications
UR3E_PP_LABELS = {
    "normal": (FaultType.NORMAL, SeverityLevel.HEALTHY),
    "anomaly_high": (FaultType.UNKNOWN, SeverityLevel.MINOR),  # Higher velocity
    "anomaly_low": (FaultType.UNKNOWN, SeverityLevel.MINOR),   # Lower velocity
}

# UR3e joint names
UR3E_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


@register_adapter("ur3e_pickplace", aliases=["ur3e_anomaly", "industrial_robotic_arm"])
class UR3ePickPlaceAdapter(BaseDatasetAdapter):
    """Adapter for UR3e Pick and Place Anomaly Detection Dataset.

    This Kaggle dataset contains multimodal data from a UR3e collaborative
    robot performing pick-and-place operations with velocity-based anomalies.

    Anomaly types (created by modifying joint velocity):
    - Normal: Default velocity (1.05 rad/s)
    - Anomaly High: Velocity higher than default
    - Anomaly Low: Velocity lower than default

    Dataset characteristics:
    - Robot: Universal Robots UR3e
    - Task: Pick and place operations
    - Data type: Multimodal time series
    - Size: ~2.8 GB
    - Format: CSV/Parquet

    Example:
        adapter = UR3ePickPlaceAdapter(data_dir="./data/ur3e_pickplace")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: {episode.fault_type.value}")
    """

    METADATA = DatasetMetadata(
        name="ur3e_pickplace",
        full_name="Industrial Robotic Arm - UR3e Pick and Place Anomaly Detection",
        description="UR3e collaborative robot pick-and-place anomaly detection dataset",
        source_url="https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-anomaly-detection",
        citation=(
            "Kayan, H. (2023). Industrial Robotic Arm - Pick and Place Dataset. "
            "Kaggle. CC BY-SA 4.0."
        ),
        license="CC BY-SA 4.0",
        num_samples=1000,  # Approximate
        file_format=".csv",
        sampling_rate_hz=125,
        machine_type="collaborative_robot",
        machine_synset="M.rob.col.ur.ur3e",
        download_size_mb=2800,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
    ):
        """Initialize UR3e Pick and Place adapter.

        Args:
            data_dir: Directory containing dataset files
            cache_dir: Optional cache directory
        """
        super().__init__(data_dir, cache_dir)

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover data files in the dataset directory."""
        # Check for various file formats
        files = []
        for pattern in ["*.csv", "*.parquet", "*.pkl"]:
            files.extend(list(self.data_dir.rglob(pattern)))

        # Filter out metadata/info files
        files = [f for f in files if "readme" not in f.name.lower()]

        logger.info(f"Discovered {len(files)} data files in {self.data_dir}")
        return sorted(files)

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a data file into RawEpisode objects.

        Args:
            file_path: Path to data file

        Yields:
            RawEpisode objects
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            yield from self._parse_csv(file_path)
        elif suffix == ".parquet":
            yield from self._parse_parquet(file_path)
        elif suffix == ".pkl":
            yield from self._parse_pickle(file_path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")

    def _parse_csv(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse CSV file."""
        try:
            import pandas as pd
        except ImportError:
            # Fall back to numpy
            data = np.loadtxt(str(file_path), delimiter=",", skiprows=1)
            yield self._create_episode_from_array(data, file_path, 0)
            return

        df = pd.read_csv(file_path)
        yield from self._parse_dataframe(df, file_path)

    def _parse_parquet(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            yield from self._parse_dataframe(df, file_path)
        except ImportError:
            logger.error("pandas required for Parquet files")
            return

    def _parse_pickle(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse pickle file."""
        import pickle

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            yield from self._parse_dict_data(data, file_path)
        elif hasattr(data, "values"):  # DataFrame
            yield from self._parse_dataframe(data, file_path)

    def _parse_dataframe(self, df: Any, file_path: Path) -> Iterable[RawEpisode]:
        """Parse pandas DataFrame."""
        # Detect label column
        label_col = None
        for col in ["label", "anomaly", "class", "target", "y"]:
            if col in df.columns:
                label_col = col
                break

        # Detect timestamp/index column
        time_col = None
        for col in ["timestamp", "time", "t", "index"]:
            if col in df.columns:
                time_col = col
                break

        # Get numeric columns for sensor data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)
        if time_col and time_col in numeric_cols:
            numeric_cols.remove(time_col)

        # Check if data is segmented by operation/episode
        if "episode" in df.columns or "operation" in df.columns:
            group_col = "episode" if "episode" in df.columns else "operation"
            for group_id, group_df in df.groupby(group_col):
                yield self._create_episode_from_df_segment(
                    group_df, numeric_cols, label_col, file_path, int(group_id)
                )
        else:
            # Single episode per file
            yield self._create_episode_from_df_segment(
                df, numeric_cols, label_col, file_path, 0
            )

    def _create_episode_from_df_segment(
        self,
        df: Any,
        numeric_cols: List[str],
        label_col: Optional[str],
        file_path: Path,
        episode_idx: int,
    ) -> RawEpisode:
        """Create RawEpisode from DataFrame segment."""
        channels = []

        for col in numeric_cols:
            data = df[col].values.astype(np.float64)

            # Determine unit based on column name
            col_lower = col.lower()
            if "position" in col_lower or "angle" in col_lower or "_q" in col_lower:
                unit = "rad"
            elif "velocity" in col_lower or "_qd" in col_lower:
                unit = "rad/s"
            elif "current" in col_lower:
                unit = "A"
            elif "force" in col_lower:
                unit = "N"
            elif "torque" in col_lower:
                unit = "Nm"
            else:
                unit = ""

            channels.append(SensorChannel(
                channel_id=f"ep{episode_idx}_{col}",
                channel_type=col,
                unit=unit,
                data=data.tolist(),
                sampling_rate_hz=UR3E_PP_SAMPLING_RATE,
            ))

        # Determine fault type from label
        fault_type = FaultType.NORMAL
        severity = SeverityLevel.HEALTHY

        if label_col:
            label = df[label_col].iloc[0] if len(df) > 0 else 0
            if isinstance(label, str):
                label_key = label.lower()
            else:
                label_key = "anomaly_high" if label == 1 else ("anomaly_low" if label == 2 else "normal")

            if label_key in UR3E_PP_LABELS:
                fault_type, severity = UR3E_PP_LABELS[label_key]
            elif label != 0:
                fault_type = FaultType.UNKNOWN
                severity = SeverityLevel.MINOR

        return RawEpisode(
            raw_id=f"ur3e_pp_{file_path.stem}_{episode_idx:05d}",
            source_dataset="ur3e_pickplace",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location="pick_place_operation",
            severity=severity,
            raw_metadata={
                "episode_index": episode_idx,
                "robot_model": "UR3e",
                "task": "pick_and_place",
                "num_timesteps": len(df),
            },
        )

    def _create_episode_from_array(
        self,
        data: np.ndarray,
        file_path: Path,
        episode_idx: int,
    ) -> RawEpisode:
        """Create RawEpisode from numpy array."""
        channels = []

        if data.ndim == 2:
            num_cols = data.shape[1]
            for col_idx in range(num_cols):
                channels.append(SensorChannel(
                    channel_id=f"ep{episode_idx}_channel{col_idx}",
                    channel_type=f"sensor_{col_idx}",
                    unit="",
                    data=data[:, col_idx].astype(np.float64).tolist(),
                    sampling_rate_hz=UR3E_PP_SAMPLING_RATE,
                ))

        return RawEpisode(
            raw_id=f"ur3e_pp_{file_path.stem}_{episode_idx:05d}",
            source_dataset="ur3e_pickplace",
            source_file=file_path.name,
            channels=channels,
            fault_type=FaultType.UNKNOWN,
            fault_location="pick_place_operation",
            severity=SeverityLevel.MINOR,
            raw_metadata={
                "episode_index": episode_idx,
                "robot_model": "UR3e",
            },
        )

    def _parse_dict_data(
        self,
        data: Dict[str, Any],
        file_path: Path,
    ) -> Iterable[RawEpisode]:
        """Parse dictionary data structure."""
        # Common dict structures
        if "X" in data and "y" in data:
            X = data["X"]
            y = data["y"]
            for idx, (sample, label) in enumerate(zip(X, y)):
                yield self._create_episode_from_sample(
                    sample, int(label), idx, file_path
                )
        elif "data" in data:
            samples = data["data"]
            labels = data.get("labels", data.get("y", [0] * len(samples)))
            for idx, (sample, label) in enumerate(zip(samples, labels)):
                yield self._create_episode_from_sample(
                    sample, int(label), idx, file_path
                )

    def _create_episode_from_sample(
        self,
        sample: np.ndarray,
        label: int,
        sample_idx: int,
        file_path: Path,
    ) -> RawEpisode:
        """Create RawEpisode from sample array."""
        channels = []

        if isinstance(sample, np.ndarray) and sample.ndim == 2:
            for col_idx in range(sample.shape[1]):
                channels.append(SensorChannel(
                    channel_id=f"sample{sample_idx}_ch{col_idx}",
                    channel_type=f"sensor_{col_idx}",
                    unit="",
                    data=sample[:, col_idx].astype(np.float64).tolist(),
                    sampling_rate_hz=UR3E_PP_SAMPLING_RATE,
                ))

        fault_type = FaultType.NORMAL if label == 0 else FaultType.UNKNOWN
        severity = SeverityLevel.HEALTHY if label == 0 else SeverityLevel.MINOR

        return RawEpisode(
            raw_id=f"ur3e_pp_{sample_idx:05d}",
            source_dataset="ur3e_pickplace",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location="pick_place_operation",
            severity=severity,
            raw_metadata={
                "label": label,
                "sample_index": sample_idx,
                "robot_model": "UR3e",
            },
        )
