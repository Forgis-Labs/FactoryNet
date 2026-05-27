"""AURSAD (Automatic Universal Robot Screwdriving Anomaly Detection) Adapter.

Adapter for the AURSAD dataset featuring anomaly detection data from
a UR3e collaborative robot performing screwdriving operations.

Dataset: https://zenodo.org/records/4487073
Paper: https://arxiv.org/abs/2102.01409
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

# AURSAD dataset constants
AURSAD_SAMPLING_RATE = 100  # 100 Hz

# Anomaly labels
AURSAD_LABELS = {
    0: ("normal", FaultType.NORMAL, SeverityLevel.HEALTHY),
    1: ("damaged_screw", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    2: ("extra_component", FaultType.UNKNOWN, SeverityLevel.MINOR),
    3: ("missing_screw", FaultType.UNKNOWN, SeverityLevel.MODERATE),
    4: ("damaged_thread", FaultType.UNKNOWN, SeverityLevel.SEVERE),
}

# Robot sensor channels available in AURSAD
AURSAD_CHANNELS = [
    # Robot joint positions (6 joints)
    "actual_q_0", "actual_q_1", "actual_q_2",
    "actual_q_3", "actual_q_4", "actual_q_5",
    # Robot joint velocities
    "actual_qd_0", "actual_qd_1", "actual_qd_2",
    "actual_qd_3", "actual_qd_4", "actual_qd_5",
    # Robot joint currents
    "actual_current_0", "actual_current_1", "actual_current_2",
    "actual_current_3", "actual_current_4", "actual_current_5",
    # TCP (Tool Center Point) position
    "actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2",
    "actual_TCP_pose_3", "actual_TCP_pose_4", "actual_TCP_pose_5",
    # TCP force
    "actual_TCP_force_0", "actual_TCP_force_1", "actual_TCP_force_2",
    "actual_TCP_force_3", "actual_TCP_force_4", "actual_TCP_force_5",
    # Screwdriver data
    "screwdriver_torque",
    "screwdriver_angle",
]


@register_adapter("aursad", aliases=["ur3e_screwdriver", "aursad_ur3e"])
class AURSADAdapter(BaseDatasetAdapter):
    """Adapter for AURSAD Universal Robot Screwdriving Dataset.

    The AURSAD dataset contains sensor data from a UR3e collaborative
    robot equipped with an OnRobot Screwdriver performing automated
    screwdriving tasks with various anomaly conditions.

    Anomaly types:
    - Normal operation (label 0)
    - Damaged screw (label 1)
    - Extra assembly component (label 2)
    - Missing screw (label 3)
    - Damaged thread (label 4)

    Dataset characteristics:
    - Robot: Universal Robots UR3e
    - End effector: OnRobot Screwdriver
    - Sampling rate: 100 Hz
    - Total samples: 2,045 operations
    - Format: HDF5

    Sensor channels include:
    - Joint positions, velocities, currents (6 joints)
    - TCP pose and force (6 DOF)
    - Screwdriver torque and angle

    Example:
        adapter = AURSADAdapter(data_dir="./data/aursad")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: {episode.fault_type.value}")
    """

    METADATA = DatasetMetadata(
        name="aursad",
        full_name="AURSAD - Universal Robot Screwdriving Anomaly Detection",
        description="UR3e collaborative robot screwdriving anomaly detection dataset",
        source_url="https://zenodo.org/records/4487073",
        citation=(
            "Leporowski, B., & Tola, D. (2021). AURSAD: Universal Robot "
            "Screwdriving Anomaly Detection Dataset. arXiv:2102.01409."
        ),
        license="CC BY 4.0",
        num_samples=2045,
        file_format=".h5",
        sampling_rate_hz=100,
        machine_type="collaborative_robot",
        machine_synset="M.rob.col.ur.ur3e",
        download_size_mb=6400,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        include_supplementary: bool = False,
    ):
        """Initialize AURSAD adapter.

        Args:
            data_dir: Directory containing AURSAD.h5 file
            cache_dir: Optional cache directory
            include_supplementary: Include loosening/picking samples
        """
        super().__init__(data_dir, cache_dir)
        self.include_supplementary = include_supplementary
        self._h5_file = None
        self._data_cache = None

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover HDF5 files in the data directory."""
        h5_files = list(self.data_dir.glob("*.h5"))
        h5_files.extend(list(self.data_dir.glob("*.hdf5")))
        logger.info(f"Discovered {len(h5_files)} HDF5 files in {self.data_dir}")
        return sorted(h5_files)

    def _load_h5_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from HDF5 file.

        Handles both simple HDF5 and pandas HDFStore formats.

        Args:
            file_path: Path to HDF5 file

        Returns:
            Dictionary with loaded data including 'data', 'labels', 'sample_nrs', 'columns'
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for AURSAD adapter. Install with: pip install h5py")

        with h5py.File(str(file_path), "r") as f:
            # Check if this is a pandas HDFStore format (has 'complete_data' group)
            if "complete_data" in f:
                return self._load_pandas_hdfstore(f)

            # Simple HDF5 format
            data = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][:]
                elif isinstance(f[key], h5py.Group):
                    data[key] = {}
                    for subkey in f[key].keys():
                        if isinstance(f[key][subkey], h5py.Dataset):
                            data[key][subkey] = f[key][subkey][:]

            return data

    def _load_pandas_hdfstore(self, f) -> Dict[str, Any]:
        """Load data from pandas HDFStore format.

        The AURSAD dataset is stored as a pandas DataFrame in HDF5 with:
        - axis0: column names
        - axis1: row indices
        - block*_items: column names per block
        - block*_values: data values per block

        Args:
            f: Open h5py.File object

        Returns:
            Dictionary with reconstructed data
        """
        import h5py

        group = f["complete_data"]

        # Get column names
        axis0 = [c.decode() if isinstance(c, bytes) else c for c in group["axis0"][:]]

        # Build column to block mapping
        block_mapping = {}  # column_name -> (block_values_key, index_in_block)

        for i in range(10):  # Check up to 10 blocks
            items_key = f"block{i}_items"
            values_key = f"block{i}_values"

            if items_key not in group:
                break

            items = [c.decode() if isinstance(c, bytes) else c for c in group[items_key][:]]
            for idx, col_name in enumerate(items):
                block_mapping[col_name] = (values_key, idx)

        # Extract key columns
        label_block, label_idx = block_mapping.get("label", (None, None))
        sample_nr_block, sample_nr_idx = block_mapping.get("sample_nr", (None, None))

        if label_block is None:
            logger.warning("No 'label' column found in AURSAD data")
            return {"columns": axis0, "block_mapping": block_mapping}

        labels = group[label_block][:, label_idx]
        sample_nrs = group[sample_nr_block][:, sample_nr_idx] if sample_nr_block else None

        # Store references to value blocks for lazy loading
        return {
            "format": "pandas_hdfstore",
            "columns": axis0,
            "block_mapping": block_mapping,
            "labels": labels,
            "sample_nrs": sample_nrs,
            "num_rows": len(labels),
            "file_path": str(f.filename),
        }

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse AURSAD HDF5 file into RawEpisode objects.

        Args:
            file_path: Path to HDF5 file

        Yields:
            RawEpisode objects
        """
        data = self._load_h5_data(file_path)

        # Check if this is pandas HDFStore format
        if data.get("format") == "pandas_hdfstore":
            yield from self._parse_pandas_hdfstore(data, file_path)
        # Extract samples - structure depends on HDF5 organization
        # AURSAD typically has 'data' and 'labels' arrays
        elif "data" in data and "labels" in data:
            yield from self._parse_structured_data(data, file_path)
        else:
            # Try alternative structure
            yield from self._parse_flat_data(data, file_path)

    def _parse_pandas_hdfstore(
        self,
        data: Dict[str, Any],
        file_path: Path,
    ) -> Iterable[RawEpisode]:
        """Parse pandas HDFStore format AURSAD data.

        Groups time series data by sample_nr and creates one episode per sample.

        Args:
            data: Dictionary with metadata from _load_pandas_hdfstore
            file_path: Path to HDF5 file

        Yields:
            RawEpisode objects
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required")

        labels = data["labels"]
        sample_nrs = data["sample_nrs"]
        block_mapping = data["block_mapping"]
        columns = data["columns"]

        if sample_nrs is None:
            logger.warning("No sample_nr column found, cannot segment episodes")
            return

        # Find unique samples
        unique_samples = np.unique(sample_nrs)
        logger.info(f"Found {len(unique_samples)} unique samples in AURSAD data")

        # Define channels to extract (subset of available columns for efficiency)
        channels_to_extract = [
            "actual_q_0", "actual_q_1", "actual_q_2",
            "actual_q_3", "actual_q_4", "actual_q_5",
            "actual_qd_0", "actual_qd_1", "actual_qd_2",
            "actual_qd_3", "actual_qd_4", "actual_qd_5",
            "actual_current_0", "actual_current_1", "actual_current_2",
            "actual_current_3", "actual_current_4", "actual_current_5",
            "actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2",
            "actual_TCP_force_0", "actual_TCP_force_1", "actual_TCP_force_2",
        ]

        # Open file for reading block values
        with h5py.File(str(file_path), "r") as f:
            group = f["complete_data"]

            # Cache block data
            block_cache = {}
            for col in channels_to_extract:
                if col in block_mapping:
                    block_key, _ = block_mapping[col]
                    if block_key not in block_cache:
                        block_cache[block_key] = group[block_key][:]

            for sample_nr in unique_samples:
                # Get row indices for this sample
                mask = sample_nrs == sample_nr
                row_indices = np.where(mask)[0]

                if len(row_indices) == 0:
                    continue

                # Get label for this sample (use mode/most common)
                sample_labels = labels[row_indices]
                label = int(np.bincount(sample_labels.astype(int)).argmax())

                # Skip supplementary samples if not requested (labels > 4)
                if not self.include_supplementary and label > 4:
                    continue

                # Extract channel data
                channels = []
                for col in channels_to_extract:
                    if col not in block_mapping:
                        continue

                    block_key, col_idx = block_mapping[col]
                    if block_key not in block_cache:
                        continue

                    channel_data = block_cache[block_key][row_indices, col_idx]

                    # Determine unit
                    if "current" in col:
                        unit = "A"
                    elif "force" in col:
                        unit = "N"
                    elif "_q_" in col:
                        unit = "rad"
                    elif "qd_" in col:
                        unit = "rad/s"
                    elif "pose" in col:
                        unit = "m"
                    else:
                        unit = ""

                    channels.append(SensorChannel(
                        channel_id=f"sample{int(sample_nr)}_{col}",
                        channel_type=col,
                        unit=unit,
                        data=channel_data.astype(np.float64).tolist(),
                        sampling_rate_hz=AURSAD_SAMPLING_RATE,
                    ))

                # Get fault info from label
                label_info = AURSAD_LABELS.get(label, ("unknown", FaultType.UNKNOWN, SeverityLevel.MINOR))
                fault_name, fault_type, severity = label_info

                yield RawEpisode(
                    raw_id=f"aursad_{int(sample_nr):05d}",
                    source_dataset="aursad",
                    source_file=file_path.name,
                    channels=channels,
                    fault_type=fault_type,
                    fault_location="screwdriving_operation",
                    severity=severity,
                    raw_metadata={
                        "label": label,
                        "label_name": fault_name,
                        "sample_nr": int(sample_nr),
                        "num_timesteps": len(row_indices),
                        "robot_model": "UR3e",
                        "end_effector": "OnRobot Screwdriver",
                        "sampling_rate_hz": AURSAD_SAMPLING_RATE,
                    },
                )

    def _parse_structured_data(
        self,
        data: Dict[str, Any],
        file_path: Path,
    ) -> Iterable[RawEpisode]:
        """Parse data with structured format (data, labels arrays)."""
        samples = data.get("data", [])
        labels = data.get("labels", [])

        if len(samples) == 0:
            logger.warning(f"No samples found in {file_path}")
            return

        for idx, (sample, label) in enumerate(zip(samples, labels)):
            label_int = int(label) if np.isscalar(label) else int(label[0])

            # Skip supplementary samples if not requested
            if not self.include_supplementary and label_int > 4:
                continue

            yield self._create_episode_from_sample(
                sample=sample,
                label=label_int,
                sample_idx=idx,
                file_path=file_path,
            )

    def _parse_flat_data(
        self,
        data: Dict[str, Any],
        file_path: Path,
    ) -> Iterable[RawEpisode]:
        """Parse data with flat structure (individual channel arrays)."""
        # Find the label array
        labels = data.get("label", data.get("labels", data.get("y", None)))
        if labels is None:
            logger.warning(f"No labels found in {file_path}")
            return

        # Build sample arrays from individual channels
        num_samples = len(labels)

        for idx in range(num_samples):
            label_int = int(labels[idx])

            if not self.include_supplementary and label_int > 4:
                continue

            # Extract channel data for this sample
            sample_data = {}
            for key, value in data.items():
                if key not in ("label", "labels", "y") and isinstance(value, np.ndarray):
                    if value.ndim == 2 and value.shape[0] == num_samples:
                        sample_data[key] = value[idx]
                    elif value.ndim == 3 and value.shape[0] == num_samples:
                        sample_data[key] = value[idx]

            yield self._create_episode_from_dict(
                sample_data=sample_data,
                label=label_int,
                sample_idx=idx,
                file_path=file_path,
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

        # Sample shape is typically (timesteps, features)
        if sample.ndim == 2:
            num_timesteps, num_features = sample.shape

            # Map features to channel names
            for feat_idx in range(min(num_features, len(AURSAD_CHANNELS))):
                channel_name = AURSAD_CHANNELS[feat_idx]
                channel_data = sample[:, feat_idx].astype(np.float64)

                # Determine unit based on channel type
                if "current" in channel_name:
                    unit = "A"
                elif "force" in channel_name:
                    unit = "N"
                elif "torque" in channel_name:
                    unit = "Nm"
                elif "angle" in channel_name or "_q_" in channel_name:
                    unit = "rad"
                elif "qd_" in channel_name:
                    unit = "rad/s"
                elif "pose" in channel_name:
                    unit = "m"
                else:
                    unit = ""

                channels.append(SensorChannel(
                    channel_id=f"sample{sample_idx}_{channel_name}",
                    channel_type=channel_name,
                    unit=unit,
                    data=channel_data.tolist(),
                    sampling_rate_hz=AURSAD_SAMPLING_RATE,
                ))

        # Get fault info from label
        label_info = AURSAD_LABELS.get(label, ("unknown", FaultType.UNKNOWN, SeverityLevel.MINOR))
        fault_name, fault_type, severity = label_info

        return RawEpisode(
            raw_id=f"aursad_{sample_idx:05d}",
            source_dataset="aursad",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location="screwdriving_operation",
            severity=severity,
            raw_metadata={
                "label": label,
                "label_name": fault_name,
                "sample_index": sample_idx,
                "robot_model": "UR3e",
                "end_effector": "OnRobot Screwdriver",
                "sampling_rate_hz": AURSAD_SAMPLING_RATE,
            },
        )

    def _create_episode_from_dict(
        self,
        sample_data: Dict[str, np.ndarray],
        label: int,
        sample_idx: int,
        file_path: Path,
    ) -> RawEpisode:
        """Create RawEpisode from dictionary of channel arrays."""
        channels = []

        for channel_name, channel_data in sample_data.items():
            if len(channel_data) == 0:
                continue

            data_flat = channel_data.flatten().astype(np.float64)

            # Determine unit
            if "current" in channel_name.lower():
                unit = "A"
            elif "force" in channel_name.lower():
                unit = "N"
            elif "torque" in channel_name.lower():
                unit = "Nm"
            else:
                unit = ""

            channels.append(SensorChannel(
                channel_id=f"sample{sample_idx}_{channel_name}",
                channel_type=channel_name,
                unit=unit,
                data=data_flat.tolist(),
                sampling_rate_hz=AURSAD_SAMPLING_RATE,
            ))

        label_info = AURSAD_LABELS.get(label, ("unknown", FaultType.UNKNOWN, SeverityLevel.MINOR))
        fault_name, fault_type, severity = label_info

        return RawEpisode(
            raw_id=f"aursad_{sample_idx:05d}",
            source_dataset="aursad",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location="screwdriving_operation",
            severity=severity,
            raw_metadata={
                "label": label,
                "label_name": fault_name,
                "sample_index": sample_idx,
                "robot_model": "UR3e",
                "end_effector": "OnRobot Screwdriver",
            },
        )

    def get_download_urls(self) -> List[str]:
        """Return download URLs."""
        return [
            "https://zenodo.org/records/4487073/files/AURSAD.h5",
        ]
