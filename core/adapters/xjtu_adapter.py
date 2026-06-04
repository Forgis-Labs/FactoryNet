"""XJTU-SY Bearing Dataset Adapter.

Adapter for the Xi'an Jiaotong University bearing dataset,
featuring run-to-failure accelerated life test data.

Dataset: https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
Download: https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK
"""
from __future__ import annotations

import logging
import re
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

# XJTU-SY dataset constants
XJTU_SAMPLING_RATE = 25600  # 25.6 kHz
XJTU_SAMPLES_PER_FILE = 32768  # 1.28 seconds per file

# Operating conditions
XJTU_CONDITIONS = {
    "35Hz12kN": {"rpm": 2100, "load_kn": 12.0},  # 35 Hz = 2100 RPM
    "37.5Hz11kN": {"rpm": 2250, "load_kn": 11.0},
    "40Hz10kN": {"rpm": 2400, "load_kn": 10.0},
}

# Bearing failure modes (documented final failure states)
XJTU_BEARING_FAILURES = {
    # Condition 1: 35Hz 12kN
    "Bearing1_1": FaultType.OUTER_RACE,
    "Bearing1_2": FaultType.OUTER_RACE,
    "Bearing1_3": FaultType.OUTER_RACE,
    "Bearing1_4": FaultType.CAGE,
    "Bearing1_5": FaultType.COMBINED,  # Inner + Outer + Cage
    # Condition 2: 37.5Hz 11kN
    "Bearing2_1": FaultType.INNER_RACE,
    "Bearing2_2": FaultType.OUTER_RACE,
    "Bearing2_3": FaultType.CAGE,
    "Bearing2_4": FaultType.OUTER_RACE,
    "Bearing2_5": FaultType.COMBINED,  # Inner + Outer
    # Condition 3: 40Hz 10kN
    "Bearing3_1": FaultType.OUTER_RACE,
    "Bearing3_2": FaultType.COMBINED,  # Ball + Outer
    "Bearing3_3": FaultType.INNER_RACE,
    "Bearing3_4": FaultType.INNER_RACE,
    "Bearing3_5": FaultType.OUTER_RACE,
}


@register_adapter("xjtu_sy", aliases=["xjtu", "xjtu_bearing"])
class XJTUAdapter(BaseDatasetAdapter):
    """Adapter for XJTU-SY Bearing Run-to-Failure Dataset.

    The XJTU-SY dataset contains complete life cycle data from
    accelerated bearing life tests. Key characteristics:
    - 15 bearings tested to failure
    - 3 operating conditions
    - Run-to-failure data showing degradation progression

    Operating conditions:
    - Condition 1: 35 Hz (2100 RPM), 12 kN radial load
    - Condition 2: 37.5 Hz (2250 RPM), 11 kN radial load
    - Condition 3: 40 Hz (2400 RPM), 10 kN radial load

    Sensor channels:
    - Horizontal acceleration
    - Vertical acceleration

    Dataset characteristics:
    - Sampling rate: 25.6 kHz
    - Samples per file: 32,768 (1.28 seconds)
    - Recording interval: 1 minute

    Example:
        adapter = XJTUAdapter(data_dir="./data/xjtu_sy")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: {episode.fault_type.value}")
    """

    METADATA = DatasetMetadata(
        name="xjtu_sy",
        full_name="XJTU-SY Bearing Run-to-Failure Dataset",
        description="Run-to-failure bearing dataset from Xi'an Jiaotong University",
        source_url="https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets",
        citation=(
            "B. Wang et al., 'A Hybrid Prognostics Approach for Estimating Remaining "
            "Useful Life of Rolling Element Bearings,' IEEE Trans. Reliability, 2020."
        ),
        license="Academic Use",
        num_samples=15000,  # Approximate (varies by bearing life)
        file_format=".csv",
        sampling_rate_hz=25600,
        machine_type="bearing_test_rig",
        machine_synset="M.tst.bea.xjtu",
        download_size_mb=5000,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        include_rul_estimate: bool = True,
    ):
        """Initialize XJTU adapter.

        Args:
            data_dir: Directory containing dataset files
            cache_dir: Optional cache directory
            include_rul_estimate: Include RUL estimation in metadata
        """
        super().__init__(data_dir, cache_dir)
        self.include_rul_estimate = include_rul_estimate
        self._bearing_file_counts: Dict[str, int] = {}

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover all CSV files in the dataset."""
        files = list(self.data_dir.rglob("*.csv"))
        logger.info(f"Discovered {len(files)} .csv files in {self.data_dir}")

        # Count files per bearing for RUL estimation
        for f in files:
            bearing_id = self._get_bearing_id(f)
            if bearing_id:
                self._bearing_file_counts[bearing_id] = (
                    self._bearing_file_counts.get(bearing_id, 0) + 1
                )

        return sorted(files)

    def _get_bearing_id(self, file_path: Path) -> Optional[str]:
        """Extract bearing ID from file path."""
        for part in file_path.parts:
            if part.startswith("Bearing"):
                return part
        return None

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a XJTU CSV file into RawEpisode(s).

        Args:
            file_path: Path to CSV file

        Yields:
            RawEpisode objects
        """
        try:
            # XJTU CSV format: horizontal_accel, vertical_accel
            data = np.loadtxt(str(file_path), delimiter=",")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return

        # Parse path for metadata
        file_info = self._parse_path(file_path)

        # Extract channels
        channels = self._extract_channels(data, file_info, file_path)

        if not channels:
            logger.warning(f"No valid channels in {file_path}")
            return

        yield self._create_episode(channels, file_info, file_path)

    def _parse_path(self, file_path: Path) -> Dict[str, Any]:
        """Parse file path for metadata.

        XJTU directory structure:
        xjtu_sy/
        ├── 35Hz12kN/
        │   ├── Bearing1_1/
        │   │   ├── 1.csv
        │   │   ├── 2.csv
        │   │   └── ...
        │   └── Bearing1_2/
        └── ...

        Args:
            file_path: Path to file

        Returns:
            Dictionary with metadata
        """
        parts = file_path.parts

        # Find operating condition
        condition = None
        rpm = 2100
        load_kn = 12.0

        for part in parts:
            if part in XJTU_CONDITIONS:
                condition = part
                rpm = XJTU_CONDITIONS[part]["rpm"]
                load_kn = XJTU_CONDITIONS[part]["load_kn"]
                break

        # Find bearing ID
        bearing_id = self._get_bearing_id(file_path)

        # Get fault type from bearing failure mode
        fault_type = XJTU_BEARING_FAILURES.get(bearing_id, FaultType.UNKNOWN)

        # Estimate severity from file number (degradation progression)
        severity = self._estimate_severity(file_path, bearing_id)

        # Extract file number for RUL
        file_num = 0
        try:
            file_num = int(file_path.stem)
        except ValueError:
            pass

        return {
            "condition": condition,
            "bearing_id": bearing_id,
            "fault_type": fault_type,
            "severity": severity,
            "rpm": rpm,
            "load_kn": load_kn,
            "file_num": file_num,
        }

    def _estimate_severity(
        self,
        file_path: Path,
        bearing_id: Optional[str],
    ) -> SeverityLevel:
        """Estimate severity based on degradation progression.

        Uses relative position in bearing life (file number / total files).

        Args:
            file_path: Path to file
            bearing_id: Bearing identifier

        Returns:
            Estimated severity level
        """
        if not bearing_id or bearing_id not in self._bearing_file_counts:
            return SeverityLevel.MINOR

        total_files = self._bearing_file_counts.get(bearing_id, 100)

        try:
            file_num = int(file_path.stem)
            progress = file_num / total_files

            if progress < 0.3:
                return SeverityLevel.HEALTHY
            elif progress < 0.6:
                return SeverityLevel.MINOR
            elif progress < 0.85:
                return SeverityLevel.MODERATE
            elif progress < 0.95:
                return SeverityLevel.SEVERE
            else:
                return SeverityLevel.CRITICAL

        except ValueError:
            return SeverityLevel.MINOR

    def _extract_channels(
        self,
        data: np.ndarray,
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> List[SensorChannel]:
        """Extract sensor channels from CSV data.

        XJTU CSV format: horizontal_accel, vertical_accel

        Args:
            data: Loaded numpy array
            file_info: Parsed metadata
            file_path: Source file path

        Returns:
            List of SensorChannel objects
        """
        channels = []

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        num_cols = data.shape[1]
        bearing_id = file_info.get("bearing_id", "unknown")

        # Horizontal acceleration (column 0)
        if num_cols >= 1:
            channels.append(SensorChannel(
                channel_id=f"{bearing_id}_horizontal",
                channel_type="vibration_horizontal",
                unit="g",
                data=data[:, 0].astype(np.float64).tolist(),
                sampling_rate_hz=XJTU_SAMPLING_RATE,
            ))

        # Vertical acceleration (column 1)
        if num_cols >= 2:
            channels.append(SensorChannel(
                channel_id=f"{bearing_id}_vertical",
                channel_type="vibration_vertical",
                unit="g",
                data=data[:, 1].astype(np.float64).tolist(),
                sampling_rate_hz=XJTU_SAMPLING_RATE,
            ))

        return channels

    def _create_episode(
        self,
        channels: List[SensorChannel],
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> RawEpisode:
        """Create RawEpisode from channels and metadata."""
        bearing_id = file_info.get("bearing_id", "unknown")
        file_num = file_info.get("file_num", 0)

        # Calculate RUL estimate if enabled
        rul_minutes = None
        if self.include_rul_estimate and bearing_id in self._bearing_file_counts:
            total_files = self._bearing_file_counts[bearing_id]
            remaining = max(0, total_files - file_num)
            rul_minutes = remaining  # 1 minute per file

        # Determine fault location
        fault_location = None
        fault_type = file_info["fault_type"]
        if fault_type == FaultType.INNER_RACE:
            fault_location = "inner_race"
        elif fault_type == FaultType.OUTER_RACE:
            fault_location = "outer_race"
        elif fault_type == FaultType.CAGE:
            fault_location = "cage"

        return RawEpisode(
            raw_id=f"xjtu_{bearing_id}_{file_num:05d}",
            source_dataset="xjtu_sy",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location=fault_location,
            severity=file_info["severity"],
            rpm=file_info["rpm"],
            raw_metadata={
                "bearing_id": bearing_id,
                "condition": file_info["condition"],
                "file_number": file_num,
                "load_kn": file_info["load_kn"],
                "sampling_rate_hz": XJTU_SAMPLING_RATE,
                "rul_estimate_minutes": rul_minutes,
            },
        )
