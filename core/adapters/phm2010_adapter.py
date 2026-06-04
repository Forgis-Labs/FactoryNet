"""PHM 2010 CNC Milling Dataset Adapter.

Adapter for the PHM 2010 Data Challenge dataset featuring
CNC milling tool wear progression data.

Dataset: https://www.phmsociety.org/competition/phm/10
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

# PHM 2010 dataset constants
PHM2010_SAMPLING_RATE = 50000  # 50 kHz
PHM2010_SPINDLE_SPEED = 10400  # RPM

# Tool wear thresholds (in mm)
WEAR_THRESHOLDS = {
    "healthy": 0.0,
    "minor": 0.05,
    "moderate": 0.10,
    "severe": 0.15,
    "critical": 0.20,  # End of life threshold
}


@register_adapter("phm_2010", aliases=["phm2010", "cnc_milling"])
class PHM2010Adapter(BaseDatasetAdapter):
    """Adapter for PHM 2010 CNC Milling Tool Wear Dataset.

    The PHM 2010 dataset contains sensor data from a high-speed
    CNC milling machine during the machining process, tracking
    tool wear progression.

    Sensor channels:
    - Force sensors (X, Y, Z directions)
    - Vibration sensors (X, Y, Z directions)
    - Acoustic emission sensor

    Dataset characteristics:
    - Sampling rate: 50 kHz
    - Spindle speed: 10,400 RPM
    - Recorded during milling operations
    - Includes flute wear measurements

    Example:
        adapter = PHM2010Adapter(data_dir="./data/phm2010")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: wear={episode.raw_metadata.get('tool_wear_mm')}")
    """

    METADATA = DatasetMetadata(
        name="phm_2010",
        full_name="PHM 2010 CNC Milling Tool Wear Dataset",
        description="CNC milling tool wear data from PHM 2010 Data Challenge",
        source_url="https://www.phmsociety.org/competition/phm/10",
        citation=(
            "PHM Society, 'PHM 2010 Data Challenge - CNC Milling Tool Wear,' "
            "2010 Conference of the Prognostics and Health Management Society."
        ),
        license="PHM Society Data Challenge License",
        num_samples=315,  # Number of cuts
        file_format=".csv",
        sampling_rate_hz=50000,
        machine_type="cnc_milling",
        machine_synset="M.tst.cnc.phm2010",
        download_size_mb=600,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        wear_labels_file: Optional[str] = None,
    ):
        """Initialize PHM 2010 adapter.

        Args:
            data_dir: Directory containing dataset files
            cache_dir: Optional cache directory
            wear_labels_file: Optional path to wear labels CSV
        """
        super().__init__(data_dir, cache_dir)
        self.wear_labels_file = wear_labels_file
        self._wear_labels: Dict[int, float] = {}
        self._load_wear_labels()

    def _load_wear_labels(self) -> None:
        """Load tool wear labels if available."""
        if self.wear_labels_file:
            labels_path = Path(self.wear_labels_file)
        else:
            # Try default locations
            labels_path = self.data_dir / "train" / "train_labels.csv"
            if not labels_path.exists():
                labels_path = self.data_dir / "wear_labels.csv"

        if labels_path.exists():
            try:
                data = np.loadtxt(str(labels_path), delimiter=",", skiprows=1)
                if data.ndim == 1:
                    # Single row
                    self._wear_labels[1] = float(data[-1])
                else:
                    # Multiple rows: cut_number, flute1, flute2, flute3
                    for row in data:
                        cut_num = int(row[0])
                        # Average wear across flutes
                        avg_wear = np.mean(row[1:])
                        self._wear_labels[cut_num] = float(avg_wear)
                logger.info(f"Loaded {len(self._wear_labels)} wear labels")
            except Exception as e:
                logger.warning(f"Could not load wear labels: {e}")

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover all data files in the dataset."""
        # PHM 2010 has various file patterns
        patterns = ["*.csv", "*.txt"]
        files = []

        for pattern in patterns:
            found = list(self.data_dir.rglob(pattern))
            # Filter out label files
            found = [f for f in found if "label" not in f.name.lower()]
            files.extend(found)

        logger.info(f"Discovered {len(files)} data files in {self.data_dir}")
        return sorted(files)

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a PHM 2010 data file into RawEpisode(s).

        Args:
            file_path: Path to data file

        Yields:
            RawEpisode objects
        """
        try:
            # Try different delimiters
            try:
                data = np.loadtxt(str(file_path), delimiter=",")
            except ValueError:
                data = np.loadtxt(str(file_path), delimiter="\t")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return

        if data.size == 0:
            return

        # Parse filename for metadata
        file_info = self._parse_filename(file_path)

        # Extract channels
        channels = self._extract_channels(data, file_info, file_path)

        if not channels:
            logger.warning(f"No valid channels in {file_path}")
            return

        yield self._create_episode(channels, file_info, file_path)

    def _parse_filename(self, file_path: Path) -> Dict[str, Any]:
        """Parse filename for metadata.

        Filename patterns:
        - c_1.csv (cut number)
        - case1_1.csv
        - etc.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with metadata
        """
        filename = file_path.stem

        # Extract cut number
        cut_match = re.search(r"(\d+)", filename)
        cut_num = int(cut_match.group(1)) if cut_match else 0

        # Get wear label if available
        wear_mm = self._wear_labels.get(cut_num, 0.0)

        # Determine severity from wear
        severity = self._wear_to_severity(wear_mm)

        # Determine fault type
        if wear_mm >= WEAR_THRESHOLDS["critical"]:
            fault_type = FaultType.UNKNOWN  # Tool failure
        elif wear_mm > WEAR_THRESHOLDS["healthy"]:
            fault_type = FaultType.UNKNOWN  # Tool wear (not a bearing fault type)
        else:
            fault_type = FaultType.NORMAL

        return {
            "cut_number": cut_num,
            "wear_mm": wear_mm,
            "severity": severity,
            "fault_type": fault_type,
        }

    def _wear_to_severity(self, wear_mm: float) -> SeverityLevel:
        """Convert tool wear to severity level.

        Args:
            wear_mm: Tool wear in mm

        Returns:
            Severity level
        """
        if wear_mm >= WEAR_THRESHOLDS["critical"]:
            return SeverityLevel.CRITICAL
        elif wear_mm >= WEAR_THRESHOLDS["severe"]:
            return SeverityLevel.SEVERE
        elif wear_mm >= WEAR_THRESHOLDS["moderate"]:
            return SeverityLevel.MODERATE
        elif wear_mm >= WEAR_THRESHOLDS["minor"]:
            return SeverityLevel.MINOR
        else:
            return SeverityLevel.HEALTHY

    def _extract_channels(
        self,
        data: np.ndarray,
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> List[SensorChannel]:
        """Extract sensor channels from data.

        PHM 2010 columns (typical):
        0: Force X
        1: Force Y
        2: Force Z
        3: Vibration X
        4: Vibration Y
        5: Vibration Z
        6: Acoustic Emission (AE)

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
        cut_num = file_info.get("cut_number", 0)

        # Channel definitions based on dataset documentation
        channel_defs = [
            ("force_x", "N", 0),
            ("force_y", "N", 1),
            ("force_z", "N", 2),
            ("vibration_x", "g", 3),
            ("vibration_y", "g", 4),
            ("vibration_z", "g", 5),
            ("acoustic_emission", "V", 6),
        ]

        for channel_type, unit, col_idx in channel_defs:
            if col_idx < num_cols:
                col_data = data[:, col_idx].astype(np.float64)
                channels.append(SensorChannel(
                    channel_id=f"cut{cut_num}_{channel_type}",
                    channel_type=channel_type,
                    unit=unit,
                    data=col_data.tolist(),
                    sampling_rate_hz=PHM2010_SAMPLING_RATE,
                ))

        return channels

    def _create_episode(
        self,
        channels: List[SensorChannel],
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> RawEpisode:
        """Create RawEpisode from channels and metadata."""
        cut_num = file_info.get("cut_number", 0)

        return RawEpisode(
            raw_id=f"phm2010_cut{cut_num:03d}",
            source_dataset="phm_2010",
            source_file=file_path.name,
            channels=channels,
            fault_type=file_info["fault_type"],
            fault_location="cutting_tool",
            severity=file_info["severity"],
            rpm=PHM2010_SPINDLE_SPEED,
            raw_metadata={
                "cut_number": cut_num,
                "tool_wear_mm": file_info["wear_mm"],
                "spindle_speed_rpm": PHM2010_SPINDLE_SPEED,
                "sampling_rate_hz": PHM2010_SAMPLING_RATE,
                "machine_type": "CNC_milling",
            },
        )
