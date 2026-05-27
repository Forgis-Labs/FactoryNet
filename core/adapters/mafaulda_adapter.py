"""MAFAULDA Dataset Adapter.

Adapter for the MAFAULDA (Machinery Fault Database) dataset,
featuring rotating machinery faults including imbalance,
misalignment, and bearing defects.

Dataset: http://www02.smt.ufrj.br/~offshore/mfs/page_01.html
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

# MAFAULDA dataset constants
MAFAULDA_SAMPLING_RATE = 50000  # 50 kHz

# Fault type mapping from directory names
MAFAULDA_FAULT_MAP = {
    "normal": FaultType.NORMAL,
    "horizontal-misalignment": FaultType.MISALIGNMENT,
    "vertical-misalignment": FaultType.MISALIGNMENT,
    "imbalance": FaultType.IMBALANCE,
    "underhang": FaultType.LOOSENESS,  # Underhang bearing
    "overhang": FaultType.LOOSENESS,   # Overhang bearing
    "outer-race": FaultType.OUTER_RACE,
    "inner-race": FaultType.INNER_RACE,
    "ball": FaultType.BALL,
    "cage": FaultType.CAGE,
}

# Severity mapping from imbalance mass
IMBALANCE_SEVERITY = {
    "6g": SeverityLevel.MINOR,
    "10g": SeverityLevel.MINOR,
    "15g": SeverityLevel.MODERATE,
    "20g": SeverityLevel.MODERATE,
    "25g": SeverityLevel.SEVERE,
    "30g": SeverityLevel.SEVERE,
    "35g": SeverityLevel.SEVERE,
}

# Misalignment severity
MISALIGNMENT_SEVERITY = {
    "0.5mm": SeverityLevel.MINOR,
    "1.0mm": SeverityLevel.MINOR,
    "1.5mm": SeverityLevel.MODERATE,
    "2.0mm": SeverityLevel.SEVERE,
}


@register_adapter("mafaulda", aliases=["mafaulda_rotating"])
class MAFAULDAAdapter(BaseDatasetAdapter):
    """Adapter for MAFAULDA Rotating Machinery Dataset.

    The MAFAULDA dataset includes:
    - Normal operation
    - Imbalance (6g to 35g masses)
    - Horizontal misalignment (0.5mm to 2.0mm)
    - Vertical misalignment (0.51mm to 1.90mm)
    - Bearing faults (inner race, outer race, ball, cage)

    Sensor channels:
    - Tachometer
    - 8 accelerometers (axial/radial/tangential)
    - Microphone

    Dataset characteristics:
    - Sampling rate: 50 kHz
    - Recording duration: 5 seconds per file
    - RPM range: 737-3686

    Example:
        adapter = MAFAULDAAdapter(data_dir="./data/mafaulda")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: {episode.fault_type.value}")
    """

    METADATA = DatasetMetadata(
        name="mafaulda",
        full_name="MAFAULDA - Machinery Fault Database",
        description="Rotating machinery fault database with multiple fault types",
        source_url="http://www02.smt.ufrj.br/~offshore/mfs/page_01.html",
        citation=(
            "M. F. R. Ribeiro et al., 'Rotating machinery fault diagnosis using "
            "similarity-based modeling,' IEEE Latin America Transactions, 2019."
        ),
        license="Academic Use",
        num_samples=1951,  # Approximate number of recordings
        file_format=".csv",
        sampling_rate_hz=50000,
        machine_type="rotating_machinery_rig",
        machine_synset="M.tst.rot.mafaulda",
        download_size_mb=3000,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        include_audio: bool = False,
    ):
        """Initialize MAFAULDA adapter.

        Args:
            data_dir: Directory containing dataset files
            cache_dir: Optional cache directory
            include_audio: Whether to include microphone channel
        """
        super().__init__(data_dir, cache_dir)
        self.include_audio = include_audio

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover all data files in the dataset."""
        # MAFAULDA uses .csv files
        csv_files = list(self.data_dir.rglob("*.csv"))

        # Also check for .wav files if audio included
        if self.include_audio:
            wav_files = list(self.data_dir.rglob("*.wav"))
            csv_files.extend(wav_files)

        logger.info(f"Discovered {len(csv_files)} files in {self.data_dir}")
        return sorted(csv_files)

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a MAFAULDA file into RawEpisode(s).

        Args:
            file_path: Path to data file

        Yields:
            RawEpisode objects
        """
        if file_path.suffix.lower() == ".wav":
            # Skip audio files in main iteration
            return

        try:
            # Load CSV data
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
        """Parse file path to extract metadata.

        MAFAULDA directory structure:
        mafaulda/
        ├── normal/
        ├── imbalance/
        │   ├── 6g/
        │   ├── 10g/
        │   └── ...
        ├── horizontal-misalignment/
        │   ├── 0.5mm/
        │   └── ...
        └── ...

        Args:
            file_path: Path to file

        Returns:
            Dictionary with metadata
        """
        parts = file_path.parts

        # Find fault type from path
        fault_type = FaultType.UNKNOWN
        severity = SeverityLevel.MINOR
        fault_location = None

        for part in parts:
            part_lower = part.lower()

            # Check fault type
            for key, ftype in MAFAULDA_FAULT_MAP.items():
                if key in part_lower:
                    fault_type = ftype
                    if "inner" in part_lower:
                        fault_location = "inner_race"
                    elif "outer" in part_lower:
                        fault_location = "outer_race"
                    break

            # Check severity from imbalance mass
            for mass, sev in IMBALANCE_SEVERITY.items():
                if mass in part:
                    severity = sev
                    break

            # Check severity from misalignment
            for dist, sev in MISALIGNMENT_SEVERITY.items():
                if dist in part:
                    severity = sev
                    break

        if fault_type == FaultType.NORMAL:
            severity = SeverityLevel.HEALTHY

        # Extract RPM from filename if present
        rpm_match = re.search(r"(\d{3,4})rpm", file_path.stem, re.IGNORECASE)
        rpm = float(rpm_match.group(1)) if rpm_match else 1500.0

        return {
            "fault_type": fault_type,
            "severity": severity,
            "fault_location": fault_location,
            "rpm": rpm,
        }

    def _extract_channels(
        self,
        data: np.ndarray,
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> List[SensorChannel]:
        """Extract sensor channels from CSV data.

        MAFAULDA CSV columns (typically):
        0: Tachometer
        1-8: Accelerometers
        9: Microphone (optional)

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

        # Channel definitions
        channel_defs = [
            ("tachometer", "pulse", 0),
            ("accel_1_axial", "g", 1),
            ("accel_1_radial", "g", 2),
            ("accel_1_tangential", "g", 3),
            ("accel_2_axial", "g", 4),
            ("accel_2_radial", "g", 5),
            ("accel_2_tangential", "g", 6),
            ("accel_3_axial", "g", 7),
            ("accel_3_radial", "g", 8),
        ]

        if self.include_audio and num_cols > 9:
            channel_defs.append(("microphone", "Pa", 9))

        for channel_type, unit, col_idx in channel_defs:
            if col_idx < num_cols:
                col_data = data[:, col_idx].astype(np.float64)
                channels.append(SensorChannel(
                    channel_id=f"{file_path.stem}_{channel_type}",
                    channel_type=channel_type,
                    unit=unit,
                    data=col_data.tolist(),
                    sampling_rate_hz=MAFAULDA_SAMPLING_RATE,
                ))

        return channels

    def _create_episode(
        self,
        channels: List[SensorChannel],
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> RawEpisode:
        """Create RawEpisode from channels and metadata."""
        return RawEpisode(
            raw_id=f"mafaulda_{file_path.stem}",
            source_dataset="mafaulda",
            source_file=file_path.name,
            channels=channels,
            fault_type=file_info["fault_type"],
            fault_location=file_info["fault_location"],
            severity=file_info["severity"],
            rpm=file_info["rpm"],
            raw_metadata={
                "sampling_rate_hz": MAFAULDA_SAMPLING_RATE,
                "recording_duration_s": len(channels[0].data) / MAFAULDA_SAMPLING_RATE if channels else 0,
            },
        )
