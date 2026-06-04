"""CWRU Bearing Dataset Adapter.

Adapter for the Case Western Reserve University Bearing Data Center dataset,
one of the most widely used benchmark datasets for bearing fault diagnosis.

Dataset: https://engineering.case.edu/bearingdatacenter
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

# CWRU dataset constants
CWRU_SAMPLING_RATES = {
    "12k": 12000,  # 12 kHz drive end samples
    "48k": 48000,  # 48 kHz drive end samples (some files)
}

# Fault diameter to severity mapping (in mils = 0.001 inch)
FAULT_SEVERITY_MAP = {
    7: SeverityLevel.MINOR,      # 0.007 inch
    14: SeverityLevel.MODERATE,  # 0.014 inch
    21: SeverityLevel.MODERATE,  # 0.021 inch
    28: SeverityLevel.SEVERE,    # 0.028 inch
}

# Motor loads (in HP)
MOTOR_LOADS = [0, 1, 2, 3]

# RPM values for each load condition
LOAD_TO_RPM = {
    0: 1797,
    1: 1772,
    2: 1750,
    3: 1730,
}

# File naming patterns for parsing
# Example: "105.mat" or "IR007_0_105.mat"
CWRU_FILE_PATTERNS = [
    # Pattern: FaultType_Diameter_RPM_FileNum.mat
    r"(?P<fault_type>IR|OR|B)(?P<diameter>\d+)_(?P<load>\d)_(?P<file_num>\d+)\.mat",
    # Pattern: Normal_Load_FileNum.mat
    r"(?P<fault_type>Normal)_(?P<load>\d)_(?P<file_num>\d+)\.mat",
    # Simple numeric pattern (need metadata lookup)
    r"(?P<file_num>\d+)\.mat",
]

# Mapping of CWRU fault codes to our FaultType enum
CWRU_FAULT_MAP = {
    "IR": FaultType.INNER_RACE,
    "OR": FaultType.OUTER_RACE,
    "B": FaultType.BALL,
    "Normal": FaultType.NORMAL,
}

# CWRU metadata lookup table for numeric-only filenames
# Based on https://engineering.case.edu/bearingdatacenter/download-data-file
CWRU_FILE_METADATA: Dict[int, Dict[str, Any]] = {
    # Normal baseline data (12k)
    97: {"fault": "Normal", "load_hp": 0, "rpm": 1797, "rate": 12000},
    98: {"fault": "Normal", "load_hp": 1, "rpm": 1772, "rate": 12000},
    99: {"fault": "Normal", "load_hp": 2, "rpm": 1750, "rate": 12000},
    100: {"fault": "Normal", "load_hp": 3, "rpm": 1730, "rate": 12000},
    # 12k Drive End Inner Race Faults
    105: {"fault": "IR", "diameter": 7, "load_hp": 0, "rpm": 1797, "location": "DE", "rate": 12000},
    106: {"fault": "IR", "diameter": 7, "load_hp": 1, "rpm": 1772, "location": "DE", "rate": 12000},
    107: {"fault": "IR", "diameter": 7, "load_hp": 2, "rpm": 1750, "location": "DE", "rate": 12000},
    108: {"fault": "IR", "diameter": 7, "load_hp": 3, "rpm": 1730, "location": "DE", "rate": 12000},
    169: {"fault": "IR", "diameter": 14, "load_hp": 0, "rpm": 1797, "location": "DE", "rate": 12000},
    170: {"fault": "IR", "diameter": 14, "load_hp": 1, "rpm": 1772, "location": "DE", "rate": 12000},
    171: {"fault": "IR", "diameter": 14, "load_hp": 2, "rpm": 1750, "location": "DE", "rate": 12000},
    172: {"fault": "IR", "diameter": 14, "load_hp": 3, "rpm": 1730, "location": "DE", "rate": 12000},
    209: {"fault": "IR", "diameter": 21, "load_hp": 0, "rpm": 1797, "location": "DE", "rate": 12000},
    210: {"fault": "IR", "diameter": 21, "load_hp": 1, "rpm": 1772, "location": "DE", "rate": 12000},
    211: {"fault": "IR", "diameter": 21, "load_hp": 2, "rpm": 1750, "location": "DE", "rate": 12000},
    212: {"fault": "IR", "diameter": 21, "load_hp": 3, "rpm": 1730, "location": "DE", "rate": 12000},
    # 12k Drive End Ball Faults
    118: {"fault": "B", "diameter": 7, "load_hp": 0, "rpm": 1797, "location": "DE", "rate": 12000},
    119: {"fault": "B", "diameter": 7, "load_hp": 1, "rpm": 1772, "location": "DE", "rate": 12000},
    120: {"fault": "B", "diameter": 7, "load_hp": 2, "rpm": 1750, "location": "DE", "rate": 12000},
    121: {"fault": "B", "diameter": 7, "load_hp": 3, "rpm": 1730, "location": "DE", "rate": 12000},
    185: {"fault": "B", "diameter": 14, "load_hp": 0, "rpm": 1797, "location": "DE", "rate": 12000},
    186: {"fault": "B", "diameter": 14, "load_hp": 1, "rpm": 1772, "location": "DE", "rate": 12000},
    187: {"fault": "B", "diameter": 14, "load_hp": 2, "rpm": 1750, "location": "DE", "rate": 12000},
    188: {"fault": "B", "diameter": 14, "load_hp": 3, "rpm": 1730, "location": "DE", "rate": 12000},
    222: {"fault": "B", "diameter": 21, "load_hp": 0, "rpm": 1797, "location": "DE", "rate": 12000},
    223: {"fault": "B", "diameter": 21, "load_hp": 1, "rpm": 1772, "location": "DE", "rate": 12000},
    224: {"fault": "B", "diameter": 21, "load_hp": 2, "rpm": 1750, "location": "DE", "rate": 12000},
    225: {"fault": "B", "diameter": 21, "load_hp": 3, "rpm": 1730, "location": "DE", "rate": 12000},
    # 12k Drive End Outer Race Faults (centered @6:00)
    130: {"fault": "OR", "diameter": 7, "load_hp": 0, "rpm": 1797, "location": "DE", "or_position": "6", "rate": 12000},
    131: {"fault": "OR", "diameter": 7, "load_hp": 1, "rpm": 1772, "location": "DE", "or_position": "6", "rate": 12000},
    132: {"fault": "OR", "diameter": 7, "load_hp": 2, "rpm": 1750, "location": "DE", "or_position": "6", "rate": 12000},
    133: {"fault": "OR", "diameter": 7, "load_hp": 3, "rpm": 1730, "location": "DE", "or_position": "6", "rate": 12000},
    197: {"fault": "OR", "diameter": 14, "load_hp": 0, "rpm": 1797, "location": "DE", "or_position": "6", "rate": 12000},
    198: {"fault": "OR", "diameter": 14, "load_hp": 1, "rpm": 1772, "location": "DE", "or_position": "6", "rate": 12000},
    199: {"fault": "OR", "diameter": 14, "load_hp": 2, "rpm": 1750, "location": "DE", "or_position": "6", "rate": 12000},
    200: {"fault": "OR", "diameter": 14, "load_hp": 3, "rpm": 1730, "location": "DE", "or_position": "6", "rate": 12000},
    234: {"fault": "OR", "diameter": 21, "load_hp": 0, "rpm": 1797, "location": "DE", "or_position": "6", "rate": 12000},
    235: {"fault": "OR", "diameter": 21, "load_hp": 1, "rpm": 1772, "location": "DE", "or_position": "6", "rate": 12000},
    236: {"fault": "OR", "diameter": 21, "load_hp": 2, "rpm": 1750, "location": "DE", "or_position": "6", "rate": 12000},
    237: {"fault": "OR", "diameter": 21, "load_hp": 3, "rpm": 1730, "location": "DE", "or_position": "6", "rate": 12000},
    # 12k Fan End Inner Race Faults
    109: {"fault": "IR", "diameter": 7, "load_hp": 0, "rpm": 1797, "location": "FE", "rate": 12000},
    110: {"fault": "IR", "diameter": 7, "load_hp": 1, "rpm": 1772, "location": "FE", "rate": 12000},
    111: {"fault": "IR", "diameter": 7, "load_hp": 2, "rpm": 1750, "location": "FE", "rate": 12000},
    112: {"fault": "IR", "diameter": 7, "load_hp": 3, "rpm": 1730, "location": "FE", "rate": 12000},
    # 12k Fan End Ball Faults
    122: {"fault": "B", "diameter": 7, "load_hp": 0, "rpm": 1797, "location": "FE", "rate": 12000},
    123: {"fault": "B", "diameter": 7, "load_hp": 1, "rpm": 1772, "location": "FE", "rate": 12000},
    124: {"fault": "B", "diameter": 7, "load_hp": 2, "rpm": 1750, "location": "FE", "rate": 12000},
    125: {"fault": "B", "diameter": 7, "load_hp": 3, "rpm": 1730, "location": "FE", "rate": 12000},
    # 12k Fan End Outer Race Faults
    135: {"fault": "OR", "diameter": 7, "load_hp": 0, "rpm": 1797, "location": "FE", "rate": 12000},
    136: {"fault": "OR", "diameter": 7, "load_hp": 1, "rpm": 1772, "location": "FE", "rate": 12000},
    137: {"fault": "OR", "diameter": 7, "load_hp": 2, "rpm": 1750, "location": "FE", "rate": 12000},
    138: {"fault": "OR", "diameter": 7, "load_hp": 3, "rpm": 1730, "location": "FE", "rate": 12000},
    # 48k Drive End samples
    278: {"fault": "Normal", "load_hp": 0, "rpm": 1797, "rate": 48000},
    279: {"fault": "Normal", "load_hp": 1, "rpm": 1772, "rate": 48000},
    280: {"fault": "Normal", "load_hp": 2, "rpm": 1750, "rate": 48000},
    281: {"fault": "Normal", "load_hp": 3, "rpm": 1730, "rate": 48000},
}


# Download URLs for CWRU dataset
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files"
CWRU_DOWNLOAD_URLS = {
    # 12k Drive End Bearing Fault Data
    "12k_DE": f"{CWRU_BASE_URL}/12k_Drive_End_Bearing_Fault_Data.zip",
    # 12k Fan End Bearing Fault Data
    "12k_FE": f"{CWRU_BASE_URL}/12k_Fan_End_Bearing_Fault_Data.zip",
    # 48k Drive End Bearing Fault Data
    "48k_DE": f"{CWRU_BASE_URL}/48k_Drive_End_Bearing_Fault_Data.zip",
    # Normal Baseline Data
    "normal": f"{CWRU_BASE_URL}/Normal_Baseline_Data.zip",
}


@register_adapter("cwru_bearing", aliases=["cwru", "case_western"])
class CWRUAdapter(BaseDatasetAdapter):
    """Adapter for CWRU Bearing Data Center dataset.

    The CWRU dataset contains vibration data from a motor driving
    mechanical shaft with two bearings (drive end and fan end).
    Faults were seeded in the bearings using electro-discharge machining.

    Dataset characteristics:
    - Sampling rates: 12 kHz (most data), 48 kHz (some files)
    - Sensor channels: DE (Drive End), FE (Fan End), BA (Base)
    - Fault types: Inner race, Outer race, Ball
    - Fault diameters: 0.007", 0.014", 0.021", 0.028"
    - Motor loads: 0-3 HP
    - Bearing type: SKF 6205-2RS

    Example:
        adapter = CWRUAdapter(data_dir="./data/cwru")
        for episode in adapter.iter_episodes(limit=10):
            print(f"{episode.raw_id}: {episode.fault_type.value}")
    """

    METADATA = DatasetMetadata(
        name="cwru_bearing",
        full_name="CWRU Bearing Data Center",
        description="Case Western Reserve University bearing fault vibration dataset",
        source_url="https://engineering.case.edu/bearingdatacenter",
        citation=(
            "W. A. Smith and R. B. Randall, 'Rolling element bearing diagnostics "
            "using the Case Western Reserve University data: A benchmark study,' "
            "Mechanical Systems and Signal Processing, vol. 64-65, pp. 100-131, 2015."
        ),
        license="Public Domain (Educational Use)",
        num_samples=161,  # Approximate number of .mat files
        file_format=".mat",
        sampling_rate_hz=12000,  # Primary sampling rate
        machine_type="bearing_test_rig",
        machine_synset="M.tst.bea.cwru",
        download_size_mb=250,
    )

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        segment_length: Optional[int] = None,
        segment_overlap: float = 0.0,
    ):
        """Initialize CWRU adapter.

        Args:
            data_dir: Directory containing .mat files
            cache_dir: Optional cache directory
            segment_length: If set, segment signals into fixed-length episodes
            segment_overlap: Overlap ratio between segments (0.0-0.5)
        """
        super().__init__(data_dir, cache_dir)
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self.METADATA

    def discover_files(self) -> List[Path]:
        """Discover all .mat files in the data directory."""
        files = list(self.data_dir.rglob("*.mat"))
        logger.info(f"Discovered {len(files)} .mat files in {self.data_dir}")
        return sorted(files)

    def get_download_urls(self) -> List[str]:
        """Return download URLs for the dataset."""
        return list(CWRU_DOWNLOAD_URLS.values())

    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a CWRU .mat file into RawEpisode(s).

        Args:
            file_path: Path to .mat file

        Yields:
            RawEpisode objects
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required for CWRU adapter. Install with: pip install scipy")

        # Load .mat file
        try:
            mat_data = loadmat(str(file_path), squeeze_me=True)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return

        # Parse filename for metadata
        file_info = self._parse_filename(file_path)

        # Extract vibration channels from .mat file
        channels = self._extract_channels(mat_data, file_info)

        if not channels:
            logger.warning(f"No valid channels found in {file_path}")
            return

        # Create episode(s)
        if self.segment_length:
            # Segment into multiple episodes
            yield from self._create_segmented_episodes(
                channels, file_info, file_path
            )
        else:
            # Single episode per file
            yield self._create_episode(channels, file_info, file_path)

    def _parse_filename(self, file_path: Path) -> Dict[str, Any]:
        """Parse CWRU filename to extract metadata.

        Args:
            file_path: Path to .mat file

        Returns:
            Dictionary with parsed metadata
        """
        filename = file_path.stem  # Remove .mat extension

        # Try to extract file number
        file_num_match = re.search(r"(\d+)$", filename)
        if file_num_match:
            file_num = int(file_num_match.group(1))
            # Look up in metadata table
            if file_num in CWRU_FILE_METADATA:
                info = CWRU_FILE_METADATA[file_num].copy()
                info["file_num"] = file_num
                return info

        # Try pattern matching
        for pattern in CWRU_FILE_PATTERNS:
            match = re.match(pattern, filename + ".mat")
            if match:
                groups = match.groupdict()
                return {
                    "fault": groups.get("fault_type", "Unknown"),
                    "diameter": int(groups.get("diameter", 0)) if groups.get("diameter") else None,
                    "load_hp": int(groups.get("load", 0)),
                    "rpm": LOAD_TO_RPM.get(int(groups.get("load", 0)), 1797),
                    "file_num": int(groups.get("file_num", 0)),
                    "rate": 12000,
                }

        # Fallback: unknown metadata
        logger.warning(f"Could not parse metadata from filename: {filename}")
        return {
            "fault": "Unknown",
            "diameter": None,
            "load_hp": 0,
            "rpm": 1797,
            "file_num": 0,
            "rate": 12000,
        }

    def _extract_channels(
        self,
        mat_data: Dict[str, Any],
        file_info: Dict[str, Any],
    ) -> List[SensorChannel]:
        """Extract vibration channels from .mat file data.

        CWRU .mat files contain arrays with names like:
        - X105_DE_time: Drive End accelerometer
        - X105_FE_time: Fan End accelerometer
        - X105_BA_time: Base accelerometer

        Args:
            mat_data: Loaded .mat file data
            file_info: Parsed file metadata

        Returns:
            List of SensorChannel objects
        """
        channels = []
        sampling_rate = file_info.get("rate", 12000)

        # Look for vibration data arrays
        # Keys typically match pattern: X{num}_{location}_time
        for key in mat_data.keys():
            if key.startswith("__"):  # Skip metadata keys
                continue

            # Check if this looks like vibration data
            if "_DE_time" in key or "_FE_time" in key or "_BA_time" in key:
                location = "DE" if "_DE_time" in key else ("FE" if "_FE_time" in key else "BA")
                channel_type = f"vibration_{location.lower()}"
            elif "DE" in key and not key.startswith("__"):
                location = "DE"
                channel_type = "vibration_de"
            elif "FE" in key and not key.startswith("__"):
                location = "FE"
                channel_type = "vibration_fe"
            elif "BA" in key and not key.startswith("__"):
                location = "BA"
                channel_type = "vibration_ba"
            else:
                # Try to extract if it's a numeric array (fallback)
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.ndim == 1 and len(data) > 1000:
                    location = "unknown"
                    channel_type = "vibration"
                else:
                    continue

            data = mat_data[key]
            if isinstance(data, np.ndarray):
                # Flatten if needed
                data = data.flatten().astype(np.float64)

                if len(data) > 100:  # Minimum viable length
                    channels.append(SensorChannel(
                        channel_id=f"{file_info.get('file_num', 0)}_{location}",
                        channel_type=channel_type,
                        unit="g",  # Acceleration in g
                        data=data.tolist(),
                        sampling_rate_hz=sampling_rate,
                    ))

        return channels

    def _create_episode(
        self,
        channels: List[SensorChannel],
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> RawEpisode:
        """Create a RawEpisode from extracted channels.

        Args:
            channels: List of sensor channels
            file_info: Parsed file metadata
            file_path: Original file path

        Returns:
            RawEpisode object
        """
        # Map fault type
        fault_str = file_info.get("fault", "Unknown")
        fault_type = CWRU_FAULT_MAP.get(fault_str, FaultType.UNKNOWN)

        # Map severity from fault diameter
        diameter = file_info.get("diameter")
        if fault_type == FaultType.NORMAL:
            severity = SeverityLevel.HEALTHY
        elif diameter:
            severity = FAULT_SEVERITY_MAP.get(diameter, SeverityLevel.MINOR)
        else:
            severity = SeverityLevel.MINOR

        # Determine fault location
        location = file_info.get("location", "DE")
        fault_location = "drive_end" if location == "DE" else "fan_end"

        return RawEpisode(
            raw_id=f"cwru_{file_info.get('file_num', file_path.stem)}",
            source_dataset="cwru_bearing",
            source_file=file_path.name,
            channels=channels,
            fault_type=fault_type,
            fault_location=fault_location,
            severity=severity,
            load_hp=file_info.get("load_hp"),
            rpm=file_info.get("rpm"),
            raw_metadata={
                "fault_diameter_mils": diameter,
                "or_position": file_info.get("or_position"),
                "sampling_rate_hz": file_info.get("rate", 12000),
                "bearing_type": "SKF 6205-2RS",
            },
        )

    def _create_segmented_episodes(
        self,
        channels: List[SensorChannel],
        file_info: Dict[str, Any],
        file_path: Path,
    ) -> Iterable[RawEpisode]:
        """Create multiple episodes by segmenting the signal.

        Args:
            channels: List of sensor channels
            file_info: Parsed file metadata
            file_path: Original file path

        Yields:
            RawEpisode objects for each segment
        """
        if not channels:
            return

        # Use first channel for segmentation reference
        ref_channel = channels[0]
        signal_length = len(ref_channel.data)

        step_size = int(self.segment_length * (1 - self.segment_overlap))
        segment_idx = 0

        for start in range(0, signal_length - self.segment_length + 1, step_size):
            end = start + self.segment_length

            # Create segmented channels
            seg_channels = []
            for ch in channels:
                seg_data = ch.data[start:end]
                seg_channels.append(SensorChannel(
                    channel_id=f"{ch.channel_id}_seg{segment_idx}",
                    channel_type=ch.channel_type,
                    unit=ch.unit,
                    data=seg_data,
                    sampling_rate_hz=ch.sampling_rate_hz,
                ))

            # Create episode for this segment
            base_episode = self._create_episode(seg_channels, file_info, file_path)
            base_episode.raw_id = f"{base_episode.raw_id}_seg{segment_idx}"
            base_episode.raw_metadata["segment_index"] = segment_idx
            base_episode.raw_metadata["segment_start_sample"] = start
            base_episode.raw_metadata["segment_end_sample"] = end

            yield base_episode
            segment_idx += 1

    def verify_integrity(self) -> bool:
        """Verify dataset integrity."""
        files = self.discover_files()
        if len(files) < 50:  # Expect at least 50 files for basic dataset
            logger.warning(f"Only found {len(files)} files, expected at least 50")
            return False
        return True
