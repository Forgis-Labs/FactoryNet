"""Base classes for dataset adapters.

Provides abstract base class and data models for adapting external
datasets to the FactoryNet schema.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class FaultType(Enum):
    """Standard fault type classification for bearing/machinery datasets."""
    NORMAL = "normal"
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"
    BALL = "ball"
    CAGE = "cage"
    COMBINED = "combined"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    LOOSENESS = "looseness"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Fault severity classification."""
    HEALTHY = "healthy"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class SensorChannel:
    """Represents a single sensor channel of time series data."""
    channel_id: str
    channel_type: str  # e.g., "vibration_x", "vibration_y", "temperature"
    unit: str  # e.g., "g", "mm/s", "°C"
    data: Sequence[float]  # Time series values
    sampling_rate_hz: float

    def __post_init__(self):
        """Validate sensor channel data."""
        if len(self.data) == 0:
            raise ValueError(f"Channel {self.channel_id} has no data")
        if self.sampling_rate_hz <= 0:
            raise ValueError(f"Invalid sampling rate: {self.sampling_rate_hz}")


@dataclass
class RawEpisode:
    """Raw episode data from an external dataset before normalization.

    This is the intermediate format between raw dataset files and
    the final FactoryNetEpisode format. Adapters produce RawEpisodes,
    which are then normalized into FactoryNetEpisodes.
    """
    # Identification
    raw_id: str  # Original ID from source dataset
    source_dataset: str  # e.g., "cwru_bearing", "paderborn_bearing"
    source_file: str  # Original filename

    # Timing
    timestamp: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Sensor data
    channels: List[SensorChannel] = field(default_factory=list)

    # Labels
    fault_type: FaultType = FaultType.UNKNOWN
    fault_location: Optional[str] = None  # e.g., "drive_end", "fan_end"
    severity: SeverityLevel = SeverityLevel.HEALTHY

    # Operating conditions
    load_hp: Optional[float] = None  # Motor load in horsepower
    rpm: Optional[float] = None  # Rotational speed

    # Additional metadata from source
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived fields."""
        if self.channels and self.duration_seconds == 0.0:
            # Compute duration from first channel
            ch = self.channels[0]
            self.duration_seconds = len(ch.data) / ch.sampling_rate_hz

    @property
    def primary_channel(self) -> Optional[SensorChannel]:
        """Get the primary (first) sensor channel."""
        return self.channels[0] if self.channels else None

    @property
    def num_samples(self) -> int:
        """Total number of samples across all channels."""
        return sum(len(ch.data) for ch in self.channels)

    def compute_checksum(self) -> str:
        """Compute MD5 checksum of episode data for deduplication."""
        hasher = hashlib.md5()
        hasher.update(self.source_dataset.encode())
        hasher.update(self.raw_id.encode())
        for ch in self.channels:
            # Hash first/last 100 values for efficiency
            data_sample = list(ch.data[:100]) + list(ch.data[-100:])
            hasher.update(str(data_sample).encode())
        return hasher.hexdigest()


@dataclass
class DatasetMetadata:
    """Metadata about an external dataset."""
    name: str  # Short identifier, e.g., "cwru_bearing"
    full_name: str  # Full dataset name
    description: str
    source_url: str
    citation: str
    license: str

    # Dataset characteristics
    num_samples: int
    file_format: str  # e.g., ".mat", ".csv"
    sampling_rate_hz: Optional[float] = None

    # Machine/equipment info
    machine_type: str = "bearing_test_rig"
    machine_synset: str = "M.tst.bea"  # Default for bearing test rigs

    # Download info
    download_size_mb: Optional[float] = None
    requires_auth: bool = False

    # Version tracking
    version: str = "1.0"
    last_updated: Optional[datetime] = None


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Each adapter handles a specific external dataset, providing methods to:
    1. Download the dataset (if needed)
    2. Discover available files/samples
    3. Parse files into RawEpisode objects
    4. Iterate over all episodes

    Usage:
        adapter = CWRUAdapter(data_dir="/path/to/data")
        for episode in adapter.iter_episodes():
            # Process episode
            pass
    """

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
    ):
        """Initialize adapter.

        Args:
            data_dir: Directory containing dataset files (or where to download)
            cache_dir: Optional cache directory for intermediate files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return metadata about this dataset."""
        pass

    @abstractmethod
    def discover_files(self) -> List[Path]:
        """Discover all data files in the dataset directory.

        Returns:
            List of paths to data files
        """
        pass

    @abstractmethod
    def parse_file(self, file_path: Path) -> Iterable[RawEpisode]:
        """Parse a single file into RawEpisode objects.

        Some files may contain multiple episodes (e.g., different fault severities).

        Args:
            file_path: Path to data file

        Yields:
            RawEpisode objects parsed from the file
        """
        pass

    def iter_episodes(
        self,
        limit: Optional[int] = None,
        file_filter: Optional[callable] = None,
    ) -> Iterable[RawEpisode]:
        """Iterate over all episodes in the dataset.

        Args:
            limit: Maximum number of episodes to yield (for demo mode)
            file_filter: Optional function to filter files

        Yields:
            RawEpisode objects
        """
        files = self.discover_files()
        if file_filter:
            files = [f for f in files if file_filter(f)]

        count = 0
        for file_path in files:
            try:
                for episode in self.parse_file(file_path):
                    yield episode
                    count += 1
                    if limit and count >= limit:
                        return
            except Exception as e:
                # Log error but continue with other files
                import logging
                logging.warning(f"Error parsing {file_path}: {e}")
                continue

    def get_download_urls(self) -> List[str]:
        """Return URLs for downloading the dataset.

        Override in subclasses that support automatic download.

        Returns:
            List of URLs to download
        """
        return []

    def verify_integrity(self) -> bool:
        """Verify dataset integrity (checksums, completeness).

        Returns:
            True if dataset is complete and valid
        """
        files = self.discover_files()
        expected = self.metadata.num_samples
        if len(files) < expected * 0.9:  # Allow 10% tolerance
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_dir={self.data_dir})"
