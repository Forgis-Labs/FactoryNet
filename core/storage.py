"""Storage module for saving FactoryNet episodes.

Handles saving episodes to the standardized directory structure
with metadata, timeseries, Q&A pairs, and semantic priors.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.normalizer import FactoryNetEpisode, SemanticPriors

logger = logging.getLogger(__name__)


def _serialize_value(obj: Any) -> Any:
    """Serialize value for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: _serialize_value(v) for k, v in obj.__dict__.items()}
    return obj


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    return _serialize_value(obj)


class EpisodeStorage:
    """Handles saving and loading FactoryNet episodes.

    Directory structure:
        factorynet_data/
        └── episodes/
            └── adapted/
                └── {dataset_name}/
                    └── {episode_id}/
                        ├── metadata.json
                        ├── timeseries.parquet
                        ├── qa_pairs.json
                        └── semantic_priors.json

    Example:
        storage = EpisodeStorage(base_dir="./factorynet_data")
        storage.save_episode(episode, qa_pairs)
    """

    def __init__(
        self,
        base_dir: str | Path,
        use_parquet: bool = True,
    ):
        """Initialize storage.

        Args:
            base_dir: Base directory for data storage
            use_parquet: Use Parquet format for timeseries (else JSON)
        """
        self.base_dir = Path(base_dir)
        self.use_parquet = use_parquet
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create base directory structure."""
        (self.base_dir / "episodes" / "adapted").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "validation_reports").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "taxonomy").mkdir(parents=True, exist_ok=True)

    def get_episode_dir(
        self,
        episode: FactoryNetEpisode,
    ) -> Path:
        """Get the directory path for an episode.

        Args:
            episode: Episode to get directory for

        Returns:
            Path to episode directory
        """
        return (
            self.base_dir / "episodes" / "adapted" /
            episode.source_dataset / episode.episode_id
        )

    def save_episode(
        self,
        episode: FactoryNetEpisode,
        qa_pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Save an episode to disk.

        Args:
            episode: Episode to save
            qa_pairs: Optional Q&A pairs to save

        Returns:
            Path to saved episode directory
        """
        episode_dir = self.get_episode_dir(episode)
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        self._save_metadata(episode, episode_dir)

        # Save timeseries
        self._save_timeseries(episode, episode_dir)

        # Save Q&A pairs
        if qa_pairs:
            self._save_qa_pairs(qa_pairs, episode_dir)

        # Save semantic priors
        if episode.semantic_priors:
            self._save_semantic_priors(episode.semantic_priors, episode_dir)

        # Save features
        if episode.features:
            self._save_features(episode, episode_dir)

        logger.debug(f"Saved episode {episode.episode_id} to {episode_dir}")
        return episode_dir

    def _save_metadata(
        self,
        episode: FactoryNetEpisode,
        episode_dir: Path,
    ) -> None:
        """Save episode metadata to JSON."""
        metadata = episode.to_metadata_dict()

        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=_json_serializer)

    def _save_timeseries(
        self,
        episode: FactoryNetEpisode,
        episode_dir: Path,
    ) -> None:
        """Save timeseries data to Parquet or JSON."""
        if not episode.steps:
            return

        # Build data structure
        data = {
            "step_index": [],
            "timestamp_offset": [],
        }

        # Add columns for each channel
        for channel in episode.channel_names:
            data[channel] = []

        # Populate data
        for step in episode.steps:
            data["step_index"].append(step.step_index)
            data["timestamp_offset"].append(step.timestamp_offset)
            for channel in episode.channel_names:
                value = step.condition_monitoring.get(channel, np.nan)
                data[channel].append(value)

        if self.use_parquet:
            self._save_timeseries_parquet(data, episode_dir)
        else:
            self._save_timeseries_json(data, episode_dir)

    def _save_timeseries_parquet(
        self,
        data: Dict[str, List],
        episode_dir: Path,
    ) -> None:
        """Save timeseries as Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert to PyArrow table
            arrays = {}
            for key, values in data.items():
                if key in ("step_index",):
                    arrays[key] = pa.array(values, type=pa.int32())
                else:
                    arrays[key] = pa.array(values, type=pa.float32())

            table = pa.table(arrays)

            # Write to Parquet
            pq.write_table(
                table,
                episode_dir / "timeseries.parquet",
                compression="snappy",
            )

        except ImportError:
            logger.warning("pyarrow not available, falling back to JSON")
            self._save_timeseries_json(data, episode_dir)

    def _save_timeseries_json(
        self,
        data: Dict[str, List],
        episode_dir: Path,
    ) -> None:
        """Save timeseries as JSON file (fallback)."""
        with open(episode_dir / "timeseries.json", "w") as f:
            json.dump(data, f, default=_json_serializer)

    def _save_qa_pairs(
        self,
        qa_pairs: List[Dict[str, Any]],
        episode_dir: Path,
    ) -> None:
        """Save Q&A pairs to JSON."""
        with open(episode_dir / "qa_pairs.json", "w") as f:
            json.dump(qa_pairs, f, indent=2, default=_json_serializer)

    def _save_semantic_priors(
        self,
        priors: SemanticPriors,
        episode_dir: Path,
    ) -> None:
        """Save semantic priors to JSON."""
        priors_dict = {
            "machine_type": priors.machine_type,
            "machine_description": priors.machine_description,
            "typical_failure_modes": priors.typical_failure_modes,
            "maintenance_recommendations": priors.maintenance_recommendations,
            "operating_conditions": priors.operating_conditions,
            "bearing_info": priors.bearing_info,
        }

        with open(episode_dir / "semantic_priors.json", "w") as f:
            json.dump(priors_dict, f, indent=2, default=_json_serializer)

    def _save_features(
        self,
        episode: FactoryNetEpisode,
        episode_dir: Path,
    ) -> None:
        """Save extracted features to JSON."""
        features_dict = episode.to_features_dict()

        with open(episode_dir / "features.json", "w") as f:
            json.dump(features_dict, f, indent=2, default=_json_serializer)

    def load_episode(self, episode_path: Path) -> Dict[str, Any]:
        """Load an episode from disk.

        Args:
            episode_path: Path to episode directory

        Returns:
            Dictionary with episode data
        """
        result = {}

        # Load metadata
        metadata_path = episode_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                result["metadata"] = json.load(f)

        # Load timeseries
        parquet_path = episode_path / "timeseries.parquet"
        json_path = episode_path / "timeseries.json"

        if parquet_path.exists():
            result["timeseries"] = self._load_timeseries_parquet(parquet_path)
        elif json_path.exists():
            with open(json_path) as f:
                result["timeseries"] = json.load(f)

        # Load Q&A pairs
        qa_path = episode_path / "qa_pairs.json"
        if qa_path.exists():
            with open(qa_path) as f:
                result["qa_pairs"] = json.load(f)

        # Load semantic priors
        priors_path = episode_path / "semantic_priors.json"
        if priors_path.exists():
            with open(priors_path) as f:
                result["semantic_priors"] = json.load(f)

        # Load features
        features_path = episode_path / "features.json"
        if features_path.exists():
            with open(features_path) as f:
                result["features"] = json.load(f)

        return result

    def _load_timeseries_parquet(self, path: Path) -> Dict[str, List]:
        """Load timeseries from Parquet file."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            return {col: table[col].to_pylist() for col in table.column_names}

        except ImportError:
            logger.error("pyarrow required to load Parquet files")
            return {}

    def list_episodes(
        self,
        dataset_name: Optional[str] = None,
    ) -> List[Path]:
        """List all saved episodes.

        Args:
            dataset_name: Optional filter by dataset

        Returns:
            List of episode directory paths
        """
        adapted_dir = self.base_dir / "episodes" / "adapted"

        if dataset_name:
            dataset_dir = adapted_dir / dataset_name
            if dataset_dir.exists():
                return sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
            return []

        episodes = []
        if adapted_dir.exists():
            for dataset_dir in adapted_dir.iterdir():
                if dataset_dir.is_dir():
                    episodes.extend([d for d in dataset_dir.iterdir() if d.is_dir()])

        return sorted(episodes)

    def get_dataset_stats(
        self,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Get statistics for a dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dictionary with dataset statistics
        """
        episodes = self.list_episodes(dataset_name)

        total_size = 0
        fault_counts: Dict[str, int] = {}

        for episode_path in episodes:
            # Count file sizes
            for f in episode_path.iterdir():
                if f.is_file():
                    total_size += f.stat().st_size

            # Count fault types from metadata
            metadata_path = episode_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    state = metadata.get("state_annotation", {}).get("state_label", "unknown")
                    fault_counts[state] = fault_counts.get(state, 0) + 1

        return {
            "dataset_name": dataset_name,
            "num_episodes": len(episodes),
            "total_size_mb": total_size / (1024 * 1024),
            "fault_distribution": fault_counts,
        }

    def save_validation_report(
        self,
        report_data: Dict[str, Any],
        dataset_name: str,
    ) -> Path:
        """Save validation report.

        Args:
            report_data: Report data dictionary
            dataset_name: Name of dataset

        Returns:
            Path to saved report
        """
        report_path = (
            self.base_dir / "validation_reports" /
            f"{dataset_name}_report.json"
        )

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=_json_serializer)

        return report_path
