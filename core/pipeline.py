"""Data pipeline orchestrator for FactoryNet.

Coordinates the full flow from dataset adapters through normalization,
validation, Q&A generation, and storage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

from tqdm import tqdm

from core.adapters.base_adapter import BaseDatasetAdapter, RawEpisode
from core.adapters.registry import AdapterRegistry
from core.normalizer import EpisodeNormalizer, FactoryNetEpisode
from core.qa_generator import QAGenerator
from core.storage import EpisodeStorage
from core.validation import (
    EpisodeValidator,
    ValidationReport,
    ValidationReportGenerator,
    ValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    # Output directory
    output_dir: str = "./factorynet_data"

    # Processing options
    extract_features: bool = True
    generate_qa: bool = True
    validate_episodes: bool = True

    # Demo mode
    demo_mode: bool = False
    demo_limit: int = 10

    # Quality gates
    sensor_completeness_threshold: float = 0.95
    label_confidence_threshold: float = 0.85

    # Q&A generation
    questions_per_category: int = 2

    # Storage
    use_parquet: bool = True

    # Logging
    verbose: bool = True


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    dataset_name: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Counts
    raw_episodes_processed: int = 0
    episodes_normalized: int = 0
    episodes_validated: int = 0
    episodes_passed: int = 0
    episodes_failed: int = 0
    episodes_saved: int = 0
    qa_pairs_generated: int = 0

    # Quality metrics
    avg_sensor_completeness: float = 0.0
    avg_label_confidence: float = 0.0
    avg_quality_score: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get pipeline duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def pass_rate(self) -> float:
        """Get validation pass rate."""
        total = self.episodes_passed + self.episodes_failed
        return self.episodes_passed / total * 100 if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "raw_episodes_processed": self.raw_episodes_processed,
            "episodes_normalized": self.episodes_normalized,
            "episodes_validated": self.episodes_validated,
            "episodes_passed": self.episodes_passed,
            "episodes_failed": self.episodes_failed,
            "episodes_saved": self.episodes_saved,
            "qa_pairs_generated": self.qa_pairs_generated,
            "pass_rate": self.pass_rate,
            "avg_sensor_completeness": self.avg_sensor_completeness,
            "avg_label_confidence": self.avg_label_confidence,
            "avg_quality_score": self.avg_quality_score,
            "error_count": len(self.errors),
        }


class DataPipeline:
    """Orchestrates the full data processing pipeline.

    Pipeline stages:
    1. Adapter: Parse raw files → RawEpisode
    2. Normalizer: RawEpisode → FactoryNetEpisode
    3. Validator: Check quality gates
    4. QA Generator: Generate Q&A pairs
    5. Storage: Save to disk

    Example:
        pipeline = DataPipeline(config=PipelineConfig(demo_mode=True))
        stats = pipeline.process_dataset("cwru_bearing", data_dir="./data/cwru")
        print(f"Processed {stats.episodes_saved} episodes")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.normalizer = EpisodeNormalizer(
            extract_features=self.config.extract_features,
        )

        self.validator = EpisodeValidator(
            sensor_completeness_threshold=self.config.sensor_completeness_threshold,
            label_confidence_threshold=self.config.label_confidence_threshold,
        )

        self.qa_generator = QAGenerator(
            questions_per_category=self.config.questions_per_category,
        )

        self.storage = EpisodeStorage(
            base_dir=self.config.output_dir,
            use_parquet=self.config.use_parquet,
        )

        self.report_generator = ValidationReportGenerator(self.validator)

    def process_dataset(
        self,
        dataset_name: str,
        data_dir: str | Path,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> PipelineStats:
        """Process a complete dataset.

        Args:
            dataset_name: Registered adapter name
            data_dir: Directory containing dataset files
            adapter_kwargs: Additional arguments for adapter

        Returns:
            PipelineStats with processing results
        """
        stats = PipelineStats(dataset_name=dataset_name)

        try:
            # Get adapter
            adapter = AdapterRegistry.get(
                dataset_name,
                data_dir=data_dir,
                **(adapter_kwargs or {}),
            )

            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"Data directory: {data_dir}")
            logger.info(f"Output directory: {self.config.output_dir}")

            if self.config.demo_mode:
                logger.info(f"Demo mode: processing {self.config.demo_limit} episodes")

            # Process episodes
            stats = self._process_adapter(adapter, stats)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            stats.errors.append(str(e))

        stats.completed_at = datetime.now()

        # Log summary
        self._log_summary(stats)

        return stats

    def process_adapter(
        self,
        adapter: BaseDatasetAdapter,
    ) -> PipelineStats:
        """Process episodes from an adapter instance.

        Args:
            adapter: Initialized adapter

        Returns:
            PipelineStats with processing results
        """
        stats = PipelineStats(dataset_name=adapter.metadata.name)
        stats = self._process_adapter(adapter, stats)
        stats.completed_at = datetime.now()
        return stats

    def _process_adapter(
        self,
        adapter: BaseDatasetAdapter,
        stats: PipelineStats,
    ) -> PipelineStats:
        """Internal method to process adapter episodes."""
        # Get episode limit
        limit = self.config.demo_limit if self.config.demo_mode else None

        # Collect episodes for validation report
        processed_episodes: List[FactoryNetEpisode] = []
        validation_results: List[ValidationResult] = []

        # Process episodes with progress bar
        episodes_iter = adapter.iter_episodes(limit=limit)

        if self.config.verbose:
            total = limit or adapter.metadata.num_samples
            episodes_iter = tqdm(
                episodes_iter,
                total=total,
                desc=f"Processing {adapter.metadata.name}",
            )

        for raw_episode in episodes_iter:
            stats.raw_episodes_processed += 1

            try:
                # Normalize
                fn_episode = self.normalizer.normalize(raw_episode)
                stats.episodes_normalized += 1

                # Validate
                if self.config.validate_episodes:
                    result = self.validator.validate(fn_episode)
                    stats.episodes_validated += 1
                    validation_results.append(result)

                    if result.valid:
                        stats.episodes_passed += 1
                    else:
                        stats.episodes_failed += 1
                        if not self.config.demo_mode:
                            # In production, skip invalid episodes
                            continue

                # Generate Q&A
                qa_pairs = []
                if self.config.generate_qa:
                    qa_pairs = self.qa_generator.generate(fn_episode)
                    stats.qa_pairs_generated += len(qa_pairs)

                # Save
                qa_dicts = [qa.to_dict() for qa in qa_pairs]
                self.storage.save_episode(fn_episode, qa_pairs=qa_dicts)
                stats.episodes_saved += 1

                processed_episodes.append(fn_episode)

            except Exception as e:
                logger.warning(f"Error processing episode {raw_episode.raw_id}: {e}")
                stats.errors.append(f"{raw_episode.raw_id}: {e}")

        # Compute aggregate metrics
        if validation_results:
            stats.avg_sensor_completeness = sum(
                r.sensor_completeness for r in validation_results
            ) / len(validation_results)
            stats.avg_label_confidence = sum(
                r.label_confidence for r in validation_results
            ) / len(validation_results)
            stats.avg_quality_score = sum(
                r.overall_quality_score for r in validation_results
            ) / len(validation_results)

        # Generate and save validation report
        if self.config.validate_episodes and processed_episodes:
            report = self.report_generator.generate_report(
                processed_episodes,
                adapter.metadata.name,
            )
            self.storage.save_validation_report(
                report.to_dict(),
                adapter.metadata.name,
            )

        return stats

    def _log_summary(self, stats: PipelineStats) -> None:
        """Log processing summary."""
        logger.info("=" * 50)
        logger.info(f"Pipeline completed for {stats.dataset_name}")
        logger.info(f"Duration: {stats.duration_seconds:.1f} seconds")
        logger.info(f"Raw episodes processed: {stats.raw_episodes_processed}")
        logger.info(f"Episodes normalized: {stats.episodes_normalized}")
        logger.info(f"Episodes passed validation: {stats.episodes_passed}")
        logger.info(f"Episodes failed validation: {stats.episodes_failed}")
        logger.info(f"Episodes saved: {stats.episodes_saved}")
        logger.info(f"Q&A pairs generated: {stats.qa_pairs_generated}")
        logger.info(f"Pass rate: {stats.pass_rate:.1f}%")

        if stats.errors:
            logger.warning(f"Errors encountered: {len(stats.errors)}")

        logger.info("=" * 50)


def run_pipeline(
    datasets: List[str],
    data_dirs: Dict[str, str],
    output_dir: str = "./factorynet_data",
    demo_mode: bool = False,
    demo_limit: int = 10,
) -> Dict[str, PipelineStats]:
    """Convenience function to run pipeline for multiple datasets.

    Args:
        datasets: List of dataset names to process
        data_dirs: Mapping of dataset names to data directories
        output_dir: Output directory
        demo_mode: Enable demo mode
        demo_limit: Episode limit in demo mode

    Returns:
        Dictionary mapping dataset names to PipelineStats
    """
    config = PipelineConfig(
        output_dir=output_dir,
        demo_mode=demo_mode,
        demo_limit=demo_limit,
    )

    pipeline = DataPipeline(config=config)
    results = {}

    for dataset_name in datasets:
        if dataset_name not in data_dirs:
            logger.warning(f"No data directory specified for {dataset_name}")
            continue

        stats = pipeline.process_dataset(
            dataset_name=dataset_name,
            data_dir=data_dirs[dataset_name],
        )
        results[dataset_name] = stats

    return results
