#!/usr/bin/env python3
"""Test script for MAFAULDA dataset download and processing.

Downloads sample MAFAULDA data and processes it through the full pipeline.
Dataset: https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset
"""
from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_from_kaggle(data_dir: Path) -> bool:
    """Download MAFAULDA from Kaggle using kagglehub.

    Args:
        data_dir: Directory to save files

    Returns:
        True if successful
    """
    try:
        import kagglehub
    except ImportError:
        logger.warning("kagglehub not installed. Run: pip install kagglehub")
        return False

    try:
        logger.info("Downloading from Kaggle (this may take a while, ~2.5GB)...")
        path = kagglehub.dataset_download("uysalserkan/fault-induction-motor-dataset")
        logger.info(f"Downloaded to: {path}")

        # Copy/link to our data directory
        import shutil
        data_dir.mkdir(parents=True, exist_ok=True)

        # Check what was downloaded
        download_path = Path(path)
        if download_path.is_dir():
            for item in download_path.iterdir():
                dest = data_dir / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                    logger.info(f"Copied: {item.name}")

        return True

    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


def download_sample_data(data_dir: Path) -> bool:
    """Download sample MAFAULDA data.

    Tries multiple sources in order:
    1. Kaggle (if credentials available)
    2. Direct URL (if available)

    Args:
        data_dir: Directory to save files

    Returns:
        True if data is available
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    csv_files = list(data_dir.rglob("*.csv"))
    if csv_files:
        logger.info(f"Found {len(csv_files)} existing CSV files in {data_dir}")
        return True

    # Try Kaggle
    logger.info("Attempting to download from Kaggle...")
    if download_from_kaggle(data_dir):
        return True

    logger.error("Could not download MAFAULDA dataset.")
    logger.error("Please download manually from:")
    logger.error("  https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset")
    logger.error(f"And extract to: {data_dir}")
    return False


def create_synthetic_test_data(data_dir: Path) -> bool:
    """Create synthetic test data for adapter testing.

    Creates CSV files that mimic MAFAULDA structure for testing.

    Args:
        data_dir: Directory to create test files

    Returns:
        True if successful
    """
    import numpy as np

    logger.info("Creating synthetic test data for adapter verification...")

    # Create directory structure
    fault_dirs = {
        "normal": (None, None),
        "imbalance/6g": ("imbalance", "6g"),
        "imbalance/20g": ("imbalance", "20g"),
        "horizontal-misalignment/1.0mm": ("misalignment", "1.0mm"),
        "underhang/outer-race": ("outer_race", None),
    }

    # MAFAULDA has 9 columns: tachometer + 8 accelerometers
    num_samples = 250000  # 5 seconds at 50kHz
    num_channels = 9

    for subdir, (fault_type, severity) in fault_dirs.items():
        dir_path = data_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create a few test files
        for i in range(3):
            rpm = 1500 + i * 100
            filename = f"{rpm}rpm_test_{i+1}.csv"
            file_path = dir_path / filename

            if file_path.exists():
                continue

            # Generate synthetic sensor data
            t = np.linspace(0, 5, num_samples)
            data = np.zeros((num_samples, num_channels))

            # Tachometer - pulse signal
            data[:, 0] = np.sin(2 * np.pi * (rpm / 60) * t)

            # Accelerometers - vibration signals with harmonics
            for ch in range(1, 9):
                base_freq = rpm / 60
                data[:, ch] = (
                    0.1 * np.sin(2 * np.pi * base_freq * t) +
                    0.05 * np.sin(2 * np.pi * 2 * base_freq * t) +
                    0.02 * np.random.randn(num_samples)
                )

                # Add fault signatures
                if fault_type == "imbalance":
                    data[:, ch] += 0.3 * np.sin(2 * np.pi * base_freq * t)
                elif fault_type == "misalignment":
                    data[:, ch] += 0.2 * np.sin(2 * np.pi * 2 * base_freq * t)
                elif fault_type == "outer_race":
                    # Simulate BPFO harmonics
                    bpfo = base_freq * 3.5
                    data[:, ch] += 0.15 * np.sin(2 * np.pi * bpfo * t)

            # Save as CSV
            np.savetxt(str(file_path), data, delimiter=",", fmt="%.6f")
            logger.info(f"Created: {file_path.relative_to(data_dir)}")

    return True


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the MAFAULDA adapter with downloaded files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing MAFAULDA adapter...")

    adapter = AdapterRegistry.get("mafaulda", data_dir=data_dir)

    # Discover files
    files = adapter.discover_files()
    logger.info(f"Discovered {len(files)} files")

    if not files:
        logger.warning("No files found!")
        return

    # Parse files
    episode_count = 0
    for file_path in files[:limit]:
        logger.info(f"\nParsing {file_path.relative_to(data_dir)}:")

        for episode in adapter.parse_file(file_path):
            episode_count += 1
            logger.info(f"  Episode: {episode.raw_id}")
            logger.info(f"    Fault type: {episode.fault_type.value}")
            logger.info(f"    Severity: {episode.severity.value}")
            logger.info(f"    Channels: {len(episode.channels)}")
            logger.info(f"    RPM: {episode.rpm}")

            if episode.channels:
                ch = episode.channels[0]
                logger.info(f"    First channel: {ch.channel_type}")
                logger.info(f"    Samples: {len(ch.data)}")
                logger.info(f"    Sampling rate: {ch.sampling_rate_hz} Hz")
                logger.info(f"    Duration: {len(ch.data) / ch.sampling_rate_hz:.2f} s")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with MAFAULDA data."""
    from core.pipeline import DataPipeline, PipelineConfig

    logger.info("\nRunning full pipeline...")

    config = PipelineConfig(
        output_dir=str(output_dir),
        extract_features=True,
        generate_qa=True,
        validate_episodes=True,
        demo_mode=True,
        demo_limit=5,
        use_parquet=True,
        verbose=True,
    )

    pipeline = DataPipeline(config=config)
    stats = pipeline.process_dataset("mafaulda", data_dir=data_dir)

    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Results:")
    logger.info(f"  Raw episodes processed: {stats.raw_episodes_processed}")
    logger.info(f"  Episodes normalized: {stats.episodes_normalized}")
    logger.info(f"  Episodes passed validation: {stats.episodes_passed}")
    logger.info(f"  Episodes saved: {stats.episodes_saved}")
    logger.info(f"  Q&A pairs generated: {stats.qa_pairs_generated}")
    logger.info(f"  Pass rate: {stats.pass_rate:.1f}%")
    logger.info(f"  Avg sensor completeness: {stats.avg_sensor_completeness:.2%}")
    logger.info(f"  Avg quality score: {stats.avg_quality_score:.2%}")

    if stats.errors:
        logger.warning(f"  Errors: {len(stats.errors)}")
        for err in stats.errors[:3]:
            logger.warning(f"    - {err}")


def verify_output(output_dir: Path) -> None:
    """Verify the output structure and content."""
    from core.storage import EpisodeStorage

    logger.info("\nVerifying output...")

    storage = EpisodeStorage(base_dir=output_dir)
    episodes = storage.list_episodes("mafaulda")

    logger.info(f"Found {len(episodes)} saved episodes")

    for episode_path in episodes[:2]:
        logger.info(f"\nEpisode: {episode_path.name}")

        # List files
        for f in episode_path.iterdir():
            size = f.stat().st_size
            logger.info(f"  {f.name}: {size:,} bytes")

        # Load and check metadata
        data = storage.load_episode(episode_path)

        if "metadata" in data:
            meta = data["metadata"]
            logger.info(f"  Episode ID: {meta.get('episode_id')}")
            logger.info(f"  Source: {meta.get('source_dataset')}")
            logger.info(f"  Duration: {meta.get('duration_seconds', 0):.2f}s")
            logger.info(f"  Machine: {meta.get('machine_synset')}")
            state = meta.get('state_annotation', {})
            logger.info(f"  State: {state.get('state_label')}")

        if "qa_pairs" in data:
            logger.info(f"  Q&A pairs: {len(data['qa_pairs'])}")

        if "timeseries" in data:
            ts = data["timeseries"]
            logger.info(f"  Timeseries columns: {list(ts.keys())[:5]}...")
            if "step_index" in ts:
                logger.info(f"  Timeseries samples: {len(ts['step_index'])}")


def main():
    """Main test function."""
    # Setup directories
    test_data_dir = PROJECT_ROOT / "test_data" / "mafaulda"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("MAFAULDA Dataset Pipeline Test")
    logger.info("=" * 60)

    # Step 1: Get data
    logger.info("\n[1/4] Obtaining MAFAULDA data...")

    # First try to download real data
    if not download_sample_data(test_data_dir):
        # Fall back to synthetic test data
        logger.info("Creating synthetic test data for adapter verification...")
        create_synthetic_test_data(test_data_dir)

    # Check if we have data
    csv_files = list(test_data_dir.rglob("*.csv"))
    if not csv_files:
        logger.error("No CSV files available, cannot continue")
        return 1

    logger.info(f"Found {len(csv_files)} CSV files")

    # Step 2: Test adapter
    logger.info("\n[2/4] Testing MAFAULDA adapter...")
    test_adapter(test_data_dir, limit=5)

    # Step 3: Run full pipeline
    logger.info("\n[3/4] Running full pipeline...")
    test_full_pipeline(test_data_dir, test_output_dir)

    # Step 4: Verify output
    logger.info("\n[4/4] Verifying output...")
    verify_output(test_output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Test completed!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
