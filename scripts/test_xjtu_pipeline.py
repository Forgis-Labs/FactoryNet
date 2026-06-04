#!/usr/bin/env python3
"""Test script for XJTU-SY bearing dataset download and processing.

Downloads sample XJTU-SY data and processes it through the full pipeline.
Dataset: https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_synthetic_test_data(data_dir: Path) -> bool:
    """Create synthetic test data mimicking XJTU-SY structure.

    XJTU-SY structure:
    xjtu_sy/
    ├── 35Hz12kN/
    │   ├── Bearing1_1/
    │   │   ├── 1.csv
    │   │   ├── 2.csv
    │   │   └── ...
    │   └── Bearing1_2/
    └── ...

    Args:
        data_dir: Directory to create test files

    Returns:
        True if successful
    """
    import numpy as np

    logger.info("Creating synthetic XJTU-SY test data for adapter verification...")

    # Operating conditions and bearings to simulate
    test_bearings = [
        ("35Hz12kN", "Bearing1_1", 2100, 12.0, "outer_race"),
        ("35Hz12kN", "Bearing1_2", 2100, 12.0, "outer_race"),
        ("37.5Hz11kN", "Bearing2_1", 2250, 11.0, "inner_race"),
        ("40Hz10kN", "Bearing3_1", 2400, 10.0, "outer_race"),
    ]

    # XJTU-SY: 25.6 kHz, 32768 samples per file (1.28 seconds)
    sampling_rate = 25600
    samples_per_file = 32768
    files_per_bearing = 10  # Simulate 10 minutes of data per bearing

    for condition, bearing_id, rpm, load_kn, fault_type in test_bearings:
        bearing_dir = data_dir / condition / bearing_id
        bearing_dir.mkdir(parents=True, exist_ok=True)

        # Create CSV files simulating degradation progression
        for file_num in range(1, files_per_bearing + 1):
            file_path = bearing_dir / f"{file_num}.csv"

            if file_path.exists():
                continue

            # Generate synthetic vibration data
            t = np.linspace(0, samples_per_file / sampling_rate, samples_per_file)
            base_freq = rpm / 60  # Shaft frequency

            # Horizontal acceleration
            horizontal = (
                0.1 * np.sin(2 * np.pi * base_freq * t) +
                0.05 * np.sin(2 * np.pi * 2 * base_freq * t) +
                0.02 * np.random.randn(samples_per_file)
            )

            # Vertical acceleration
            vertical = (
                0.08 * np.sin(2 * np.pi * base_freq * t + np.pi/4) +
                0.04 * np.sin(2 * np.pi * 2 * base_freq * t) +
                0.02 * np.random.randn(samples_per_file)
            )

            # Add fault signature that increases with progression
            progress = file_num / files_per_bearing
            if fault_type == "outer_race":
                bpfo = base_freq * 3.5  # Ball pass frequency outer
                fault_amplitude = 0.1 * progress ** 2
                horizontal += fault_amplitude * np.sin(2 * np.pi * bpfo * t)
                vertical += fault_amplitude * 0.8 * np.sin(2 * np.pi * bpfo * t)
            elif fault_type == "inner_race":
                bpfi = base_freq * 5.5  # Ball pass frequency inner
                fault_amplitude = 0.12 * progress ** 2
                horizontal += fault_amplitude * np.sin(2 * np.pi * bpfi * t)
                vertical += fault_amplitude * 0.7 * np.sin(2 * np.pi * bpfi * t)

            # Stack and save as CSV
            data = np.column_stack([horizontal, vertical])
            np.savetxt(str(file_path), data, delimiter=",", fmt="%.8f")

        logger.info(f"Created: {condition}/{bearing_id}/ ({files_per_bearing} files)")

    return True


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the XJTU-SY adapter with data files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing XJTU-SY adapter...")

    adapter = AdapterRegistry.get("xjtu_sy", data_dir=data_dir)

    # Discover files
    files = adapter.discover_files()
    logger.info(f"Discovered {len(files)} CSV files")

    if not files:
        logger.warning("No files found!")
        return

    # Parse files
    episode_count = 0
    for file_path in files[:limit]:
        rel_path = file_path.relative_to(data_dir)
        logger.info(f"\nParsing {rel_path}:")

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

            if episode.raw_metadata:
                logger.info(f"    Bearing: {episode.raw_metadata.get('bearing_id', 'N/A')}")
                logger.info(f"    RUL estimate: {episode.raw_metadata.get('rul_estimate_minutes', 'N/A')} min")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with XJTU-SY data."""
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
    stats = pipeline.process_dataset("xjtu_sy", data_dir=data_dir)

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
    episodes = storage.list_episodes("xjtu_sy")

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
    test_data_dir = PROJECT_ROOT / "test_data" / "xjtu_sy"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("XJTU-SY Bearing Dataset Pipeline Test")
    logger.info("=" * 60)

    # Step 1: Get data
    logger.info("\n[1/4] Obtaining XJTU-SY data...")

    # Check if data already exists
    csv_files = list(test_data_dir.rglob("*.csv"))
    if csv_files:
        logger.info(f"Found {len(csv_files)} existing CSV files")
    else:
        # Create synthetic test data
        logger.info("Creating synthetic test data for adapter verification...")
        create_synthetic_test_data(test_data_dir)

    # Check if we have data
    csv_files = list(test_data_dir.rglob("*.csv"))
    if not csv_files:
        logger.error("No CSV files available, cannot continue")
        return 1

    logger.info(f"Found {len(csv_files)} CSV files")

    # Step 2: Test adapter
    logger.info("\n[2/4] Testing XJTU-SY adapter...")
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
