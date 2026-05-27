#!/usr/bin/env python3
"""Test script for PHM 2010 CNC milling dataset download and processing.

Downloads sample PHM 2010 data and processes it through the full pipeline.
Dataset: https://www.phmsociety.org/competition/phm/10
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_synthetic_test_data(data_dir: Path) -> bool:
    """Create synthetic test data mimicking PHM 2010 structure.

    PHM 2010 structure:
    phm2010/
    ├── train/
    │   ├── c_1.csv      # Cut 1 sensor data (7 columns)
    │   ├── c_2.csv      # Cut 2 sensor data
    │   ├── ...
    │   └── train_labels.csv  # Wear measurements per cut
    └── test/
        └── ...

    Columns: force_x, force_y, force_z, vibration_x, vibration_y, vibration_z, AE

    Args:
        data_dir: Directory to create test files

    Returns:
        True if successful
    """
    logger.info("Creating synthetic PHM 2010 test data for adapter verification...")

    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    sampling_rate = 50000  # 50 kHz
    spindle_rpm = 10400
    duration_sec = 0.5  # Short cuts for testing
    num_samples = int(sampling_rate * duration_sec)
    num_cuts = 20  # Simulate 20 cuts with progressive wear

    # Time array
    t = np.linspace(0, duration_sec, num_samples)
    spindle_freq = spindle_rpm / 60  # Hz

    # Generate wear labels (progressive wear over cuts)
    wear_labels = []
    max_wear = 0.25  # mm - end of life

    for cut_num in range(1, num_cuts + 1):
        # Simulate progressive tool wear
        progress = (cut_num - 1) / (num_cuts - 1)
        wear_mm = max_wear * (progress ** 1.5)  # Non-linear wear progression

        wear_labels.append([cut_num, wear_mm * 0.9, wear_mm * 1.0, wear_mm * 1.1])

        file_path = train_dir / f"c_{cut_num}.csv"
        if file_path.exists():
            continue

        # Generate sensor data with wear-dependent characteristics
        data = np.zeros((num_samples, 7))

        # Base cutting forces (increase with wear)
        force_base = 50 + 100 * progress
        force_noise = 10 + 20 * progress

        # Force X (feed direction)
        data[:, 0] = (
            force_base * np.sin(2 * np.pi * spindle_freq * t) +
            force_noise * np.random.randn(num_samples)
        )

        # Force Y (cross-feed)
        data[:, 1] = (
            force_base * 0.6 * np.sin(2 * np.pi * spindle_freq * t + np.pi/4) +
            force_noise * 0.8 * np.random.randn(num_samples)
        )

        # Force Z (axial)
        data[:, 2] = (
            force_base * 0.4 * np.sin(2 * np.pi * spindle_freq * t + np.pi/2) +
            force_noise * 0.6 * np.random.randn(num_samples)
        )

        # Vibration signals (increase with wear)
        vib_base = 0.5 + 2.0 * progress
        vib_noise = 0.1 + 0.5 * progress

        # Vibration X
        data[:, 3] = (
            vib_base * np.sin(2 * np.pi * spindle_freq * t) +
            vib_base * 0.5 * np.sin(2 * np.pi * 2 * spindle_freq * t) +  # 2x harmonic
            vib_noise * np.random.randn(num_samples)
        )

        # Vibration Y
        data[:, 4] = (
            vib_base * 0.8 * np.sin(2 * np.pi * spindle_freq * t + np.pi/6) +
            vib_base * 0.4 * np.sin(2 * np.pi * 2 * spindle_freq * t) +
            vib_noise * np.random.randn(num_samples)
        )

        # Vibration Z
        data[:, 5] = (
            vib_base * 0.6 * np.sin(2 * np.pi * spindle_freq * t + np.pi/3) +
            vib_noise * np.random.randn(num_samples)
        )

        # Acoustic Emission (high frequency content increases with wear)
        ae_carrier = 20000  # 20 kHz carrier
        ae_base = 0.1 + 0.5 * progress
        data[:, 6] = (
            ae_base * np.sin(2 * np.pi * ae_carrier * t) *
            (1 + 0.3 * np.sin(2 * np.pi * spindle_freq * t)) +
            0.05 * np.random.randn(num_samples)
        )

        # Add impulses for worn tool (chatter)
        if progress > 0.5:
            num_impulses = int(5 * progress)
            impulse_locs = np.random.randint(0, num_samples, num_impulses)
            impulse_amp = 3.0 * progress
            for loc in impulse_locs:
                end = min(loc + 100, num_samples)
                data[loc:end, 3:6] += impulse_amp * np.exp(-np.linspace(0, 5, end - loc))[:, np.newaxis]

        # Save as CSV
        np.savetxt(str(file_path), data, delimiter=",", fmt="%.6f")
        logger.info(f"Created: c_{cut_num}.csv (wear={wear_mm:.3f}mm)")

    # Save wear labels
    labels_path = train_dir / "train_labels.csv"
    if not labels_path.exists():
        with open(labels_path, "w") as f:
            f.write("cut,flute1,flute2,flute3\n")
            for row in wear_labels:
                f.write(f"{int(row[0])},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f}\n")
        logger.info(f"Created: train_labels.csv ({len(wear_labels)} entries)")

    return True


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the PHM 2010 adapter with data files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing PHM 2010 adapter...")

    adapter = AdapterRegistry.get("phm_2010", data_dir=data_dir)

    # Discover files
    files = adapter.discover_files()
    logger.info(f"Discovered {len(files)} data files")

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
                logger.info(f"    Duration: {len(ch.data) / ch.sampling_rate_hz:.3f} s")

            if episode.raw_metadata:
                logger.info(f"    Cut number: {episode.raw_metadata.get('cut_number', 'N/A')}")
                logger.info(f"    Tool wear: {episode.raw_metadata.get('tool_wear_mm', 'N/A'):.4f} mm")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with PHM 2010 data."""
    from core.pipeline import DataPipeline, PipelineConfig

    logger.info("\nRunning full pipeline...")

    config = PipelineConfig(
        output_dir=str(output_dir),
        extract_features=True,
        generate_qa=True,
        validate_episodes=True,
        demo_mode=True,
        demo_limit=10,
        use_parquet=True,
        verbose=True,
    )

    pipeline = DataPipeline(config=config)
    stats = pipeline.process_dataset("phm_2010", data_dir=data_dir)

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
    episodes = storage.list_episodes("phm_2010")

    logger.info(f"Found {len(episodes)} saved episodes")

    for episode_path in episodes[:3]:
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
            logger.info(f"  Duration: {meta.get('duration_seconds', 0):.3f}s")
            logger.info(f"  Machine: {meta.get('machine_synset')}")
            state = meta.get('state_annotation', {})
            logger.info(f"  State: {state.get('state_label')}")

            # Show tool wear from raw metadata
            raw_meta = meta.get('raw_metadata', {})
            if 'tool_wear_mm' in raw_meta:
                logger.info(f"  Tool wear: {raw_meta['tool_wear_mm']:.4f} mm")

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
    test_data_dir = PROJECT_ROOT / "test_data" / "phm2010"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("PHM 2010 CNC Milling Dataset Pipeline Test")
    logger.info("=" * 60)

    # Step 1: Get data
    logger.info("\n[1/4] Obtaining PHM 2010 data...")

    # Check if data already exists
    csv_files = list(test_data_dir.rglob("*.csv"))
    csv_files = [f for f in csv_files if "label" not in f.name.lower()]

    if csv_files:
        logger.info(f"Found {len(csv_files)} existing CSV files")
    else:
        # Create synthetic test data
        logger.info("Creating synthetic test data for adapter verification...")
        create_synthetic_test_data(test_data_dir)

    # Check if we have data
    csv_files = list(test_data_dir.rglob("*.csv"))
    csv_files = [f for f in csv_files if "label" not in f.name.lower()]

    if not csv_files:
        logger.error("No CSV files available, cannot continue")
        return 1

    logger.info(f"Found {len(csv_files)} CSV files")

    # Step 2: Test adapter
    logger.info("\n[2/4] Testing PHM 2010 adapter...")
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
