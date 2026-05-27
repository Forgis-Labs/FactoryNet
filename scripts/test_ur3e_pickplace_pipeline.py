#!/usr/bin/env python3
"""Test script for UR3e Pick-Place anomaly detection dataset.

Downloads sample UR3e data and processes it through the full pipeline.
Dataset: https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-anomaly-detection
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

# UR3e joint names
UR3E_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def create_synthetic_test_data(data_dir: Path) -> bool:
    """Create synthetic test data mimicking UR3e Pick-Place structure.

    UR3e Pick-Place structure:
    ur3e_pickplace/
    ├── normal/
    │   ├── operation_001.csv
    │   ├── operation_002.csv
    │   └── ...
    ├── anomaly_high/
    │   └── ...
    └── anomaly_low/
        └── ...

    Columns per joint: position (rad), velocity (rad/s), current (A)
    Plus: timestamp, label

    Args:
        data_dir: Directory to create test files

    Returns:
        True if successful
    """
    logger.info("Creating synthetic UR3e Pick-Place test data for adapter verification...")

    # Simulation parameters
    sampling_rate = 125  # Hz (typical UR robot rate)
    duration_sec = 5.0  # 5 seconds per pick-place operation
    num_samples = int(sampling_rate * duration_sec)
    ops_per_class = 10  # Operations per anomaly class

    # Time array
    t = np.linspace(0, duration_sec, num_samples)

    # Pick-place motion frequency
    motion_freq = 0.5  # Complete cycle in 2 seconds

    # Columns
    columns = ["timestamp"]
    for joint in UR3E_JOINTS:
        columns.extend([f"{joint}_position", f"{joint}_velocity", f"{joint}_current"])
    columns.append("label")

    # Create directories and data for each class
    classes = [
        ("normal", 0, 1.0),  # (folder, label, velocity_multiplier)
        ("anomaly_high", 1, 1.4),  # Higher velocity
        ("anomaly_low", 2, 0.6),  # Lower velocity
    ]

    for class_name, label, vel_mult in classes:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for op_num in range(1, ops_per_class + 1):
            file_path = class_dir / f"operation_{op_num:03d}.csv"
            if file_path.exists():
                continue

            # Generate data
            data = np.zeros((num_samples, len(columns)))
            data[:, 0] = t  # Timestamp

            col_idx = 1
            for j, joint in enumerate(UR3E_JOINTS):
                # Joint position (rad) - pick-place motion
                if joint in ["base", "wrist3"]:
                    # Rotational joints - sinusoidal motion
                    amplitude = 0.5 + 0.2 * j
                    phase = j * np.pi / 6
                    position = amplitude * np.sin(2 * np.pi * motion_freq * t + phase)
                elif joint in ["shoulder", "elbow"]:
                    # Main arm joints - larger motion for pick-place
                    amplitude = 0.8 + 0.1 * j
                    position = amplitude * np.sin(2 * np.pi * motion_freq * t)
                else:
                    # Wrist joints
                    amplitude = 0.3
                    position = amplitude * np.sin(2 * np.pi * 2 * motion_freq * t)

                # Add some noise
                position += 0.01 * np.random.randn(num_samples)
                data[:, col_idx] = position

                # Joint velocity (rad/s) - derivative of position
                velocity = amplitude * 2 * np.pi * motion_freq * np.cos(2 * np.pi * motion_freq * t)
                velocity *= vel_mult  # Apply velocity modification for anomalies
                velocity += 0.05 * np.random.randn(num_samples)
                data[:, col_idx + 1] = velocity

                # Motor current (A) - proportional to velocity/torque
                base_current = 0.5 + 0.1 * j
                current = base_current + 0.3 * np.abs(velocity) + 0.05 * np.random.randn(num_samples)
                data[:, col_idx + 2] = current

                col_idx += 3

            # Label column
            data[:, -1] = label

            # Save as CSV with headers
            header = ",".join(columns)
            np.savetxt(str(file_path), data, delimiter=",", fmt="%.6f", header=header, comments="")
            logger.info(f"Created: {class_name}/operation_{op_num:03d}.csv (label={label})")

    return True


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the UR3e Pick-Place adapter with data files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing UR3e Pick-Place adapter...")

    adapter = AdapterRegistry.get("ur3e_pickplace", data_dir=data_dir)

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

            if episode.channels:
                ch = episode.channels[0]
                logger.info(f"    First channel: {ch.channel_type}")
                logger.info(f"    Samples: {len(ch.data)}")
                logger.info(f"    Sampling rate: {ch.sampling_rate_hz} Hz")
                logger.info(f"    Duration: {len(ch.data) / ch.sampling_rate_hz:.2f} s")

            if episode.raw_metadata:
                logger.info(f"    Robot model: {episode.raw_metadata.get('robot_model', 'N/A')}")
                logger.info(f"    Task: {episode.raw_metadata.get('task', 'N/A')}")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with UR3e Pick-Place data."""
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
    stats = pipeline.process_dataset("ur3e_pickplace", data_dir=data_dir)

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
    episodes = storage.list_episodes("ur3e_pickplace")

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
            logger.info(f"  Duration: {meta.get('duration_seconds', 0):.2f}s")
            logger.info(f"  Machine: {meta.get('machine_synset')}")
            state = meta.get('state_annotation', {})
            logger.info(f"  State: {state.get('state_label')}")

            raw_meta = meta.get('raw_metadata', {})
            logger.info(f"  Robot model: {raw_meta.get('robot_model', 'N/A')}")

        if "qa_pairs" in data:
            logger.info(f"  Q&A pairs: {len(data['qa_pairs'])}")

        if "timeseries" in data:
            ts = data["timeseries"]
            # Show joint-related columns
            joint_cols = [k for k in ts.keys() if any(j in k for j in UR3E_JOINTS)]
            logger.info(f"  Joint channels: {len(joint_cols)}")
            if "step_index" in ts:
                logger.info(f"  Timeseries samples: {len(ts['step_index'])}")


def main():
    """Main test function."""
    # Setup directories
    test_data_dir = PROJECT_ROOT / "test_data" / "ur3e_pickplace"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("UR3e Pick-Place Anomaly Detection Pipeline Test")
    logger.info("=" * 60)

    # Step 1: Get data
    logger.info("\n[1/4] Obtaining UR3e Pick-Place data...")

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
    logger.info("\n[2/4] Testing UR3e Pick-Place adapter...")
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
