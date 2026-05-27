#!/usr/bin/env python3
"""Test script for PHM 2021 SCARA robot dataset download and processing.

Downloads sample PHM 2021 data and processes it through the full pipeline.
Dataset: https://data.phmsociety.org/2021-phm-conference-data-challenge/
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

# PHM 2021 fault labels
FAULT_LABELS = {
    0: "healthy",
    1: "robot_arm_mode",
    2: "pressure_leakage",
    3: "conveyor_speed",
    4: "combined_1_2",
    5: "combined_1_3",
    6: "combined_2_3",
    7: "combined_1_2_3",
    8: "intermittent_1",
    9: "intermittent_2",
}


def create_synthetic_test_data(data_dir: Path) -> bool:
    """Create synthetic test data mimicking PHM 2021 SCARA structure.

    PHM 2021 structure:
    phm2021/
    ├── train/
    │   ├── exp_01.csv
    │   ├── exp_02.csv
    │   └── ...
    ├── test/
    │   └── ...
    └── labels.csv

    Signals include:
    - Pressure, Vacuum (pneumatic system)
    - Temperature, Humidity (environment)
    - RobotPosition, RobotVelocity, MotorCurrent (robot)
    - CPUTemperature, ProcessMemoryConsumption (system)

    Args:
        data_dir: Directory to create test files

    Returns:
        True if successful
    """
    logger.info("Creating synthetic PHM 2021 SCARA test data for adapter verification...")

    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    sampling_rate = 10  # Hz (low rate, industrial PLC data)
    duration_sec = 120  # 2 minutes per experiment
    num_samples = int(sampling_rate * duration_sec)
    num_experiments = 15  # Multiple experiments covering different fault types

    # Time array
    t = np.linspace(0, duration_sec, num_samples)

    # Signal columns
    columns = [
        "Time", "Pressure", "Vacuum", "FuseHeatSlope",
        "Temperature", "Humidity",
        "RobotPosition_X", "RobotPosition_Y", "RobotPosition_Z",
        "RobotVelocity_X", "RobotVelocity_Y", "RobotVelocity_Z",
        "MotorCurrent_1", "MotorCurrent_2", "MotorCurrent_3", "MotorCurrent_4",
        "CPUTemperature", "ProcessMemoryConsumption"
    ]

    # Labels for experiments
    exp_labels = []

    for exp_num in range(1, num_experiments + 1):
        file_path = train_dir / f"exp_{exp_num:02d}.csv"
        if file_path.exists():
            continue

        # Assign fault label (cycle through different faults)
        label = exp_num % 10

        # Generate data
        data = np.zeros((num_samples, len(columns)))
        data[:, 0] = t  # Time column

        # Base signals
        cycle_freq = 0.1  # Pick-place cycle frequency

        # Pressure (bar) - pneumatic system
        base_pressure = 6.0
        data[:, 1] = base_pressure + 0.2 * np.sin(2 * np.pi * cycle_freq * t) + 0.05 * np.random.randn(num_samples)

        # Vacuum (bar)
        data[:, 2] = -0.8 + 0.1 * np.sin(2 * np.pi * cycle_freq * t) + 0.02 * np.random.randn(num_samples)

        # Fuse Heat Slope
        data[:, 3] = 2.5 + 0.3 * np.sin(2 * np.pi * 0.01 * t) + 0.1 * np.random.randn(num_samples)

        # Temperature (°C)
        data[:, 4] = 22.0 + 2.0 * np.sin(2 * np.pi * 0.001 * t) + 0.5 * np.random.randn(num_samples)

        # Humidity (%)
        data[:, 5] = 45.0 + 5.0 * np.sin(2 * np.pi * 0.001 * t) + 1.0 * np.random.randn(num_samples)

        # Robot positions (m) - SCARA motion
        data[:, 6] = 0.3 * np.sin(2 * np.pi * cycle_freq * t)  # X
        data[:, 7] = 0.2 * np.cos(2 * np.pi * cycle_freq * t)  # Y
        data[:, 8] = 0.05 * np.abs(np.sin(2 * np.pi * 2 * cycle_freq * t))  # Z (pick-place)

        # Robot velocities (m/s)
        data[:, 9] = 0.3 * 2 * np.pi * cycle_freq * np.cos(2 * np.pi * cycle_freq * t)
        data[:, 10] = -0.2 * 2 * np.pi * cycle_freq * np.sin(2 * np.pi * cycle_freq * t)
        data[:, 11] = 0.05 * 2 * 2 * np.pi * cycle_freq * np.cos(2 * np.pi * 2 * cycle_freq * t)

        # Motor currents (A)
        for j in range(4):
            data[:, 12 + j] = 1.5 + 0.5 * np.sin(2 * np.pi * cycle_freq * t + j * np.pi/4) + 0.1 * np.random.randn(num_samples)

        # CPU Temperature (°C)
        data[:, 16] = 55.0 + 5.0 * np.sin(2 * np.pi * 0.01 * t) + 1.0 * np.random.randn(num_samples)

        # Process Memory (MB)
        data[:, 17] = 512.0 + 50.0 * np.sin(2 * np.pi * 0.005 * t) + 10.0 * np.random.randn(num_samples)

        # Apply fault signatures based on label
        if label == 1:  # Robot arm mode modification
            data[:, 9:12] *= 0.7  # Slower velocities
            data[:, 12:16] *= 0.8  # Lower currents
        elif label == 2:  # Pressure leakage
            data[:, 1] -= 1.5 + 0.5 * (t / duration_sec)  # Pressure drops over time
            data[:, 2] += 0.3  # Vacuum affected
        elif label == 3:  # Conveyor speed alteration
            data[:, 6:9] *= 1.3  # Faster positions
            data[:, 9:12] *= 1.4  # Faster velocities
        elif label == 4:  # Combined 1+2
            data[:, 9:12] *= 0.7
            data[:, 1] -= 1.0
        elif label == 5:  # Combined 1+3
            data[:, 9:12] *= 0.8
            data[:, 6:9] *= 1.2
        elif label == 6:  # Combined 2+3
            data[:, 1] -= 1.0
            data[:, 6:9] *= 1.2
        elif label == 7:  # Combined 1+2+3 (severe)
            data[:, 9:12] *= 0.7
            data[:, 1] -= 1.5
            data[:, 6:9] *= 1.3
        elif label == 8:  # Intermittent 1
            fault_mask = (np.sin(2 * np.pi * 0.02 * t) > 0.5)
            data[fault_mask, 9:12] *= 0.6
        elif label == 9:  # Intermittent 2
            fault_mask = (np.sin(2 * np.pi * 0.03 * t) > 0.7)
            data[fault_mask, 1] -= 2.0

        # Save as CSV with headers
        header = ",".join(columns)
        np.savetxt(str(file_path), data, delimiter=",", fmt="%.6f", header=header, comments="")
        exp_labels.append((exp_num, label))
        logger.info(f"Created: exp_{exp_num:02d}.csv (label={label}: {FAULT_LABELS.get(label, 'unknown')})")

    # Save labels file
    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        with open(labels_path, "w") as f:
            f.write("experiment_id,label\n")
            for exp_id, label in exp_labels:
                f.write(f"{exp_id},{label}\n")
        logger.info(f"Created: labels.csv ({len(exp_labels)} entries)")

    return True


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the PHM 2021 SCARA adapter with data files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing PHM 2021 SCARA adapter...")

    adapter = AdapterRegistry.get("phm2021_scara", data_dir=data_dir)

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
                logger.info(f"    Duration: {len(ch.data) / ch.sampling_rate_hz:.1f} s")

            if episode.raw_metadata:
                logger.info(f"    Experiment ID: {episode.raw_metadata.get('experiment_id', 'N/A')}")
                logger.info(f"    Fault name: {episode.raw_metadata.get('fault_name', 'N/A')}")
                logger.info(f"    Robot type: {episode.raw_metadata.get('robot_type', 'N/A')}")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with PHM 2021 SCARA data."""
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
    stats = pipeline.process_dataset("phm2021_scara", data_dir=data_dir)

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
    episodes = storage.list_episodes("phm2021_scara")

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
            logger.info(f"  Duration: {meta.get('duration_seconds', 0):.1f}s")
            logger.info(f"  Machine: {meta.get('machine_synset')}")
            state = meta.get('state_annotation', {})
            logger.info(f"  State: {state.get('state_label')}")

            # Show fault info
            raw_meta = meta.get('raw_metadata', {})
            if 'fault_name' in raw_meta:
                logger.info(f"  Fault name: {raw_meta['fault_name']}")

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
    test_data_dir = PROJECT_ROOT / "test_data" / "phm2021_scara"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("PHM 2021 SCARA Robot Dataset Pipeline Test")
    logger.info("=" * 60)

    # Step 1: Get data
    logger.info("\n[1/4] Obtaining PHM 2021 SCARA data...")

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
    logger.info("\n[2/4] Testing PHM 2021 SCARA adapter...")
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
