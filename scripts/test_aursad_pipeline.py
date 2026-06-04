#!/usr/bin/env python3
"""Test script for AURSAD robot dataset download and processing.

Downloads AURSAD data and processes it through the full pipeline.
Dataset: https://zenodo.org/records/4487073
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

# AURSAD dataset info
AURSAD_URLS = {
    "AURSAD.h5": {
        "url": "https://zenodo.org/records/4487073/files/AURSAD.h5?download=1",
        "description": "Full AURSAD dataset (UR3e screwdriving, ~6.4GB)",
        "size_mb": 6400,
    },
}


def check_h5py_available() -> bool:
    """Check if h5py is installed."""
    try:
        import h5py
        logger.info(f"h5py version: {h5py.__version__}")
        return True
    except ImportError:
        logger.error("h5py not installed. Run: pip install h5py")
        return False


def download_aursad_file(data_dir: Path) -> Path | None:
    """Download AURSAD HDF5 file.

    Args:
        data_dir: Directory to save files

    Returns:
        Path to downloaded file or None
    """
    from core.download_manager import DownloadManager

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    local_path = data_dir / "AURSAD.h5"
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"AURSAD.h5 already exists ({size_mb:.1f} MB)")
        return local_path

    manager = DownloadManager(cache_dir=data_dir)

    info = AURSAD_URLS["AURSAD.h5"]
    logger.info(f"Downloading AURSAD.h5: {info['description']}")
    logger.warning(f"This is a large file (~{info['size_mb']} MB). Download may take a while...")

    result = manager.download(
        url=info["url"],
        filename="AURSAD.h5",
        progress=True,
    )

    if result.success:
        logger.info(f"Downloaded: {result.local_path} ({result.size_bytes:,} bytes)")
        return result.local_path
    else:
        logger.error(f"Failed to download: {result.error}")
        return None


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the AURSAD adapter with downloaded files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing AURSAD adapter...")

    adapter = AdapterRegistry.get("aursad", data_dir=data_dir)

    # Discover files
    files = adapter.discover_files()
    logger.info(f"Discovered {len(files)} HDF5 files")

    if not files:
        logger.warning("No HDF5 files found!")
        return

    # Parse first file with limit
    episode_count = 0
    for file_path in files[:1]:
        logger.info(f"\nParsing {file_path.name}:")

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

            # Show metadata
            if episode.raw_metadata:
                logger.info(f"    Label: {episode.raw_metadata.get('label_name', 'N/A')}")
                logger.info(f"    Robot: {episode.raw_metadata.get('robot_model', 'N/A')}")

            if episode_count >= limit:
                break

        if episode_count >= limit:
            break

    logger.info(f"\nTotal episodes parsed (limited): {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with AURSAD data."""
    from core.pipeline import DataPipeline, PipelineConfig

    logger.info("\nRunning full pipeline...")

    config = PipelineConfig(
        output_dir=str(output_dir),
        extract_features=True,
        generate_qa=True,
        validate_episodes=True,
        demo_mode=True,
        demo_limit=5,  # Limit to 5 episodes for testing
        use_parquet=True,
        verbose=True,
    )

    pipeline = DataPipeline(config=config)
    stats = pipeline.process_dataset("aursad", data_dir=data_dir)

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
    episodes = storage.list_episodes("aursad")

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
            if data["qa_pairs"]:
                qa = data["qa_pairs"][0]
                logger.info(f"    Sample Q: {qa.get('question', '')[:60]}...")

        if "timeseries" in data:
            ts = data["timeseries"]
            logger.info(f"  Timeseries columns: {list(ts.keys())[:5]}...")
            if "step_index" in ts:
                logger.info(f"  Timeseries samples: {len(ts['step_index'])}")


def main():
    """Main test function."""
    # Setup directories
    test_data_dir = PROJECT_ROOT / "test_data" / "aursad"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("AURSAD Robot Dataset Pipeline Test")
    logger.info("=" * 60)

    # Check dependencies
    if not check_h5py_available():
        return 1

    # Step 1: Download AURSAD file
    logger.info("\n[1/4] Downloading AURSAD dataset...")
    downloaded = download_aursad_file(test_data_dir)

    if not downloaded:
        logger.error("AURSAD file not available, cannot continue")
        return 1

    # Step 2: Test adapter
    logger.info("\n[2/4] Testing AURSAD adapter...")
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
