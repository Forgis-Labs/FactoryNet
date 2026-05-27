#!/usr/bin/env python3
"""Test script for CWRU data download and processing.

Downloads sample CWRU .mat files and processes them through the full pipeline.
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

# CWRU sample file URLs
CWRU_SAMPLE_FILES = {
    # Normal baseline data
    "97.mat": {
        "url": "https://engineering.case.edu/sites/default/files/97.mat",
        "description": "Normal baseline - 0 HP",
    },
    "98.mat": {
        "url": "https://engineering.case.edu/sites/default/files/98.mat",
        "description": "Normal baseline - 1 HP",
    },
    # Inner race fault - 0.007" diameter
    "105.mat": {
        "url": "https://engineering.case.edu/sites/default/files/105.mat",
        "description": "Inner race fault 0.007\" - 0 HP",
    },
    "106.mat": {
        "url": "https://engineering.case.edu/sites/default/files/106.mat",
        "description": "Inner race fault 0.007\" - 1 HP",
    },
}


def download_sample_files(data_dir: Path, limit: int = 4) -> list[Path]:
    """Download sample CWRU files.

    Args:
        data_dir: Directory to save files
        limit: Maximum number of files to download

    Returns:
        List of downloaded file paths
    """
    from core.download_manager import DownloadManager

    data_dir.mkdir(parents=True, exist_ok=True)
    manager = DownloadManager(cache_dir=data_dir)

    downloaded = []
    for i, (filename, info) in enumerate(CWRU_SAMPLE_FILES.items()):
        if i >= limit:
            break

        logger.info(f"Downloading {filename}: {info['description']}")
        result = manager.download(
            url=info["url"],
            filename=filename,
            progress=True,
        )

        if result.success:
            logger.info(f"  Downloaded: {result.local_path} ({result.size_bytes} bytes)")
            downloaded.append(result.local_path)
        else:
            logger.error(f"  Failed: {result.error}")

    return downloaded


def test_adapter(data_dir: Path) -> None:
    """Test the CWRU adapter with downloaded files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing CWRU adapter...")

    adapter = AdapterRegistry.get("cwru_bearing", data_dir=data_dir)

    # Discover files
    files = adapter.discover_files()
    logger.info(f"Discovered {len(files)} .mat files")

    # Parse first few files
    episode_count = 0
    for file_path in files[:4]:
        logger.info(f"\nParsing {file_path.name}:")

        for episode in adapter.parse_file(file_path):
            episode_count += 1
            logger.info(f"  Episode: {episode.raw_id}")
            logger.info(f"  Fault type: {episode.fault_type.value}")
            logger.info(f"  Severity: {episode.severity.value}")
            logger.info(f"  Channels: {len(episode.channels)}")

            if episode.channels:
                ch = episode.channels[0]
                logger.info(f"  First channel: {ch.channel_type}")
                logger.info(f"  Samples: {len(ch.data)}")
                logger.info(f"  Sampling rate: {ch.sampling_rate_hz} Hz")
                logger.info(f"  Duration: {len(ch.data) / ch.sampling_rate_hz:.2f} s")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with CWRU data."""
    from core.pipeline import DataPipeline, PipelineConfig

    logger.info("\nRunning full pipeline...")

    config = PipelineConfig(
        output_dir=str(output_dir),
        extract_features=True,
        generate_qa=True,
        validate_episodes=True,
        demo_mode=True,
        demo_limit=4,
        use_parquet=True,
        verbose=True,
    )

    pipeline = DataPipeline(config=config)
    stats = pipeline.process_dataset("cwru_bearing", data_dir=data_dir)

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
    episodes = storage.list_episodes("cwru_bearing")

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
            logger.info(f"  State: {meta.get('state_annotation', {}).get('state_label')}")

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
    test_data_dir = PROJECT_ROOT / "test_data" / "cwru"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("CWRU Pipeline Test")
    logger.info("=" * 60)

    # Step 1: Download sample files
    logger.info("\n[1/4] Downloading sample CWRU files...")
    downloaded = download_sample_files(test_data_dir, limit=4)

    if not downloaded:
        logger.error("No files downloaded, cannot continue")
        return 1

    # Step 2: Test adapter
    logger.info("\n[2/4] Testing CWRU adapter...")
    test_adapter(test_data_dir)

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
