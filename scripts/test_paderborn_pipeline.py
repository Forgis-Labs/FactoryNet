#!/usr/bin/env python3
"""Test script for Paderborn bearing dataset download and processing.

Downloads sample Paderborn data and processes it through the full pipeline.
Dataset: https://zenodo.org/records/15845309
"""
from __future__ import annotations

import logging
import shutil
import subprocess
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

# Paderborn sample files from Zenodo
PADERBORN_SAMPLE_FILES = {
    "K001.rar": {
        "url": "https://zenodo.org/records/15845309/files/K001.rar?download=1",
        "description": "Healthy bearing K001",
        "fault_type": "normal",
    },
    "K002.rar": {
        "url": "https://zenodo.org/records/15845309/files/K002.rar?download=1",
        "description": "Healthy bearing K002",
        "fault_type": "normal",
    },
    "KA01.rar": {
        "url": "https://zenodo.org/records/15845309/files/KA01.rar?download=1",
        "description": "Outer race fault KA01",
        "fault_type": "outer_race",
    },
    "KI01.rar": {
        "url": "https://zenodo.org/records/15845309/files/KI01.rar?download=1",
        "description": "Inner race fault KI01",
        "fault_type": "inner_race",
    },
}


def check_unrar_available() -> bool:
    """Check if unrar or 7z is available for extraction."""
    # Try unrar
    try:
        result = subprocess.run(["unrar"], capture_output=True)
        logger.info("unrar is available")
        return True
    except FileNotFoundError:
        pass

    # Try 7z
    try:
        result = subprocess.run(["7z"], capture_output=True)
        logger.info("7z is available")
        return True
    except FileNotFoundError:
        pass

    # Try Python rarfile
    try:
        import rarfile
        logger.info("rarfile Python module is available")
        return True
    except ImportError:
        pass

    logger.error("No RAR extraction tool found. Install unrar, 7z, or: pip install rarfile")
    return False


def extract_rar(rar_path: Path, output_dir: Path) -> bool:
    """Extract RAR file to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try user's UnRAR.exe first (downloaded to ~/.unrar)
    user_unrar = Path.home() / ".unrar" / "UnRAR.exe"
    if user_unrar.exists():
        try:
            result = subprocess.run(
                [str(user_unrar), "x", "-y", str(rar_path), str(output_dir) + "\\"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"Extracted with UnRAR.exe: {rar_path.name}")
                return True
            else:
                logger.warning(f"UnRAR.exe failed: {result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"UnRAR.exe error: {e}")

    # Try unrar in PATH
    try:
        result = subprocess.run(
            ["unrar", "x", "-y", str(rar_path), str(output_dir) + "/"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Extracted with unrar: {rar_path.name}")
            return True
    except FileNotFoundError:
        pass

    # Try 7z
    try:
        result = subprocess.run(
            ["7z", "x", f"-o{output_dir}", "-y", str(rar_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Extracted with 7z: {rar_path.name}")
            return True
    except FileNotFoundError:
        pass

    # Try Python rarfile with custom unrar path
    try:
        import rarfile
        rarfile.UNRAR_TOOL = str(user_unrar) if user_unrar.exists() else "unrar"
        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(str(output_dir))
        logger.info(f"Extracted with rarfile: {rar_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {rar_path}: {e}")

    return False


def download_sample_files(data_dir: Path, limit: int = 4) -> list[Path]:
    """Download and extract sample Paderborn files.

    Args:
        data_dir: Directory to save files
        limit: Maximum number of files to download

    Returns:
        List of extracted directories
    """
    from core.download_manager import DownloadManager

    data_dir.mkdir(parents=True, exist_ok=True)
    rar_dir = data_dir / "rar_files"
    rar_dir.mkdir(exist_ok=True)

    manager = DownloadManager(cache_dir=rar_dir)
    extracted_dirs = []

    for i, (filename, info) in enumerate(PADERBORN_SAMPLE_FILES.items()):
        if i >= limit:
            break

        # Check if already extracted
        bearing_code = filename.replace(".rar", "")
        extract_dir = data_dir / bearing_code

        if extract_dir.exists() and list(extract_dir.glob("*.mat")):
            logger.info(f"{bearing_code} already extracted ({len(list(extract_dir.glob('*.mat')))} .mat files)")
            extracted_dirs.append(extract_dir)
            continue

        logger.info(f"Downloading {filename}: {info['description']}")
        rar_path = rar_dir / filename

        if not rar_path.exists():
            result = manager.download(
                url=info["url"],
                filename=filename,
                progress=True,
            )

            if not result.success:
                logger.error(f"Failed to download: {result.error}")
                continue

            rar_path = result.local_path

        # Extract
        logger.info(f"Extracting {filename}...")
        if extract_rar(rar_path, data_dir):
            # Find extracted directory
            if extract_dir.exists():
                extracted_dirs.append(extract_dir)
            else:
                # Files might be extracted directly
                mat_files = list(data_dir.glob(f"{bearing_code}*.mat"))
                if mat_files:
                    extracted_dirs.append(data_dir)

    return extracted_dirs


def test_adapter(data_dir: Path, limit: int = 5) -> None:
    """Test the Paderborn adapter with downloaded files."""
    from core.adapters import AdapterRegistry

    logger.info("Testing Paderborn adapter...")

    adapter = AdapterRegistry.get("paderborn_bearing", data_dir=data_dir)

    # Discover files
    files = adapter.discover_files()
    logger.info(f"Discovered {len(files)} .mat files")

    if not files:
        logger.warning("No .mat files found!")
        return

    # Parse files
    episode_count = 0
    for file_path in files[:limit]:
        logger.info(f"\nParsing {file_path.name}:")

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
                logger.info(f"    Bearing code: {episode.raw_metadata.get('bearing_code', 'N/A')}")

    logger.info(f"\nTotal episodes parsed: {episode_count}")


def test_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Test the full pipeline with Paderborn data."""
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
    stats = pipeline.process_dataset("paderborn_bearing", data_dir=data_dir)

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
    episodes = storage.list_episodes("paderborn_bearing")

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
    test_data_dir = PROJECT_ROOT / "test_data" / "paderborn"
    test_output_dir = PROJECT_ROOT / "test_output"

    logger.info("=" * 60)
    logger.info("Paderborn Bearing Dataset Pipeline Test")
    logger.info("=" * 60)

    # Check extraction tools
    if not check_unrar_available():
        logger.warning("Continuing without RAR extraction capability...")
        logger.warning("Please install unrar, 7z, or run: pip install rarfile")

    # Step 1: Download sample files
    logger.info("\n[1/4] Downloading sample Paderborn files...")
    extracted = download_sample_files(test_data_dir, limit=4)

    if not extracted:
        logger.error("No files extracted, cannot continue")
        logger.info("Checking for existing .mat files...")

        # Check if any .mat files exist already
        mat_files = list(test_data_dir.rglob("*.mat"))
        if mat_files:
            logger.info(f"Found {len(mat_files)} existing .mat files")
        else:
            return 1

    # Step 2: Test adapter
    logger.info("\n[2/4] Testing Paderborn adapter...")
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
