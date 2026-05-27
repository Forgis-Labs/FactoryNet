#!/usr/bin/env python3
"""CLI entry point for processing open datasets into FactoryNet format.

This script provides a command-line interface for:
- Processing specific datasets (CWRU, Paderborn, MAFAULDA, etc.)
- Demo mode for quick validation
- Batch processing of multiple datasets
- Validation report generation

Usage:
    # Process CWRU dataset in demo mode
    python scripts/process_datasets.py --dataset cwru_bearing --demo --demo-limit 10

    # Process full CWRU dataset
    python scripts/process_datasets.py --dataset cwru_bearing --data-dir ./data/cwru

    # Process multiple datasets
    python scripts/process_datasets.py --dataset cwru_bearing paderborn_bearing --demo

    # List available adapters
    python scripts/process_datasets.py --list-adapters
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.adapters.registry import AdapterRegistry
from core.pipeline import DataPipeline, PipelineConfig, PipelineStats


# Configure logging
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def list_adapters() -> None:
    """List all available dataset adapters."""
    # Import adapters to trigger registration
    try:
        import core.adapters  # noqa: F401
    except ImportError as e:
        print(f"Warning: Could not import all adapters: {e}")

    adapters = AdapterRegistry.list_adapters()

    print("\nAvailable Dataset Adapters:")
    print("=" * 50)

    for name in adapters:
        try:
            metadata = AdapterRegistry.get_metadata(name)
            if metadata:
                print(f"\n{name}:")
                print(f"  Full name: {metadata.full_name}")
                print(f"  Samples: ~{metadata.num_samples}")
                print(f"  Format: {metadata.file_format}")
                print(f"  URL: {metadata.source_url}")
            else:
                print(f"\n{name}: (metadata not available)")
        except Exception as e:
            print(f"\n{name}: Error loading metadata - {e}")

    print("\n" + "=" * 50)
    print(f"Total adapters: {len(adapters)}")


def process_dataset(
    dataset_name: str,
    data_dir: str,
    config: PipelineConfig,
) -> PipelineStats:
    """Process a single dataset.

    Args:
        dataset_name: Registered adapter name
        data_dir: Directory containing dataset files
        config: Pipeline configuration

    Returns:
        PipelineStats with processing results
    """
    pipeline = DataPipeline(config=config)
    return pipeline.process_dataset(dataset_name, data_dir)


def process_multiple(
    datasets: List[str],
    data_dirs: Dict[str, str],
    config: PipelineConfig,
) -> Dict[str, PipelineStats]:
    """Process multiple datasets.

    Args:
        datasets: List of dataset names
        data_dirs: Mapping of dataset names to directories
        config: Pipeline configuration

    Returns:
        Dictionary of dataset name to stats
    """
    results = {}
    pipeline = DataPipeline(config=config)

    for dataset_name in datasets:
        if dataset_name not in data_dirs:
            logging.warning(f"No data directory for {dataset_name}, skipping")
            continue

        print(f"\nProcessing {dataset_name}...")
        try:
            stats = pipeline.process_dataset(
                dataset_name,
                data_dirs[dataset_name],
            )
            results[dataset_name] = stats
        except Exception as e:
            logging.error(f"Failed to process {dataset_name}: {e}")
            continue

    return results


def print_summary(results: Dict[str, PipelineStats]) -> None:
    """Print summary of all processed datasets."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    total_episodes = 0
    total_saved = 0
    total_qa = 0

    for dataset_name, stats in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Raw episodes: {stats.raw_episodes_processed}")
        print(f"  Saved: {stats.episodes_saved}")
        print(f"  Pass rate: {stats.pass_rate:.1f}%")
        print(f"  Q&A pairs: {stats.qa_pairs_generated}")
        print(f"  Duration: {stats.duration_seconds:.1f}s")

        if stats.errors:
            print(f"  Errors: {len(stats.errors)}")

        total_episodes += stats.raw_episodes_processed
        total_saved += stats.episodes_saved
        total_qa += stats.qa_pairs_generated

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_episodes} raw episodes, {total_saved} saved, {total_qa} Q&A pairs")
    print("=" * 60)


def save_results(
    results: Dict[str, PipelineStats],
    output_file: str,
) -> None:
    """Save processing results to JSON file."""
    data = {
        name: stats.to_dict()
        for name, stats in results.items()
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process open datasets into FactoryNet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        help="Dataset(s) to process (e.g., cwru_bearing, paderborn_bearing)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing dataset files",
    )

    parser.add_argument(
        "--data-dirs",
        type=str,
        help="JSON file mapping dataset names to directories",
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./factorynet_data",
        help="Output directory (default: ./factorynet_data)",
    )

    # Demo mode
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Enable demo mode (process limited episodes)",
    )

    parser.add_argument(
        "--demo-limit",
        type=int,
        default=10,
        help="Number of episodes in demo mode (default: 10)",
    )

    # Processing options
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip Q&A generation",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation",
    )

    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip feature extraction",
    )

    parser.add_argument(
        "--json-timeseries",
        action="store_true",
        help="Use JSON instead of Parquet for timeseries",
    )

    # Quality gates
    parser.add_argument(
        "--sensor-threshold",
        type=float,
        default=0.95,
        help="Sensor completeness threshold (default: 0.95)",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Label confidence threshold (default: 0.85)",
    )

    # Utility options
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="List available dataset adapters",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file",
    )

    parser.add_argument(
        "--results-file",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # Import adapters
    try:
        import core.adapters  # noqa: F401
    except ImportError as e:
        logger.warning(f"Could not import adapters: {e}")

    # List adapters mode
    if args.list_adapters:
        list_adapters()
        return 0

    # Validate arguments
    if not args.dataset:
        parser.error("--dataset is required (or use --list-adapters)")

    # Build data directories mapping
    data_dirs: Dict[str, str] = {}

    if args.data_dirs:
        # Load from JSON file
        with open(args.data_dirs) as f:
            data_dirs = json.load(f)
    elif args.data_dir:
        # Use single directory for all datasets
        for dataset in args.dataset:
            data_dirs[dataset] = args.data_dir
    else:
        # Use default directories based on dataset names
        for dataset in args.dataset:
            default_dir = f"./data/{dataset}"
            if Path(default_dir).exists():
                data_dirs[dataset] = default_dir
            else:
                logger.warning(
                    f"No data directory specified for {dataset}. "
                    f"Expected: {default_dir}"
                )

    if not data_dirs:
        parser.error(
            "No valid data directories found. Use --data-dir or --data-dirs"
        )

    # Build pipeline config
    config = PipelineConfig(
        output_dir=args.output_dir,
        extract_features=not args.no_features,
        generate_qa=not args.no_qa,
        validate_episodes=not args.no_validate,
        demo_mode=args.demo,
        demo_limit=args.demo_limit,
        sensor_completeness_threshold=args.sensor_threshold,
        label_confidence_threshold=args.confidence_threshold,
        use_parquet=not args.json_timeseries,
        verbose=True,
    )

    logger.info("FactoryNet Data Pipeline")
    logger.info(f"Datasets: {args.dataset}")
    logger.info(f"Output: {args.output_dir}")
    if args.demo:
        logger.info(f"Demo mode: {args.demo_limit} episodes per dataset")

    # Process datasets
    if len(args.dataset) == 1:
        # Single dataset
        dataset = args.dataset[0]
        if dataset not in data_dirs:
            logger.error(f"No data directory for {dataset}")
            return 1

        stats = process_dataset(dataset, data_dirs[dataset], config)
        results = {dataset: stats}
    else:
        # Multiple datasets
        results = process_multiple(args.dataset, data_dirs, config)

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.results_file:
        save_results(results, args.results_file)

    # Check for failures
    failed = sum(1 for s in results.values() if s.episodes_saved == 0)
    if failed > 0:
        logger.warning(f"{failed} dataset(s) had no saved episodes")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
