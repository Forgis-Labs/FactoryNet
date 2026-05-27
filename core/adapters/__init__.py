"""Dataset adapters for FactoryNet.

This module provides adapters for converting external bearing,
machinery fault, and robot datasets into the FactoryNet schema.

Supported datasets:

Bearing/Machinery:
- CWRU Bearing (Case Western Reserve University)
- Paderborn Bearing (KAt-DataCenter)
- MAFAULDA (Machinery Fault Database)
- XJTU-SY Bearing (Xi'an Jiaotong University)
- PHM 2010 CNC Milling

Industrial Robots:
- AURSAD (UR3e Screwdriving Anomaly Detection)
- UR3e Pick and Place (Kaggle Industrial Robotic Arm)
- PHM 2021 SCARA Robot (PHM Europe Data Challenge)

Usage:
    from core.adapters import AdapterRegistry

    # List available adapters
    print(AdapterRegistry.list_adapters())

    # Get an adapter by name
    adapter = AdapterRegistry.get("cwru_bearing", data_dir="/path/to/data")

    # Iterate over episodes
    for episode in adapter.iter_episodes(limit=10):
        print(episode.raw_id, episode.fault_type)
"""
from .base_adapter import (
    BaseDatasetAdapter,
    DatasetMetadata,
    FaultType,
    RawEpisode,
    SensorChannel,
    SeverityLevel,
)
from .registry import AdapterRegistry, register_adapter

# Import adapters to trigger registration
# These imports have side effects (register decorators)
# Wrapped in try/except to allow module import even if dependencies are missing
import logging as _logging

_logger = _logging.getLogger(__name__)

# Bearing datasets
try:
    from . import cwru_adapter
except ImportError as e:
    _logger.debug(f"Could not import cwru_adapter: {e}")

try:
    from . import paderborn_adapter
except ImportError as e:
    _logger.debug(f"Could not import paderborn_adapter: {e}")

try:
    from . import mafaulda_adapter
except ImportError as e:
    _logger.debug(f"Could not import mafaulda_adapter: {e}")

try:
    from . import xjtu_adapter
except ImportError as e:
    _logger.debug(f"Could not import xjtu_adapter: {e}")

try:
    from . import phm2010_adapter
except ImportError as e:
    _logger.debug(f"Could not import phm2010_adapter: {e}")

# Robot datasets
try:
    from . import aursad_adapter
except ImportError as e:
    _logger.debug(f"Could not import aursad_adapter: {e}")

try:
    from . import ur3e_pickplace_adapter
except ImportError as e:
    _logger.debug(f"Could not import ur3e_pickplace_adapter: {e}")

try:
    from . import phm2021_scara_adapter
except ImportError as e:
    _logger.debug(f"Could not import phm2021_scara_adapter: {e}")

__all__ = [
    # Base classes
    "BaseDatasetAdapter",
    "DatasetMetadata",
    "FaultType",
    "RawEpisode",
    "SensorChannel",
    "SeverityLevel",
    # Registry
    "AdapterRegistry",
    "register_adapter",
]
