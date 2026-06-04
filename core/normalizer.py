"""Normalizer for converting RawEpisode to FactoryNetEpisode.

Transforms adapter output into the standardized FactoryNet schema format.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.adapters.base_adapter import FaultType, RawEpisode, SeverityLevel
from core.feature_extraction import (
    BEARING_GEOMETRIES,
    VibrationFeatureExtractor,
    VibrationFeatures,
    compute_statistics,
)

logger = logging.getLogger(__name__)


# Mapping from FaultType to state synsets
FAULT_TYPE_TO_SYNSET = {
    FaultType.NORMAL: "S.nom.run",
    FaultType.INNER_RACE: "S.flt.mec.wea.bea.inn",
    FaultType.OUTER_RACE: "S.flt.mec.wea.bea.out",
    FaultType.BALL: "S.flt.mec.wea.bea.bal",
    FaultType.CAGE: "S.flt.mec.wea.bea.cag",
    FaultType.COMBINED: "S.flt.mec.wea.bea",
    FaultType.IMBALANCE: "S.flt.mec.imb",
    FaultType.MISALIGNMENT: "S.flt.mec.mis",
    FaultType.LOOSENESS: "S.flt.mec.loo",
    FaultType.UNKNOWN: "S.unk",
}

# Mapping from SeverityLevel to severity value
SEVERITY_TO_VALUE = {
    SeverityLevel.HEALTHY: 0.0,
    SeverityLevel.MINOR: 0.25,
    SeverityLevel.MODERATE: 0.5,
    SeverityLevel.SEVERE: 0.75,
    SeverityLevel.CRITICAL: 1.0,
}

# Dataset to machine synset mapping
DATASET_MACHINE_SYNSETS = {
    # Bearing test rigs
    "cwru_bearing": "M.tst.bea.cwru",
    "paderborn_bearing": "M.tst.bea.paderborn",
    "mafaulda": "M.tst.rot.mafaulda",
    "xjtu_sy": "M.tst.bea.xjtu",
    "phm_2010": "M.tst.cnc.phm2010",
    "nasa_ims": "M.tst.bea.ims",
    # Robot datasets
    "aursad": "M.rob.col.ur.ur3e",
    "ur3e_pickplace": "M.rob.col.ur.ur3e",
    "phm2021_scara": "M.rob.ind.scara.phm2021",
}


@dataclass
class TimeStep:
    """Single time step in the episode."""
    step_index: int
    timestamp_offset: float  # Seconds from episode start

    # Sensor values at this time step
    condition_monitoring: Dict[str, float] = field(default_factory=dict)
    # E.g., {"vibration_de": 0.5, "vibration_fe": 0.3}


@dataclass
class StateAnnotation:
    """State annotation for the episode."""
    state_synset: str  # e.g., "S.flt.mec.wea.bea.inn"
    state_label: str   # Human-readable label
    confidence: float  # 0.0 to 1.0
    severity: float    # 0.0 to 1.0

    # Symptom synsets observed
    symptoms: List[str] = field(default_factory=list)


@dataclass
class SemanticPriors:
    """Domain knowledge and semantic priors for the episode."""
    machine_type: str
    machine_description: str
    typical_failure_modes: List[str]
    maintenance_recommendations: List[str]
    operating_conditions: Dict[str, Any]
    bearing_info: Optional[Dict[str, Any]] = None


@dataclass
class FactoryNetEpisode:
    """Normalized episode in the FactoryNet schema format.

    This is the final format that gets saved to the dataset.
    """
    # Identification
    episode_id: str  # "FN-ADAPTED-000001"
    source: str  # "adapted"
    source_dataset: str  # "cwru_bearing"
    source_file: str

    # Machine info
    machine_synset: str
    machine_instance: str

    # Timing
    timestamp_start: datetime
    timestamp_end: datetime
    duration_seconds: float
    sampling_rate_hz: float

    # Time series data
    steps: List[TimeStep]
    channel_names: List[str]
    channel_units: Dict[str, str]

    # State and labels
    state_annotation: StateAnnotation
    cause_synset: Optional[str] = None

    # Extracted features
    features: Dict[str, VibrationFeatures] = field(default_factory=dict)

    # Domain knowledge
    semantic_priors: Optional[SemanticPriors] = None

    # Operating conditions
    load_hp: Optional[float] = None
    rpm: Optional[float] = None

    # Raw metadata preserved
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata.json format."""
        return {
            "episode_id": self.episode_id,
            "source": self.source,
            "source_dataset": self.source_dataset,
            "source_file": self.source_file,
            "machine_synset": self.machine_synset,
            "machine_instance": self.machine_instance,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat(),
            "duration_seconds": self.duration_seconds,
            "sampling_rate_hz": self.sampling_rate_hz,
            "num_timesteps": len(self.steps),
            "channel_names": self.channel_names,
            "channel_units": self.channel_units,
            "state_annotation": {
                "state_synset": self.state_annotation.state_synset,
                "state_label": self.state_annotation.state_label,
                "confidence": self.state_annotation.confidence,
                "severity": self.state_annotation.severity,
                "symptoms": self.state_annotation.symptoms,
            },
            "cause_synset": self.cause_synset,
            "load_hp": self.load_hp,
            "rpm": self.rpm,
            "raw_metadata": self.raw_metadata,
        }

    def to_features_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert features to dictionary format."""
        result = {}
        for channel_name, feat in self.features.items():
            result[channel_name] = feat.to_dict()
        return result


class EpisodeNormalizer:
    """Normalizes RawEpisode objects to FactoryNetEpisode format.

    Handles:
    - Feature extraction from raw signals
    - Mapping fault types to taxonomy synsets
    - Generating unique episode IDs
    - Computing derived fields

    Example:
        normalizer = EpisodeNormalizer()
        for raw_episode in adapter.iter_episodes():
            fn_episode = normalizer.normalize(raw_episode)
            # Save fn_episode...
    """

    def __init__(
        self,
        id_prefix: str = "FN-ADAPTED",
        extract_features: bool = True,
        bearing_type: str = "6205",
    ):
        """Initialize normalizer.

        Args:
            id_prefix: Prefix for generated episode IDs
            extract_features: Whether to extract vibration features
            bearing_type: Default bearing type for fault frequency calculation
        """
        self.id_prefix = id_prefix
        self.extract_features = extract_features
        self.bearing_type = bearing_type
        self._episode_counter = 0
        self._seen_ids = set()

    def normalize(
        self,
        raw_episode: RawEpisode,
        episode_id: Optional[str] = None,
    ) -> FactoryNetEpisode:
        """Normalize a RawEpisode to FactoryNetEpisode format.

        Args:
            raw_episode: Input episode from adapter
            episode_id: Optional explicit episode ID

        Returns:
            Normalized FactoryNetEpisode
        """
        # Generate episode ID
        if episode_id is None:
            episode_id = self._generate_episode_id(raw_episode)

        # Get sampling rate from first channel
        sampling_rate = 12000.0  # Default
        if raw_episode.channels:
            sampling_rate = raw_episode.channels[0].sampling_rate_hz

        # Build time steps
        steps, channel_names = self._build_time_steps(raw_episode)

        # Build channel units mapping
        channel_units = {}
        for ch in raw_episode.channels:
            channel_units[ch.channel_type] = ch.unit

        # Create state annotation
        state_annotation = self._create_state_annotation(raw_episode)

        # Extract features
        features = {}
        if self.extract_features:
            features = self._extract_features(raw_episode)

        # Build semantic priors
        semantic_priors = self._build_semantic_priors(raw_episode)

        # Determine timestamps
        now = datetime.now()
        timestamp_start = raw_episode.timestamp or now
        duration = raw_episode.duration_seconds
        timestamp_end = timestamp_start

        # Get machine synset
        machine_synset = DATASET_MACHINE_SYNSETS.get(
            raw_episode.source_dataset,
            "M.tst.bea"
        )

        return FactoryNetEpisode(
            episode_id=episode_id,
            source="adapted",
            source_dataset=raw_episode.source_dataset,
            source_file=raw_episode.source_file,
            machine_synset=machine_synset,
            machine_instance=f"{raw_episode.source_dataset}_{raw_episode.raw_id}",
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_seconds=duration,
            sampling_rate_hz=sampling_rate,
            steps=steps,
            channel_names=channel_names,
            channel_units=channel_units,
            state_annotation=state_annotation,
            cause_synset=self._infer_cause_synset(raw_episode),
            features=features,
            semantic_priors=semantic_priors,
            load_hp=raw_episode.load_hp,
            rpm=raw_episode.rpm,
            raw_metadata=raw_episode.raw_metadata,
        )

    def _generate_episode_id(self, raw_episode: RawEpisode) -> str:
        """Generate a unique episode ID."""
        self._episode_counter += 1

        # Create ID with counter
        episode_id = f"{self.id_prefix}-{self._episode_counter:06d}"

        # Ensure uniqueness
        while episode_id in self._seen_ids:
            self._episode_counter += 1
            episode_id = f"{self.id_prefix}-{self._episode_counter:06d}"

        self._seen_ids.add(episode_id)
        return episode_id

    def _build_time_steps(
        self,
        raw_episode: RawEpisode,
    ) -> tuple[List[TimeStep], List[str]]:
        """Build time steps from sensor channels.

        Args:
            raw_episode: Input episode

        Returns:
            Tuple of (time_steps, channel_names)
        """
        if not raw_episode.channels:
            return [], []

        # Get reference channel for timing
        ref_channel = raw_episode.channels[0]
        num_samples = len(ref_channel.data)
        dt = 1.0 / ref_channel.sampling_rate_hz

        # Build channel name list
        channel_names = [ch.channel_type for ch in raw_episode.channels]

        # Create time steps
        steps = []
        for i in range(num_samples):
            condition_data = {}
            for ch in raw_episode.channels:
                if i < len(ch.data):
                    condition_data[ch.channel_type] = ch.data[i]

            steps.append(TimeStep(
                step_index=i,
                timestamp_offset=i * dt,
                condition_monitoring=condition_data,
            ))

        return steps, channel_names

    def _create_state_annotation(
        self,
        raw_episode: RawEpisode,
    ) -> StateAnnotation:
        """Create state annotation from raw episode labels.

        Args:
            raw_episode: Input episode

        Returns:
            StateAnnotation object
        """
        # Map fault type to synset
        state_synset = FAULT_TYPE_TO_SYNSET.get(
            raw_episode.fault_type,
            "S.unk"
        )

        # Create human-readable label
        fault_str = raw_episode.fault_type.value.replace("_", " ").title()
        if raw_episode.fault_location:
            label = f"{fault_str} ({raw_episode.fault_location})"
        else:
            label = fault_str

        # Map severity
        severity = SEVERITY_TO_VALUE.get(raw_episode.severity, 0.0)

        # Confidence based on data source (adapted datasets = high confidence)
        confidence = 0.95

        # Infer symptoms from fault type
        symptoms = self._infer_symptoms(raw_episode)

        return StateAnnotation(
            state_synset=state_synset,
            state_label=label,
            confidence=confidence,
            severity=severity,
            symptoms=symptoms,
        )

    def _infer_symptoms(self, raw_episode: RawEpisode) -> List[str]:
        """Infer symptom synsets from fault type.

        Args:
            raw_episode: Input episode

        Returns:
            List of symptom synsets
        """
        symptoms = []

        if raw_episode.fault_type == FaultType.INNER_RACE:
            symptoms.append("Y.vib.hig.bea.bpfi")
        elif raw_episode.fault_type == FaultType.OUTER_RACE:
            symptoms.append("Y.vib.hig.bea.bpfo")
        elif raw_episode.fault_type == FaultType.BALL:
            symptoms.append("Y.vib.hig.bea.bsf")
        elif raw_episode.fault_type == FaultType.CAGE:
            symptoms.append("Y.vib.hig.bea.ftf")
        elif raw_episode.fault_type == FaultType.IMBALANCE:
            symptoms.append("Y.vib.hig.1x")
        elif raw_episode.fault_type == FaultType.MISALIGNMENT:
            symptoms.append("Y.vib.hig.2x")

        # Add high vibration symptom for all faults
        if raw_episode.fault_type != FaultType.NORMAL:
            symptoms.append("Y.vib.hig")

        return symptoms

    def _infer_cause_synset(self, raw_episode: RawEpisode) -> Optional[str]:
        """Infer cause synset from fault type.

        Args:
            raw_episode: Input episode

        Returns:
            Cause synset or None
        """
        # Most bearing faults in datasets are seeded defects (for research)
        if raw_episode.fault_type != FaultType.NORMAL:
            # Assume artificial/seeded defect
            return "C.tes.art"

        return None

    def _extract_features(
        self,
        raw_episode: RawEpisode,
    ) -> Dict[str, VibrationFeatures]:
        """Extract vibration features from all channels.

        Args:
            raw_episode: Input episode

        Returns:
            Dictionary mapping channel names to features
        """
        features = {}

        # Get bearing geometry
        bearing_geometry = BEARING_GEOMETRIES.get(self.bearing_type)

        for channel in raw_episode.channels:
            extractor = VibrationFeatureExtractor(
                sampling_rate_hz=channel.sampling_rate_hz,
                bearing_geometry=bearing_geometry,
            )

            try:
                channel_features = extractor.extract(
                    signal=channel.data,
                    rpm=raw_episode.rpm,
                )
                features[channel.channel_type] = channel_features
            except Exception as e:
                logger.warning(f"Feature extraction failed for {channel.channel_id}: {e}")

        return features

    def _build_semantic_priors(
        self,
        raw_episode: RawEpisode,
    ) -> SemanticPriors:
        """Build semantic priors for the episode.

        Args:
            raw_episode: Input episode

        Returns:
            SemanticPriors object
        """
        machine_descriptions = {
            "cwru_bearing": "CWRU bearing test rig with 2HP motor and SKF bearings",
            "paderborn_bearing": "Paderborn bearing test rig with modular bearing unit",
            "mafaulda": "MAFAULDA rotating machinery fault simulator",
            "xjtu_sy": "XJTU-SY accelerated bearing life test rig",
            "phm_2010": "PHM 2010 CNC milling machine tool wear dataset",
        }

        typical_failures = {
            "cwru_bearing": ["Inner race fault", "Outer race fault", "Ball fault"],
            "paderborn_bearing": ["Inner race fault", "Outer race fault", "Combined fault"],
            "mafaulda": ["Imbalance", "Misalignment", "Looseness", "Bearing fault"],
            "xjtu_sy": ["Progressive bearing degradation", "Spalling", "Fatigue"],
            "phm_2010": ["Tool wear", "Tool breakage"],
        }

        maintenance_recs = [
            "Monitor vibration trends for early fault detection",
            "Replace bearing when fault frequencies become dominant",
            "Check lubrication condition periodically",
        ]

        operating_conditions = {
            "load_hp": raw_episode.load_hp,
            "rpm": raw_episode.rpm,
        }

        bearing_info = None
        if self.bearing_type in BEARING_GEOMETRIES:
            geom = BEARING_GEOMETRIES[self.bearing_type]
            bearing_info = {
                "type": self.bearing_type,
                "num_balls": geom.num_balls,
                "ball_diameter_mm": geom.ball_diameter_mm,
                "pitch_diameter_mm": geom.pitch_diameter_mm,
            }

        return SemanticPriors(
            machine_type="bearing_test_rig",
            machine_description=machine_descriptions.get(
                raw_episode.source_dataset,
                "Bearing test rig"
            ),
            typical_failure_modes=typical_failures.get(
                raw_episode.source_dataset,
                ["Bearing fault"]
            ),
            maintenance_recommendations=maintenance_recs,
            operating_conditions=operating_conditions,
            bearing_info=bearing_info,
        )

    def reset_counter(self) -> None:
        """Reset the episode counter."""
        self._episode_counter = 0
        self._seen_ids.clear()
