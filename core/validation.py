"""Validation module for FactoryNet episodes.

Provides schema validation, quality gates, and validation reporting.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.normalizer import FactoryNetEpisode

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Fails validation
    WARNING = "warning"  # Passes but flagged
    INFO = "info"        # Informational


class ValidationCategory(Enum):
    """Category of validation check."""
    SCHEMA = "schema"           # Schema compliance
    COMPLETENESS = "completeness"  # Data completeness
    QUALITY = "quality"         # Data quality
    CONSISTENCY = "consistency"  # Internal consistency
    TAXONOMY = "taxonomy"       # Taxonomy compliance


@dataclass
class ValidationIssue:
    """Single validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    field: str
    message: str
    value: Any = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "field": self.field,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validating an episode."""
    episode_id: str
    valid: bool  # Passes all quality gates
    issues: List[ValidationIssue] = field(default_factory=list)

    # Quality metrics
    sensor_completeness: float = 0.0  # 0-1
    label_confidence: float = 0.0     # 0-1
    feature_completeness: float = 0.0  # 0-1
    overall_quality_score: float = 0.0  # 0-1

    # Timestamps
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "sensor_completeness": self.sensor_completeness,
            "label_confidence": self.label_confidence,
            "feature_completeness": self.feature_completeness,
            "overall_quality_score": self.overall_quality_score,
            "validated_at": self.validated_at.isoformat(),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class ValidationReport:
    """Aggregated validation report for multiple episodes."""
    dataset_name: str
    total_episodes: int
    valid_episodes: int
    invalid_episodes: int

    # Aggregated metrics
    avg_sensor_completeness: float
    avg_label_confidence: float
    avg_quality_score: float

    # Issue breakdown
    error_counts: Dict[str, int] = field(default_factory=dict)
    warning_counts: Dict[str, int] = field(default_factory=dict)

    # Episode results
    episode_results: List[ValidationResult] = field(default_factory=list)

    # Timestamps
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def pass_rate(self) -> float:
        """Percentage of episodes that passed validation."""
        if self.total_episodes == 0:
            return 0.0
        return self.valid_episodes / self.total_episodes * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_episodes": self.total_episodes,
            "valid_episodes": self.valid_episodes,
            "invalid_episodes": self.invalid_episodes,
            "pass_rate": self.pass_rate,
            "avg_sensor_completeness": self.avg_sensor_completeness,
            "avg_label_confidence": self.avg_label_confidence,
            "avg_quality_score": self.avg_quality_score,
            "error_counts": self.error_counts,
            "warning_counts": self.warning_counts,
            "generated_at": self.generated_at.isoformat(),
            "episode_results": [r.to_dict() for r in self.episode_results],
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EpisodeValidator:
    """Validates FactoryNetEpisode objects against schema and quality requirements.

    Quality gates:
    - Sensor completeness >= 95%
    - Label confidence >= 85%

    Example:
        validator = EpisodeValidator()
        result = validator.validate(episode)
        if result.valid:
            # Save episode
        else:
            # Log issues
            for issue in result.issues:
                print(f"{issue.severity.value}: {issue.message}")
    """

    # Default quality gate thresholds
    DEFAULT_SENSOR_COMPLETENESS_THRESHOLD = 0.95
    DEFAULT_LABEL_CONFIDENCE_THRESHOLD = 0.85
    DEFAULT_MIN_SAMPLES = 100
    DEFAULT_MAX_NAN_RATIO = 0.05

    # Valid synset prefixes
    VALID_STATE_PREFIXES = {"S.nom", "S.flt", "S.deg", "S.unk"}
    VALID_MACHINE_PREFIXES = {"M.rob", "M.cnc", "M.tst", "M.sen"}
    VALID_CAUSE_PREFIXES = {"C.mai", "C.ope", "C.env", "C.tes", "C.unk"}
    VALID_SYMPTOM_PREFIXES = {"Y.vib", "Y.the", "Y.aco", "Y.vis"}

    def __init__(
        self,
        sensor_completeness_threshold: float = DEFAULT_SENSOR_COMPLETENESS_THRESHOLD,
        label_confidence_threshold: float = DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        max_nan_ratio: float = DEFAULT_MAX_NAN_RATIO,
    ):
        """Initialize validator.

        Args:
            sensor_completeness_threshold: Minimum sensor completeness (0-1)
            label_confidence_threshold: Minimum label confidence (0-1)
            min_samples: Minimum number of time steps
            max_nan_ratio: Maximum ratio of NaN values allowed
        """
        self.sensor_completeness_threshold = sensor_completeness_threshold
        self.label_confidence_threshold = label_confidence_threshold
        self.min_samples = min_samples
        self.max_nan_ratio = max_nan_ratio

    def validate(self, episode: FactoryNetEpisode) -> ValidationResult:
        """Validate a single episode.

        Args:
            episode: Episode to validate

        Returns:
            ValidationResult with issues and quality metrics
        """
        issues: List[ValidationIssue] = []

        # Run validation checks
        issues.extend(self._validate_schema(episode))
        issues.extend(self._validate_completeness(episode))
        issues.extend(self._validate_quality(episode))
        issues.extend(self._validate_taxonomy(episode))
        issues.extend(self._validate_consistency(episode))

        # Compute quality metrics
        sensor_completeness = self._compute_sensor_completeness(episode)
        label_confidence = episode.state_annotation.confidence
        feature_completeness = self._compute_feature_completeness(episode)

        # Overall quality score
        overall_score = (
            sensor_completeness * 0.4 +
            label_confidence * 0.3 +
            feature_completeness * 0.3
        )

        # Determine if episode passes quality gates
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        passes_gates = (
            sensor_completeness >= self.sensor_completeness_threshold and
            label_confidence >= self.label_confidence_threshold
        )
        valid = not has_errors and passes_gates

        return ValidationResult(
            episode_id=episode.episode_id,
            valid=valid,
            issues=issues,
            sensor_completeness=sensor_completeness,
            label_confidence=label_confidence,
            feature_completeness=feature_completeness,
            overall_quality_score=overall_score,
        )

    def _validate_schema(self, episode: FactoryNetEpisode) -> List[ValidationIssue]:
        """Validate schema compliance."""
        issues = []

        # Required fields
        if not episode.episode_id:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                field="episode_id",
                message="Episode ID is required",
            ))

        if not episode.source_dataset:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                field="source_dataset",
                message="Source dataset is required",
            ))

        if not episode.machine_synset:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                field="machine_synset",
                message="Machine synset is required",
            ))

        if episode.sampling_rate_hz <= 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                field="sampling_rate_hz",
                message="Sampling rate must be positive",
                value=episode.sampling_rate_hz,
            ))

        if episode.duration_seconds < 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                field="duration_seconds",
                message="Duration cannot be negative",
                value=episode.duration_seconds,
            ))

        return issues

    def _validate_completeness(self, episode: FactoryNetEpisode) -> List[ValidationIssue]:
        """Validate data completeness."""
        issues = []

        # Check minimum samples
        if len(episode.steps) < self.min_samples:
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                field="steps",
                message=f"Episode has fewer than {self.min_samples} time steps",
                value=len(episode.steps),
            ))

        # Check for empty channels
        if not episode.channel_names:
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                field="channel_names",
                message="No sensor channels found",
            ))

        # Check state annotation
        if not episode.state_annotation.state_synset:
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                field="state_annotation.state_synset",
                message="State synset is empty",
            ))

        return issues

    def _validate_quality(self, episode: FactoryNetEpisode) -> List[ValidationIssue]:
        """Validate data quality."""
        issues = []

        # Check for NaN values in steps
        nan_count = 0
        total_values = 0

        for step in episode.steps:
            for value in step.condition_monitoring.values():
                total_values += 1
                if value is None or (isinstance(value, float) and not (value == value)):  # NaN check
                    nan_count += 1

        if total_values > 0:
            nan_ratio = nan_count / total_values
            if nan_ratio > self.max_nan_ratio:
                issues.append(ValidationIssue(
                    category=ValidationCategory.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    field="steps",
                    message=f"High NaN ratio ({nan_ratio:.2%}) in sensor data",
                    value=nan_ratio,
                    suggestion="Check data source for missing values",
                ))

        # Check confidence threshold
        if episode.state_annotation.confidence < self.label_confidence_threshold:
            issues.append(ValidationIssue(
                category=ValidationCategory.QUALITY,
                severity=ValidationSeverity.WARNING,
                field="state_annotation.confidence",
                message=f"Label confidence below threshold ({self.label_confidence_threshold})",
                value=episode.state_annotation.confidence,
            ))

        return issues

    def _validate_taxonomy(self, episode: FactoryNetEpisode) -> List[ValidationIssue]:
        """Validate taxonomy compliance."""
        issues = []

        # Validate state synset format
        state_synset = episode.state_annotation.state_synset
        if state_synset:
            prefix = ".".join(state_synset.split(".")[:2])
            if prefix not in self.VALID_STATE_PREFIXES:
                issues.append(ValidationIssue(
                    category=ValidationCategory.TAXONOMY,
                    severity=ValidationSeverity.WARNING,
                    field="state_annotation.state_synset",
                    message=f"Unknown state synset prefix: {prefix}",
                    value=state_synset,
                    suggestion=f"Valid prefixes: {self.VALID_STATE_PREFIXES}",
                ))

        # Validate machine synset format
        machine_synset = episode.machine_synset
        if machine_synset:
            prefix = ".".join(machine_synset.split(".")[:2])
            if prefix not in self.VALID_MACHINE_PREFIXES:
                issues.append(ValidationIssue(
                    category=ValidationCategory.TAXONOMY,
                    severity=ValidationSeverity.WARNING,
                    field="machine_synset",
                    message=f"Unknown machine synset prefix: {prefix}",
                    value=machine_synset,
                ))

        # Validate cause synset if present
        if episode.cause_synset:
            prefix = ".".join(episode.cause_synset.split(".")[:2])
            if prefix not in self.VALID_CAUSE_PREFIXES:
                issues.append(ValidationIssue(
                    category=ValidationCategory.TAXONOMY,
                    severity=ValidationSeverity.WARNING,
                    field="cause_synset",
                    message=f"Unknown cause synset prefix: {prefix}",
                    value=episode.cause_synset,
                ))

        # Validate symptom synsets
        for symptom in episode.state_annotation.symptoms:
            prefix = ".".join(symptom.split(".")[:2])
            if prefix not in self.VALID_SYMPTOM_PREFIXES:
                issues.append(ValidationIssue(
                    category=ValidationCategory.TAXONOMY,
                    severity=ValidationSeverity.INFO,
                    field="state_annotation.symptoms",
                    message=f"Unknown symptom synset prefix: {prefix}",
                    value=symptom,
                ))

        return issues

    def _validate_consistency(self, episode: FactoryNetEpisode) -> List[ValidationIssue]:
        """Validate internal consistency."""
        issues = []

        # Check duration matches steps
        if episode.steps and episode.sampling_rate_hz > 0:
            expected_duration = len(episode.steps) / episode.sampling_rate_hz
            if abs(episode.duration_seconds - expected_duration) > 1.0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONSISTENCY,
                    severity=ValidationSeverity.WARNING,
                    field="duration_seconds",
                    message=f"Duration mismatch: stated {episode.duration_seconds:.2f}s, computed {expected_duration:.2f}s",
                    value=episode.duration_seconds,
                ))

        # Check channel names match step data
        if episode.steps and episode.channel_names:
            step_channels = set()
            for step in episode.steps[:10]:  # Sample first 10
                step_channels.update(step.condition_monitoring.keys())

            declared_channels = set(episode.channel_names)
            missing = declared_channels - step_channels
            if missing and step_channels:  # Only flag if we have some data
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONSISTENCY,
                    severity=ValidationSeverity.INFO,
                    field="channel_names",
                    message=f"Declared channels not found in data: {missing}",
                    value=list(missing),
                ))

        return issues

    def _compute_sensor_completeness(self, episode: FactoryNetEpisode) -> float:
        """Compute sensor data completeness ratio."""
        if not episode.steps or not episode.channel_names:
            return 0.0

        expected = len(episode.steps) * len(episode.channel_names)
        actual = 0

        for step in episode.steps:
            for channel in episode.channel_names:
                value = step.condition_monitoring.get(channel)
                if value is not None and (not isinstance(value, float) or value == value):
                    actual += 1

        return actual / expected if expected > 0 else 0.0

    def _compute_feature_completeness(self, episode: FactoryNetEpisode) -> float:
        """Compute feature extraction completeness."""
        if not episode.channel_names:
            return 0.0

        expected = len(episode.channel_names)
        actual = len(episode.features)

        return actual / expected if expected > 0 else 0.0


class ValidationReportGenerator:
    """Generates aggregated validation reports for datasets."""

    def __init__(self, validator: Optional[EpisodeValidator] = None):
        """Initialize report generator.

        Args:
            validator: Validator to use (creates default if None)
        """
        self.validator = validator or EpisodeValidator()

    def generate_report(
        self,
        episodes: List[FactoryNetEpisode],
        dataset_name: str,
    ) -> ValidationReport:
        """Generate validation report for a set of episodes.

        Args:
            episodes: List of episodes to validate
            dataset_name: Name of the dataset

        Returns:
            ValidationReport with aggregated results
        """
        results = []
        valid_count = 0
        invalid_count = 0

        total_sensor_completeness = 0.0
        total_label_confidence = 0.0
        total_quality_score = 0.0

        error_counts: Dict[str, int] = {}
        warning_counts: Dict[str, int] = {}

        for episode in episodes:
            result = self.validator.validate(episode)
            results.append(result)

            if result.valid:
                valid_count += 1
            else:
                invalid_count += 1

            total_sensor_completeness += result.sensor_completeness
            total_label_confidence += result.label_confidence
            total_quality_score += result.overall_quality_score

            # Count issues by message
            for issue in result.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    error_counts[issue.message] = error_counts.get(issue.message, 0) + 1
                elif issue.severity == ValidationSeverity.WARNING:
                    warning_counts[issue.message] = warning_counts.get(issue.message, 0) + 1

        n = len(episodes)
        return ValidationReport(
            dataset_name=dataset_name,
            total_episodes=n,
            valid_episodes=valid_count,
            invalid_episodes=invalid_count,
            avg_sensor_completeness=total_sensor_completeness / n if n > 0 else 0.0,
            avg_label_confidence=total_label_confidence / n if n > 0 else 0.0,
            avg_quality_score=total_quality_score / n if n > 0 else 0.0,
            error_counts=error_counts,
            warning_counts=warning_counts,
            episode_results=results,
        )
