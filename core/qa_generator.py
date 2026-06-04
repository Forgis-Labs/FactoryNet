"""Q&A Generator for FactoryNet episodes.

Generates template-based question-answer pairs from episode data
with difficulty levels, criticality tags, and expertise requirements.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.adapters.base_adapter import FaultType, SeverityLevel
from core.normalizer import FactoryNetEpisode


class Difficulty(Enum):
    """Question difficulty level."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Criticality(Enum):
    """Answer criticality for safety/operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExpertiseArea(Enum):
    """Required expertise area."""
    VIBRATION_ANALYSIS = "vibration_analysis"
    BEARING_DIAGNOSTICS = "bearing_diagnostics"
    MACHINERY_MAINTENANCE = "machinery_maintenance"
    SIGNAL_PROCESSING = "signal_processing"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"


@dataclass
class QAPair:
    """Question-answer pair with metadata."""
    question: str
    answer: str
    question_type: str  # e.g., "fault_identification", "severity_assessment"

    # Assessment metadata
    difficulty: Difficulty = Difficulty.MEDIUM
    criticality: Criticality = Criticality.MEDIUM
    expertise_areas: List[ExpertiseArea] = field(default_factory=list)

    # Context
    context_required: bool = True  # Whether timeseries context is needed
    reasoning_required: bool = False  # Whether step-by-step reasoning expected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "difficulty": self.difficulty.value,
            "criticality": self.criticality.value,
            "expertise_areas": [e.value for e in self.expertise_areas],
            "context_required": self.context_required,
            "reasoning_required": self.reasoning_required,
        }


# Question templates organized by type
QUESTION_TEMPLATES = {
    "fault_identification": [
        {
            "template": "Based on the vibration signal from {sensor_location}, what type of bearing fault is present?",
            "answer_template": "The vibration signal indicates {fault_type}. Key indicators include {indicators}.",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.HIGH,
            "expertise": [ExpertiseArea.VIBRATION_ANALYSIS, ExpertiseArea.BEARING_DIAGNOSTICS],
        },
        {
            "template": "Analyze the vibration signature and identify the fault condition.",
            "answer_template": "The analysis indicates {fault_type} with {severity} severity. {reasoning}",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.HIGH,
            "expertise": [ExpertiseArea.VIBRATION_ANALYSIS],
        },
        {
            "template": "What fault pattern can you identify from this bearing vibration data?",
            "answer_template": "{fault_type}. The characteristic features are: {features}.",
            "difficulty": Difficulty.EASY,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.BEARING_DIAGNOSTICS],
        },
    ],
    "severity_assessment": [
        {
            "template": "What is the severity level of the detected bearing fault?",
            "answer_template": "The fault severity is {severity}. {reasoning}",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.HIGH,
            "expertise": [ExpertiseArea.BEARING_DIAGNOSTICS, ExpertiseArea.PREDICTIVE_MAINTENANCE],
        },
        {
            "template": "How urgent is maintenance action required based on this data?",
            "answer_template": "Maintenance urgency: {urgency}. The {severity} fault level suggests {recommendation}.",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.CRITICAL,
            "expertise": [ExpertiseArea.MACHINERY_MAINTENANCE, ExpertiseArea.PREDICTIVE_MAINTENANCE],
        },
        {
            "template": "Rate the condition of this bearing on a scale from healthy to critical.",
            "answer_template": "Bearing condition: {severity}. {reasoning}",
            "difficulty": Difficulty.EASY,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.BEARING_DIAGNOSTICS],
        },
    ],
    "feature_analysis": [
        {
            "template": "What does the RMS value of {rms:.4f} indicate about the vibration level?",
            "answer_template": "The RMS value of {rms:.4f} indicates {interpretation}. {comparison}",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.VIBRATION_ANALYSIS, ExpertiseArea.SIGNAL_PROCESSING],
        },
        {
            "template": "The kurtosis value is {kurtosis:.2f}. What does this suggest about the signal?",
            "answer_template": "A kurtosis of {kurtosis:.2f} {interpretation}. {significance}",
            "difficulty": Difficulty.HARD,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.SIGNAL_PROCESSING, ExpertiseArea.VIBRATION_ANALYSIS],
        },
        {
            "template": "Interpret the dominant frequency of {freq:.1f} Hz in relation to the {rpm} RPM shaft speed.",
            "answer_template": "The dominant frequency of {freq:.1f} Hz corresponds to {interpretation}. {significance}",
            "difficulty": Difficulty.HARD,
            "criticality": Criticality.HIGH,
            "expertise": [ExpertiseArea.VIBRATION_ANALYSIS, ExpertiseArea.SIGNAL_PROCESSING],
        },
    ],
    "root_cause": [
        {
            "template": "What is the most likely root cause of this {fault_type}?",
            "answer_template": "The most likely cause is {cause}. Contributing factors include: {factors}.",
            "difficulty": Difficulty.HARD,
            "criticality": Criticality.HIGH,
            "expertise": [ExpertiseArea.ROOT_CAUSE_ANALYSIS, ExpertiseArea.BEARING_DIAGNOSTICS],
        },
        {
            "template": "What failure mechanism is indicated by this vibration pattern?",
            "answer_template": "The pattern indicates {mechanism}. This typically results from {causes}.",
            "difficulty": Difficulty.HARD,
            "criticality": Criticality.HIGH,
            "expertise": [ExpertiseArea.ROOT_CAUSE_ANALYSIS],
        },
    ],
    "maintenance_action": [
        {
            "template": "What maintenance action should be taken for this bearing condition?",
            "answer_template": "Recommended action: {action}. Priority: {priority}. {details}",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.CRITICAL,
            "expertise": [ExpertiseArea.MACHINERY_MAINTENANCE, ExpertiseArea.PREDICTIVE_MAINTENANCE],
        },
        {
            "template": "Should this bearing be replaced immediately, scheduled for replacement, or monitored?",
            "answer_template": "{recommendation}. Rationale: {rationale}",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.CRITICAL,
            "expertise": [ExpertiseArea.MACHINERY_MAINTENANCE],
        },
        {
            "template": "What is the recommended monitoring interval for this bearing?",
            "answer_template": "Recommended interval: {interval}. {reasoning}",
            "difficulty": Difficulty.EASY,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.PREDICTIVE_MAINTENANCE],
        },
    ],
    "operating_conditions": [
        {
            "template": "How does the {load_hp} HP load affect the bearing fault characteristics?",
            "answer_template": "At {load_hp} HP load, {effect}. {recommendation}",
            "difficulty": Difficulty.HARD,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.MACHINERY_MAINTENANCE, ExpertiseArea.BEARING_DIAGNOSTICS],
        },
        {
            "template": "Is the bearing operating within normal parameters at {rpm} RPM?",
            "answer_template": "{assessment}. At {rpm} RPM, {analysis}.",
            "difficulty": Difficulty.MEDIUM,
            "criticality": Criticality.MEDIUM,
            "expertise": [ExpertiseArea.MACHINERY_MAINTENANCE],
        },
    ],
}


# Fault type specific answer components
FAULT_INDICATORS = {
    FaultType.INNER_RACE: [
        "elevated BPFI (Ball Pass Frequency Inner) amplitude",
        "modulation with shaft speed",
        "harmonics at multiples of BPFI",
    ],
    FaultType.OUTER_RACE: [
        "elevated BPFO (Ball Pass Frequency Outer) amplitude",
        "stationary defect pattern",
        "consistent amplitude levels",
    ],
    FaultType.BALL: [
        "elevated BSF (Ball Spin Frequency) amplitude",
        "2x BSF sidebands",
        "non-synchronous vibration",
    ],
    FaultType.CAGE: [
        "elevated FTF (Fundamental Train Frequency) amplitude",
        "sub-synchronous vibration",
        "unstable amplitude modulation",
    ],
    FaultType.NORMAL: [
        "baseline vibration levels",
        "no significant fault frequencies",
        "normal kurtosis values",
    ],
}

SEVERITY_DESCRIPTIONS = {
    SeverityLevel.HEALTHY: ("healthy", "Continue normal monitoring"),
    SeverityLevel.MINOR: ("minor", "Increase monitoring frequency"),
    SeverityLevel.MODERATE: ("moderate", "Schedule maintenance within 2-4 weeks"),
    SeverityLevel.SEVERE: ("severe", "Schedule maintenance within 1 week"),
    SeverityLevel.CRITICAL: ("critical", "Immediate action required"),
}

URGENCY_MAPPING = {
    SeverityLevel.HEALTHY: "None - routine monitoring only",
    SeverityLevel.MINOR: "Low - monitor for progression",
    SeverityLevel.MODERATE: "Medium - plan maintenance",
    SeverityLevel.SEVERE: "High - schedule prompt maintenance",
    SeverityLevel.CRITICAL: "Immediate - risk of failure",
}


class QAGenerator:
    """Generates question-answer pairs from FactoryNet episodes.

    Uses template-based generation with difficulty and criticality tags.
    Generates diverse questions covering fault identification, severity
    assessment, feature analysis, root cause, and maintenance actions.

    Example:
        generator = QAGenerator()
        qa_pairs = generator.generate(episode)
        for qa in qa_pairs:
            print(f"Q: {qa.question}")
            print(f"A: {qa.answer}")
    """

    def __init__(
        self,
        questions_per_category: int = 2,
        include_reasoning: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize Q&A generator.

        Args:
            questions_per_category: Max questions per category
            include_reasoning: Include reasoning-required questions
            seed: Random seed for reproducibility
        """
        self.questions_per_category = questions_per_category
        self.include_reasoning = include_reasoning

        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        episode: FactoryNetEpisode,
        categories: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """Generate Q&A pairs for an episode.

        Args:
            episode: Episode to generate questions for
            categories: Optional list of question categories to include

        Returns:
            List of QAPair objects
        """
        qa_pairs = []

        # Determine fault type from state annotation
        fault_type = self._infer_fault_type(episode)
        severity = self._infer_severity(episode)

        # Get features if available
        features = self._get_primary_features(episode)

        # Select categories
        if categories is None:
            categories = list(QUESTION_TEMPLATES.keys())

        for category in categories:
            templates = QUESTION_TEMPLATES.get(category, [])
            selected = random.sample(
                templates,
                min(self.questions_per_category, len(templates))
            )

            for template_info in selected:
                qa = self._generate_from_template(
                    template_info,
                    episode,
                    fault_type,
                    severity,
                    features,
                    category,
                )
                if qa:
                    qa_pairs.append(qa)

        return qa_pairs

    def _infer_fault_type(self, episode: FactoryNetEpisode) -> FaultType:
        """Infer fault type from episode state annotation."""
        synset = episode.state_annotation.state_synset

        if "inn" in synset:
            return FaultType.INNER_RACE
        elif "out" in synset:
            return FaultType.OUTER_RACE
        elif "bal" in synset:
            return FaultType.BALL
        elif "cag" in synset:
            return FaultType.CAGE
        elif "nom" in synset or "healthy" in episode.state_annotation.state_label.lower():
            return FaultType.NORMAL
        else:
            return FaultType.UNKNOWN

    def _infer_severity(self, episode: FactoryNetEpisode) -> SeverityLevel:
        """Infer severity level from episode."""
        severity_val = episode.state_annotation.severity

        if severity_val >= 0.9:
            return SeverityLevel.CRITICAL
        elif severity_val >= 0.7:
            return SeverityLevel.SEVERE
        elif severity_val >= 0.4:
            return SeverityLevel.MODERATE
        elif severity_val > 0:
            return SeverityLevel.MINOR
        else:
            return SeverityLevel.HEALTHY

    def _get_primary_features(self, episode: FactoryNetEpisode) -> Dict[str, float]:
        """Get features from primary channel."""
        features = {}

        if episode.features:
            # Get first channel features
            for channel_name, channel_features in episode.features.items():
                features = channel_features.to_dict()
                break

        return features

    def _generate_from_template(
        self,
        template_info: Dict[str, Any],
        episode: FactoryNetEpisode,
        fault_type: FaultType,
        severity: SeverityLevel,
        features: Dict[str, float],
        category: str,
    ) -> Optional[QAPair]:
        """Generate Q&A from a template."""
        try:
            # Build context for template formatting
            context = self._build_context(
                episode, fault_type, severity, features
            )

            # Format question
            question = template_info["template"].format(**context)

            # Format answer
            answer = self._generate_answer(
                template_info, fault_type, severity, features, context
            )

            return QAPair(
                question=question,
                answer=answer,
                question_type=category,
                difficulty=template_info.get("difficulty", Difficulty.MEDIUM),
                criticality=template_info.get("criticality", Criticality.MEDIUM),
                expertise_areas=template_info.get("expertise", []),
                context_required=True,
                reasoning_required=self.include_reasoning and template_info.get(
                    "difficulty", Difficulty.MEDIUM
                ) == Difficulty.HARD,
            )

        except (KeyError, ValueError) as e:
            # Template formatting failed, skip this question
            return None

    def _build_context(
        self,
        episode: FactoryNetEpisode,
        fault_type: FaultType,
        severity: SeverityLevel,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build context dictionary for template formatting."""
        # Determine sensor location from channels
        sensor_location = "drive end"
        if episode.channel_names:
            if any("fe" in ch.lower() for ch in episode.channel_names):
                sensor_location = "fan end"
            elif any("de" in ch.lower() for ch in episode.channel_names):
                sensor_location = "drive end"

        # Build fault type description
        fault_desc = fault_type.value.replace("_", " ")

        # Severity description
        severity_desc, recommendation = SEVERITY_DESCRIPTIONS.get(
            severity, ("unknown", "Investigate further")
        )

        context = {
            "sensor_location": sensor_location,
            "fault_type": fault_desc,
            "severity": severity_desc,
            "rpm": episode.rpm or 1797,
            "load_hp": episode.load_hp or 0,
            "recommendation": recommendation,
            "urgency": URGENCY_MAPPING.get(severity, "Unknown"),
        }

        # Add features if available
        context["rms"] = features.get("rms", 0.01)
        context["kurtosis"] = features.get("kurtosis", 3.0)
        context["freq"] = features.get("dominant_frequency_hz", 100.0)
        context["crest_factor"] = features.get("crest_factor", 3.0)

        return context

    def _generate_answer(
        self,
        template_info: Dict[str, Any],
        fault_type: FaultType,
        severity: SeverityLevel,
        features: Dict[str, float],
        context: Dict[str, Any],
    ) -> str:
        """Generate answer text."""
        template = template_info.get("answer_template", "")

        # Add fault-specific indicators
        indicators = FAULT_INDICATORS.get(fault_type, ["characteristic vibration pattern"])
        context["indicators"] = ", ".join(random.sample(indicators, min(2, len(indicators))))

        # Add features description
        context["features"] = self._describe_features(features, fault_type)

        # Add reasoning
        context["reasoning"] = self._generate_reasoning(fault_type, severity, features)

        # Add maintenance details
        context["action"] = self._get_maintenance_action(fault_type, severity)
        context["priority"] = URGENCY_MAPPING.get(severity, "Medium")
        context["details"] = f"Based on {severity.value} severity level."

        # Cause and mechanism
        context["cause"] = self._get_likely_cause(fault_type)
        context["factors"] = self._get_contributing_factors(fault_type)
        context["mechanism"] = self._get_failure_mechanism(fault_type)
        context["causes"] = self._get_mechanism_causes(fault_type)

        # Operating condition effects
        context["effect"] = self._get_load_effect(severity)
        context["assessment"] = "Yes" if severity == SeverityLevel.HEALTHY else "No - fault detected"
        context["analysis"] = self._get_rpm_analysis(context["rpm"], fault_type)

        # Monitoring interval
        context["interval"] = self._get_monitoring_interval(severity)

        # Interpretation for features
        context["interpretation"] = self._interpret_rms(features.get("rms", 0.01))
        context["comparison"] = self._compare_to_baseline(features.get("rms", 0.01))
        context["significance"] = self._get_frequency_significance(fault_type)

        # Kurtosis interpretation
        kurtosis = features.get("kurtosis", 3.0)
        if kurtosis > 5:
            context["interpretation"] = "indicates impulsive behavior typical of bearing faults"
        elif kurtosis > 3.5:
            context["interpretation"] = "suggests developing fault with periodic impacts"
        else:
            context["interpretation"] = "is near the normal Gaussian value of 3.0"

        # Recommendation
        context["rationale"] = f"The {severity.value} condition and {fault_type.value} pattern indicate this action."

        try:
            return template.format(**context)
        except KeyError:
            return f"Analysis indicates {fault_type.value} with {severity.value} severity."

    def _describe_features(self, features: Dict[str, float], fault_type: FaultType) -> str:
        """Generate feature description."""
        parts = []

        if "rms" in features:
            rms = features["rms"]
            if rms > 0.5:
                parts.append(f"elevated RMS of {rms:.3f}")
            else:
                parts.append(f"RMS of {rms:.3f}")

        if "kurtosis" in features:
            kurt = features["kurtosis"]
            if kurt > 5:
                parts.append(f"high kurtosis ({kurt:.1f})")
            elif kurt > 3.5:
                parts.append(f"moderate kurtosis ({kurt:.1f})")

        if fault_type != FaultType.NORMAL:
            parts.append("characteristic fault frequencies")

        return ", ".join(parts) if parts else "vibration analysis results"

    def _generate_reasoning(
        self,
        fault_type: FaultType,
        severity: SeverityLevel,
        features: Dict[str, float],
    ) -> str:
        """Generate reasoning text."""
        if fault_type == FaultType.INNER_RACE:
            return "The BPFI amplitude and shaft speed modulation confirm inner race defect."
        elif fault_type == FaultType.OUTER_RACE:
            return "The BPFO presence with consistent amplitude indicates outer race fault."
        elif fault_type == FaultType.BALL:
            return "The BSF and its harmonics indicate rolling element damage."
        elif fault_type == FaultType.CAGE:
            return "Sub-synchronous vibration at FTF indicates cage degradation."
        elif fault_type == FaultType.NORMAL:
            return "No significant fault frequencies detected; baseline vibration levels observed."
        else:
            return "Vibration pattern indicates potential fault condition."

    def _get_maintenance_action(self, fault_type: FaultType, severity: SeverityLevel) -> str:
        """Get recommended maintenance action."""
        if severity == SeverityLevel.CRITICAL:
            return "Immediate bearing replacement"
        elif severity == SeverityLevel.SEVERE:
            return "Schedule bearing replacement within 1 week"
        elif severity == SeverityLevel.MODERATE:
            return "Plan bearing replacement; increase monitoring"
        elif severity == SeverityLevel.MINOR:
            return "Monitor condition; prepare replacement bearing"
        else:
            return "Continue routine monitoring"

    def _get_likely_cause(self, fault_type: FaultType) -> str:
        """Get likely root cause."""
        causes = {
            FaultType.INNER_RACE: "fatigue spalling from cyclic stress concentration",
            FaultType.OUTER_RACE: "localized fatigue failure in load zone",
            FaultType.BALL: "surface fatigue from rolling contact stress",
            FaultType.CAGE: "wear from inadequate lubrication or contamination",
            FaultType.NORMAL: "N/A - bearing operating normally",
        }
        return causes.get(fault_type, "unknown failure mechanism")

    def _get_contributing_factors(self, fault_type: FaultType) -> str:
        """Get contributing factors."""
        factors = {
            FaultType.INNER_RACE: "misalignment, excessive preload, contamination",
            FaultType.OUTER_RACE: "improper mounting, static overload, corrosion",
            FaultType.BALL: "lubrication breakdown, contamination, material defects",
            FaultType.CAGE: "poor lubrication, high speed, improper clearance",
            FaultType.NORMAL: "N/A",
        }
        return factors.get(fault_type, "operating conditions, maintenance history")

    def _get_failure_mechanism(self, fault_type: FaultType) -> str:
        """Get failure mechanism description."""
        mechanisms = {
            FaultType.INNER_RACE: "rolling contact fatigue with subsurface crack propagation",
            FaultType.OUTER_RACE: "Hertzian contact stress exceeding material limits",
            FaultType.BALL: "surface pitting and spalling from contact fatigue",
            FaultType.CAGE: "wear and plastic deformation from friction",
            FaultType.NORMAL: "no failure mechanism present",
        }
        return mechanisms.get(fault_type, "progressive degradation")

    def _get_mechanism_causes(self, fault_type: FaultType) -> str:
        """Get mechanism causes."""
        return "cyclic loading, material fatigue, and environmental factors"

    def _get_load_effect(self, severity: SeverityLevel) -> str:
        """Get load effect description."""
        if severity in (SeverityLevel.SEVERE, SeverityLevel.CRITICAL):
            return "higher loads accelerate fault progression"
        elif severity == SeverityLevel.MODERATE:
            return "load affects fault frequency amplitude"
        else:
            return "load is within normal operating range"

    def _get_rpm_analysis(self, rpm: float, fault_type: FaultType) -> str:
        """Get RPM analysis."""
        if fault_type == FaultType.NORMAL:
            return f"vibration levels are normal for {rpm} RPM operation"
        else:
            return f"fault frequencies scale with the {rpm} RPM shaft speed"

    def _get_monitoring_interval(self, severity: SeverityLevel) -> str:
        """Get recommended monitoring interval."""
        intervals = {
            SeverityLevel.HEALTHY: "Monthly",
            SeverityLevel.MINOR: "Weekly",
            SeverityLevel.MODERATE: "Every 3 days",
            SeverityLevel.SEVERE: "Daily",
            SeverityLevel.CRITICAL: "Continuous until replacement",
        }
        return intervals.get(severity, "Weekly")

    def _interpret_rms(self, rms: float) -> str:
        """Interpret RMS value."""
        if rms > 1.0:
            return "very high vibration requiring immediate attention"
        elif rms > 0.5:
            return "elevated vibration suggesting fault presence"
        elif rms > 0.2:
            return "moderate vibration within acceptable limits"
        else:
            return "low vibration indicating good condition"

    def _compare_to_baseline(self, rms: float) -> str:
        """Compare RMS to typical baseline."""
        if rms > 0.5:
            return "This exceeds typical baseline values by a significant margin."
        elif rms > 0.2:
            return "This is slightly above typical baseline values."
        else:
            return "This is within expected baseline range."

    def _get_frequency_significance(self, fault_type: FaultType) -> str:
        """Get frequency significance."""
        if fault_type == FaultType.INNER_RACE:
            return "This frequency relationship confirms inner race fault."
        elif fault_type == FaultType.OUTER_RACE:
            return "This matches the expected BPFO for outer race defect."
        elif fault_type == FaultType.BALL:
            return "The frequency indicates ball spin frequency fault."
        else:
            return "The frequency pattern is consistent with the identified condition."
