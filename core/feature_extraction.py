"""Feature extraction for vibration and condition monitoring data.

Provides signal processing and feature extraction utilities for
bearing fault detection and machinery condition monitoring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VibrationFeatures:
    """Extracted features from vibration signal."""
    # Time domain features
    mean: float = 0.0
    std: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    peak_to_peak: float = 0.0
    crest_factor: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    shape_factor: float = 0.0
    impulse_factor: float = 0.0
    clearance_factor: float = 0.0

    # Frequency domain features
    dominant_frequency_hz: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_spread_hz: float = 0.0
    spectral_energy: float = 0.0

    # Bearing characteristic frequencies (if computed)
    bpfo_amplitude: float = 0.0  # Ball Pass Frequency Outer
    bpfi_amplitude: float = 0.0  # Ball Pass Frequency Inner
    bsf_amplitude: float = 0.0   # Ball Spin Frequency
    ftf_amplitude: float = 0.0   # Fundamental Train Frequency

    # FFT data for storage
    fft_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        result = {
            "mean": self.mean,
            "std": self.std,
            "rms": self.rms,
            "peak": self.peak,
            "peak_to_peak": self.peak_to_peak,
            "crest_factor": self.crest_factor,
            "kurtosis": self.kurtosis,
            "skewness": self.skewness,
            "shape_factor": self.shape_factor,
            "impulse_factor": self.impulse_factor,
            "clearance_factor": self.clearance_factor,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_spread_hz": self.spectral_spread_hz,
            "spectral_energy": self.spectral_energy,
            "bpfo_amplitude": self.bpfo_amplitude,
            "bpfi_amplitude": self.bpfi_amplitude,
            "bsf_amplitude": self.bsf_amplitude,
            "ftf_amplitude": self.ftf_amplitude,
        }
        return result


@dataclass
class BearingGeometry:
    """Bearing geometry parameters for fault frequency calculation."""
    num_balls: int  # Number of rolling elements
    ball_diameter_mm: float  # Ball/roller diameter
    pitch_diameter_mm: float  # Pitch circle diameter
    contact_angle_deg: float = 0.0  # Contact angle in degrees

    def calculate_fault_frequencies(self, rpm: float) -> Dict[str, float]:
        """Calculate theoretical bearing fault frequencies.

        Args:
            rpm: Rotational speed in RPM

        Returns:
            Dictionary with BPFO, BPFI, BSF, FTF frequencies in Hz
        """
        shaft_freq = rpm / 60.0  # Hz
        n = self.num_balls
        d = self.ball_diameter_mm
        D = self.pitch_diameter_mm
        theta = np.radians(self.contact_angle_deg)

        # Fault frequencies
        bpfo = (n / 2) * shaft_freq * (1 - (d / D) * np.cos(theta))
        bpfi = (n / 2) * shaft_freq * (1 + (d / D) * np.cos(theta))
        bsf = (D / (2 * d)) * shaft_freq * (1 - ((d / D) * np.cos(theta)) ** 2)
        ftf = (shaft_freq / 2) * (1 - (d / D) * np.cos(theta))

        return {
            "bpfo": bpfo,
            "bpfi": bpfi,
            "bsf": bsf,
            "ftf": ftf,
        }


# Common bearing geometries
BEARING_GEOMETRIES = {
    # CWRU bearings (SKF 6205-2RS)
    "6205": BearingGeometry(
        num_balls=9,
        ball_diameter_mm=7.94,
        pitch_diameter_mm=39.04,
        contact_angle_deg=0.0,
    ),
    # Deep groove ball bearings
    "6206": BearingGeometry(
        num_balls=9,
        ball_diameter_mm=9.53,
        pitch_diameter_mm=46.64,
        contact_angle_deg=0.0,
    ),
    "6208": BearingGeometry(
        num_balls=9,
        ball_diameter_mm=12.7,
        pitch_diameter_mm=60.0,
        contact_angle_deg=0.0,
    ),
}


class VibrationFeatureExtractor:
    """Extract features from vibration signals for fault detection.

    Computes time-domain, frequency-domain, and bearing-specific features
    from raw vibration signals.

    Example:
        extractor = VibrationFeatureExtractor(sampling_rate_hz=12000)
        features = extractor.extract(signal_data)
        print(f"RMS: {features.rms}, Kurtosis: {features.kurtosis}")
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        bearing_geometry: Optional[BearingGeometry] = None,
        fft_window: str = "hann",
        fft_size: Optional[int] = None,
    ):
        """Initialize feature extractor.

        Args:
            sampling_rate_hz: Signal sampling rate in Hz
            bearing_geometry: Optional bearing parameters for fault frequencies
            fft_window: Window function for FFT ('hann', 'hamming', 'rectangular')
            fft_size: FFT size (default: next power of 2)
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.bearing_geometry = bearing_geometry
        self.fft_window = fft_window
        self.fft_size = fft_size

    def extract(
        self,
        signal: Sequence[float],
        rpm: Optional[float] = None,
        compute_fft: bool = True,
    ) -> VibrationFeatures:
        """Extract features from vibration signal.

        Args:
            signal: 1D array of vibration values
            rpm: Optional rotational speed for bearing frequencies
            compute_fft: Whether to compute frequency domain features

        Returns:
            VibrationFeatures with all computed features
        """
        x = np.asarray(signal, dtype=np.float64)

        if len(x) == 0:
            return VibrationFeatures()

        # Remove DC offset
        x = x - np.mean(x)

        features = VibrationFeatures()

        # Time domain features
        features = self._compute_time_features(x, features)

        # Frequency domain features
        if compute_fft and len(x) > 10:
            features = self._compute_freq_features(x, features)

            # Bearing fault frequencies
            if self.bearing_geometry and rpm:
                features = self._compute_bearing_features(
                    x, features, rpm, self.bearing_geometry
                )

        return features

    def _compute_time_features(
        self,
        x: np.ndarray,
        features: VibrationFeatures,
    ) -> VibrationFeatures:
        """Compute time-domain features.

        Args:
            x: Signal array (DC removed)
            features: Features object to update

        Returns:
            Updated features
        """
        n = len(x)
        if n == 0:
            return features

        # Basic statistics
        features.mean = float(np.mean(x))
        features.std = float(np.std(x))
        features.rms = float(np.sqrt(np.mean(x ** 2)))
        features.peak = float(np.max(np.abs(x)))
        features.peak_to_peak = float(np.max(x) - np.min(x))

        # Shape factors
        if features.rms > 0:
            features.crest_factor = features.peak / features.rms
            features.shape_factor = features.rms / (np.mean(np.abs(x)) + 1e-10)
        else:
            features.crest_factor = 0.0
            features.shape_factor = 0.0

        # Higher order statistics
        if features.std > 0:
            x_norm = (x - np.mean(x)) / features.std
            features.kurtosis = float(np.mean(x_norm ** 4))
            features.skewness = float(np.mean(x_norm ** 3))
        else:
            features.kurtosis = 0.0
            features.skewness = 0.0

        # Impulse factor
        mean_abs = np.mean(np.abs(x))
        if mean_abs > 0:
            features.impulse_factor = features.peak / mean_abs
        else:
            features.impulse_factor = 0.0

        # Clearance factor
        sqrt_mean = np.mean(np.sqrt(np.abs(x))) ** 2
        if sqrt_mean > 0:
            features.clearance_factor = features.peak / sqrt_mean
        else:
            features.clearance_factor = 0.0

        return features

    def _compute_freq_features(
        self,
        x: np.ndarray,
        features: VibrationFeatures,
    ) -> VibrationFeatures:
        """Compute frequency-domain features.

        Args:
            x: Signal array
            features: Features object to update

        Returns:
            Updated features
        """
        n = len(x)

        # Determine FFT size
        if self.fft_size:
            nfft = self.fft_size
        else:
            nfft = 2 ** int(np.ceil(np.log2(n)))

        # Apply window
        if self.fft_window == "hann":
            window = np.hanning(n)
        elif self.fft_window == "hamming":
            window = np.hamming(n)
        else:
            window = np.ones(n)

        x_windowed = x * window

        # Compute FFT
        fft_result = np.fft.rfft(x_windowed, nfft)
        fft_magnitude = np.abs(fft_result) / n
        fft_freq = np.fft.rfftfreq(nfft, 1.0 / self.sampling_rate_hz)

        # Store FFT data (downsampled for storage efficiency)
        max_freq_idx = min(len(fft_freq), 1000)  # Limit to 1000 points
        features.fft_frequencies = fft_freq[:max_freq_idx].astype(np.float32)
        features.fft_magnitudes = fft_magnitude[:max_freq_idx].astype(np.float32)

        # Dominant frequency
        if len(fft_magnitude) > 1:
            # Skip DC component
            peak_idx = np.argmax(fft_magnitude[1:]) + 1
            features.dominant_frequency_hz = float(fft_freq[peak_idx])

        # Spectral centroid (center of mass)
        total_energy = np.sum(fft_magnitude)
        if total_energy > 0:
            features.spectral_centroid_hz = float(
                np.sum(fft_freq * fft_magnitude) / total_energy
            )

            # Spectral spread (standard deviation around centroid)
            features.spectral_spread_hz = float(np.sqrt(
                np.sum(((fft_freq - features.spectral_centroid_hz) ** 2) * fft_magnitude) / total_energy
            ))

        # Total spectral energy
        features.spectral_energy = float(np.sum(fft_magnitude ** 2))

        return features

    def _compute_bearing_features(
        self,
        x: np.ndarray,
        features: VibrationFeatures,
        rpm: float,
        geometry: BearingGeometry,
    ) -> VibrationFeatures:
        """Compute bearing fault frequency amplitudes.

        Args:
            x: Signal array
            features: Features object to update
            rpm: Rotational speed in RPM
            geometry: Bearing geometry parameters

        Returns:
            Updated features with bearing frequency amplitudes
        """
        # Calculate theoretical fault frequencies
        fault_freqs = geometry.calculate_fault_frequencies(rpm)

        # Get FFT magnitude at each fault frequency
        fft_freq = features.fft_frequencies
        fft_mag = features.fft_magnitudes

        if len(fft_freq) == 0:
            return features

        freq_resolution = fft_freq[1] - fft_freq[0] if len(fft_freq) > 1 else 1.0

        def get_amplitude_at_freq(target_freq: float, tolerance_bins: int = 2) -> float:
            """Get max amplitude near target frequency."""
            idx = int(target_freq / freq_resolution) if freq_resolution > 0 else 0
            start = max(0, idx - tolerance_bins)
            end = min(len(fft_mag), idx + tolerance_bins + 1)
            if start < end:
                return float(np.max(fft_mag[start:end]))
            return 0.0

        features.bpfo_amplitude = get_amplitude_at_freq(fault_freqs["bpfo"])
        features.bpfi_amplitude = get_amplitude_at_freq(fault_freqs["bpfi"])
        features.bsf_amplitude = get_amplitude_at_freq(fault_freqs["bsf"])
        features.ftf_amplitude = get_amplitude_at_freq(fault_freqs["ftf"])

        return features

    def extract_batch(
        self,
        signals: List[Sequence[float]],
        rpm: Optional[float] = None,
    ) -> List[VibrationFeatures]:
        """Extract features from multiple signals.

        Args:
            signals: List of signal arrays
            rpm: Optional rotational speed

        Returns:
            List of VibrationFeatures
        """
        return [self.extract(s, rpm=rpm) for s in signals]


def compute_envelope_spectrum(
    signal: Sequence[float],
    sampling_rate_hz: float,
    bandpass_low_hz: Optional[float] = None,
    bandpass_high_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute envelope spectrum for bearing fault detection.

    The envelope spectrum is effective for detecting bearing faults
    by demodulating the high-frequency carrier signal.

    Args:
        signal: Input vibration signal
        sampling_rate_hz: Sampling rate
        bandpass_low_hz: Lower bandpass cutoff (default: 1/4 Nyquist)
        bandpass_high_hz: Upper bandpass cutoff (default: 3/4 Nyquist)

    Returns:
        Tuple of (frequencies, envelope_spectrum)
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)

    nyquist = sampling_rate_hz / 2

    # Default bandpass range
    if bandpass_low_hz is None:
        bandpass_low_hz = nyquist / 4
    if bandpass_high_hz is None:
        bandpass_high_hz = nyquist * 3 / 4

    # Simple bandpass using FFT
    fft_result = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, 1.0 / sampling_rate_hz)

    # Zero out frequencies outside bandpass
    mask = (np.abs(freqs) < bandpass_low_hz) | (np.abs(freqs) > bandpass_high_hz)
    fft_filtered = fft_result.copy()
    fft_filtered[mask] = 0

    # Inverse FFT to get filtered signal
    x_filtered = np.real(np.fft.ifft(fft_filtered))

    # Compute envelope using Hilbert transform approximation
    # (analytical signal magnitude)
    analytic = np.fft.ifft(2 * fft_filtered * (freqs > 0))
    envelope = np.abs(analytic)

    # Compute envelope spectrum
    env_fft = np.fft.rfft(envelope - np.mean(envelope))
    env_freq = np.fft.rfftfreq(n, 1.0 / sampling_rate_hz)
    env_spectrum = np.abs(env_fft) / n

    return env_freq, env_spectrum


def compute_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Compute basic statistics for a signal.

    Args:
        values: 1D array of values

    Returns:
        Dictionary with mean, std, min, max, rms
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "rms": 0.0}

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }
