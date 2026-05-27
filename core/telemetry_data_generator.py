from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# When running from within the `core` directory.
from telemetry_dataset import TelemetryData  


from tqdm.auto import tqdm


@dataclass
class SineProperties:
    amplitude: float
    frequency: float
    phase: float
    noise_std: float


@dataclass
class StepProperties:
    step_size: float
    step_time: float
    noise_std: float


@dataclass
class RandomWalkProperties:
    noise_std: float
    drift: float

@dataclass
class SeriesMetadata:
    domain: Literal["industrial", "vibration", "stock"]
    subtype: str
    properties: Dict[str, float]
    anomalies: List[Dict[str, Any]]


class TimeseriesGenerator:
    def __init__(
        self,
        n_timeseries: int,
        time_duration: float,
        frequency: float = 1.0,
        seed: int = 42,
    ):
        """
        n_timeseries: how many independent time series to generate
        time_duration: total duration in seconds
        frequency: sampling frequency (Hz)
        """
        self.n_timeseries = n_timeseries
        self.time_duration = float(time_duration)
        self.frequency = float(frequency)
        self.time_series_length = int(self.time_duration * self.frequency)
        self.seed = seed

        self.random_state = np.random.RandomState(seed)

        # Shared time axis for all generators
        # endpoint=False -> exact regular sampling at dt = 1/frequency
        self.time = np.linspace(0.0, self.time_duration, self.time_series_length, endpoint=False)

    # -------------------------------------------------------------------------
    # Sinusoidal generator
    # -------------------------------------------------------------------------#
    def _generate_sinusoidal_timeseries(
        self,
        freq_range: Tuple[float, float] = (0.1, 10.0),
        amp_range: Tuple[float, float] = (0.1, 5.0),
        phase_range: Tuple[float, float] = (0.0, 2 * np.pi),
        noise_std_range: Tuple[float, float] = (0.0, 0.1),
        return_properties: bool = False,
    ) -> (
        List[np.ndarray]
        | Tuple[List[np.ndarray], List[SineProperties]]
    ):
        """
        Generate n_timeseries sinusoids with random properties for each series.

        Vectorized implementation: all parameters and series are generated in
        one shot instead of Python loops.

        Each timeseries gets:
          - amplitude ~ U(amp_range)
          - frequency ~ U(freq_range)
          - phase     ~ U(phase_range)
          - noise_std ~ U(noise_std_range)
        """
        n = self.n_timeseries
        if n <= 0:
            return ([], []) if return_properties else []

        rs = self.random_state

        amplitude = rs.uniform(*amp_range, size=n)
        frequency = rs.uniform(*freq_range, size=n)
        phase = rs.uniform(*phase_range, size=n)
        noise_std = rs.uniform(*noise_std_range, size=n)

        # time: (T,) -> broadcast to (n, T)
        t = self.time[None, :]  # (1, T)
        angles = 2 * np.pi * frequency[:, None] * t + phase[:, None]  # (n, T)
        clean = amplitude[:, None] * np.sin(angles)  # (n, T)

        noise = rs.normal(0.0, 1.0, size=(n, self.time_series_length)) * noise_std[:, None]
        values_2d = clean + noise  # (n, T)

        timeseries = [values_2d[i] for i in range(n)]
        properties = [
            SineProperties(
                amplitude=float(amplitude[i]),
                frequency=float(frequency[i]),
                phase=float(phase[i]),
                noise_std=float(noise_std[i]),
            )
            for i in range(n)
        ]

        if return_properties:
            return timeseries, properties
        return timeseries

    # -------------------------------------------------------------------------
    # Step generator
    # -------------------------------------------------------------------------
    def _generate_step_timeseries(
        self,
        step_size_range: Tuple[float, float] = (0.5, 2.0),
        step_time_range: Tuple[float, float] = (0.2, 0.8),
        noise_std_range: Tuple[float, float] = (0.0, 0.01),
        return_properties: bool = False,
    ) -> (
        List[np.ndarray]
        | Tuple[List[np.ndarray], List[StepProperties]]
    ):
        """
        Generate n_timeseries step responses with random properties for each series.

        Vectorized implementation using a broadcasted step mask.

        Each timeseries gets:
          - step_size ~ U(step_size_range)
          - step_time (fraction of total duration) ~ U(step_time_range)
          - noise_std ~ U(noise_std_range)
        """
        n = self.n_timeseries
        if n <= 0:
            return ([], []) if return_properties else []

        rs = self.random_state

        step_size = rs.uniform(*step_size_range, size=n)
        step_fraction = rs.uniform(*step_time_range, size=n)  # fraction of time_duration
        noise_std = rs.uniform(*noise_std_range, size=n)

        # Convert step_fraction (0–1) into actual time value for metadata
        step_time_values = step_fraction * self.time_duration  # (n,)

        # time: (T,) -> broadcast to (n, T)
        t = self.time[None, :]  # (1, T)
        # Mask is 1.0 after the step, 0.0 before
        mask = (t >= step_time_values[:, None]).astype(float)  # (n, T)
        values_2d = mask * step_size[:, None]  # (n, T)

        noise = rs.normal(0.0, 1.0, size=(n, self.time_series_length)) * noise_std[:, None]
        values_2d = values_2d + noise

        timeseries = [values_2d[i] for i in range(n)]
        properties = [
            StepProperties(
                step_size=float(step_size[i]),
                step_time=float(step_time_values[i]),
                noise_std=float(noise_std[i]),
            )
            for i in range(n)
        ]

        if return_properties:
            return timeseries, properties
        return timeseries

    # -------------------------------------------------------------------------
    # Random walk generator
    # -------------------------------------------------------------------------
    def _generate_random_walk_timeseries(
        self,
        noise_std_range: Tuple[float, float] = (0.01, 0.2),
        drift_range: Tuple[float, float] = (-0.001, 0.001),
        return_properties: bool = False,
    ) -> (
        List[np.ndarray]
        | Tuple[List[np.ndarray], List[RandomWalkProperties]]
    ):
        """
        Generate n_timeseries random walks.

        Vectorized implementation: all random walks in one shot.

        Each timeseries gets:
          - noise_std ~ U(noise_std_range)
          - drift     ~ U(drift_range)  (constant drift per step)
        """
        n = self.n_timeseries
        if n <= 0:
            return ([], []) if return_properties else []

        rs = self.random_state

        noise_std = rs.uniform(*noise_std_range, size=n)
        drift = rs.uniform(*drift_range, size=n)

        # Base standard normal increments for all series, then scale and shift
        base = rs.normal(0.0, 1.0, size=(n, self.time_series_length))
        increments = base * noise_std[:, None] + drift[:, None]  # (n, T)
        values_2d = np.cumsum(increments, axis=1)  # (n, T)

        timeseries = [values_2d[i] for i in range(n)]
        properties = [
            RandomWalkProperties(
                noise_std=float(noise_std[i]),
                drift=float(drift[i]),
            )
            for i in range(n)
        ]

        if return_properties:
            return timeseries, properties
        return timeseries



    def _generate_mixed_realistic(
        self,
        domains: Tuple[str, ...] = ("industrial", "vibration", "stock"),
        return_metadata: bool = True,
    ) -> (
        List[TelemetryData]
        | Tuple[List[TelemetryData], List[SeriesMetadata]]
    ):
        """
        Generate n_timeseries from a mixture of 'industrial', 'vibration', 'stock'.

        Each series:
          - randomly picks a domain from `domains`
          - calls the appropriate domain-specific generator
          - returns TimeSeries plus rich metadata
        """
        ts_list: List[TelemetryData] = []
        meta_list: List[SeriesMetadata] = []

        for i in range(self.n_timeseries):
            domain = self.random_state.choice(domains)
            if domain == "industrial":
                values, meta = self._generate_industrial_sensor_like()
            elif domain == "vibration":
                values, meta = self._generate_vibration_like()
            elif domain == "stock":
                values, meta = self._generate_stock_like()
            else:
                raise ValueError(f"Unknown domain: {domain}")

            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

            telemetry = TelemetryData(
                id=f"{domain}_{i}",
                time=self.time.copy(),
                timeseries=values.astype(float),
                metadata={
                    "domain": meta.domain,
                    "subtype": meta.subtype,
                    "properties": meta.properties,
                    "anomalies": meta.anomalies,
                },
                statistics=stats,
            )

            ts_list.append(telemetry)
            meta_list.append(meta)

        if return_metadata:
            return ts_list, meta_list
        return ts_list

    # -------------------------------------------------------------------------
    # High-level dataset generator with type proportions
    # -------------------------------------------------------------------------
    def iter_telemetry(
        self,
        total: int,
        type_proportions: Dict[str, float],
    ):
        """
        Streaming generator version of `generate_telemetry`.

        Yields `TelemetryData` objects one by one instead of returning a
        fully materialized list, which keeps peak memory usage much lower
        for very large datasets.
        """

        if total <= 0:
            raise ValueError("total must be positive")
        if not type_proportions:
            raise ValueError("type_proportions must not be empty")

        valid_types = {"sine", "step", "random_walk", "industrial", "vibration", "stock"}
        unknown = set(type_proportions) - valid_types
        if unknown:
            raise ValueError(f"Unknown series types in type_proportions: {sorted(unknown)}")

        # Normalize proportions to sum to 1.0
        total_prop = float(sum(type_proportions.values()))
        if total_prop <= 0.0:
            raise ValueError("Sum of type_proportions must be positive")

        normalized = {k: v / total_prop for k, v in type_proportions.items()}

        # Convert proportions to integer counts whose sum is exactly `total`
        raw_counts = {k: total * p for k, p in normalized.items()}
        int_counts = {k: int(np.floor(c)) for k, c in raw_counts.items()}
        assigned = sum(int_counts.values())
        remaining = total - assigned

        if remaining > 0:
            # Distribute the remaining series to types with largest fractional parts
            remainders = sorted(
                raw_counts.items(),
                key=lambda kv: kv[1] - int_counts[kv[0]],
                reverse=True,
            )
            for k, _ in remainders:
                if remaining <= 0:
                    break
                int_counts[k] += 1
                remaining -= 1

        current_global_index = 0
        pbar = tqdm(total=total, desc="Generating telemetry", unit="series")
        original_n = self.n_timeseries

        # Helper to wrap a value array and metadata into TelemetryData
        def _make_telemetry(
            kind: str,
            values: np.ndarray,
            domain: str,
            subtype: str,
            properties: Dict[str, float],
            anomalies: List[Dict[str, Any]],
        ) -> TelemetryData:
            nonlocal current_global_index

            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

            telemetry = TelemetryData(
                id=f"{kind}_{current_global_index}",
                time=self.time.copy(),
                timeseries=values.astype(np.float32),
                metadata={
                    "domain": domain,
                    "subtype": subtype,
                    "properties": properties,
                    "anomalies": anomalies,
                },
                statistics=stats,
            )

            current_global_index += 1
            pbar.update(1)
            return telemetry

        try:
            # Sine series
            n_sine = int_counts.get("sine", 0)
            if n_sine > 0:
                self.n_timeseries = n_sine
                sine_values, sine_props = self._generate_sinusoidal_timeseries(return_properties=True)
                for values, props in zip(sine_values, sine_props):
                    yield _make_telemetry(
                        kind="sine",
                        values=values,
                        domain="synthetic",
                        subtype="sine",
                        properties={
                            "amplitude": props.amplitude,
                            "frequency": props.frequency,
                            "phase": props.phase,
                            "noise_std": props.noise_std,
                        },
                        anomalies=[],
                    )

            # Step series
            n_step = int_counts.get("step", 0)
            if n_step > 0:
                self.n_timeseries = n_step
                step_values, step_props = self._generate_step_timeseries(return_properties=True)
                for values, props in zip(step_values, step_props):
                    yield _make_telemetry(
                        kind="step",
                        values=values,
                        domain="synthetic",
                        subtype="step",
                        properties={
                            "step_size": props.step_size,
                            "step_time": props.step_time,
                            "noise_std": props.noise_std,
                        },
                        anomalies=[],
                    )

            # Random walk series
            n_rw = int_counts.get("random_walk", 0)
            if n_rw > 0:
                self.n_timeseries = n_rw
                rw_values, rw_props = self._generate_random_walk_timeseries(return_properties=True)
                for values, props in zip(rw_values, rw_props):
                    yield _make_telemetry(
                        kind="random_walk",
                        values=values,
                        domain="synthetic",
                        subtype="random_walk",
                        properties={
                            "noise_std": props.noise_std,
                            "drift": props.drift,
                        },
                        anomalies=[],
                    )

            # Industrial / vibration / stock series
            for kind in ("industrial", "vibration", "stock"):
                n_kind = int_counts.get(kind, 0)
                for _ in range(n_kind):
                    if kind == "industrial":
                        values, meta = self._generate_industrial_sensor_like()
                    elif kind == "vibration":
                        values, meta = self._generate_vibration_like()
                    else:  # stock
                        values, meta = self._generate_stock_like()

                    yield _make_telemetry(
                        kind=kind,
                        values=values,
                        domain=meta.domain,
                        subtype=meta.subtype,
                        properties=meta.properties,
                        anomalies=meta.anomalies,
                    )
        finally:
            # Restore original n_timeseries and close progress bar
            self.n_timeseries = original_n
            pbar.close()

    def generate_telemetry(
        self,
        total: int,
        type_proportions: Dict[str, float],
        shuffle: bool = True,
    ) -> List[TelemetryData]:
        """
        Generate `total` telemetry series with a given composition of types.

        `type_proportions` maps a type name to a relative proportion. Supported
        types:
          - \"sine\"
          - \"step\"
          - \"random_walk\"
          - \"industrial\"
          - \"vibration\"
          - \"stock\"

        The values in `type_proportions` are non‑negative and will be
        normalized so their sum is 1.0. For example, to generate 1M series
        with 3% steps, 20% sinusoids and the rest random walks:

            gen.generate_telemetry(
                total=1_000_000,
                type_proportions={\"step\": 0.03, \"sine\": 0.20, \"random_walk\": 0.77},
            )
        """
        series = list(self.iter_telemetry(total=total, type_proportions=type_proportions))

        if shuffle and len(series) > 1:
            perm = self.random_state.permutation(len(series))
            series = [series[i] for i in perm]

        return series

    # Industrial sensor–like
    def _generate_industrial_sensor_like(
        self,
    ) -> Tuple[np.ndarray, SeriesMetadata]:
        """
        Think: temperature/flow/pressure sensor.
        Components:
          - baseline + slow drift
          - a few setpoint changes (steps)
          - white noise
          - occasional spikes / short faults
        """
        rs = self.random_state

        # Baseline and drift
        baseline = rs.uniform(0.0, 10.0)
        drift_per_sec = rs.uniform(-0.01, 0.01)  # slow drift
        drift = drift_per_sec * self.time

        # Low-frequency process variation (like slow oscillation)
        proc_amp = rs.uniform(0.1, 2.0)
        proc_freq = rs.uniform(0.01, 0.2)
        proc_phase = rs.uniform(0.0, 2 * np.pi)
        process = proc_amp * np.sin(2 * np.pi * proc_freq * self.time + proc_phase)

        # Setpoint changes: random number of steps
        n_steps = rs.randint(0, 4)
        setpoint = np.zeros_like(self.time)
        step_mags = []
        step_times = []

        current_level = 0.0
        for _ in range(n_steps):
            step_time = rs.uniform(0.1 * self.time_duration, 0.9 * self.time_duration)
            idx = int(step_time * self.frequency)
            mag = rs.uniform(-3.0, 3.0)
            current_level += mag
            setpoint[idx:] += mag
            step_mags.append(mag)
            step_times.append(step_time)

        # Measurement noise
        noise_std = rs.uniform(0.01, 0.2)
        noise = rs.normal(0.0, noise_std, size=self.time_series_length)

        # Occasional spikes / glitches
        n_spikes = rs.randint(0, 5)
        spikes = np.zeros_like(self.time)
        spike_info = []
        for _ in range(n_spikes):
            idx = rs.randint(0, self.time_series_length)
            height = rs.uniform(2.0, 8.0) * rs.choice([-1, 1])
            spikes[idx] += height
            spike_info.append({"t": float(self.time[idx]), "height": float(height)})

        values = baseline + drift + process + setpoint + noise + spikes

        meta = SeriesMetadata(
            domain="industrial",
            subtype="sensor_generic",
            properties={
                "baseline": baseline,
                "drift_per_sec": drift_per_sec,
                "proc_amp": proc_amp,
                "proc_freq": proc_freq,
                "noise_std": noise_std,
                "n_steps": float(n_steps),
                "n_spikes": float(n_spikes),
            },
            anomalies=[
                *(dict(kind="step", t=t, mag=m) for t, m in zip(step_times, step_mags)),
                *(dict(kind="spike", **s) for s in spike_info),
            ],
        )

        return values, meta

    # Vibration-like (e.g., rotating machinery)
    def _generate_vibration_like(
        self,
    ) -> Tuple[np.ndarray, SeriesMetadata]:
        """
        Think: accelerometer on a bearing housing.
        Components:
          - Sum of a few sinusoids (fundamental + harmonics)
          - broadband noise
          - possible amplitude modulation
          - impulsive events (fault impacts)
        """
        rs = self.random_state

        # Base rotation frequency (in Hz relative to sampling)
        base_freq = rs.uniform(1.0, self.frequency / 8.0)  # keep under Nyquist
        n_harmonics = rs.randint(1, 4)

        signal = np.zeros_like(self.time)
        freqs = []
        amps = []
        phases = []

        for h in range(1, n_harmonics + 1):
            f = base_freq * h
            a = rs.uniform(0.1, 1.0) / h  # decreasing amplitude with harmonic
            p = rs.uniform(0.0, 2 * np.pi)
            signal += a * np.sin(2 * np.pi * f * self.time + p)
            freqs.append(f)
            amps.append(a)
            phases.append(p)

        # Amplitude modulation to mimic load changes
        mod_freq = rs.uniform(0.01, 0.2)
        mod_depth = rs.uniform(0.0, 0.7)
        modulation = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * self.time)
        signal *= modulation

        # Broadband noise
        noise_std = rs.uniform(0.01, 0.1)
        noise = rs.normal(0.0, noise_std, size=self.time_series_length)

        # Impulsive events (fault impacts)
        n_impacts = rs.randint(0, 10)
        impacts = np.zeros_like(self.time)
        impact_info = []

        for _ in range(n_impacts):
            center_t = rs.uniform(0.1 * self.time_duration, 0.9 * self.time_duration)
            center_idx = int(center_t * self.frequency)
            width = rs.randint(1, int(0.01 * self.time_series_length) + 2)
            height = rs.uniform(0.5, 3.0)

            start = max(0, center_idx - width // 2)
            end = min(self.time_series_length, center_idx + width // 2)
            impacts[start:end] += height * np.hanning(end - start)
            impact_info.append({"t": float(center_t), "height": float(height)})

        values = signal + noise + impacts

        meta = SeriesMetadata(
            domain="vibration",
            subtype="rotating_machine",
            properties={
                "base_freq": base_freq,
                "n_harmonics": float(n_harmonics),
                "mod_freq": mod_freq,
                "mod_depth": mod_depth,
                "noise_std": noise_std,
                "n_impacts": float(n_impacts),
            },
            anomalies=[dict(kind="impact", **imp) for imp in impact_info],
        )

        return values, meta

    # Stock-like (price series)
    def _generate_stock_like(
        self,
    ) -> Tuple[np.ndarray, SeriesMetadata]:
        """
        Simple geometric Brownian motion with:
          - piecewise volatility regimes
          - possible jumps (price gaps)

        Implementation is vectorized over time (no Python loop over timesteps)
        so that long series remain reasonably fast to generate.
        """
        rs = self.random_state

        dt = 1.0 / max(self.frequency, 1.0)

        # Base drift & volatility
        mu = rs.uniform(-0.05, 0.15)    # per unit time
        sigma_low = rs.uniform(0.05, 0.2)
        sigma_high = sigma_low * rs.uniform(1.5, 3.0)

        # Volatility regimes
        values = np.zeros(self.time_series_length)
        S0 = rs.uniform(10.0, 200.0)
        values[0] = S0

        vol = np.full(self.time_series_length, sigma_low)
        n_regimes = rs.randint(1, 4)
        regime_info = []

        for _ in range(n_regimes):
            start_t = rs.uniform(0.0, self.time_duration * 0.8)
            end_t = start_t + rs.uniform(0.1 * self.time_duration, 0.5 * self.time_duration)
            end_t = min(end_t, self.time_duration)

            start_idx = int(start_t * self.frequency)
            end_idx = int(end_t * self.frequency)
            if start_idx >= end_idx:
                continue

            high = bool(rs.randint(0, 2))
            vol[start_idx:end_idx] = sigma_high if high else sigma_low
            regime_info.append(
                {
                    "t_start": float(self.time[start_idx]),
                    "t_end": float(self.time[end_idx - 1]),
                    "sigma": float(sigma_high if high else sigma_low),
                }
            )

        # GBM dynamics (vectorized over time)
        # vol[1:] is sigma_t for t>=1
        sigma_t = vol[1:]
        eps = rs.normal(0.0, 1.0, size=self.time_series_length - 1)
        drift_term = (mu - 0.5 * sigma_t**2) * dt
        diffusion_term = sigma_t * np.sqrt(dt) * eps
        log_returns = drift_term + diffusion_term
        returns = np.exp(log_returns)
        values[1:] = S0 * np.cumprod(returns)

        # Jumps (gaps)
        n_jumps = rs.randint(0, 5)
        jumps_info = []

        for _ in range(n_jumps):
            idx = rs.randint(1, self.time_series_length)  # not at t=0
            jump_pct = rs.uniform(-0.2, 0.2)
            values[idx:] *= 1.0 + jump_pct
            jumps_info.append(
                {
                    "t": float(self.time[idx]),
                    "jump_pct": float(jump_pct),
                }
            )

        meta = SeriesMetadata(
            domain="stock",
            subtype="GBM",
            properties={
                "mu": mu,
                "sigma_low": sigma_low,
                "sigma_high": sigma_high,
                "n_regimes": float(len(regime_info)),
                "n_jumps": float(n_jumps),
            },
            anomalies=[
                *(dict(kind="regime", **r) for r in regime_info),
                *(dict(kind="jump", **j) for j in jumps_info),
            ],
        )

        return values, meta


if __name__ == "__main__":
    # Simple example: generate mixed realistic telemetry and build an HF dataset.
    generator = TimeseriesGenerator(
        n_timeseries=5,
        time_duration=10.0,
        frequency=100.0,
    )

    mixed_telemetry = generator.generate_telemetry(
        total=5,
        type_proportions={"industrial": 0.4, "vibration": 0.4, "stock": 0.2},
    )

    from telemetry_dataset import build_telemetry_dataset

    ds = build_telemetry_dataset(mixed_telemetry)
    print(ds)
    print("First row:")
    print(ds[0])

    # Optional: quick plot of the first generated series.
    first = mixed_telemetry[0]
    plt.plot(first.time, first.timeseries)
    plt.title(f"Example telemetry: {first.metadata.get('domain')} / {first.metadata.get('subtype')}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()
