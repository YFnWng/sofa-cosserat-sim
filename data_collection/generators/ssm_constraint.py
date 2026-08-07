"""Rich bent-configuration excitation for the SSM insertion constraint.

This generator is deliberately different from the component-identification
trajectory.  It holds a nonzero tendon force long enough for the catheter to
settle into a bent configuration *before* probing insertion and rotation.  The
resulting matrix recording tests whether the near-null relation between
interface axial velocity and material insertion is a fixed straight-rod
coincidence or a configuration-dependent tangent constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import InputGenerator


@dataclass(frozen=True)
class _Episode:
    name: str
    duration: float
    command: Callable[[float], np.ndarray]


class SSMConstraintGenerator(InputGenerator):
    """Excite insertion/rotation on several settled, tendon-bent branches.

    Each probe starts and ends at zero command.  Within a probe the tendon is
    ramped first, held while base motion is applied, and unloaded last.  This
    ordering separates the settled SSM tangent from the entry/exit transient.
    Episode labels are recorded in HDF5 so diagnostics can compare them.
    """

    def __init__(
        self,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        dt: float,
        joint_max_speeds: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(joint_lower_limits, joint_upper_limits, dt)
        del seed
        speeds = (np.asarray(joint_max_speeds, dtype=float)
                  if joint_max_speeds is not None
                  else np.array([0.03, 40.0, 10.0]))
        positive = np.maximum(self.joint_upper, 0.0)
        insertion_range = max(float(positive[0]), 1e-12)
        rotation_range = max(
            float(self.joint_upper[1] - self.joint_lower[1]), 1e-12)
        tendon_range = max(float(positive[2]), 1e-12)

        # Conservative amplitudes keep the smooth probes below nominal joint
        # speed limits while spanning two materially different bend levels.
        self._insertion_amp = min(0.70 * insertion_range, 0.75 * speeds[0])
        self._rotation_amp = min(0.30 * rotation_range, 0.45 * speeds[1])
        self._tendon_high = min(0.75 * tendon_range, 1.5 * speeds[2])
        self._tendon_low = 0.5 * self._tendon_high

        self._episodes = [
            _Episode("settle_zero", 3.0, lambda _s: np.zeros(3)),
            _Episode("low_bend_hold", 10.0,
                     lambda s: self._bent_probe(s, self._tendon_low, "hold")),
            _Episode("low_bend_insertion", 14.0,
                     lambda s: self._bent_probe(
                         s, self._tendon_low, "insertion")),
            _Episode("high_bend_hold", 10.0,
                     lambda s: self._bent_probe(s, self._tendon_high, "hold")),
            _Episode("high_bend_insertion_slow", 16.0,
                     lambda s: self._bent_probe(
                         s, self._tendon_high, "insertion")),
            _Episode("high_bend_insertion_chirp", 16.0,
                     lambda s: self._bent_probe(
                         s, self._tendon_high, "insertion_chirp")),
            _Episode("high_bend_rotation", 16.0,
                     lambda s: self._bent_probe(
                         s, self._tendon_high, "rotation")),
            _Episode("high_bend_insertion_rotation", 18.0,
                     lambda s: self._bent_probe(
                         s, self._tendon_high, "combined")),
            _Episode("varying_bend_insertion", 18.0,
                     self._varying_bend_insertion),
            _Episode("fast_transient_bend_insertion", 12.0,
                     self._transient_probe),
            _Episode("final_ringdown", 5.0, lambda _s: np.zeros(3)),
        ]
        self._starts = np.cumsum(
            [0.0] + [episode.duration for episode in self._episodes[:-1]])
        self._total_time = float(sum(e.duration for e in self._episodes))

    @staticmethod
    def _smooth01(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return 0.5 - 0.5*np.cos(np.pi*x)

    @classmethod
    def _plateau(cls, s: float, edge: float = 0.25) -> float:
        if s < edge:
            return cls._smooth01(s / edge)
        if s > 1.0 - edge:
            return cls._smooth01((1.0 - s) / edge)
        return 1.0

    @classmethod
    def _probe_window(cls, s: float) -> tuple[float, float]:
        """Return normalized probe time and an interior smooth window.

        Motion occurs only after tendon loading and stops before unloading.
        """
        start, stop = 0.27, 0.73
        tau = (s - start) / (stop - start)
        if not 0.0 <= tau <= 1.0:
            return float(np.clip(tau, 0.0, 1.0)), 0.0
        return tau, float(np.sin(np.pi*tau)**2)

    def _bent_probe(self, s: float, tendon: float, kind: str) -> np.ndarray:
        tendon_command = tendon * self._plateau(s)
        tau, window = self._probe_window(s)
        insertion = 0.0
        rotation = 0.0
        if kind == "insertion":
            insertion = self._insertion_amp * 0.5 * (
                1.0 - np.cos(2*np.pi*tau)) if window > 0.0 else 0.0
        elif kind == "insertion_chirp":
            phase = 2*np.pi * (0.5*tau + 2.5*tau*tau)
            insertion = (0.55*self._insertion_amp*window
                         * (1.0 + 0.8*np.sin(phase)))
        elif kind == "rotation":
            rotation = self._rotation_amp * window * np.sin(4*np.pi*tau)
        elif kind == "combined":
            insertion = self._insertion_amp * 0.5 * (
                1.0 - np.cos(2*np.pi*tau)) if window > 0.0 else 0.0
            rotation = (0.75*self._rotation_amp*window
                        * np.sin(6*np.pi*tau + 0.4))
        elif kind != "hold":
            raise ValueError(f"unknown bent probe {kind!r}")
        return np.array([insertion, rotation, tendon_command])

    def _varying_bend_insertion(self, s: float) -> np.ndarray:
        envelope = self._plateau(s, edge=0.15)
        tendon = self._tendon_high * envelope * (0.65 + 0.25*np.sin(4*np.pi*s))
        insertion = (0.65*self._insertion_amp*envelope
                     * (0.65 + 0.35*np.sin(6*np.pi*s + 0.3)))
        return np.array([insertion, 0.0, tendon])

    def _transient_probe(self, s: float) -> np.ndarray:
        # Deliberately retain a dynamic episode.  The diagnostic reports it
        # separately rather than silently conflating it with settled SSM data.
        window = float(np.sin(np.pi*np.clip(s, 0.0, 1.0))**2)
        tendon = self._tendon_high * window * (0.60 + 0.30*np.sin(4*np.pi*s))
        insertion = (0.55*self._insertion_amp*window
                     * (0.65 + 0.35*np.sin(10*np.pi*s + 0.2)))
        return np.array([insertion, 0.0, tendon])

    def _episode_at(self, t: float):
        clipped = float(np.clip(t, 0.0, self._total_time))
        index = int(np.searchsorted(self._starts, clipped, side="right") - 1)
        index = min(max(index, 0), len(self._episodes) - 1)
        local = clipped - float(self._starts[index])
        return index, local, self._episodes[index]

    def step(self, t: float) -> np.ndarray:
        _, local, episode = self._episode_at(t)
        command = episode.command(local / episode.duration)
        command = np.clip(command, self.joint_lower, self.joint_upper)
        return self.wrap_rotation(command)

    def is_done(self, t: float) -> bool:
        return t >= self._total_time

    def episode_index(self, t: float) -> int:
        return self._episode_at(t)[0]

    def episode_time(self, t: float) -> float:
        return self._episode_at(t)[1]

    @property
    def episode_names(self):
        return [episode.name for episode in self._episodes]

    @property
    def episode_durations(self):
        return [episode.duration for episode in self._episodes]
