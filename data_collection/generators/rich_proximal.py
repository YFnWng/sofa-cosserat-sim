"""Persistent excitation for learning latent proximal rollout dynamics.

This complements ``proximal_identification``: that trajectory isolates SOFA
force fields, whereas this one deliberately visits broad coupled states,
velocities, and loading histories. Named episodes support episode holdout.
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
    command: Callable[[float, float], np.ndarray]


class RichProximalGenerator(InputGenerator):
    """Seeded excitation of insertion, rotation, and tendon-force commands.

    Units are metres, degrees, and newtons. Smooth windows make commands and
    first derivatives zero at episode boundaries. Rotation deliberately crosses
    +/-180 degrees and is wrapped only at the public output.
    """

    def __init__(self, joint_lower_limits: np.ndarray,
                 joint_upper_limits: np.ndarray, dt: float,
                 joint_max_speeds: np.ndarray | None = None,
                 seed: int = 0) -> None:
        super().__init__(joint_lower_limits, joint_upper_limits, dt)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        positive = np.maximum(self.joint_upper, 0.0)
        self._ins_max = float(positive[0])
        self._rot_max = float(min(abs(self.joint_lower[1]),
                                  abs(self.joint_upper[1])))
        self._tendon_max = float(positive[2])
        self._speeds = (np.asarray(joint_max_speeds, dtype=float)
                        if joint_max_speeds is not None
                        else np.array([0.03, 40.0, 10.0]))
        self._time_scale = max(
            1.0, 0.03/max(self._speeds[0], 1e-12),
            40.0/max(self._speeds[1], 1e-12),
            10.0/max(self._speeds[2], 1e-12))
        def duration(seconds: float) -> float:
            return seconds*self._time_scale

        # Incommensurate frequencies avoid repeatedly tracing one short orbit.
        self._cycles = np.array([1.0, 1.7, 2.6, 3.35])
        self._weights = np.array([1.0, 0.65, 0.40, 0.25])
        self._phases_a = rng.uniform(-np.pi, np.pi, size=(3, 4))
        self._phases_b = rng.uniform(-np.pi, np.pi, size=(3, 4))

        self._episodes = [
            _Episode("settle_zero", duration(3.0), self._zero),
            _Episode("insertion_broadband_zero_tendon", duration(28.0),
                     self._insertion_only),
            _Episode("rotation_wrap_reversal_zero_tendon", duration(48.0),
                     self._rotation_wrap_reversal),
            _Episode("tendon_broadband_fixed_base", duration(28.0),
                     self._tendon_only),
            _Episode("coupled_persistent_excitation_a", duration(40.0),
                     self._coupled_a),
            _Episode("coupled_persistent_excitation_b", duration(40.0),
                     self._coupled_b),
            _Episode("repeated_state_opposite_history", duration(80.0),
                     self._history_loop),
            _Episode("loaded_unload_ringdown", duration(20.0),
                     self._loaded_ringdown),
            _Episode("final_settle_zero", duration(4.0), self._zero),
        ]
        self._starts = np.cumsum(
            [0.0] + [episode.duration for episode in self._episodes[:-1]])
        self._total_time = float(sum(e.duration for e in self._episodes))

    @staticmethod
    def _smooth01(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return 0.5 - 0.5 * np.cos(np.pi*x)

    @staticmethod
    def _window(s: float) -> float:
        return float(np.sin(np.pi*np.clip(s, 0.0, 1.0))**2)

    @staticmethod
    def _zero(_s: float, _duration: float) -> np.ndarray:
        return np.zeros(3)

    def _series(self, s: float, phases: np.ndarray) -> float:
        terms = self._weights*np.sin(2*np.pi*self._cycles*s + phases)
        return float(np.sum(terms)/np.sum(self._weights))

    def _insertion_only(self, s: float, _duration: float) -> np.ndarray:
        w = self._window(s)
        insertion = self._ins_max*w*(0.48 + 0.34*self._series(
            s, self._phases_a[0]))
        return np.array([insertion, 0.0, 0.0])

    def _rotation_wrap_reversal(self, s: float, _duration: float) -> np.ndarray:
        # Smoothly crosses pi and reverses without a discontinuous target pose.
        excursion = min(0.78*self._rot_max, 250.0)
        rotation = excursion*self._window(s)*np.sin(2*np.pi*s - 0.35)
        return np.array([0.0, rotation, 0.0])

    def _tendon_only(self, s: float, _duration: float) -> np.ndarray:
        w = self._window(s)
        tendon = self._tendon_max*w*(0.42 + 0.28*self._series(
            s, self._phases_a[2]))
        return np.array([0.0, 0.0, tendon])

    def _coupled(self, s: float, phases: np.ndarray,
                 shift: float) -> np.ndarray:
        w = self._window(s)
        insertion = self._ins_max*w*(0.45 + 0.30*self._series(s, phases[0]))
        rotation = min(0.42*self._rot_max, 135.0)*w*self._series(s, phases[1])
        tendon = self._tendon_max*w*(0.40 + 0.30*self._series(
            s + shift, phases[2]))
        return np.array([insertion, rotation, tendon])

    def _coupled_a(self, s: float, _duration: float) -> np.ndarray:
        return self._coupled(s, self._phases_a, 0.07)

    def _coupled_b(self, s: float, _duration: float) -> np.ndarray:
        return self._coupled(s, self._phases_b, 0.19)

    def _history_loop(self, s: float, _duration: float) -> np.ndarray:
        # Revisit one target through different paths to expose latent memory.
        target = np.array([0.55*self._ins_max,
                           min(0.28*self._rot_max, 90.0),
                           0.30*self._tendon_max])
        waypoints = np.array([
            [0.0, 0.0, 0.0], target,
            [0.25*self._ins_max, -min(0.32*self._rot_max, 110.0),
             0.70*self._tendon_max], target,
            [0.78*self._ins_max, min(0.50*self._rot_max, 170.0),
             0.10*self._tendon_max], target,
            [0.18*self._ins_max, -min(0.48*self._rot_max, 160.0),
             0.55*self._tendon_max], target,
            [0.0, 0.0, 0.0],
        ])
        position = np.clip(s, 0.0, 1.0)*(len(waypoints) - 1)
        index = min(int(position), len(waypoints) - 2)
        alpha = self._smooth01(position - index)
        return (1-alpha)*waypoints[index] + alpha*waypoints[index + 1]

    def _loaded_ringdown(self, s: float, _duration: float) -> np.ndarray:
        loaded = np.array([0.45*self._ins_max,
                           min(0.30*self._rot_max, 100.0),
                           0.60*self._tendon_max])
        if s < 0.25:
            return loaded*self._smooth01(s/0.25)
        if s < 0.50:
            result = loaded.copy()
            result[2] *= 1-self._smooth01((s-0.25)/0.25)
            return result
        if s < 0.75:
            result = loaded.copy()
            result[2] = 0.0
            return result
        result = loaded.copy()
        result[2] = 0.0
        return result*(1-self._smooth01((s-0.75)/0.25))

    def _episode_at(self, t: float):
        clipped = float(np.clip(t, 0.0, self._total_time))
        index = int(np.searchsorted(self._starts, clipped, side="right") - 1)
        index = min(max(index, 0), len(self._episodes)-1)
        local = clipped-float(self._starts[index])
        return index, local, self._episodes[index]

    def step(self, t: float) -> np.ndarray:
        _, local, episode = self._episode_at(t)
        command = episode.command(local/episode.duration, episode.duration)
        return self.wrap_rotation(np.clip(
            command, self.joint_lower, self.joint_upper))

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
