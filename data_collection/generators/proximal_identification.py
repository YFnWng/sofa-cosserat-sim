"""Episode-based excitation for identifying reduced proximal mechanics.

Unlike the general simultaneous sinusoidal generator, this trajectory first
excites insertion, rotation, and tendon channels separately and then introduces
their pairwise interactions.  Except for the explicitly named tendon-loading
episodes, ``zero_tendon`` means zero commanded tendon force, not zero tendon
velocity at a nonzero force.
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


class ProximalIdentificationGenerator(InputGenerator):
    """Structured single-channel, ring-down, and interaction excitation.

    The default suite lasts roughly two minutes.  Each episode returns to zero
    or uses smooth entry/exit envelopes so its initial condition is explicit.
    Rotation commands are allowed to cross the wrapped +/-180 degree boundary;
    the base force field evaluates the command-to-base error on the circle.

    Dense matrices are normally unnecessary at every 10 ms step for this long
    trajectory.  ``COLLECT_A_INTERVAL=5`` or ``10`` is a useful first recording,
    while a shorter selected episode can subsequently be recollected densely.
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
        del seed  # Reserved for reproducible future multi-amplitude variants.
        speeds = (np.asarray(joint_max_speeds, dtype=float)
                  if joint_max_speeds is not None
                  else np.array([0.03, 40.0, 10.0]))

        positive_range = np.maximum(self.joint_upper, 0.0)
        ins_range = max(float(positive_range[0]), 1e-12)
        rot_range = max(float(self.joint_upper[1] - self.joint_lower[1]), 1e-12)
        tendon_range = max(float(positive_range[2]), 1e-12)

        self._ins_amp = min(0.75 * ins_range, 0.8 * speeds[0] * 10.0 / np.pi)
        self._rot_amp = min(0.35 * rot_range, 0.8 * speeds[1] * 16.0
                            / (2.0 * np.pi))
        self._turn_amp = min(0.75 * max(abs(float(self.joint_lower[1])),
                                        abs(float(self.joint_upper[1]))),
                             0.8 * speeds[1] * 28.0 / np.pi)
        # The tendon ring-down ramp occupies approximately three seconds.
        self._tendon_amp = min(0.6 * tendon_range,
                               0.8 * speeds[2] * 3.0 * 2.0 / np.pi)
        self._mid_insertion = 0.55 * ins_range

        z = self._zero
        self._episodes = [
            _Episode("settle_zero", 2.0, z),
            _Episode("insertion_slow_zero_tendon", 10.0,
                     self._insertion_slow),
            _Episode("insertion_chirp_zero_tendon", 10.0,
                     self._insertion_chirp),
            _Episode("rotation_slow_zero_tendon", 16.0,
                     self._rotation_slow),
            _Episode("rotation_pi_crossing_zero_tendon", 28.0,
                     self._rotation_pi_crossing),
            _Episode("rotation_chirp_zero_tendon", 14.0,
                     self._rotation_chirp),
            _Episode("tendon_load_hold_unload_ringdown_home", 12.0,
                     self._tendon_ringdown_home),
            _Episode("tendon_ringdown_mid_insertion", 14.0,
                     self._tendon_ringdown_mid),
            _Episode("tendon_insertion_interaction", 12.0,
                     self._tendon_insertion),
            _Episode("tendon_rotation_interaction", 14.0,
                     self._tendon_rotation),
            _Episode("insertion_rotation_interaction_zero_tendon", 14.0,
                     self._insertion_rotation),
            _Episode("final_settle_zero", 2.0, z),
        ]
        self._starts = np.cumsum(
            [0.0] + [episode.duration for episode in self._episodes[:-1]])
        self._total_time = float(sum(e.duration for e in self._episodes))

    @staticmethod
    def _smooth01(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return 0.5 - 0.5 * np.cos(np.pi * x)

    @staticmethod
    def _window(s: float) -> float:
        return float(np.sin(np.pi * np.clip(s, 0.0, 1.0)) ** 2)

    @staticmethod
    def _zero(_s: float, _duration: float) -> np.ndarray:
        return np.zeros(3)

    def _insertion_slow(self, s: float, _duration: float) -> np.ndarray:
        return np.array([self._ins_amp * 0.5 * (1.0 - np.cos(2*np.pi*s)),
                         0.0, 0.0])

    def _insertion_chirp(self, s: float, _duration: float) -> np.ndarray:
        phase = 2*np.pi * (0.5*s + 2.5*s*s)
        value = 0.5 * self._ins_amp * self._window(s) * (1.0 + np.sin(phase))
        return np.array([value, 0.0, 0.0])

    def _rotation_slow(self, s: float, _duration: float) -> np.ndarray:
        return np.array([0.0, self._rot_amp * np.sin(2*np.pi*s), 0.0])

    def _rotation_pi_crossing(self, s: float, _duration: float) -> np.ndarray:
        # Smoothly traverse beyond +pi and return. ``step`` wraps the stored
        # motor angle, while the physical target orientation remains continuous
        # on SO(3); downstream preprocessing must unwrap this channel.
        angle = self._turn_amp * 0.5 * (1.0 - np.cos(2*np.pi*s))
        return np.array([0.0, angle, 0.0])

    def _rotation_chirp(self, s: float, _duration: float) -> np.ndarray:
        phase = 2*np.pi * (0.5*s + 3.0*s*s)
        return np.array([0.0, self._rot_amp * self._window(s)*np.sin(phase), 0.0])

    def _tendon_pulse(self, s: float) -> float:
        # Load, deliberately hold a nonzero force (zero velocity), unload, then
        # leave a long interval at exactly zero force for passive ring-down.
        if s < 0.25:
            return self._tendon_amp * self._smooth01(s / 0.25)
        if s < 0.40:
            return self._tendon_amp
        if s < 0.50:
            return self._tendon_amp * (1.0 - self._smooth01((s - 0.40) / 0.10))
        return 0.0

    def _tendon_ringdown_home(self, s: float, _duration: float) -> np.ndarray:
        return np.array([0.0, 0.0, self._tendon_pulse(s)])

    def _plateau(self, s: float) -> float:
        if s < 0.15:
            return self._mid_insertion * self._smooth01(s / 0.15)
        if s < 0.90:
            return self._mid_insertion
        return self._mid_insertion * (1.0 - self._smooth01((s - 0.90) / 0.10))

    def _tendon_ringdown_mid(self, s: float, _duration: float) -> np.ndarray:
        # Delay the tendon pulse until the insertion plateau is established.
        tendon_s = (s - 0.18) / 0.72
        tendon = self._tendon_pulse(tendon_s) if 0.0 <= tendon_s <= 1.0 else 0.0
        return np.array([self._plateau(s), 0.0, tendon])

    def _tendon_insertion(self, s: float, _duration: float) -> np.ndarray:
        w = self._window(s)
        return np.array([
            0.6*self._ins_amp*w*(0.7 + 0.3*np.sin(4*np.pi*s)),
            0.0,
            0.65*self._tendon_amp*w*(0.65 + 0.35*np.sin(6*np.pi*s + 0.4)),
        ])

    def _tendon_rotation(self, s: float, _duration: float) -> np.ndarray:
        w = self._window(s)
        return np.array([
            0.0,
            0.65*self._rot_amp*w*np.sin(4*np.pi*s),
            0.65*self._tendon_amp*w*(0.65 + 0.35*np.sin(6*np.pi*s + 0.7)),
        ])

    def _insertion_rotation(self, s: float, _duration: float) -> np.ndarray:
        w = self._window(s)
        return np.array([
            0.6*self._ins_amp*w*(0.7 + 0.3*np.sin(4*np.pi*s)),
            0.65*self._rot_amp*w*np.sin(6*np.pi*s + 0.3),
            0.0,
        ])

    def _episode_at(self, t: float):
        clipped = float(np.clip(t, 0.0, self._total_time))
        index = int(np.searchsorted(self._starts, clipped, side="right") - 1)
        index = min(max(index, 0), len(self._episodes) - 1)
        local = clipped - float(self._starts[index])
        return index, local, self._episodes[index]

    def step(self, t: float) -> np.ndarray:
        _, local, episode = self._episode_at(t)
        s = local / episode.duration
        command = episode.command(s, episode.duration)
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
