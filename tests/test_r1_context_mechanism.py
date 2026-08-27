#!/usr/bin/env python3
"""Deterministic unit tests for frozen R1 context/mechanism definitions."""

from __future__ import annotations

import unittest

import numpy as np

from tools.r1_context_mechanism_core import calculate_hlc_option_b, calculate_tsb_option_a


def time_grid(seconds: float = 5.0) -> np.ndarray:
    return np.arange(0.0, seconds + 0.05, 0.1)


def speed_from_accel(accel: np.ndarray, initial: float = 10.0) -> np.ndarray:
    speed = np.empty(len(accel), dtype=float)
    speed[0] = initial
    for index in range(1, len(accel)):
        speed[index] = max(0.1, speed[index - 1] + accel[index - 1] * 0.1)
    return speed


class HlcOptionBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = time_grid()
        self.speed = np.full(len(self.time), 8.0)

    def test_monotonic_decisive_transition(self) -> None:
        p = np.clip((self.time - 0.5) / 2.0, 0.0, 1.0)
        result = calculate_hlc_option_b(self.time, p, self.speed)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["hesitation_retreat_count"], 0)
        self.assertGreater(result["monotonic_transition_fraction"], 0.99)

    def test_single_retreat(self) -> None:
        p = np.zeros_like(self.time)
        p[(self.time >= 0.5) & (self.time < 1.2)] = np.linspace(0.0, 0.4, np.sum((self.time >= 0.5) & (self.time < 1.2)))
        p[(self.time >= 1.2) & (self.time < 1.7)] = np.linspace(0.4, 0.2, np.sum((self.time >= 1.2) & (self.time < 1.7)))
        tail = self.time >= 1.7
        p[tail] = np.linspace(0.2, 1.0, np.sum(tail))
        result = calculate_hlc_option_b(self.time, p, self.speed)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["hesitation_retreat_count"], 1)

    def test_two_retreats(self) -> None:
        p = np.interp(self.time, [0, .5, 1.1, 1.6, 2.1, 2.6, 4.2, 5.0], [0, 0, .4, .2, .6, .4, 1, 1])
        result = calculate_hlc_option_b(self.time, p, self.speed)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["hesitation_retreat_count"], 2)

    def test_jitter_only(self) -> None:
        p = np.clip((self.time - .5) / 2.0 + .008 * np.sin(2 * np.pi * self.time / .2), 0.0, 1.0)
        result = calculate_hlc_option_b(self.time, p, self.speed)
        self.assertEqual(result["hesitation_retreat_count"], 0)

    def test_unfinished_transition(self) -> None:
        p = np.clip(self.time / 8.0, 0.0, .7)
        result = calculate_hlc_option_b(self.time, p, self.speed)
        self.assertEqual(result["status"], "UNFINISHED_TRANSITION")

    def test_low_speed(self) -> None:
        p = np.clip((self.time - .5) / 2.0, 0.0, 1.0)
        speed = self.speed.copy()
        speed[10:16] = .5
        self.assertEqual(calculate_hlc_option_b(self.time, p, speed)["status"], "LOW_SPEED_TRANSITION")

    def test_map_invalid(self) -> None:
        p = np.clip((self.time - .5) / 2.0, 0.0, 1.0)
        self.assertEqual(calculate_hlc_option_b(self.time, p, self.speed, map_valid=False)["status"], "MAP_INVALID")


class TsbOptionATests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = time_grid()

    def test_one_brake_phase(self) -> None:
        accel = np.zeros(len(self.time)); accel[10:20] = -1.0
        result = calculate_tsb_option_a(self.time, speed_from_accel(accel))
        self.assertEqual(result["brake_phase_count"], 1)

    def test_two_phases_with_release(self) -> None:
        accel = np.zeros(len(self.time)); accel[10:16] = -1.0; accel[16:21] = .4; accel[21:27] = -1.0
        result = calculate_tsb_option_a(self.time, speed_from_accel(accel))
        self.assertEqual(result["brake_phase_count"], 2)
        self.assertGreaterEqual(result["interstage_release_fraction"], .15)

    def test_chattering_merges(self) -> None:
        accel = np.zeros(len(self.time)); accel[10:14] = -1.0; accel[14:16] = -.4; accel[16:20] = -1.0
        result = calculate_tsb_option_a(self.time, speed_from_accel(accel))
        self.assertEqual(result["brake_phase_count"], 1)

    def test_three_true_phases(self) -> None:
        # Each physical segment is deliberately longer than the median3 edge
        # loss, so all three are genuine 0.3 s+ Option-A phases.
        accel = np.zeros(len(self.time)); accel[8:16] = -1.0; accel[16:24] = .4; accel[24:32] = -1.0; accel[32:40] = .4; accel[40:48] = -1.0
        result = calculate_tsb_option_a(self.time, speed_from_accel(accel))
        self.assertEqual(result["brake_phase_count"], 3)

    def test_weak_release(self) -> None:
        accel = np.zeros(len(self.time)); accel[10:16] = -1.0; accel[16:21] = .1; accel[21:27] = -1.0
        result = calculate_tsb_option_a(self.time, speed_from_accel(accel))
        self.assertEqual(result["brake_phase_count"], 2)
        self.assertLess(result["interstage_release_fraction"], .15)

    def test_low_speed_endstop(self) -> None:
        accel = np.zeros(len(self.time)); speed = np.full(len(self.time), .5)
        self.assertEqual(calculate_tsb_option_a(self.time, speed)["status"], "LOW_SPEED_ENDSTOP")

    def test_truncated_second_phase(self) -> None:
        accel = np.zeros(len(self.time)); accel[10:16] = -1.0; accel[16:21] = .4; accel[-2:] = -1.0
        result = calculate_tsb_option_a(self.time, speed_from_accel(accel))
        self.assertEqual(result["brake_phase_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
