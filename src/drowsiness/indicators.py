"""Time-based drowsiness indicators.

Each indicator turns a per-frame observation into a score in [0, 1]. They
are driven by timestamps, not frame counts, so behaviour does not depend on
the camera's frame rate.

Each indicator also has ``face_lost(t)``, called by the pipeline when the
face has been missing for longer than a short grace period. It clears the
continuous timers (closure, yawn, tilt), so time that was not observed is
never counted, and lets the windowed scores age out naturally instead of
freezing or jumping when the face comes back.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Frames further apart than this (e.g. after the face was lost) only count
# for this long, so gaps don't inflate PERCLOS.
_MAX_FRAME_GAP_S = 0.5


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class EyeState:
    score: float = 0.0
    closed: bool = False
    perclos: float = 0.0
    closed_duration_s: float = 0.0
    microsleep: bool = False


class EyeClosureIndicator:
    """Combines a PERCLOS-like closed-eye ratio (fatigue) with continuous-closure
    time (microsleep).

    "Closed" means EAR below the calibrated threshold (by default 60% of the
    open-eye EAR), which is looser than the P80 criterion of classic PERCLOS.
    """

    def __init__(
        self, window_s: float, perclos_max: float, microsleep_s: float, debounce_s: float = 0.0
    ):
        self._window_s = window_s
        self._perclos_max = perclos_max
        self._microsleep_s = microsleep_s
        self._debounce_s = debounce_s
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[float, float, bool]] = deque()  # (t, dt, closed)
        self._closed_time = 0.0
        self._closed_samples = 0
        self._last_t: float | None = None
        self._closed_since: float | None = None
        self._open_since: float | None = None
        self.state = EyeState()

    def update(self, t: float, closed: bool) -> EyeState:
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), _MAX_FRAME_GAP_S)
        self._last_t = t
        self._samples.append((t, dt, closed))
        if closed:
            self._closed_time += dt
            self._closed_samples += 1
        self._expire(t)

        if closed:
            self._open_since = None
            if self._closed_since is None:
                self._closed_since = t
        elif self._closed_since is not None:
            # Only end the closure once the eyes have stayed open long enough.
            if self._open_since is None:
                self._open_since = t
            if t - self._open_since >= self._debounce_s:
                self._closed_since = self._open_since = None
        closed_duration = 0.0 if self._closed_since is None else t - self._closed_since
        return self._set_state(closed, closed_duration)

    def face_lost(self, t: float) -> EyeState:
        """The eyes cannot be seen: end any closure and keep only the ratio."""
        self._closed_since = self._open_since = None
        # The next frame after the gap starts fresh instead of being credited
        # with the gap's duration.
        self._last_t = None
        self._expire(t)
        return self._set_state(False, 0.0)

    def _expire(self, t: float) -> None:
        while self._samples and t - self._samples[0][0] > self._window_s:
            _, old_dt, old_closed = self._samples.popleft()
            if old_closed:
                self._closed_time -= old_dt
                self._closed_samples -= 1
        if self._closed_samples == 0:
            self._closed_time = 0.0  # drop accumulated floating-point drift

    def _set_state(self, closed: bool, closed_duration: float) -> EyeState:
        # Dividing by the full window (not the elapsed time) keeps PERCLOS
        # conservative right after start-up.
        perclos = max(self._closed_time, 0.0) / self._window_s
        score = max(
            _clip01(perclos / self._perclos_max), _clip01(closed_duration / self._microsleep_s)
        )
        microsleep = closed_duration >= self._microsleep_s
        self.state = EyeState(score, closed, perclos, closed_duration, microsleep)
        return self.state


@dataclass(frozen=True)
class YawnState:
    score: float = 0.0
    yawning: bool = False
    yawn_count: int = 0


class YawnIndicator:
    """Counts sustained wide mouth openings within a sliding window."""

    def __init__(
        self,
        mar_threshold: float,
        min_duration_s: float,
        window_s: float,
        yawns_for_max: int,
        debounce_s: float = 0.0,
    ):
        self._mar_threshold = mar_threshold
        self._min_duration_s = min_duration_s
        self._window_s = window_s
        self._yawns_for_max = yawns_for_max
        self._debounce_s = debounce_s
        self.reset()

    def reset(self) -> None:
        self._yawn_times: deque[float] = deque()
        self._open_since: float | None = None
        self._below_since: float | None = None
        self._counted = False
        self.state = YawnState()

    def update(self, t: float, mar: float) -> YawnState:
        if mar > self._mar_threshold:
            self._below_since = None
            if self._open_since is None:
                self._open_since = t
            if not self._counted and t - self._open_since >= self._min_duration_s:
                self._yawn_times.append(t)
                self._counted = True
        elif self._open_since is not None:
            # A brief dip below the threshold does not abort the yawn.
            if self._below_since is None:
                self._below_since = t
            if t - self._below_since >= self._debounce_s:
                self._end_opening()
        return self._set_state(t)

    def face_lost(self, t: float) -> YawnState:
        """The mouth cannot be seen: abort a yawn in progress; counted yawns stay."""
        self._end_opening()
        return self._set_state(t)

    def _end_opening(self) -> None:
        self._open_since = self._below_since = None
        self._counted = False

    def _set_state(self, t: float) -> YawnState:
        while self._yawn_times and t - self._yawn_times[0] > self._window_s:
            self._yawn_times.popleft()
        count = len(self._yawn_times)
        self.state = YawnState(_clip01(count / self._yawns_for_max), self._counted, count)
        return self.state


@dataclass(frozen=True)
class HeadState:
    score: float = 0.0
    tilted: bool = False
    tilt_duration_s: float = 0.0


class HeadTiltIndicator:
    """Measures how long the head stays nodded forward or tilted sideways."""

    def __init__(
        self,
        pitch_threshold_deg: float,
        roll_threshold_deg: float,
        max_duration_s: float,
        debounce_s: float = 0.0,
    ):
        self._pitch_threshold = pitch_threshold_deg
        self._roll_threshold = roll_threshold_deg
        self._max_duration_s = max_duration_s
        self._debounce_s = debounce_s
        self.reset()

    def reset(self) -> None:
        self._tilted_since: float | None = None
        self._upright_since: float | None = None
        self.state = HeadState()

    def update(self, t: float, pitch_offset: float, roll_offset: float) -> HeadState:
        tilted = pitch_offset > self._pitch_threshold or abs(roll_offset) > self._roll_threshold
        if tilted:
            self._upright_since = None
            if self._tilted_since is None:
                self._tilted_since = t
        elif self._tilted_since is not None:
            # Only end the tilt once the head has stayed upright long enough,
            # so a one-frame pose flicker does not reset the duration.
            if self._upright_since is None:
                self._upright_since = t
            if t - self._upright_since >= self._debounce_s:
                self._tilted_since = self._upright_since = None
        duration = 0.0 if self._tilted_since is None else t - self._tilted_since
        self.state = HeadState(_clip01(duration / self._max_duration_s), tilted, duration)
        return self.state

    def face_lost(self, t: float) -> HeadState:
        """The head pose cannot be measured: end any tilt in progress."""
        self._tilted_since = self._upright_since = None
        self.state = HeadState()
        return self.state
