"""Structured MIDI input model and tick<->seconds timing for pysf2.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

pysf2 renders *structured* events, not MIDI files: any SMF parser (mido,
pretty_midi, ...) can be adapted to the small dataclasses below. Keeping
file I/O out of the library keeps the runtime dependency list at exactly
one package (numpy).

Tick discipline
---------------
Everything is expressed in raw SMF ticks plus a tempo map; :class:`TimeMap`
does the piecewise-linear tick<->seconds conversion that honors tempo
changes exactly (a single linear ``tick / ppq * sec_per_beat`` scaling is
wrong for any file with tempo changes).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

#: Special ``ChannelEvent.cc`` values for non-CC channel events.
EV_PROGRAM = 128      #: program change; value = program number 0..127
EV_PITCHBEND = 129    #: pitch bend; value = raw 14-bit 0..16383 (8192 = center)

#: CC numbers the engine reads: bank select MSB, mod wheel, channel volume,
#: pan, mod X/Y, sostenuto, and the pitch-bend range coarse/fine pair.
TRACKED_CCS = (0, 1, 7, 10, 11, 32, 64, 100, 101)

#: Default SMPTE tempo when no tempo is given (500000 us == 120 BPM).
DEFAULT_US_PER_QUARTER = 500_000


@dataclass(frozen=True)
class ChannelEvent:
    """One program-change, pitch-bend, or tracked CC event on a channel.

    Events are stored per channel in chronological (non-decreasing tick)
    order so the engine can bisect to the state in effect at any note-on.
    """

    tick: int
    cc: int       #: CC number 0..127, or :data:`EV_PROGRAM` / :data:`EV_PITCHBEND`
    value: int    #: program / CC value 0..127 / pitch-bend raw 0..16383


@dataclass(frozen=True)
class TempoChange:
    """A set-tempo meta event: ``us_per_quarter`` microseconds per quarter."""

    tick: int
    us_per_quarter: int


@dataclass(frozen=True)
class MidiNote:
    """A single pitched note, in canonical tick units.

    ``start_tick`` / ``end_tick`` are the raw note-on / note-off ticks. When
    the note is held by a sustain pedal, ``visual_end_tick`` (>= end_tick) is
    the tick at which its audio tail should end (the CC64 release); ``None``
    means the effective end is ``end_tick``.
    """

    channel: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    is_percussion: bool = False
    sustained: bool = False
    visual_end_tick: int | None = None

    def __post_init__(self) -> None:
        if self.end_tick < self.start_tick:
            raise ValueError(
                f"note ch{self.channel} p{self.pitch}: end_tick "
                f"({self.end_tick}) < start_tick ({self.start_tick})"
            )
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch out of range 0..127: {self.pitch}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"channel out of range 0..15: {self.channel}")

    @property
    def effective_end_tick(self) -> int:
        """Tick at which the note (audibly) ends, sustain-aware."""
        return self.visual_end_tick if self.visual_end_tick is not None else self.end_tick


@dataclass
class MidiInput:
    """Fully structured MIDI, ready for rendering (no file I/O involved)."""

    notes: list[MidiNote] = field(default_factory=list)
    #: Per-channel program/CC/pitch-bend events (tick-ordered), see
    #: :class:`ChannelEvent`. Empty dict = GM defaults on every channel.
    channel_events: dict[int, list[ChannelEvent]] = field(default_factory=dict)
    tempo_map: list[TempoChange] = field(default_factory=list)
    ppq: int = 480
    total_ticks: int = 0


# --- tick <-> seconds ------------------------------------------------------------


def _sec_per_tick(us_per_quarter: int, ppq: int) -> float:
    """Seconds elapsed per single tick at the given tempo."""
    if ppq <= 0:
        raise ValueError(f"ppq must be > 0, got {ppq}")
    return us_per_quarter / (ppq * 1_000_000.0)


@dataclass(frozen=True)
class TimeMap:
    """Immutable tick<->seconds mapper built from a tempo map.

    A sorted list of (tick, seconds) breakpoints is precomputed at each tempo
    change and interpolated linearly *within* each segment. The slope of a
    segment reflects the tempo in effect for that segment, which makes the
    mapping exact at every boundary.
    """

    ppq: int
    tempo_map: tuple[TempoChange, ...]
    total_ticks: int
    # Parallel, sorted-by-tick arrays of breakpoints.
    _brk_ticks: tuple[int, ...]
    _brk_secs: tuple[float, ...]

    @classmethod
    def build(cls, tempo_map: list[TempoChange], ppq: int, total_ticks: int) -> TimeMap:
        changes = list(tempo_map)
        if not changes:
            changes = [TempoChange(0, DEFAULT_US_PER_QUARTER)]
        # Guarantee a change at tick 0.
        if changes[0].tick != 0:
            changes.insert(0, TempoChange(0, changes[0].us_per_quarter))
        # NOTE: we require callers to pass tempo changes in chronological
        # (non-decreasing tick) order -- SMF parsers yield meta events in
        # file order, which is chronological. We do not silently sort, so an
        # out-of-order map is a loud error rather than a quiet mis-render.

        brk_ticks = [changes[0].tick]
        brk_secs = [0.0]
        for i in range(1, len(changes)):
            prev = changes[i - 1]
            cur = changes[i]
            if cur.tick < prev.tick:
                raise ValueError("tempo_map must be non-decreasing in tick")
            dt = (cur.tick - prev.tick) * _sec_per_tick(prev.us_per_quarter, ppq)
            brk_ticks.append(cur.tick)
            brk_secs.append(brk_secs[-1] + dt)

        # Extend the last segment to cover total_ticks at the final tempo.
        last_us = changes[-1].us_per_quarter
        if total_ticks > brk_ticks[-1]:
            dt = (total_ticks - brk_ticks[-1]) * _sec_per_tick(last_us, ppq)
            brk_ticks.append(int(total_ticks))
            brk_secs.append(brk_secs[-1] + dt)
        elif not brk_ticks:
            brk_ticks.append(0)
            brk_secs.append(0.0)

        return cls(
            ppq=ppq,
            tempo_map=tuple(changes),
            total_ticks=int(total_ticks),
            _brk_ticks=tuple(int(t) for t in brk_ticks),
            _brk_secs=tuple(float(s) for s in brk_secs),
        )

    def duration(self) -> float:
        """Total duration in seconds of ``total_ticks`` ticks."""
        return self.tick_to_seconds(self.total_ticks)

    def tick_to_seconds(self, tick: int) -> float:
        """Map a tick to seconds (piecewise-linear, exact at tempo boundaries)."""
        brk_t = self._brk_ticks
        brk_s = self._brk_secs
        if tick <= brk_t[0]:
            return 0.0
        if tick >= brk_t[-1]:
            return brk_s[-1]
        i = bisect.bisect_right(brk_t, tick) - 1  # brk_t[i] <= tick < brk_t[i+1]
        t0, t1 = brk_t[i], brk_t[i + 1]
        s0, s1 = brk_s[i], brk_s[i + 1]
        if t1 == t0:
            return s0
        return s0 + (tick - t0) * (s1 - s0) / (t1 - t0)

    def seconds_to_tick(self, sec: float) -> float:
        """Inverse of :meth:`tick_to_seconds` (piecewise-linear)."""
        brk_t = self._brk_ticks
        brk_s = self._brk_secs
        if sec <= 0.0:
            return 0.0
        if sec >= brk_s[-1]:
            return float(brk_t[-1])
        i = bisect.bisect_right(brk_s, sec) - 1  # brk_s[i] <= sec < brk_s[i+1]
        t0, t1 = brk_t[i], brk_t[i + 1]
        s0, s1 = brk_s[i], brk_s[i + 1]
        if s1 == s0:
            return float(t0)
        return t0 + (sec - s0) * (t1 - t0) / (s1 - s0)

    def tempos(self) -> list[TempoChange]:
        return list(self.tempo_map)
