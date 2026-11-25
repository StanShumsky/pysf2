"""pysf2 -- pure-Python + numpy SoundFont 2 (.sf2) renderer.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Public surface::

    from pysf2 import get_soundfont, render

    sf, regions, matcher = get_soundfont("font.sf2")   # parsed + cached
    audio = render(midi_input, "font.sf2")             # RenderedAudio

``render`` also accepts a font *path* (loads + caches it for you). Output is
a :class:`RenderedAudio`: stereo ``float32`` waveform, channel-first
``(2, N)``, plus the sample rate.

The input is a structured :class:`MidiInput` (notes + per-channel
program/CC/pitch-bend events + tempo map, all in raw SMF ticks) -- see
:mod:`pysf2.events`. MIDI *file* parsing is deliberately host-side.
"""
from __future__ import annotations

import os

from .events import (
    EV_PITCHBEND,
    EV_PROGRAM,
    TRACKED_CCS,
    ChannelEvent,
    MidiInput,
    MidiNote,
    TempoChange,
    TimeMap,
)
from .parse import SoundFont, SoundFontParseError, load_soundfont
from .regions import Region, build_regions
from .render import ChState, RegionMatcher, RenderedAudio, render_midi

__all__ = [
    "SoundFont",
    "SoundFontParseError",
    "load_soundfont",
    "Region",
    "build_regions",
    "RegionMatcher",
    "ChState",
    "RenderedAudio",
    "render_midi",
    "render",
    "get_soundfont",
    "MidiInput",
    "MidiNote",
    "ChannelEvent",
    "TempoChange",
    "TimeMap",
    "EV_PROGRAM",
    "EV_PITCHBEND",
    "TRACKED_CCS",
]

__version__ = "0.1.0"

_MAX_CACHED_FONTS = 3
_font_cache: dict[tuple, tuple[SoundFont, list, RegionMatcher]] = {}


def get_soundfont(path) -> tuple[SoundFont, list[list[Region]], RegionMatcher]:
    """Load (and cache) a SoundFont, returning ``(font, regions, matcher)``.

    Cached by (absolute path, mtime_ns, size) so edits on disk invalidate the
    entry. Raises :class:`SoundFontParseError` on malformed files.
    """
    p = os.path.abspath(os.fspath(path))
    st = os.stat(p)
    key = (p, st.st_mtime_ns, st.st_size)
    hit = _font_cache.get(key)
    if hit is not None:
        return hit
    sf = load_soundfont(p)
    regions = build_regions(sf)
    entry = (sf, regions, RegionMatcher(regions))
    while len(_font_cache) >= _MAX_CACHED_FONTS:
        _font_cache.pop(next(iter(_font_cache)))
    _font_cache[key] = entry
    return entry


def render(
    midi: MidiInput,
    soundfont,
    sample_rate: int = 44100,
    tail_sec: float = 2.0,
    volume: float = 1.0,
    include_percussion: bool = True,
) -> RenderedAudio:
    """Render ``midi`` (a :class:`MidiInput`) through a SoundFont.

    ``soundfont`` may be a :class:`SoundFont` or a path to a ``.sf2`` file
    (loaded + cached). Returns a :class:`RenderedAudio`.
    """
    if isinstance(soundfont, (str, os.PathLike)):
        sf, _regions, matcher = get_soundfont(soundfont)
    elif isinstance(soundfont, SoundFont):
        matcher = RegionMatcher(build_regions(soundfont))
        sf = soundfont
    else:
        raise TypeError(
            f"soundfont must be a SoundFont or a file path, got {type(soundfont).__name__}"
        )
    return render_midi(
        midi, sf, matcher,
        sample_rate=sample_rate, tail_sec=tail_sec, volume=volume,
        include_percussion=include_percussion,
    )
