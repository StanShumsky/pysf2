"""Build ``pysf2.MidiInput`` directly -- no MIDI files, no mido.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

The engine consumes structured tick-based events, so the tests generate
them with a tiny DSL. Item ticks are *deltas* (SMF convention, same
semantics the host pack's tests used for mido messages); they are
cumulated in list order, so items must already be in chronological
order (equal ticks keep list order -- on before off, like a parser):

    build_input([
        (0,   ("program", 0, 0)),
        (0,   ("note_on", 0, 60, 100)),
        (960, ("note_off", 0, 60)),
    ], ppq=480, tempo_us=500_000)

Kinds: ``("note_on", ch, note, vel)`` | ``("note_off", ch, note)`` |
``("cc", ch, cc, value)`` | ``("program", ch, program)`` |
``("bend", ch, raw14)``. Channel 9 notes get ``is_percussion=True``
(GM convention, mirroring the host pack's parser).
"""
from __future__ import annotations

from pysf2 import (
    EV_PITCHBEND,
    EV_PROGRAM,
    ChannelEvent,
    MidiInput,
    MidiNote,
    TempoChange,
)


def build_input(items, ppq: int = 480, tempo_us: int = 500_000) -> MidiInput:
    """items: list of ``(delta_tick, kind-tuple)`` in chronological order.

    500000 us/quarter == 120 BPM (the default of every test).
    """
    notes: list[MidiNote] = []
    open_idx: dict[tuple[int, int], int] = {}
    ch_events: dict[int, list[ChannelEvent]] = {}
    tick = 0

    def add_ev(ch: int, ev: ChannelEvent) -> None:
        ch_events.setdefault(ch, []).append(ev)

    for dt, it in items:
        if dt < 0:
            raise ValueError(f"negative delta tick: {dt}")
        tick += dt
        kind = it[0]
        if kind == "note_on":
            _, ch, note, vel = it
            open_idx[(ch, note)] = len(notes)
            notes.append(MidiNote(
                ch, note, vel, tick, tick, is_percussion=(ch == 9)))
        elif kind == "note_off":
            _, ch, note = it
            i = open_idx.pop((ch, note), None)
            if i is not None:
                n = notes[i]
                notes[i] = MidiNote(n.channel, n.pitch, n.velocity, n.start_tick,
                                    tick, is_percussion=n.is_percussion)
        elif kind == "cc":
            _, ch, cc, value = it
            add_ev(ch, ChannelEvent(tick, cc, value))
        elif kind == "program":
            _, ch, program = it
            add_ev(ch, ChannelEvent(tick, EV_PROGRAM, program))
        elif kind == "bend":
            _, ch, raw = it
            add_ev(ch, ChannelEvent(tick, EV_PITCHBEND, raw))
        else:
            raise ValueError(f"unknown event kind: {kind!r}")

    return MidiInput(
        notes=notes,
        channel_events=ch_events,
        tempo_map=[TempoChange(0, tempo_us)],
        ppq=ppq,
        total_ticks=tick,
    )
