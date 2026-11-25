"""Offline SoundFont 2 rendering: voice synthesis and stereo mix (numpy only).

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Pipeline per note
-----------------
1. Resolve the channel state in effect at note-on (program, bank select, CC
   volume / mod wheel, pitch bend) from ``MidiInput.channel_events`` (bisect).
2. Select the preset ``(bank, program)`` -- percussion channels default to
   bank 128 when the file never sends CC0 -- then the narrowest region whose
   key / velocity range contains the note (key/velocity substitution applied
   afterwards, per the spec).
3. Sample readout: linearly interpolated resample whose phase is a ``cumsum``
   of per-sample rates, which makes vibrato / modulation-LFO pitch modulation
   exact; loop wrapping (continuous and while-key-held); clamped / zeroed past
   the sample end.
4. Piecewise-linear volume envelope (delay -> attack -> hold -> decay ->
   sustain -> release on note-off) via ``np.interp``.
5. Resonant 4-pole low-pass (two cascaded RBJ 2-pole biquads approximating
   the spec's idealized filter) via a block-looped direct-form numpy
   biquad chain (validated against ``scipy.signal.lfilter`` to <1e-5 abs).
   The default cut (13500 cents ~ 20 kHz) is a no-op, so unfiltered fonts
   skip the pass entirely.
6. Explicit preset / instrument modulators evaluated at note-on with the
   spec source encoding (8.2): note-on velocity / key number, pitch wheel
   and its sensitivity, channel pressure, arbitrary CC sources (bit 7),
   direction / polarity bits, and the spec 8.3 transform (linear or
   absolute value).
7. Cosine pan law, channel volume, exclusive-class voice release, float32
   stereo mix, peak guard (no up-normalization: fonts carry their own
   dynamics).

v1 simplifications (documented; no audible impact on typical fonts)
--------------------------------------------------------------------
* modulation envelope ignored (volume envelope only), so mod-env
  destinations (7, 11) read 0;
* reverb / chorus sends not implemented (no shared FX buses);
* time-varying filter cut (LFO -> filterCut) uses the note-on value;
* LFO rate / delay modulators (dest 21-24) are not applied;
* non-linear modulator curves (concave / convex / switch, bit 10) are read
  as linear; secondary (amount) modulator sources read 1;
* poly-pressure and link modulator sources read 0;
* resampling is linear (no band-limiting), so extreme pitch shifts alias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .events import EV_PITCHBEND, EV_PROGRAM, TRACKED_CCS, MidiInput, MidiNote, TimeMap
from .parse import SoundFont
from .regions import Region, cents_to_hz, tc_to_sec

__all__ = ["render_midi", "RenderedAudio", "RegionMatcher", "ChState"]

#: GM default values for tracked CCs (used when a file never sets one).
_CC_DEFAULTS = {0: 0, 1: 0, 7: 100, 10: 0, 11: 0, 32: 0, 100: 2, 101: 0}


# --- channel state -----------------------------------------------------------


@dataclass(frozen=True)
class ChState:
    """Channel state snapshot at one tick."""

    program: int = 0
    bank: int = 0
    volume: int = 100
    mod_wheel: int = 0
    bend_cents: float = 0.0
    bend_range_cents: float = 200.0


class _ChannelReader:
    """Bisect lookup of per-channel program / CC / bend state by tick."""

    __slots__ = ("_cc", "ticks")

    def __init__(self, events: list) -> None:
        self._cc: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for e in events:  # already tick-ordered (parser emits in event order)
            ta, va = self._cc.get(e.cc, ([], []))
            ta.append(e.tick)
            va.append(e.value)
            self._cc[e.cc] = (ta, va)
        self._cc = {
            cc: (np.asarray(ta, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for cc, (ta, va) in self._cc.items()
        }

    def has_cc(self, cc: int) -> bool:
        return cc in self._cc

    def cc_at(self, tick: int, cc: int, default: int) -> int:
        arr = self._cc.get(cc)
        if arr is None:
            return default
        ta, va = arr
        i = int(np.searchsorted(ta, tick, side="right")) - 1
        return int(va[i]) if i >= 0 else default

    def state_at(self, tick: int) -> ChState:
        coarse = self.cc_at(tick, 100, _CC_DEFAULTS[100])
        fine = self.cc_at(tick, 101, _CC_DEFAULTS[101])
        range_cents = float(coarse * 100 + fine)
        bend_raw = self.cc_at(tick, EV_PITCHBEND, 8192)
        bend_cents = (bend_raw - 8192) / 8192.0 * range_cents
        return ChState(
            program=self.cc_at(tick, EV_PROGRAM, 0),
            bank=self.cc_at(tick, 0, _CC_DEFAULTS[0]),
            volume=self.cc_at(tick, 7, _CC_DEFAULTS[7]),
            mod_wheel=self.cc_at(tick, 1, _CC_DEFAULTS[1]),
            bend_cents=bend_cents,
            bend_range_cents=range_cents,
        )

    def cc_snapshot(self, tick: int) -> dict[int, int]:
        return {
            cc: self.cc_at(tick, cc, _CC_DEFAULTS.get(cc, 0)) for cc in TRACKED_CCS
        }


# --- explicit modulators (note-on evaluation) --------------------------------


@dataclass
class _Params:
    """Region parameters adjusted by explicit modulators at note-on."""

    key: int
    vel: int
    tune_cents: float = 0.0      # extra static pitch (dest pitch)
    vib_depth: float = 0.0       # extra vibrato-LFO depth (cents)
    mod_depth_pitch: float = 0.0  # extra mod-LFO pitch depth (cents)
    mod_depth_vol: float = 0.0   # extra mod-LFO volume depth (cB)
    fc_cents: float = 0.0
    q_cb: float = 0.0
    att_cb: float = 0.0
    pan: float = 0.0


def _xform(x: float, kind: int) -> float:
    """SF2 2.01 modulator transform (spec 8.3) on the normalized value.

    0 = linear (pass through), 2 = absolute value; kinds 1 / 3 are
    undefined in 2.01 and read as linear.
    """
    if kind == 2:
        return abs(x)
    return x


def _src_norm(src: int, raw: float) -> float:
    """Direction (bit 8) / polarity (bit 9) normalization (spec 8.2.2-3).

    Unipolar: 0 -> 0, 127 -> 127/128; D=1 flips the ramp ((127-v)/128).
    Bipolar:  0 -> -1, 127 -> +1 around 63.5; D=1 negates.
    """
    if (src >> 9) & 1:  # P: bipolar
        x = (raw - 63.5) / 63.5
        return -x if (src >> 8) & 1 else x
    return raw / 128.0 if not ((src >> 8) & 1) else (127.0 - raw) / 128.0


def _src_raw(src: int, state: ChState, key: int, vel: int,
             cc_snap: dict[int, int]) -> float:
    """Raw value of the note-on source in the 0..127 domain (pitch wheel:
    63.5 = center).  Spec 8.2.1 palettes: bit 7 selects the MIDI CC palette
    (index = CC number), otherwise the general palette (2 note-on velocity,
    3 note-on key number, 13 channel pressure, 14 pitch wheel, 16 pitch
    wheel sensitivity, 0 no controller = value 1)."""
    if src & 0x80:  # CC palette
        return float(cc_snap.get(src & 0x7F, 0))
    sid = src & 0x7F
    if sid == 0:    # no controller: spec says value 1
        return 1.0
    if sid == 2:    # note-on velocity
        return float(vel)
    if sid == 3:    # note-on key number
        return float(key)
    if sid == 13:   # channel pressure (CC10)
        return float(cc_snap.get(10, 0))
    if sid == 14:   # pitch wheel (63.5 = center, full scale 63.5)
        frac = state.bend_cents / state.bend_range_cents if state.bend_range_cents else 0.0
        return float(min(127.0, max(0.0, 63.5 * (1.0 + frac))))
    if sid == 16:   # pitch-wheel sensitivity (RPN 0, semitones)
        return float(state.bend_range_cents / 100.0)
    return 0.0  # 10 poly pressure / 127 link / reserved: v1 reads 0


def _eval_modulators(mods: list[tuple], state: ChState, key: int, vel: int,
                     cc_snap: dict[int, int]) -> _Params:
    """Apply explicit zone modulators to the region parameters (note-on).

    ``mods`` are raw pmod/imod 5-tuples (src, dest, amount, amt_src,
    transform).  Destinations are spec generator ids: 48 initial
    attenuation (cB), 8 initial filter fc (cents), 9 initial filter Q (cB),
    5 mod LFO to pitch (cents), 6 vibrato LFO to pitch (cents), 13 mod LFO
    to volume (cB), 17 pan (0.1%), 51/52 coarse (st) / fine (cents) tune.
    """
    p = _Params(key=key, vel=vel)
    for src, dest, amount, _amt_src, trans in mods:
        if (src >> 11) & 0x1F:  # undefined source types: spec says ignore
            continue
        v = _xform(_src_norm(src, _src_raw(src, state, key, vel, cc_snap)),
                   trans) * amount
        if dest == 48:      # initial attenuation (cB)
            p.att_cb += v
        elif dest == 8:     # initial filter fc (cents)
            p.fc_cents += v
        elif dest == 9:     # initial filter Q (cB)
            p.q_cb += v
        elif dest == 6:     # vibrato LFO to pitch (cents)
            p.vib_depth += v
        elif dest == 5:     # mod LFO to pitch (cents)
            p.mod_depth_pitch += v
        elif dest == 13:    # mod LFO to volume (cB)
            p.mod_depth_vol += v
        elif dest == 17:    # pan (0.1% units)
            p.pan += v * 0.001
        elif dest == 51:    # coarse tune (semitones)
            p.tune_cents += v * 100.0
        elif dest == 52:    # fine tune (cents)
            p.tune_cents += v
    return p


# --- 4-pole RBJ low-pass ------------------------------------------------------


def _rbj4(x: np.ndarray, sr: int, fc: float, q: float) -> np.ndarray:
    """4-pole resonant low-pass: two cascaded RBJ 2-pole biquads (the spec's
    filter is idealized; this is a standard stable -24 dB/oct realization).
    The normalized pole radius is < 1 for every clamped fc, and a Q floor
    keeps the damping term bounded, so the cascade never over-rings."""
    w = 2.0 * math.pi * min(max(fc, 10.0), sr * 0.49) / sr
    cw = math.cos(w)
    sw = math.sin(w)
    q = max(q, sw / 3.9)  # keep the damping term < 2 (stable everywhere)
    a = sw / (2.0 * q)
    b0 = (1.0 - cw) / 2.0
    b = [b0, 2.0 * b0, b0]
    aa = [1.0 + a, -2.0 * cw, 1.0 - a]
    b = [v / aa[0] for v in b]
    aa = [1.0, aa[1] / aa[0], aa[2] / aa[0]]
    return _df1(b, aa, _df1(b, aa, x))


def _df1(b: list[float], a: list[float], x: np.ndarray) -> np.ndarray:
    """Direct-form I biquad, block-looped."""
    out = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1v, b2v = b
    a1v, a2v = a[1], a[2]
    for s in range(0, len(x), 4096):
        seg = x[s : s + 4096]
        o = out[s : s + len(seg)]
        for i in range(len(seg)):
            v = b0 * seg[i] + b1v * x1 + b2v * x2 - a1v * y1 - a2v * y2
            x2, x1 = x1, seg[i]
            y2, y1 = y1, v
            o[i] = v
    return out


# --- voices -------------------------------------------------------------------


class _Voice:
    """One scheduled note-on, with everything resolved before rendering."""

    __slots__ = (
        "note", "region", "on_tick", "off_tick", "state", "params", "cc_snap",
    )

    def __init__(self, note, region, on_tick, off_tick, state, params, cc_snap):
        self.note = note
        self.region = region
        self.on_tick = on_tick
        self.off_tick = off_tick
        self.state = state
        self.params = params
        self.cc_snap = cc_snap


def _env_at(t: float, td: float, ta: float, th: float, tdec: float, sus: float) -> float:
    """Pre-release volume envelope value (continuous by construction)."""
    if t <= td:
        return 0.0
    if t <= td + ta:
        return (t - td) / ta if ta > 0 else 1.0
    if t <= td + ta + th:
        return 1.0
    t4 = td + ta + th + tdec
    if t <= t4:
        return sus if tdec <= 0 else 1.0 - (1.0 - sus) * ((t - td - ta - th) / tdec)
    return sus


def _render_voice(v: _Voice, sf: SoundFont, tm: TimeMap, sr: int, n: int,
                  buf: np.ndarray) -> None:
    r = v.region
    p = v.params
    state = v.state
    smpl = sf.smpl
    total = len(smpl)

    on_s = tm.tick_to_seconds(v.on_tick)
    off_s = tm.tick_to_seconds(v.off_tick)
    off_time = max(0.0, off_s - on_s)
    off_idx = int(off_time * sr)
    i_start = int(on_s * sr)
    if i_start >= n:
        return

    env = r.env
    t_rel = tc_to_sec(env.release)
    t_delay = tc_to_sec(env.delay)
    t_atk = tc_to_sec(env.attack)
    t_hold = tc_to_sec(env.hold)
    t_dec = tc_to_sec(env.decay)
    t_sus_end = t_delay + t_atk + t_hold + t_dec
    t_end = off_time + t_rel

    # --- length: envelope end, capped by sample end for non-looping voices
    loops = r.loops
    # readout rate: input frames per output sample. (sample_rate / sr) keeps
    # real-time playback; the pitch term shifts frequency.
    ratio = (r.sample_rate / sr) * 2.0 ** (
        ((p.key - r.keycenter) * r.keytrack + r.transpose * 100 + r.tune + p.tune_cents
         + state.bend_cents) / 1200.0
    )
    if loops or t_rel <= 0.0:
        L = int(max(0.0, t_end) * sr) + 1
    else:
        sample_end_s = (r.end - r.offset) / max(ratio * sr, 1e-6)
        L = int(min(t_end, sample_end_s) * sr) + 1
    L = min(L, n - i_start)
    if L <= 1 or t_end <= 0.0:
        return

    t_axis = np.arange(L, dtype=np.float64) / sr

    # --- volume envelope (piecewise linear via np.interp)
    sus = 10.0 ** (env.sustain / 200.0)
    val_off = _env_at(off_time, t_delay, t_atk, t_hold, t_dec, sus)
    bp_t = (0.0, t_delay, td2 := t_delay + t_atk, td2 + t_hold,
            t_sus_end, off_time, off_time + t_rel)
    bp_v = (0.0, 0.0, 1.0, 1.0, sus, val_off, 0.0)
    # Collapse to unique x, keeping the group's first ("just before") and last
    # ("just after") value -- this preserves the sustain plateau when phases
    # are instant and lets an instant release drop to 0 without ramping.
    groups: list[tuple] = []
    for tt, vv in zip(bp_t, bp_v, strict=False):
        tt = max(0.0, min(tt, L / sr))
        if groups and tt <= groups[-1][-1][0] + 1e-9:
            groups[-1].append((tt, vv))
        else:
            groups.append([(tt, vv)])
    pts, vals = [], []
    for g in groups:
        pts.append(g[0][0])
        vals.append(g[0][1])
        if len(g) > 1:
            pts.append(g[-1][0])
            vals.append(g[-1][1])
    env_arr = np.interp(t_axis, pts, vals)

    # --- LFO pitch modulation (per-sample cents) and mod-LFO volume (cB)
    vib_depth = r.vib_to_pitch + p.vib_depth
    mod_depth_pitch = r.mod_to_pitch + p.mod_depth_pitch
    mod_depth_vol = r.mod_to_volume + p.mod_depth_vol
    cents_lfo = None
    if vib_depth or mod_depth_pitch:
        cents_lfo = np.zeros(L, dtype=np.float64)
        if vib_depth:
            ph = 2.0 * np.pi * cents_to_hz(r.vib_freq_cents) * t_axis
            ph[ t_axis < max(0.0, tc_to_sec(r.vib_delay)) ] = 0.0
            cents_lfo += vib_depth * np.sin(ph)
        if mod_depth_pitch:
            ph = 2.0 * np.pi * cents_to_hz(r.mod_freq_cents) * t_axis
            ph[ t_axis < max(0.0, tc_to_sec(r.mod_delay)) ] = 0.0
            cents_lfo += mod_depth_pitch * np.sin(ph)
    vol_lfo_gain = None
    if mod_depth_vol:
        ph = 2.0 * np.pi * cents_to_hz(r.mod_freq_cents) * t_axis
        ph[t_axis < max(0.0, tc_to_sec(r.mod_delay))] = 0.0
        vol_lfo_gain = (10.0 ** (mod_depth_vol * np.sin(ph) / 200.0)).astype(np.float32)

    # --- resample phase (cumsum of per-sample rates) + loop wrap
    base_rate = (r.sample_rate / sr) * 2.0 ** (
        ((p.key - r.keycenter) * r.keytrack + r.transpose * 100 + r.tune
         + p.tune_cents + state.bend_cents) / 1200.0
    )
    if cents_lfo is None:
        pos = r.offset + base_rate * np.arange(L, dtype=np.float64)
    else:
        pos = r.offset + np.cumsum(base_rate * np.exp(0.6931471805599453 * cents_lfo / 1200.0))
    end = min(r.end, total)
    if end <= r.offset:
        end = min(r.offset + 1, total)
    dead = None
    if loops:
        ls = min(r.loop_start, end)
        le = min(r.loop_end, end)
        if le > ls:
            ll = le - ls
            wrapped = np.where(pos >= le, ls + (pos - ls) % ll, pos)
            if r.loop_mode == 3:  # while-key-held: no wrap after note-off
                idx = np.where(np.arange(L) >= off_idx, pos, wrapped)
                dead = idx >= end  # play-through past the sample end is silent
            else:
                idx = wrapped
        else:
            idx = pos
            dead = idx >= end
    else:
        idx = pos
        dead = idx >= end

    i0 = np.clip(np.floor(idx).astype(np.int64), 0, total - 1)
    i1 = np.minimum(i0 + 1, total - 1)
    frac = (idx - i0).astype(np.float32)
    x = (smpl[i0] * (1.0 - frac) + smpl[i1] * frac).astype(np.float32)
    if dead is not None and dead.any():
        x[dead] = 0.0

    # --- 4-pole RBJ low-pass (default cut ~20 kHz => skipped, i.e. no-op)
    fc = cents_to_hz(r.fc_cents + p.fc_cents)
    if fc < sr * 0.45:
        q = min(max(10.0 ** ((r.q_cb + p.q_cb) / 200.0), 0.1), 100.0)
        x = _rbj4(x, sr, fc, q).astype(np.float32)

    # --- gains: velocity, channel volume (CC7), region attenuation
    att_cb = r.attenuation_db * 10.0 + p.att_cb
    gain = (
        (p.vel / 127.0)
        * (state.volume / 127.0)
        * 10.0 ** (-att_cb / 200.0)
    )
    if vol_lfo_gain is not None:
        x *= vol_lfo_gain
    x = (x * env_arr.astype(np.float32) * gain).astype(np.float32)

    # --- pan + mix
    ang = (r.pan + p.pan + 0.5) * 0.5 * math.pi
    gl, gr = math.cos(ang), math.sin(ang)
    buf[0, i_start : i_start + L] += (x * gl).astype(np.float32)
    buf[1, i_start : i_start + L] += (x * gr).astype(np.float32)


# --- fast region matching ------------------------------------------------------


class RegionMatcher:
    """Per-preset numpy arrays for fast key/velocity region lookup.

    Selects the *narrowest* matching region (key width, then velocity width,
    then file order) -- the "most specific" rule reference loaders use.
    """

    def __init__(self, regions: list[list[Region]]) -> None:
        self._data = []
        for regs in regions:
            if not regs:
                self._data.append(None)
                continue
            self._data.append((
                np.array([r.key_lo for r in regs], dtype=np.int16),
                np.array([r.key_hi for r in regs], dtype=np.int16),
                np.array([r.vel_lo for r in regs], dtype=np.int16),
                np.array([r.vel_hi for r in regs], dtype=np.int16),
                np.array([r.key_hi - r.key_lo for r in regs], dtype=np.int32),
                np.array([r.vel_hi - r.vel_lo for r in regs], dtype=np.int32),
                regs,
            ))

    def match(self, preset_index: int, key: int, vel: int) -> Region | None:
        d = self._data[preset_index]
        if d is None:
            return None
        klo, khi, vlo, vhi, kw, vw, regs = d
        m = (klo <= key) & (khi >= key) & (vlo <= vel) & (vhi >= vel)
        if not m.any():
            return None
        cand = np.flatnonzero(m)
        order = np.lexsort((cand, vw[m], kw[m]))
        return regs[int(cand[order[0]])]


# --- output container --------------------------------------------------------


@dataclass(frozen=True)
class RenderedAudio:
    """Offline render result: stereo ``float32`` waveform, channel-first.

    ``waveform`` has shape ``(2, N)`` -- row 0 is left, row 1 is right.
    """

    waveform: np.ndarray
    sample_rate: int

    def to_mono(self) -> np.ndarray:
        """Mid-side collapse to mono ``(N,)`` float32."""
        return (self.waveform[0] + self.waveform[1]) * 0.5


# --- top-level render ----------------------------------------------------------


def render_midi(
    data: MidiInput,
    sf: SoundFont,
    matcher: RegionMatcher,
    sample_rate: int = 44100,
    tail_sec: float = 2.0,
    volume: float = 1.0,
    include_percussion: bool = True,
) -> dict:
    """Render a structured :class:`MidiInput` through a SoundFont bank.

    Returns a :class:`RenderedAudio` (stereo ``float32``, channel-first
    ``(2, N)``). Notes without a matching preset/region are skipped silently
    (they would be silent in a SoundFont too).
    """
    tm = TimeMap.build(data.tempo_map, data.ppq, data.total_ticks)
    n = int(max(1, (tm.duration() + tail_sec) * sample_rate))
    buf = np.zeros((2, n), dtype=np.float32)

    voices: list[_Voice] = []
    per_ch: dict[int, list[MidiNote]] = {}
    for note in data.notes:
        if note.is_percussion and not include_percussion:
            continue
        per_ch.setdefault(note.channel, []).append(note)

    for ch, notes in per_ch.items():
        reader = _ChannelReader(data.channel_events.get(ch, []))
        active_classes: dict[int, list[_Voice]] = {}
        for note in notes:
            state = reader.state_at(note.start_tick)
            bank = state.bank
            if note.is_percussion and not reader.has_cc(0) and \
                    sf.preset_index(128, state.program) is not None:
                bank = 128  # GM: drum channel defaults to bank 128
            pi = sf.preset_index(bank, state.program)
            if pi is None:
                continue
            region = matcher.match(pi, note.pitch, note.velocity)
            if region is None:
                continue
            key = region.keynum_sub if region.keynum_sub >= 0 else note.pitch
            vel = region.vel_sub if region.vel_sub >= 0 else note.velocity
            cc_snap = reader.cc_snapshot(note.start_tick)
            params = _eval_modulators(region.modulators, state, key, vel, cc_snap)
            v = _Voice(
                note=note,
                region=region,
                on_tick=note.start_tick,
                off_tick=note.effective_end_tick,
                state=state,
                params=params,
                cc_snap=cc_snap,
            )
            ec = region.exclusive_class
            if ec:
                keep = [old for old in active_classes.get(ec, ())
                        if old.off_tick > note.start_tick]
                for old in keep:  # release all same-class voices now
                    old.off_tick = note.start_tick
                keep.append(v)
                active_classes[ec] = keep
            voices.append(v)

    for v in voices:
        _render_voice(v, sf, tm, sample_rate, n, buf)

    peak = float(np.abs(buf).max()) if n else 0.0
    if peak > 1.0:
        buf *= (0.99 / peak)
    return RenderedAudio(waveform=buf, sample_rate=int(sample_rate))
