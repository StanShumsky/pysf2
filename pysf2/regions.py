"""Zone matching for the SoundFont 2.00/2.01 model.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

A *preset zone* (segment of the preset's pgen list, ending at an
``instrument`` generator) selects an instrument; an *instrument zone* (segment
of an instrument's igen list, ending at a ``sampleID`` generator) selects a
sample. The final audible *region* is the merge of: default values, the
instrument's global zone, the instrument zone, and the preset zone (value
generators add; range generators intersect; index generators set). This
follows the SF2 2.01 spec sections 7/9.4 and the behaviour of reference
loaders (cf. TinySoundFont, MIT).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parse import SoundFont

# --- generator ids (SoundFont 2.00/2.01 table, spec 8.1.2) -------------------
GEN_STARTADDRS = 0
GEN_ENDADDRS = 1
GEN_STARTLOOP = 2
GEN_ENDLOOP = 3
GEN_STARTADDRS_COARSE = 4
GEN_MODLFO_TO_PITCH = 5
GEN_VIBLFO_TO_PITCH = 6
GEN_MODENV_TO_PITCH = 7
GEN_INITIAL_FILTER_FC = 8
GEN_INITIAL_FILTER_Q = 9
GEN_MODLFO_TO_FILTER = 10
GEN_MODENV_TO_FILTER = 11
GEN_ENDADDRS_COARSE = 12
GEN_MODLFO_TO_VOLUME = 13
GEN_CHORUS_SEND = 15
GEN_REVERB_SEND = 16
GEN_PAN = 17
GEN_DELAY_MODLFO = 21
GEN_FREQ_MODLFO = 22
GEN_DELAY_VIBLFO = 23
GEN_FREQ_VIBLFO = 24
GEN_DELAY_MODENV = 25
GEN_ATTACK_MODENV = 26
GEN_HOLD_MODENV = 27
GEN_DECAY_MODENV = 28
GEN_SUSTAIN_MODENV = 29
GEN_RELEASE_MODENV = 30
GEN_KNUM_MODENV_HOLD = 31
GEN_KNUM_MODENV_DECAY = 32
GEN_DELAY_VOLENV = 33
GEN_ATTACK_VOLENV = 34
GEN_HOLD_VOLENV = 35
GEN_DECAY_VOLENV = 36
GEN_SUSTAIN_VOLENV = 37
GEN_RELEASE_VOLENV = 38
GEN_KNUM_VOLENV_HOLD = 39
GEN_KNUM_VOLENV_DECAY = 40
GEN_INSTRUMENT = 41
GEN_KEYRANGE = 43
GEN_VELRANGE = 44
GEN_STARTLOOP_COARSE = 45
GEN_KEYNUM = 46
GEN_VELOCITY = 47
GEN_INITIAL_ATTENUATION = 48
GEN_ENDLOOP_COARSE = 50
GEN_COARSE_TUNE = 51
GEN_FINE_TUNE = 52
GEN_SAMPLEID = 53
GEN_SAMPLEMODES = 54
GEN_SCALE_TUNING = 56
GEN_EXCLUSIVE_CLASS = 57
GEN_OVERRIDING_ROOTKEY = 58

_TC_INSTANT = -12000.0  # timecents ~= 0.98 ms; spec convention for "instant"


@dataclass
class Env:
    """Volume-envelope parameters, times in timecents (0 = 1 s), sustain in cB."""

    delay: float = _TC_INSTANT
    attack: float = _TC_INSTANT
    hold: float = _TC_INSTANT
    decay: float = _TC_INSTANT
    sustain: float = 0.0
    release: float = _TC_INSTANT
    keynum_to_hold: int = 0
    keynum_to_decay: int = 0


def tc_to_sec(timecents: float) -> float:
    """Absolute timecents to seconds (0 -> 1 s, -12000 -> ~0.98 ms)."""
    if timecents <= -11950.0:
        return 0.0
    return 2.0 ** (timecents / 1200.0)


def cents_to_hz(cents: float) -> float:
    """Absolute cents to hertz (0 -> 8.176 Hz, 13500 -> ~20 kHz)."""
    return 8.176 * 2.0 ** (cents / 1200.0)


@dataclass
class Region:
    """One fully-resolved, playable zone (sample + parameters)."""

    preset_index: int
    sample_id: int
    # sample addressing (absolute frames into SoundFont.smpl)
    offset: int = 0
    end: int = 0
    loop_start: int = 0
    loop_end: int = 0
    loop_mode: int = 0          # 0 none, 1 continuous, 3 while-held
    sample_rate: int = 44100
    # pitch
    keycenter: int = 60         # root key (sample played unshifted here)
    transpose: int = 0          # coarseTune, semitones
    tune: int = 0               # fineTune + shdr pitchCorrection, cents
    keytrack: int = 100         # scaleTuning, cents/key (100 = semitone)
    # amplitude
    attenuation_db: float = 0.0
    env: Env = field(default_factory=Env)
    # filter / effects
    fc_cents: int = 13500
    q_cb: int = 0
    pan: float = 0.0            # -0.5 (L) .. +0.5 (R)
    # LFOs
    vib_delay: float = _TC_INSTANT
    vib_freq_cents: int = 0
    vib_to_pitch: int = 0       # cents
    mod_delay: float = _TC_INSTANT
    mod_freq_cents: int = 0
    mod_to_pitch: int = 0
    mod_to_filter: int = 0
    mod_to_volume: int = 0      # cB
    # zone matching
    key_lo: int = 0
    key_hi: int = 127
    vel_lo: int = 0
    vel_hi: int = 127
    exclusive_class: int = 0
    keynum_sub: int = -1        # substitution (forced key), -1 = none
    vel_sub: int = -1
    # explicit modulators merged from instrument + preset level
    modulators: list[tuple[int, int, int, int, int]] = field(default_factory=list)

    @property
    def loops(self) -> bool:
        return self.loop_mode in (1, 3) and self.loop_start < self.loop_end


# --- generator accumulation ---------------------------------------------------

class _Acc:
    """Mutable zone accumulator (one per zone being built)."""

    __slots__ = (
        "key_lo", "key_hi", "vel_lo", "vel_hi",
        "off", "end", "lstart", "lend",
        "keycenter", "transpose", "tune", "keytrack",
        "att_db", "pan", "fc", "q",
        "vib_delay", "vib_freq", "vib_pitch", "mod_delay", "mod_freq",
        "mod_pitch", "mod_filter", "mod_volume",
        "e_delay", "e_attack", "e_hold", "e_decay", "e_sustain", "e_release",
        "e_knh", "e_knd",
        "exclusive", "keynum_sub", "vel_sub",
        "_set",
    )

    #: timecents generators: a zone *sets* the value (replacing the "instant"
    #: default), it does not add to the -12000 sentinel.
    _TC_FIELDS = ("e_delay", "e_attack", "e_hold", "e_decay", "e_release",
                  "vib_delay", "mod_delay")

    def __init__(self) -> None:
        self.key_lo, self.key_hi = 0, 127
        self.vel_lo, self.vel_hi = 0, 127
        self.off = self.end = self.lstart = self.lend = 0
        self.keycenter = -1
        self.transpose = self.tune = 0
        # keytrack / fc hold the *delta* from the spec default (100 cents/key,
        # 13500 cents); the default is applied in _finalize so that merging
        # zones (which adds) never double-counts it.
        self.keytrack = 0
        self.att_db = 0.0
        self.pan = 0.0
        self.fc = 0
        self.q = 0
        self.vib_delay = self.mod_delay = _TC_INSTANT
        self.vib_freq = self.mod_freq = 0
        self.vib_pitch = self.mod_pitch = self.mod_filter = self.mod_volume = 0
        self.e_delay = self.e_attack = self.e_hold = self.e_decay = self.e_release = _TC_INSTANT
        self.e_sustain = 0.0
        self.e_knh = self.e_knd = 0
        self.exclusive = 0
        self.keynum_sub = self.vel_sub = -1
        self._set: set = set()  # timecents fields explicitly set by a zone

    def _set_tc(self, f: str, a: int) -> None:
        setattr(self, f, float(a))
        self._set.add(f)

    def tc(self, f: str) -> float:
        return getattr(self, f) if f in self._set else _TC_INSTANT

    def apply(self, gen: int, amount: int) -> None:
        a = amount
        if gen == GEN_KEYRANGE:
            self.key_lo = a & 0xFF
            self.key_hi = (a >> 8) & 0xFF
        elif gen == GEN_VELRANGE:
            self.vel_lo = a & 0xFF
            self.vel_hi = (a >> 8) & 0xFF
        elif gen == GEN_STARTADDRS:
            self.off += a
        elif gen == GEN_ENDADDRS:
            self.end += a
        elif gen == GEN_STARTLOOP:
            self.lstart += a
        elif gen == GEN_ENDLOOP:
            self.lend += a
        elif gen == GEN_STARTADDRS_COARSE:
            self.off += a * 32768
        elif gen == GEN_ENDADDRS_COARSE:
            self.end += a * 32768
        elif gen == GEN_STARTLOOP_COARSE:
            self.lstart += a * 32768
        elif gen == GEN_ENDLOOP_COARSE:
            self.lend += a * 32768
        elif gen == GEN_MODLFO_TO_PITCH:
            self.mod_pitch += a
        elif gen == GEN_VIBLFO_TO_PITCH:
            self.vib_pitch += a
        elif gen == GEN_INITIAL_FILTER_FC:
            self.fc += a
        elif gen == GEN_INITIAL_FILTER_Q:
            self.q += a
        elif gen == GEN_MODLFO_TO_FILTER:
            self.mod_filter += a
        elif gen == GEN_MODLFO_TO_VOLUME:
            self.mod_volume += a
        elif gen == GEN_PAN:
            self.pan += a * 0.001
        elif gen == GEN_DELAY_MODLFO:
            self._set_tc("mod_delay", a)
        elif gen == GEN_FREQ_MODLFO:
            self.mod_freq += a
        elif gen == GEN_DELAY_VIBLFO:
            self._set_tc("vib_delay", a)
        elif gen == GEN_FREQ_VIBLFO:
            self.vib_freq += a
        elif gen == GEN_DELAY_VOLENV:
            self._set_tc("e_delay", a)
        elif gen == GEN_ATTACK_VOLENV:
            self._set_tc("e_attack", a)
        elif gen == GEN_HOLD_VOLENV:
            self._set_tc("e_hold", a)
        elif gen == GEN_DECAY_VOLENV:
            self._set_tc("e_decay", a)
        elif gen == GEN_SUSTAIN_VOLENV:
            self.e_sustain += a
        elif gen == GEN_RELEASE_VOLENV:
            self._set_tc("e_release", a)
        elif gen == GEN_KNUM_VOLENV_HOLD:
            self.e_knh += a
        elif gen == GEN_KNUM_VOLENV_DECAY:
            self.e_knd += a
        elif gen == GEN_KEYNUM:
            if 0 <= a <= 127:
                self.keynum_sub = a
        elif gen == GEN_VELOCITY:
            if 0 <= a <= 127:
                self.vel_sub = a
        elif gen == GEN_INITIAL_ATTENUATION:
            self.att_db += a * 0.1
        elif gen == GEN_COARSE_TUNE:
            self.transpose += a
        elif gen == GEN_FINE_TUNE:
            self.tune += a
        elif gen == GEN_SCALE_TUNING:
            self.keytrack += a
        elif gen == GEN_EXCLUSIVE_CLASS:
            self.exclusive = a
        elif gen == GEN_OVERRIDING_ROOTKEY:
            if 0 <= a <= 127:
                self.keycenter = a
        # other generators (LFO delay/freq sign handling, mod-env, effects sends,
        # reserved, ...) are ignored in v1.

    def merge(self, other: _Acc) -> None:
        """Add a higher-precedence zone (value gens add; ranges intersect)."""
        self.key_lo = max(self.key_lo, other.key_lo)
        self.key_hi = min(self.key_hi, other.key_hi)
        self.vel_lo = max(self.vel_lo, other.vel_lo)
        self.vel_hi = min(self.vel_hi, other.vel_hi)
        for f in _Acc.__slots__:
            if f in ("key_lo", "key_hi", "vel_lo", "vel_hi", "_set"):
                continue
            if f in self._TC_FIELDS:
                if f in other._set:
                    setattr(self, f, getattr(other, f))
                    self._set.add(f)
            elif f == "att_db" or f == "pan":
                setattr(self, f, getattr(self, f) + getattr(other, f))
            elif f in ("keycenter", "keynum_sub", "vel_sub"):
                if getattr(other, f) >= 0:
                    setattr(self, f, getattr(other, f))
            elif f == "exclusive":
                if other.exclusive:
                    self.exclusive = other.exclusive
            else:
                setattr(self, f, getattr(self, f) + getattr(other, f))


def _gen_iter(gens: np.ndarray, start: int, stop: int):
    """Yield (gen_id, signed_amount) for a gen-list window."""
    for i in range(start, min(stop, len(gens))):
        gid = int(gens[i][0])
        amt = int(gens[i][1])
        if amt >= 0x8000:
            amt -= 0x10000
        yield gid, amt


def build_regions(sf: SoundFont) -> list[list[Region]]:
    """Return per-preset-index lists of fully resolved regions."""
    regions: list[list[Region]] = [[] for _ in range(len(sf.presets))]
    if not sf.presets or len(sf.ibags) == 0:
        return regions

    pbags, pgens = sf.pbags, sf.pgens
    ibags, igens = sf.ibags, sf.igens
    pmods, imods = sf.pmods, sf.imods
    total = len(sf.smpl)

    def _mod_window(mods: list, bag: np.ndarray, i: int) -> list[tuple]:
        m0 = int(bag[i][1])
        m1 = int(bag[i + 1][1]) if i + 1 < len(bag) else len(mods)
        m0, m1 = min(m0, len(mods)), min(m1, len(mods))
        if m1 <= m0:
            return []
        return [
            (m.src, m.dest, m.amount, m.amt_src, m.transform)
            for m in mods[m0:m1]
        ]

    for pi, preset in enumerate(sf.presets):
        p0 = min(preset.bag_ndx, len(pbags))
        p1 = max(p0 + 1, min(preset.bag_end, len(pbags)))
        global_acc = _Acc()
        for bi in range(p0, p1):
            preset_acc = _Acc()
            preset_acc.merge(global_acc)
            had_instrument = False
            g0 = int(pbags[bi]["gen"])
            g1 = int(pbags[bi + 1]["gen"]) if bi + 1 < len(pbags) else len(pgens)
            preset_mods = _mod_window(pmods, pbags, bi)

            for gid, amt in _gen_iter(pgens, g0, g1):
                if gid == GEN_INSTRUMENT:
                    had_instrument = True
                    which = amt
                    if which < 0 or which >= len(sf.instruments):
                        continue
                    inst = sf.instruments[which]
                    i0 = min(inst.bag_ndx, len(ibags))
                    i1 = max(i0 + 1, min(inst.bag_end, len(ibags)))
                    inst_global = _Acc()
                    first_ibag = True
                    for ii in range(i0, i1):
                        zone = _Acc()
                        zone.merge(inst_global)
                        zg0 = int(ibags[ii]["gen"])
                        zg1 = int(ibags[ii + 1]["gen"]) if ii + 1 < len(ibags) else len(igens)
                        sample_id = -1
                        zone_mods = _mod_window(imods, ibags, ii)
                        for zg, za in _gen_iter(igens, zg0, zg1):
                            if zg == GEN_SAMPLEID:
                                sample_id = za
                                break
                            zone.apply(zg, za)
                        if first_ibag and sample_id < 0:
                            # instrument's global zone
                            inst_global = zone
                        first_ibag = False
                        if not (0 <= sample_id < len(sf.samples)):
                            continue
                        # preset zone range filter
                        if (zone.key_hi < preset_acc.key_lo or zone.key_lo > preset_acc.key_hi
                                or zone.vel_hi < preset_acc.vel_lo or zone.vel_lo > preset_acc.vel_hi):
                            continue
                        zone.merge(preset_acc)
                        _finalize(sf, pi, zone, sample_id, zone_mods + preset_mods, regions[pi], total)
                else:
                    preset_acc.apply(gid, amt)
            if bi == p0 and not had_instrument:
                # preset's global zone
                global_acc = preset_acc
    return regions


def _finalize(sf: SoundFont, pi: int, z: _Acc, sample_id: int,
              modulators: list[tuple], out: list[Region], total: int) -> None:
    s = sf.samples[sample_id]
    # shdr start/end/loop bounds: start & loop_start inclusive, end &
    # end_loop exclusive (spec 8.2.1); coarse/addr gens adjust by frame.
    off = max(0, min(z.off + s.start, total))
    end = max(0, min(z.end + s.end, total))
    if end <= off:
        end = min(off + 1, total)
    lstart = max(0, min(z.lstart + s.start_loop, total))
    lend = max(0, min(z.lend + s.end_loop, total))
    keycenter = z.keycenter if z.keycenter >= 0 else s.original_key
    out.append(Region(
        preset_index=pi,
        sample_id=sample_id,
        offset=max(0, off),
        end=end,
        loop_start=lstart,
        loop_end=lend,
        loop_mode=s.sample_type & 3,
        sample_rate=s.sample_rate or 44100,
        keycenter=keycenter,
        transpose=z.transpose,
        tune=z.tune + s.pitch_correction,
        keytrack=100 + z.keytrack,
        attenuation_db=z.att_db,
        env=Env(z.tc("e_delay"), z.tc("e_attack"), z.tc("e_hold"), z.tc("e_decay"),
                z.e_sustain, z.tc("e_release"), z.e_knh, z.e_knd),
        fc_cents=13500 + z.fc,
        q_cb=z.q,
        pan=max(-0.5, min(0.5, z.pan)),
        vib_delay=z.tc("vib_delay"),
        vib_freq_cents=z.vib_freq,
        vib_to_pitch=z.vib_pitch,
        mod_delay=z.tc("mod_delay"),
        mod_freq_cents=z.mod_freq,
        mod_to_pitch=z.mod_pitch,
        mod_to_filter=z.mod_filter,
        mod_to_volume=z.mod_volume,
        key_lo=z.key_lo,
        key_hi=z.key_hi,
        vel_lo=z.vel_lo,
        vel_hi=z.vel_hi,
        exclusive_class=z.exclusive,
        keynum_sub=z.keynum_sub,
        vel_sub=z.vel_sub,
        modulators=modulators,
    ))
