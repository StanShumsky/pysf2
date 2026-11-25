"""Tests for pysf2 -- the pure-Python + numpy SoundFont 2 engine.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Layout
------
* unit    -- timecents/cents math, SF2 source transforms, volume envelope,
             region matcher, parser validation, channel-state reader;
* e2e     -- render the minimal 2.01 test font (``tests/_sf2gen.py``) at
             44.1 kHz and check pitch / velocity / release / loop / key
             range / percussion bank / CC7 / exclusive class via FFT;
* smoke   -- parse + render GeneralUser_GS (skipped when the font is absent).

The fixture font is built fresh per session (no binary fixture committed):

    sample 0  440 Hz, 2 s @ 22 kHz, loop 0.25-1.0 s, key 60  -> preset (0,0) rel 0.5 s
    sample 1  220 Hz, 1 s, no loop                          -> preset (0,1) key 60 only
    sample 2  330 Hz, 1 s, no loop                           -> preset (0,2) excl class 1,
                                                            regions 60-63 / 72-75, rel 0.3 s
    sample 3   80 Hz, 1 s, no loop                           -> preset (128,0) "Drums"
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest
from _notes import build_input
from _sf2gen import build_test_font

from pysf2 import EV_PITCHBEND, get_soundfont, load_soundfont, render
from pysf2.parse import SoundFontParseError
from pysf2.regions import cents_to_hz, tc_to_sec
from pysf2.render import (
    ChState,
    RegionMatcher,
    _ChannelReader,
    _env_at,
    _eval_modulators,
    _xform,
)

SR_OUT = 44100  # render rate; the font is 22050 Hz


# --- fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def font_path(tmp_path_factory) -> str:
    p = str(tmp_path_factory.mktemp("sf2") / "test.sf2")
    build_test_font(p)
    return p


def _render(tmp_path, items, font_path, **kw):
    """Build MidiInput directly (no file round-trip), render, and return
    (mono np array, sample_rate)."""
    kw.setdefault("sample_rate", SR_OUT)
    kw.setdefault("tail_sec", 1.5)
    data = build_input(items)
    out = render(data, font_path, **kw)
    w = out.waveform
    assert w.shape[0] == 2
    assert w.dtype == np.float32
    assert isinstance(out.sample_rate, int)
    return w.mean(axis=0), out.sample_rate


def _seg(mono, a, b, sr=SR_OUT):
    return mono[int(a * sr): int(b * sr)]


def _dom(x, fmin, fmax, sr=SR_OUT):
    fr = np.fft.rfftfreq(x.size, 1 / sr)
    s = np.abs(np.fft.rfft(x))
    m = (fr >= fmin) & (fr <= fmax)
    fr, s = fr[m], s[m]
    assert m.any()
    return float(fr[np.argmax(s)])


def _band_rms(x, fmin, fmax, sr=SR_OUT):
    # Hann-windowed: kills the rectangular-window leakage that a partial /
    # ramped tone would otherwise smear across the band.
    fr = np.fft.rfftfreq(x.size, 1 / sr)
    s = np.abs(np.fft.rfft(np.hanning(x.size) * x))
    m = (fr >= fmin) & (fr <= fmax)
    return float(np.sqrt(np.mean(s[m] ** 2)))


# --- unit: unit conversion -------------------------------------------------------

def test_tc_to_sec():
    assert tc_to_sec(0) == 1.0
    assert tc_to_sec(-12000) == 0.0
    assert tc_to_sec(-11950) == 0.0          # spec "instant" threshold
    assert tc_to_sec(-1200) == pytest.approx(0.5, rel=1e-6)    # 2^(-1)
    assert tc_to_sec(1200) == pytest.approx(2.0, rel=1e-6)     # 2^(+1)


def test_cents_to_hz():
    assert cents_to_hz(0) == pytest.approx(8.176)
    assert cents_to_hz(13500) == pytest.approx(20000.0, rel=0.01)   # spec "20 kHz"
    assert cents_to_hz(1200) == pytest.approx(cents_to_hz(0) * 2, rel=1e-9)


def test_xform_transforms():
    assert _xform(64.0, 0) == 64.0          # linear: pass through
    assert _xform(-0.5, 0) == -0.5
    assert _xform(-1.5, 2) == 1.5           # absolute value
    assert _xform(0.25, 2) == 0.25
    # kinds 1 / 3 are undefined in 2.01 -> linear
    assert _xform(-2.0, 1) == -2.0
    assert _xform(-3.0, 3) == -3.0


def test_env_at_phases():
    # instant delay/attack, hold 0.2, decay 0.3 -> sustain 0.5
    f = lambda t: _env_at(t, 0.0, 0.0, 0.2, 0.3, 0.5)   # noqa: E731
    assert f(0.0) == 0.0
    assert f(0.1) == pytest.approx(1.0)
    assert f(0.2) == pytest.approx(1.0)
    assert f(0.35) == pytest.approx(0.75)
    assert f(0.6) == pytest.approx(0.5)
    assert f(2.0) == pytest.approx(0.5)
    # real attack ramp
    g = lambda t: _env_at(t, 0.0, 0.4, 0.0, 0.0, 0.0)   # noqa: E731
    assert g(0.2) == pytest.approx(0.5)
    assert g(0.4) == pytest.approx(1.0)


# --- unit: parser ----------------------------------------------------------------

def test_parse_rejects_garbage(tmp_path):
    p = tmp_path / "bad.sf2"
    p.write_bytes(b"RIFF" + struct.pack("<I", 4) + b"WAVE" + b"data")
    with pytest.raises(SoundFontParseError):
        load_soundfont(p)
    p2 = tmp_path / "missing.sf2"
    with pytest.raises(SoundFontParseError):
        load_soundfont(p2)


def test_parse_test_font(font_path):
    sf = load_soundfont(font_path)
    assert sf.version == (2, 0)
    assert len(sf.presets) == 4
    assert len(sf.instruments) == 4
    assert sf.n_samples == 4
    assert sf.preset_index(0, 0) == 0
    assert sf.preset_index(128, 0) == 3
    assert sf.preset_index(1, 0) is None
    # last sample's exclusive end == total concatenated frames
    last = sf.samples[3]
    assert last.end == len(sf.smpl)
    assert sf.smpl.dtype == np.float32
    assert np.abs(sf.smpl).max() <= 1.0


def test_get_soundfont_cache(font_path):
    a = get_soundfont(font_path)
    b = get_soundfont(font_path)
    assert a is b                      # cached (same identity)
    assert isinstance(a[2], RegionMatcher)


# --- unit: region matcher ----------------------------------------------------------

def test_matcher_narrowest(font_path):
    sf, regions, matcher = get_soundfont(font_path)
    # preset 3 (index 3) = "Drums": one region, whole key range
    r = matcher.match(3, 36, 100)
    assert r is not None and r.sample_id == 3
    # preset 2 ("Excl"): keys 60-63 -> sample 2; key 64 -> no region
    r60 = matcher.match(2, 60, 100)
    r64 = matcher.match(2, 64, 100)
    r72 = matcher.match(2, 72, 100)
    assert r60 is not None and r60.sample_id == 2 and r60.exclusive_class == 1
    assert r64 is None
    assert r72 is not None and r72.key_lo == 72 and r72.key_hi == 75


# --- unit: channel reader -------------------------------------------------------------

class _E:
    def __init__(self, tick, cc, value):
        self.tick, self.cc, self.value = tick, cc, value


def test_channel_reader_state():
    evs = [
        _E(0, 0, 2),                     # bank (CC0)
        _E(100, 7, 50),                  # channel volume
        _E(200, 7, 100),                 # volume back to max
        _E(200, 0, 0),                   # bank reset
        _E(300, 100, 2),                 # bend range coarse -> 200 cents
        _E(300, 101, 0),
        _E(400, EV_PITCHBEND, 12288),    # +4096/8192 of 200 cents
    ]
    rd = _ChannelReader(evs)
    s1 = rd.state_at(50)     # before the volume event: GM default 100
    assert s1.volume == 100 and s1.bank == 2
    s2 = rd.state_at(150)    # volume 50, bank still 2
    assert s2.volume == 50 and s2.bank == 2
    s3 = rd.state_at(250)    # bank reset to 0, volume 100
    assert s3.volume == 100 and s3.bank == 0
    s4 = rd.state_at(500)    # bend +4096/8192 of 200 cents = +100 cents
    assert s4.bend_range_cents == pytest.approx(200.0)
    assert s4.bend_cents == pytest.approx(100.0)


def test_eval_modulators_dest_pitch(font_path):
    # key number (src 3, unipolar) -> fine tune (dest 52), 100 cents:
    # 64 -> 64/128 * 100 cents
    mods = [(3, 52, 100, 0, 0)]
    p = _eval_modulators(mods, ChState(), key=64, vel=100, cc_snap={})
    assert p.tune_cents == pytest.approx(64 / 128.0 * 100.0)
    p2 = _eval_modulators(mods, ChState(), key=127, vel=100, cc_snap={})
    assert p2.tune_cents == pytest.approx(127 / 128.0 * 100.0)


def test_eval_modulators_sources(font_path):
    # direction bit (D=1) flips the unipolar ramp: src 0x0103 (key, D=1)
    mods = [(0x103, 48, 128, 0, 0)]          # -> initial attenuation, 128 cB
    p0 = _eval_modulators(mods, ChState(), key=0, vel=100, cc_snap={})
    assert p0.att_cb == pytest.approx(127.0)  # (127-0)/128 * 128
    p1 = _eval_modulators(mods, ChState(), key=127, vel=100, cc_snap={})
    assert p1.att_cb == pytest.approx(0.0)    # (127-127)/128 * 128
    # CC palette (bit 7, index = CC number): CC7 volume -> filter Q (dest 9)
    p2 = _eval_modulators([(0x87, 9, 50, 0, 0)], ChState(), key=60, vel=100,
                          cc_snap={7: 100})
    assert p2.q_cb == pytest.approx(100 / 128.0 * 50.0)
    # bipolar (P=1) + absolute-value transform: key -> vibrato depth (dest 6)
    mods3 = [(0x203, 6, 30, 0, 2)]
    p3 = _eval_modulators(mods3, ChState(), key=0, vel=100, cc_snap={})
    assert p3.vib_depth == pytest.approx(30.0)     # |-1| * 30
    p4 = _eval_modulators(mods3, ChState(), key=127, vel=100, cc_snap={})
    assert p4.vib_depth == pytest.approx(30.0)     # |+1| * 30
    # pitch wheel (src 14, bipolar, spec 8.4.10 form 0x020E): +half range
    st = ChState(bend_cents=100.0, bend_range_cents=200.0)
    p5 = _eval_modulators([(0x20E, 52, 1000, 0, 0)], st, key=60, vel=100,
                          cc_snap={})
    assert p5.tune_cents == pytest.approx(500.0)
    # pitch-wheel sensitivity (src 16): 2 semitones -> 2/128
    p6 = _eval_modulators([(0x10, 48, 10, 0, 0)], st, key=60, vel=100,
                          cc_snap={})
    assert p6.att_cb == pytest.approx(2 / 128.0 * 10.0)


# --- e2e: pitch / velocity / envelope / loop ---------------------------------------

def test_pitch_exact(tmp_path, font_path):
    # preset 0 (440 Hz sample, keycenter 60): note 69 = +9 st -> 740 Hz
    mono, _ = _render(tmp_path, [
        (0, ("program", 0, 0)),
        (0, ("note_on", 0, 69, 100)),
        (960, ("note_off", 0, 69)),
    ], font_path, tail_sec=0.5)
    assert _dom(_seg(mono, 0.1, 0.9), 700, 780) == pytest.approx(740.0, abs=1.3)


def test_velocity_scaling(tmp_path, font_path):
    items = lambda vel: [                                   # noqa: E731
        (0, ("program", 0, 0)),
        (0, ("note_on", 0, 60, vel)),
        (480, ("note_off", 0, 60)),
    ]
    m127, _ = _render(tmp_path, items(127), font_path, tail_sec=0.3)
    m63, _ = _render(tmp_path, items(63), font_path, tail_sec=0.3)
    r = _band_rms(_seg(m127, 0.1, 0.4), 400, 480) / _band_rms(_seg(m63, 0.1, 0.4), 400, 480)
    assert r == pytest.approx(127 / 63, rel=0.01)


def test_release_tail(tmp_path, font_path):
    # preset 0 release = 0.5 s; note 60 on at 0, off at 1.0 s (tick 960)
    mono, _ = _render(tmp_path, [
        (0, ("program", 0, 0)),
        (0, ("note_on", 0, 60, 100)),
        (960, ("note_off", 0, 60)),
    ], font_path, tail_sec=1.0)
    full = _band_rms(_seg(mono, 0.3, 0.8), 400, 480)
    tail = _band_rms(_seg(mono, 1.15, 1.35), 400, 480)   # mid-ramp
    dead = _band_rms(_seg(mono, 1.6, 2.0), 400, 480)     # past release end
    assert full > 10.0
    assert 0 < tail < full
    assert dead < full * 0.02


def test_looping_continuity(tmp_path, font_path):
    # 4 s note on a 2 s looping sample: RMS steady across the loop boundary
    mono, _ = _render(tmp_path, [
        (0, ("program", 0, 0)),
        (0, ("note_on", 0, 60, 100)),
        (3840, ("note_off", 0, 60)),
    ], font_path, tail_sec=1.0)
    a = float(np.sqrt(np.mean(_seg(mono, 1.5, 2.0) ** 2)))
    b = float(np.sqrt(np.mean(_seg(mono, 2.5, 3.0) ** 2)))
    assert a > 0.05 and b > 0.05
    assert a / b == pytest.approx(1.0, rel=0.02)
    # no mid-note dropout anywhere
    for t in np.arange(0.1, 3.9, 0.25):
        assert float(np.abs(_seg(mono, t, t + 0.1)).max()) > 0.02


def test_keyrange_matching(tmp_path, font_path):
    # preset 1 region covers key 60 only
    items = lambda note: [                                  # noqa: E731
        (0, ("program", 0, 1)),
        (0, ("note_on", 0, note, 100)),
        (480, ("note_off", 0, note)),
    ]
    m60, _ = _render(tmp_path, items(60), font_path, tail_sec=0.3)
    m61, _ = _render(tmp_path, items(61), font_path, tail_sec=0.3)
    assert _dom(_seg(m60, 0.1, 0.4), 200, 240) == pytest.approx(220.0, abs=1.3)
    assert float(np.abs(m61).max()) < 1e-6


def test_unknown_program_silent(tmp_path, font_path):
    mono, _ = _render(tmp_path, [
        (0, ("program", 0, 99)),
        (0, ("note_on", 0, 60, 100)),
        (480, ("note_off", 0, 60)),
    ], font_path, tail_sec=0.3)
    assert float(np.abs(mono).max()) < 1e-6


# --- e2e: percussion / CC7 / exclusive class ----------------------------------------

def test_percussion_bank128(tmp_path, font_path):
    # channel 9, no CC0 -> bank 128; program 0 = "Drums" (80 Hz sample, key 60)
    # note 36 = -24 st -> 20 Hz
    mono, _ = _render(tmp_path, [
        (0, ("note_on", 9, 36, 100)),
        (480, ("note_off", 9, 36)),
    ], font_path, tail_sec=0.3)
    assert _dom(_seg(mono, 0.05, 0.45), 10, 40) == pytest.approx(20.0, abs=2.5)


def test_cc7_volume(tmp_path, font_path):
    items = lambda vol: [                                   # noqa: E731
        (0, ("program", 0, 0)),
        (0, ("cc", 0, 7, vol)),
        (10, ("note_on", 0, 60, 100)),
        (490, ("note_off", 0, 60)),
    ]
    m127, _ = _render(tmp_path, items(127), font_path, tail_sec=0.3)
    m63, _ = _render(tmp_path, items(63), font_path, tail_sec=0.3)
    r = _band_rms(_seg(m127, 0.1, 0.4), 400, 480) / _band_rms(_seg(m63, 0.1, 0.4), 400, 480)
    assert r == pytest.approx(127 / 63, rel=0.01)


def test_exclusive_class(tmp_path, font_path):
    """120 BPM, ppq 480 (1 tick = 1/960 s).

    v1 = note 72 (region 72-75 -> 660 Hz voice):  on t=0.0, off t=2.0 s
    v2 = note 60 (region 60-63 -> 330 Hz voice):  on t=0.5, off t=1.0 s
    Both regions are exclusive class 1, release 0.3 s, so v1 must be released
    at t=0.5 when v2 strikes and be silent from ~0.8 s on -- even though its
    note_off is at 2.0 s (it would otherwise play to ~2.3 s).
    """
    mono, _ = _render(tmp_path, [
        (0, ("program", 0, 2)),
        (0, ("note_on", 0, 72, 100)),
        (480, ("note_on", 0, 60, 100)),
        (480, ("note_off", 0, 60)),
        (1440, ("note_off", 0, 72)),
    ], font_path, tail_sec=1.0)
    full = _band_rms(_seg(mono, 0.05, 0.45), 600, 720)      # v1 alone, steady
    assert full > 10.0
    # v1 alone at full level
    assert _dom(_seg(mono, 0.05, 0.45), 600, 720) == pytest.approx(660.0, abs=1.3)
    # after v2 strikes: 330 present, 660 in release ramp
    assert _dom(_seg(mono, 0.85, 0.95), 250, 450) == pytest.approx(330.0, abs=1.3)
    # the proof: nothing at 660 after the release tail ends (t >= 0.8 s) --
    # more than 60 dB below the steady level; any residual is window leakage
    # from v2's 330 Hz release ramp, which overlaps [1.0, 1.3] inside the
    # measurement window
    assert _band_rms(_seg(mono, 1.0, 3.0), 600, 720) < full * 1e-3
    # v2's own tail also ends (off 1.0 + release 0.3 -> silent after 1.3 s)
    assert _band_rms(_seg(mono, 1.4, 3.0), 290, 370) == pytest.approx(0.0, abs=1e-4)
    assert float(np.abs(_seg(mono, 1.5, 1.9)).max()) < 1e-6


# --- e2e: render options -------------------------------------------------------------

def test_volume_and_percussion_flags(tmp_path, font_path):
    items = [
        (0, ("note_on", 9, 36, 100)),
        (480, ("note_off", 9, 36)),
    ]
    full, _ = _render(tmp_path, items, font_path, tail_sec=0.3)
    half, _ = _render(tmp_path, items, font_path, tail_sec=0.3, volume=0.5)
    no_p, _ = _render(tmp_path, items, font_path, tail_sec=0.3, include_percussion=False)
    assert np.abs(half).max() == pytest.approx(np.abs(full).max(), rel=0.02)
    assert float(np.abs(no_p).max()) < 1e-6


# --- smoke: GeneralUser GS (skip if the font is not on disk) ---------------------------
# Smoke-test font comes from the PYSF2_FONT env var only (no machine-specific
# paths in the suite); the test skips when unset or the file is missing.
_GU_FONT = os.environ.get("PYSF2_FONT", "")
_GU_FONT = _GU_FONT if os.path.isfile(_GU_FONT) else None


@pytest.mark.skipif(_GU_FONT is None, reason="PYSF2_FONT not set or font not present")
def test_generaluser_smoke():
    sf, regions, _m = get_soundfont(_GU_FONT)
    assert len(sf.presets) > 200
    assert sf.n_samples > 500
    total_regions = sum(len(r) for r in regions)
    assert total_regions > 1000

    data = build_input([
        (0, ("program", 0, 0)),   # grand piano
        (0, ("note_on", 0, 60, 100)),
        (0, ("note_on", 0, 67, 90)),
        (1440, ("note_off", 0, 60)),
        (1440, ("note_off", 0, 67)),
    ])
    out = render(data, sf, tail_sec=1.0)
    w = out.waveform.mean(axis=0)
    assert np.isfinite(w).all()
    assert float(np.sqrt(np.mean(w ** 2))) > 1e-3
    # C4 fundamental: peak of 200-320 Hz band within a few Hz of 261.6
    seg = _seg(w, 0.1, 1.2)
    fr = np.fft.rfftfreq(seg.size, 1 / 44100)
    s = np.abs(np.fft.rfft(seg))
    m = (fr >= 200) & (fr <= 320)
    peak = float(fr[m][np.argmax(s[m])])
    assert 255.0 <= peak <= 268.0

