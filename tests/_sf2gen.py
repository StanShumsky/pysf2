"""Minimal SoundFont 2.01 (``.sf2``) test-font builder (numpy + stdlib only).

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Builds a small, fully valid 2.01-layout font on the fly (no binary fixture to
commit) with known ground truth for the engine tests:

* sample 0 -- 440 Hz sine, 2.0 s @ 22050 Hz, original key 60, continuous loop
  0.25 s..1.0 s  ->  preset (0, 0) "Sine Loop" (release 0.5 s)
* sample 1 -- 220 Hz sine, 1.0 s, no loop  ->  preset (0, 1) "Sine One",
  region restricted to key 60 only (range matching)
* sample 2 -- 330 Hz sine, 1.0 s  ->  preset (0, 2) "Excl": two regions,
  keys 60..63 and 72..75, both exclusive class 1 (release 0.3 s)
* sample 3 -- 80 Hz sine, 1.0 s    ->  preset (128, 0) "Drums" (bank-select test)

Layout: ``RIFF sfbk`` with ``hdrl/ifil`` + ``pdta`` (phdr pbag pmod pgen inst
ibag imod igen) + ``sdta/smpl`` -- the standard flat layout (the parser also
accepts pdtr/pdts nesting).
"""
from __future__ import annotations

import struct

import numpy as np

# generator ids (subset, SF2 2.01 spec table 8.1.2)
GEN_INSTRUMENT = 41
GEN_KEYRANGE = 43
GEN_EXCLUSIVE_CLASS = 57
GEN_RELEASE_VOLENV = 38
GEN_SAMPLEID = 53


def _tc(sec: float) -> int:
    """Seconds -> timecents (0 = 1 s, negative = < 1 s)."""
    return int(round(1200.0 * np.log2(sec))) if sec > 0 else -12000


def _u16(v: int) -> int:
    return v & 0xFFFF


def _name(s: str) -> bytes:
    b = s.encode("ascii")[:20]
    return b + b"\0" * (20 - len(b))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack("<I", len(payload)) + payload


def _list(form: bytes, payload: bytes) -> bytes:
    """A ``LIST`` chunk: tag LIST, size (incl. form), 4cc form, sub-chunks."""
    body = form + payload
    return b"LIST" + struct.pack("<I", len(body)) + body


def _sine(freq: float, sec: float, sr: int) -> np.ndarray:
    t = np.arange(int(sec * sr), dtype=np.float64) / sr
    x = 0.6 * np.sin(2.0 * np.pi * freq * t)
    return (np.clip(x, -1, 1) * 32767).astype("<i2")


def build_test_font(path: str, sr: int = 22050) -> dict:
    """Write a test font to ``path``; return per-sample frame counts."""
    sines = {
        0: _sine(440.0, 2.0, sr),
        1: _sine(220.0, 1.0, sr),
        2: _sine(330.0, 1.0, sr),
        3: _sine(80.0, 1.0, sr),
    }
    # shdr: (start, end, loop_start, loop_end, orig_key, sample_type)
    starts, hdrs, acc = [], [], 0
    for i in range(4):
        n = len(sines[i])
        starts.append(acc)
        acc += n
    hdrs = [
        (starts[0], starts[0] + len(sines[0]), int(0.25 * sr), int(1.0 * sr), 60, 1),
        (starts[1], starts[1] + len(sines[1]), 0, 0, 60, 0),
        (starts[2], starts[2] + len(sines[2]), 0, 0, 60, 0),
        (starts[3], starts[3] + len(sines[3]), 0, 0, 60, 0),
    ]
    shdr = b""
    for (start, end, ls, le, key, stype) in hdrs:
        shdr += _name("tone") + struct.pack(
            "<IIIIIBbHH", start, end, ls, le, sr, key, 0, 0, stype
        )

    # instruments: 0 -> sample 0; 1 -> sample 1; 2 -> two key-ranged regions
    # on sample 2 (exclusive class 1); 3 -> sample 3
    # instrument -> [bag, ...]; each bag = [(gen_id, amount), ...]
    inst_bags = {
        0: [[(GEN_SAMPLEID, 0)]],
        1: [[(GEN_SAMPLEID, 1)]],
        2: [[(GEN_KEYRANGE, (63 << 8) | 60), (GEN_EXCLUSIVE_CLASS, 1),
             (GEN_SAMPLEID, 2)],
            [(GEN_KEYRANGE, (75 << 8) | 72), (GEN_EXCLUSIVE_CLASS, 1),
             (GEN_SAMPLEID, 2)]],
        3: [[(GEN_SAMPLEID, 3)]],
    }
    ibag, igen = b"", b""
    bag_ndx = []  # instrument -> first ibag index (bags are contiguous per inst)
    for i in range(4):
        bag_ndx.append(len(ibag) // 4)
        for gens in inst_bags[i]:
            # each bag's gen window starts at the current igen count; the window
            # runs to the next bag's index (no in-list terminator in the format)
            ibag += struct.pack("<HH", len(igen) // 4, 0)
            for gid, amt in gens:
                igen += struct.pack("<HH", gid, _u16(amt))
    inst = b""
    for i, nm in enumerate(("inst0", "inst1", "inst2", "inst3")):
        inst += _name(nm) + struct.pack("<H", bag_ndx[i])

    # presets: (name, bank, preset, release_tc or None, extra preset gens, inst)
    presets = [
        ("Sine Loop", 0, 0, _tc(0.5), [], 0),
        ("Sine One", 0, 1, None, [(GEN_KEYRANGE, (60 << 8) | 60)], 1),
        ("Excl", 0, 2, _tc(0.3), [], 2),
        ("Drums", 128, 0, None, [], 3),
    ]
    phdr, pbag, pgen = b"", b"", b""
    for i, (nm, bank, preset, rel, extra, which) in enumerate(presets):
        # bag window = [this bag idx, next preset's bag idx) -> use i directly
        phdr += _name(nm) + struct.pack("<HHHIII", preset, bank, i, 0, 0, 0)
        pbag += struct.pack("<HH", len(pgen) // 4, 0)  # gen window start
        if rel is not None:
            pgen += struct.pack("<HH", GEN_RELEASE_VOLENV, _u16(rel))
        for gid, amt in extra:
            pgen += struct.pack("<HH", gid, _u16(amt))
        pgen += struct.pack("<HH", GEN_INSTRUMENT, _u16(which))

    pdta = (b"phdr" + struct.pack("<I", len(phdr)) + phdr
            + b"pbag" + struct.pack("<I", len(pbag)) + pbag
            + b"pmod" + struct.pack("<I", 0)
            + b"pgen" + struct.pack("<I", len(pgen)) + pgen
            + b"inst" + struct.pack("<I", len(inst)) + inst
            + b"ibag" + struct.pack("<I", len(ibag)) + ibag
            + b"imod" + struct.pack("<I", 0)
            + b"igen" + struct.pack("<I", len(igen)) + igen
            + b"shdr" + struct.pack("<I", len(shdr)) + shdr)
    hdrl = _chunk(b"ifil", struct.pack("<HH", 2, 0)) + _chunk(b"sbna", b"")
    smpl = np.concatenate(list(sines.values())).tobytes()
    sdta = _chunk(b"smpl", smpl)
    body = _list(b"hdrl", hdrl) + _list(b"pdta", pdta) + _list(b"sdta", sdta)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + b"sfbk" + body)
    return {"frames": [len(v) for v in sines.values()], "sr": sr}
