"""SoundFont 2.00/2.01 (``.sf2``) file parser -- pure Python + numpy.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Layout reference: *SoundFont 2.01 Technical Specification* (Creative, 1998),
verified byte-for-byte against real fonts (e.g. GeneralUser GS):

=====================  ==========================================================
chunk                    record (little endian)
=====================  ==========================================================
``phdr``   (38 B)       name[20], preset u16, bank u16, presetBagNdx u16,
                        library u32, genre u32, morphology u32
``pbag``   (4 B)        genNdx u16, modNdx u16
``pmod``   (10 B)       src u16, dest u16, amount i16, amtSrc u16, transform u16
``pgen``   (4 B)        genOper u16, amount u16 (signed short or lo/hi range)
``inst``   (22 B)       name[20], instBagNdx u16
``ibag``   (4 B)        genNdx u16, modNdx u16
``imod``   (10 B)       src u16, dest u16, amount i16, amtSrc u16, transform u16
``igen``   (4 B)        genOper u16, amount u16
``shdr``   (46 B)       name[20], start u32, end u32, startLoop u32, endLoop u32,
                        sampleRate u32, originalKey u8, pitchCorrection i8,
                        sampleLink u16, sampleType u16
``smpl``   (2 B each)   int16 linear PCM, all samples concatenated
=====================  ==========================================================

The standard (and overwhelmingly common) layout places every pdta sub-chunk
flat inside ``LIST pdta`` (``phdr pbag pmod pgen inst ibag imod igen shdr``) and
the sample data in ``LIST sdta / smpl``; some fonts nest them in ``pdtr`` /
``pdts`` lists instead. Chunk tags are collected wherever they appear, so both
layouts are handled. Bag lists are terminated by the *next* bag's index (the
format has no in-list terminator marker).

Note on ``.sf3``: SF3 files with Vorbis-compressed samples are **not** supported
(this parser reads 16-bit PCM only); uncompressed SF3 files parse fine.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SoundFontParseError",
    "Sf2Sample",
    "Sf2Instrument",
    "Sf2Preset",
    "Sf2Modulator",
    "SoundFont",
    "load_soundfont",
    "GEN_END",
]


#: Sentinel generator id (0x2FFFE cannot occur in a 16-bit record; kept for
#: compatibility with tools that emit it as a terminator).
GEN_END = 0x2FFFE


#: Generator record: 16-bit op + 16-bit amount (signed value or lo/hi range).
_GEN_DT = np.dtype([("id", "<u2"), ("amt", "<u2")])


class SoundFontParseError(ValueError):
    """Raised when a file cannot be parsed as a SoundFont 2 file."""


def _zstr(b: bytes, n: int) -> str:
    """Decode a fixed-width zero-padded ASCII field."""
    return b[:n].split(b"\0", 1)[0].decode("ascii", "replace")


@dataclass(frozen=True)
class Sf2Sample:
    """One ``shdr`` record (sample header)."""

    name: str
    start: int
    end: int
    start_loop: int
    end_loop: int
    sample_rate: int
    original_key: int
    pitch_correction: int
    sample_link: int
    sample_type: int


def s16(v: int) -> int:
    """Interpret a 16-bit unsigned value as signed."""
    return v - 0x10000 if v >= 0x8000 else v


@dataclass
class Sf2Modulator:
    """One ``pmod``/``imod`` record."""

    src: int
    dest: int
    amount: int
    amt_src: int
    transform: int

    @property
    def src_index(self) -> int:
        return self.src & 0x7F

    @property
    def src_cc(self) -> bool:
        return bool(self.src & 0x80)

    @property
    def src_dir(self) -> int:
        return (self.src >> 8) & 0x1

    @property
    def src_pol(self) -> int:
        return (self.src >> 9) & 0x1

    @property
    def src_type(self) -> int:
        return (self.src >> 10) & 0x3F


@dataclass
class Sf2Preset:
    """One ``phdr`` record plus resolved bag window into the pbag list."""

    name: str
    preset: int
    bank: int
    bag_ndx: int
    bag_end: int  # exclusive; = next preset's bag_ndx (or list end)


@dataclass
class Sf2Instrument:
    """One ``inst`` record plus resolved bag window into the ibag list."""

    name: str
    bag_ndx: int
    bag_end: int


@dataclass
class SoundFont:
    """A fully parsed SoundFont 2 bank."""

    path: str
    version: tuple[int, int]
    presets: list[Sf2Preset]
    instruments: list[Sf2Instrument]
    samples: list[Sf2Sample]
    pbags: np.ndarray  # (N, 2) u16: genNdx, modNdx
    pmods: list[Sf2Modulator]
    pgens: np.ndarray  # (N,) structured u16 id/amt
    ibags: np.ndarray
    imods: list[Sf2Modulator]
    igens: np.ndarray
    smpl: np.ndarray  # float32, all samples concatenated, in [-1, 1]
    presets_by_id: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def preset_index(self, bank: int, preset: int) -> int | None:
        """Index of the (bank, preset) program, or None."""
        return self.presets_by_id.get((bank, preset))


def _walk_into(data: bytes, off: int, end: int, found: dict[str, bytes]) -> None:
    """Collect chunk payloads by 4cc tag (first occurrence wins).

    Handles ``LIST``/``RIFF`` nesting and 16-bit size padding. Non-printable
tags or malformed sizes stop the walk (lenient, like most SF2 loaders).
    """
    while off + 8 <= end:
        tag = data[off : off + 4]
        if len(tag) < 4 or not all(32 <= b < 127 for b in tag):
            break
        (size,) = struct.unpack_from("<I", data, off + 4)
        body = off + 8
        if body + size > end:
            break
        tag_s = tag.decode("latin1")
        if tag_s in ("LIST", "RIFF"):
            form = data[body : body + 4]
            if form and all(32 <= b < 127 for b in form):
                _walk_into(data, body + 4, body + size, found)
        elif tag_s not in found:
            found[tag_s] = data[body : body + size]
        off = body + size
        if off & 1:
            off += 1


def _walk_chunks(data: bytes, off: int, end: int) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    _walk_into(data, off, end, found)
    return found


def _split_records(buf: bytes, rec_size: int, what: str) -> bytes:
    n = len(buf) // rec_size
    if n == 0:
        raise SoundFontParseError(f"Chunk '{what}' is empty or too small.")
    return buf[: n * rec_size]


def load_soundfont(path: str | os.PathLike) -> SoundFont:
    """Parse a SoundFont 2 file into a :class:`SoundFont`.

    Raises :class:`SoundFontParseError` on unreadable / malformed files.
    """
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise SoundFontParseError(f"SoundFont file not found: {p}")
    with open(p, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] not in (b"sfbk", b"sf2k"):
        raise SoundFontParseError(
            f"Not a SoundFont file (missing RIFF/'sfbk' header): {p}"
        )
    chunks = _walk_chunks(data, 12, len(data))

    for required in ("phdr", "pbag", "pgen", "inst", "ibag", "igen", "shdr", "smpl"):
        if required not in chunks:
            raise SoundFontParseError(
                f"Missing required chunk '{required}' in SoundFont file: {p}"
            )

    # --- ifil version (optional) ---
    version = (2, 0)
    ifil = chunks.get("ifil")
    if ifil and len(ifil) >= 4:
        major, minor = struct.unpack_from("<HH", ifil, 0)
        version = (major, minor)

    # --- phdr (38 B) ---
    phdr = _split_records(chunks["phdr"], 38, "phdr")
    n_presets = len(phdr) // 38
    presets = []
    for i in range(n_presets):
        e = phdr[i * 38 : (i + 1) * 38]
        presets.append(
            Sf2Preset(
                name=_zstr(e, 20),
                preset=struct.unpack_from("<H", e, 20)[0],
                bank=struct.unpack_from("<H", e, 22)[0],
                bag_ndx=struct.unpack_from("<H", e, 24)[0],
                bag_end=0,  # resolved below
            )
        )

    # --- inst (22 B) ---
    inst = _split_records(chunks["inst"], 22, "inst")
    n_inst = len(inst) // 22
    instruments = []
    for i in range(n_inst):
        e = inst[i * 22 : (i + 1) * 22]
        instruments.append(
            Sf2Instrument(
                name=_zstr(e, 20),
                bag_ndx=struct.unpack_from("<H", e, 20)[0],
                bag_end=0,  # resolved below
            )
        )

    # --- pbag / ibag (4 B) and bag-window resolution ---
    pbag = _split_records(chunks["pbag"], 4, "pbag")
    pbags = np.frombuffer(pbag, dtype=np.dtype([("gen", "<u2"), ("mod", "<u2")])).copy()
    n_pbags = len(pbags)
    ibag = _split_records(chunks["ibag"], 4, "ibag")
    ibags = np.frombuffer(ibag, dtype=np.dtype([("gen", "<u2"), ("mod", "<u2")])).copy()
    n_ibags = len(ibags)

    # Preset / instrument bag windows: [bag_ndx, next record's bag_ndx in
    # FILE order), clamped to the bag list length.  The spec orders phdr / inst
    # records by (bank, preset) so file order normally equals bag order -- but
    # some fonts (e.g. GeneralUser's 'sfbk' build) break that (a (0,0) preset
    # can appear at the very end), so only file order is consistent: the bag
    # regions are laid out sequentially in the file and bagNdx is monotonic in
    # file order.
    for pos in range(n_presets):
        nxt = presets[pos + 1].bag_ndx if pos + 1 < n_presets else n_pbags
        presets[pos].bag_end = max(presets[pos].bag_ndx, min(nxt, n_pbags))
    for pos in range(n_inst):
        nxt = instruments[pos + 1].bag_ndx if pos + 1 < n_inst else n_ibags
        instruments[pos].bag_end = max(instruments[pos].bag_ndx, min(nxt, n_ibags))

    # --- modulators (10 B) ---
    def _parse_mods(buf: bytes | None) -> list[Sf2Modulator]:
        if not buf:
            return []
        out = []
        for i in range(len(buf) // 10):
            e = buf[i * 10 : (i + 1) * 10]
            src, dest, amt, amt_src, trans = struct.unpack_from("<HHhHH", e, 0)
            out.append(Sf2Modulator(src, dest, amt, amt_src, trans))
        return out

    pmods = _parse_mods(chunks.get("pmod"))
    imods = _parse_mods(chunks.get("imod"))

    # --- generators (4 B) ---
    def _parse_gens(buf: bytes) -> np.ndarray:
        buf = _split_records(buf, 4, "pgen/igen")
        return np.frombuffer(buf, dtype=_GEN_DT).copy()

    pgens = _parse_gens(chunks["pgen"])
    igens = _parse_gens(chunks["igen"])

    # --- shdr (46 B) ---
    shdr = _split_records(chunks["shdr"], 46, "shdr")
    samples = []
    for i in range(len(shdr) // 46):
        e = shdr[i * 46 : (i + 1) * 46]
        (start, end, lstart, lend, sr) = struct.unpack_from("<IIIII", e, 20)
        key = e[40]
        corr = e[41] - 256 if e[41] > 127 else e[41]
        link, stype = struct.unpack_from("<HH", e, 42)
        samples.append(
            Sf2Sample(
                name=_zstr(e, 20),
                start=start,
                end=end,
                start_loop=lstart,
                end_loop=lend,
                sample_rate=sr,
                original_key=key,
                pitch_correction=corr,
                sample_link=link,
                sample_type=stype,
            )
        )

    # --- smpl: int16 -> float32 ---
    smpl_bytes = chunks["smpl"][: len(chunks["smpl"]) - (len(chunks["smpl"]) % 2)]
    smpl = (np.frombuffer(smpl_bytes, dtype="<i2") / 32767.0).astype(np.float32)

    presets_by_id: dict[tuple[int, int], int] = {}
    for i, pr in enumerate(presets):
        presets_by_id.setdefault((pr.bank, pr.preset), i)

    return SoundFont(
        path=os.path.abspath(p),
        version=version,
        presets=presets,
        instruments=instruments,
        samples=samples,
        pbags=pbags,
        pmods=pmods,
        pgens=pgens,
        ibags=ibags,
        imods=imods,
        igens=igens,
        smpl=smpl,
        presets_by_id=presets_by_id,
    )
