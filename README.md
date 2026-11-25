# pysf2

Pure-Python + numpy **SoundFont 2 (`.sf2`) renderer**: parses SF2 2.00/2.01
fonts and renders structured, tick-based MIDI events to offline stereo audio.

**One runtime dependency: `numpy`.** No scipy (the 4-pole filter is a
numpy biquad chain validated against `scipy.signal.lfilter` to <1e-5 abs),
no mido (MIDI *file* I/O stays host-side -- the engine consumes structured
events, see below).

```
pip install pysf2
```

## Features

* **Parser** -- SF2 2.00/2.01, flat *and* nested (`pdtr`/`pdts`) chunk
  layouts, verified byte-for-byte against GeneralUser GS. Uncompressed SF3
  parses; Vorbis-compressed SF3 is out of scope (16-bit PCM only).
* **Zone/region model** -- preset/instrument zone merging per the spec
  (value generators add, range generators intersect, index generators set),
  narrowest-region selection, key/velocity substitution, exclusive classes
  with early release of same-class voices.
* **Renderer (offline)**
  * linearly-interpolated resample with `cumsum` phase integration, so
    vibrato / modulation-LFO pitch modulation is exact;
  * continuous and while-key-held loop wrapping;
  * timecents volume envelope (delay/attack/hold/decay/sustain/release);
  * pitch and volume LFOs with delay, rate, and depth modulators;
  * explicit note-on modulators: all spec sources (key, velocity, pitch
    wheel + its sensitivity, CC palette via bit 7, ...), polarity/direction
    bits, linear/absolute-value transforms;
  * resonant 4-pole low-pass (two cascaded RBJ 2-pole biquads) -- skipped
    entirely when the region cut is the 20 kHz default;
  * key tracking, transpose, tune, velocity -> attenuation, channel volume
    (CC7), pan (cosine law), GM percussion channel 9 -> bank 128
    auto-select.
* **Input model** -- plain dataclasses in raw SMF ticks + a tempo map;
  `TimeMap` does exact piecewise-linear tick<->seconds conversion.

## Quick start

```python
import numpy as np
from pysf2 import (
    ChannelEvent,
    MidiInput,
    MidiNote,
    TempoChange,
    get_soundfont,
    load_soundfont,
    render,
)

# 1. Parse the font (cached by path + mtime + size).
font = load_soundfont("GeneralUser_GS.sf2")

# 2. Structured events (any SMF parser can produce these).
midi = MidiInput(
    ppq=480,
    total_ticks=1920,
    tempo_map=[TempoChange(0, 500_000)],  # 120 BPM
    notes=[
        MidiNote(channel=0, pitch=60, velocity=100, start_tick=0, end_tick=960),
        MidiNote(channel=0, pitch=64, velocity=100, start_tick=960, end_tick=1920),
    ],
    # channel_events: per-channel tick-ordered program/CC/pitch-bend events;
    # empty = GM defaults (program 0, CC7 100, bend range 2 semitones).
)

# 3. Render (a font path works too -- it loads + caches for you).
audio = render(midi, font, sample_rate=44100, tail_sec=2.0)
audio.waveform      # (2, N) float32, channel-first: row 0 = L, row 1 = R
audio.sample_rate   # 44100
mono = audio.to_mono()
```

## API

| Symbol | Purpose |
| --- | --- |
| `load_soundfont(path) -> SoundFont` | Parse one font (no caching). |
| `get_soundfont(path) -> (SoundFont, regions, RegionMatcher)` | Parse + build regions + matcher, LRU-cached (3 fonts) by (path, mtime, size). |
| `render(midi, soundfont, sample_rate=44100, tail_sec=2.0, volume=1.0, include_percussion=True) -> RenderedAudio` | Render; `soundfont` is a `SoundFont` or a path. |
| `MidiInput` / `MidiNote` / `ChannelEvent` / `TempoChange` | The structured input model. |
| `TimeMap` | Exact tick<->seconds mapper for a tempo map. |
| `EV_PROGRAM` / `EV_PITCHBEND` / `TRACKED_CCS` | `ChannelEvent.cc` convention (CC 0..127, 128 = program, 129 = pitch bend). |
| `SoundFont` / `SoundFontParseError` | Parsed font container / validation error. |
| `Region` / `build_regions` / `RegionMatcher` | Region model + fast numpy matcher (advanced use). |

`RenderedAudio` fields: `waveform` ((2, N) float32, channel-first),
`sample_rate` (int); method `to_mono()`.

Notes without a matching preset/region are skipped silently (they would be
silent in a SoundFont too). Output is peak-guarded (scaled to 0.99 if any
sample exceeds 1.0) but never up-normalized: fonts carry their own dynamics.

## v1 simplifications (documented; no audible impact on typical fonts)

* modulation envelope ignored (volume envelope only) -- mod-env
  destinations read 0;
* reverb / chorus sends not implemented (no shared FX buses);
* time-varying filter cut (LFO -> filterCut) uses the note-on value;
* LFO rate/delay modulators (dest 21-24) not applied;
* non-linear modulator curves (concave/convex/switch, bit 10) read as
  linear; secondary (amount) modulator sources read 1;
* poly-pressure and link modulator sources read 0.

## Performance note

The 4-pole RBJ pass is a block-looped direct-form numpy biquad (pure Python
sample loop, ~30 ms per second of audio per *filtered* voice). It runs only
when a region sets a filter cut below ~20 kHz; typical piano fonts skip it
entirely. For offline rendering this is negligible; if you render long,
dense, heavily-filtered material and it becomes a bottleneck, that is the
one place to reach for a compiled filter (the numerics match
`scipy.signal.lfilter` to <1e-5 abs, so such an acceleration is drop-in).

## Development

```
pip install -e . pytest
pytest            # 22 tests: unit + FFT e2e against a generated 2.01 font
```

The test font is built fresh by `tests/_sf2gen.py` (no binary fixture
committed). A GeneralUser GS smoke test runs when `PYSF2_FONT` points to a
real font on disk, and is skipped otherwise (so CI on a bare runner stays
green).

## License

Apache License 2.0 -- see [LICENSE](LICENSE) and [NOTICE](NOTICE).
