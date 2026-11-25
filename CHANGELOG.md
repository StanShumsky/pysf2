# Changelog

## 0.1.0 (2026-11-25)

Initial release.

* SF2 2.00/2.01 parser (flat and nested `pdta`/`pdtr`/`pdts` layouts,
  16-bit PCM; uncompressed SF3 parses, Vorbis SF3 does not).
* Zone/region model: preset/instrument zone merging, narrowest-region
  selection, key/velocity substitution, exclusive classes with early release.
* Offline renderer: timecents volume envelope, pitch/volume LFOs with exact
  phase integration, note-on modulators (CC palette, pitch-wheel forms,
  polarity/direction, linear/abs transforms), RBJ-style 4-pole resonant
  low-pass (numpy biquad chain, validated against `scipy.signal.lfilter` to
  <1e-5 abs), cosine pan law, GM channel-9 -> bank-128 auto-select.
* Structured tick-based input model (`MidiNote` / `ChannelEvent` /
  `TempoChange` / `TimeMap`) -- no MIDI file I/O in the library.
* Runtime dependency: numpy only.
