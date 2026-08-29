# STRUM Roadmap

## Status

The core pipeline is **production**: drums + guitar + bass + vocals + keys all
generate playable Clone Hero / YARG charts with cross-instrument grid
alignment. Verified on a held-out test set of 9 paired audio + ground-truth
chart pairs (see `docs/ARCHITECTURE.md` and the Performance section in
[README.md](../README.md)).

## Shipped

- ✅ Two-stage drum onset detector + ensemble lane classifier (8 lanes)
- ✅ Six audio-coupled rescue passes (onset / cymbal / tom / drumsep)
- ✅ Hybrid guitar & bass (V2 onset CRNN + Spotify Basic Pitch + fret mapping)
- ✅ Whisper + pYIN vocal pipeline with LRCLIB lyrics
- ✅ Spectral keyboard detection with Pro Keys output
- ✅ Cross-instrument BPM refinement + phase shift + 32nd-note snap with
  per-lane roll detection
- ✅ Difficulty reduction (Expert / Hard / Medium / Easy)
- ✅ Clone Hero / YARG packaging (`notes.mid`, `song.ini`, album art)
- ✅ Backend selection via env vars (`STRUM_GUITAR_BACKEND`,
  `STRUM_BASS_BACKEND`, `STRUM_FRET_MAPPER`, `STRUM_V12C_VARIANT`)
- ✅ Training pipelines for every model with W&B logging

## In Flight

- 🔄 Per-instrument benchmark harness (drums F1 verified; guitar / bass /
  vocals / keys numbers being collected on the GT test set)
- 🔄 Hugging Face Hub model release (`opria123/strum`)
- 🔄 OCTAVE chart-editor integration

## Planned

- ☐ Whitepaper writeup (problem framing, alignment story, results,
  limitations)
- ☐ Streaming inference mode (chunked Demucs + incremental rescue passes)
- ☐ Pro Guitar (string + fret inference, not just 5-fret)
- ☐ Web demo (Hugging Face Space) — drag-and-drop a song, get a chart
- ☐ Genre-specific drum classifier variants (metal, jazz, electronic)
- ☐ Multi-take training data from charter community

## Vocals: lyric synchronisation

Vocal charting currently takes its word timings from Whisper. Those are accurate
to roughly a syllable, which is enough for a playable chart but visibly loose
against the beat on sustained notes.

**Evaluated: `Karaoke-Timed-Lyrics-Qwen3-0.6B`** (and its GGUF quantisations).
Not usable for this. It is a 0.6B *text* model fine-tuned from Qwen3-0.6B: it
takes chat messages and emits text, and consumes no audio at any point. It can
produce plausible-looking timings for a set of lyrics, but nothing ties those
timings to the recording in front of it, so they would be invented rather than
measured. Synchronisation is an alignment problem, and alignment needs the
audio.

Worth trying instead, in rough order of effort:

- ☐ **Forced alignment against fetched lyrics.** `src/lyrics/fetcher.py` already
  retrieves the real words. Aligning known text to audio is a much easier and
  better-posed problem than transcribing it, and it removes Whisper's
  mishearings from the chart entirely. `ctc-forced-aligner` or WhisperX's
  alignment stage are the obvious candidates.
- ☐ **Onset snapping against the isolated vocal.** The karaoke pass already
  produces a clean lead-vocal stem; vocal onsets in it are a strong timing
  signal, and the charter's `dynamic_alignment` does a limited version of this
  already.
- ☐ **Syllable-level alignment**, so held notes get their syllables spread
  across the sustain rather than bunched at the onset.

## Hardware

- **Training**: NVIDIA DGX Spark (GB10, 12 GB)
- **Inference**: any CUDA GPU; CPU works but ~10× slower

## Tracking

W&B project: `strum`. Runs are tagged by model family
(`drums-v14-*`, `onset-classifier-v15-*`, `guitar-v2-*`, etc.).
