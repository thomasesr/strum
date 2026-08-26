# Separation pipeline

This fork replaces STRUM's single Demucs pass with a two-stage front end.

```
mix.mp3
  │
  ├─ Stage 0  Mel-Band RoFormer karaoke  ──► lead_vocals.wav   → PART VOCALS
  │                                       └► instrumental.wav  (backing vox still in)
  │              (optional 2nd pass)     ──► backing_vocals.wav → HARM2 / HARM3
  │
  └─ Stage 1  6-stem separation on the instrumental
               demucs (htdemucs_6s)  ── or ──  bs_roformer_sw
                 └► drums · bass · guitar · piano · other
```

Both stages degrade to the previous behaviour instead of failing: if a model is
missing or inference errors out, the stage logs a warning and the pipeline
continues with the raw mix / with Demucs.

## Stage 0 — karaoke pre-separation

**Why.** The lead vocal is the loudest and most spectrally dominant source in a
typical mix, and Demucs' vocal branch is markedly weaker than a dedicated
Mel-Band RoFormer. Whatever vocal Demucs fails to pull out lands in `other`,
`guitar` and `piano` — and basic-pitch happily transcribes a sung melody as a
very plausible guitar line. Removing the lead vocal with a specialist model
first is the single cheapest way to kill those phantom notes.

Karaoke models are trained to remove the **lead** vocal only, so backing vocals
stay in `instrumental` by design.

**Trade-off to know.** Those surviving backing vocals still bleed into the
Stage 1 `other` / `guitar` stems — the same failure mode, smaller. Turning on
`backing_split` runs a second, full-vocal RoFormer over `instrumental`, which:

- gives Stage 1 a completely vocal-free mix, and
- yields a clean `backing_vocals.wav` to chart HARM2/HARM3 from.

It costs one extra separation pass. Off by default so the default path matches
the single-pass design; recommended on for anything you intend to ship.

### Backends

| Backend | How | Use when |
|---|---|---|
| `audio-separator` (default) | `pip install audio-separator`, weights auto-download | Default. Zero setup. |
| `msst` | subprocess into a [ZFTurbo MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) checkout | You want becruily / gabox karaoke checkpoints, which are not in the audio-separator registry. |

Known checkpoints:

- `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` — default, in the
  audio-separator registry. Crisp, but sometimes leaves lead-vocal residue.
- [`mel_band_roformer_karaoke_becruily.ckpt`](https://huggingface.co/becruily/mel-band-roformer-karaoke)
  (+ `config_karaoke_becruily.yaml`) — newer; run via the `msst` backend.

## Stage 1 — 6-stem separation

`STRUM_SEPARATOR` picks the model. Both backends emit the same six stem names,
so every downstream stage is unaffected by the choice.

- `demucs` (default) — `htdemucs_6s` through the Demucs Python API.
- `bs_roformer_sw` — [BS-RoFormer SW](https://huggingface.co/jarredou/BS-ROFO-SW-Fixed)
  (`bs_6stem_fixed.ckpt` + `bs_6stem_fixed_config.yaml`) through MSST. Holds up
  better on dense mixes, which matters most for `guitar` and `other`.

The `STRUM_DRUMS_DEMUCS` specialist pass still runs on top of either backend:
the drums models were tuned on `htdemucs_ft` drum stems, so that stem is
re-extracted with Demucs regardless of the Stage 1 choice.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `STRUM_KARAOKE` | `1` | Enable Stage 0. |
| `STRUM_KARAOKE_BACKEND` | `audio-separator` | `audio-separator` or `msst`. |
| `STRUM_KARAOKE_MODEL` | aufr33/viperx karaoke | Registry filename (audio-separator backend). |
| `STRUM_KARAOKE_BACKING_SPLIT` | `0` | Second pass isolating backing vocals. |
| `STRUM_KARAOKE_BACKING_MODEL` | BS-RoFormer ep317 | Full-vocal model for that pass. |
| `STRUM_KARAOKE_CKPT` / `_CFG` | — | Karaoke weights for the `msst` backend. |
| `STRUM_KARAOKE_BACKING_CKPT` / `_CFG` | — | Backing-split weights for the `msst` backend. |
| `STRUM_KARAOKE_MSST` | `STRUM_MSST_DIR` | MSST checkout for Stage 0. |
| `STRUM_SEPARATOR` | `demucs` | `demucs` or `bs_roformer_sw`. |
| `STRUM_ROFORMER_CKPT` / `_CFG` | — | BS-RoFormer SW weights. Required for that backend. |
| `STRUM_ROFORMER_MSST` | `STRUM_MSST_DIR` | MSST checkout for Stage 1. |
| `STRUM_MSST_DIR` | legacy drumsep path | Shared MSST checkout fallback. |

## Examples

Default (karaoke + Demucs):

```bash
pip install -e '.[full]'
python scripts/batch_pipeline.py --input song.mp3 --output out/
```

Recommended quality setup — backing vocals split out, BS-RoFormer SW for stems:

```bash
export STRUM_MSST_DIR=~/Music-Source-Separation-Training
export STRUM_KARAOKE_BACKING_SPLIT=1
export STRUM_SEPARATOR=bs_roformer_sw
export STRUM_ROFORMER_CKPT=~/models/bs_6stem_fixed.ckpt
export STRUM_ROFORMER_CFG=~/models/bs_6stem_fixed_config.yaml
python scripts/batch_pipeline.py --input song.mp3 --output out/
```

Reproduce upstream behaviour exactly:

```bash
STRUM_KARAOKE=0 python scripts/batch_pipeline.py --input song.mp3 --output out/
```

## Caching

Every stage caches into the song's `stems/` folder and short-circuits on a hit,
since reruns during charting are the common case. Delete `stems/karaoke/` to
force Stage 0 to re-run, or `stems/*.wav` for Stage 1.
