"""guitar_basicpitch.py — Spike: Basic Pitch + rule-based fret mapper.

Pipeline:
  guitar.wav  →  Basic Pitch (polyphonic transcription)  →  notes
                                                          ↓
                                          group by onset (≤30 ms window)
                                                          ↓
                            per-song percentile bins →  G/R/Y/B/O
                                                          ↓
                                       Clone Hero notes.mid (PART GUITAR)

Operates on already-separated stems under <pred-dir>/<song>/stems/guitar.wav,
writes notes.mid into <out-dir>/<song>/notes.mid.

Usage:
  python scripts/guitar_basicpitch.py \
      --stems-root /mnt/ml-data/benchmark-pred-v3p1 \
      --out-dir    /mnt/ml-data/benchmark-pred-bp
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import mido

# Suppress basic_pitch's optional-backend warnings
logging.getLogger("root").setLevel(logging.ERROR)

from basic_pitch.inference import predict as _basic_pitch_predict  # noqa: E402

from src.preprocessing.gpu import free_memory, rss_gb  # noqa: E402

# Same name the module uses further down, so this is the same logger object.
_bp_log = logging.getLogger("guitar_basicpitch")


def predict(*args, **kwargs):
    """basic-pitch inference, with the memory it leaves behind reclaimed.

    The guitar stage runs this more than once per song, and TensorFlow does not
    return what it allocates when a call finishes. Stacked on everything PyTorch
    is already holding, that is what has been getting the process OOM-killed, so
    each call reports what it cost and collects afterwards.
    """
    before = rss_gb()
    try:
        return _basic_pitch_predict(*args, **kwargs)
    finally:
        after = rss_gb()
        if before is not None and after is not None:
            _bp_log.info(
                f"  basic-pitch RSS: {before:.2f} -> {after:.2f} GB "
                f"(+{after - before:.2f} GB)"
            )
        free_memory("basic-pitch")

# Optional torch import for learned mapper
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

log = logging.getLogger("guitar_basicpitch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FRET_TO_MIDI = {0: 96, 1: 97, 2: 98, 3: 99, 4: 100}  # G,R,Y,B,O
CHORD_GROUP_SEC = 0.080
DEFAULT_NOTE_DUR_SEC = 0.05
DEFAULT_TEMPO_BPM = 120.0


# ─────────────────────────── Mapping helpers ────────────────────────────────
def group_notes_by_onset(notes, window_sec: float = CHORD_GROUP_SEC):
    """Group Basic Pitch notes into onset clusters.

    Each note tuple: (start, end, pitch, amplitude, pitch_bends).
    Returns list of dicts: {time, pitches: list[int], durations: list[float]}.
    """
    notes = sorted(notes, key=lambda n: n[0])
    onsets: list[dict] = []
    cur_t = -1e9
    cur: dict | None = None
    for start, end, pitch, amp, _ in notes:
        if start - cur_t > window_sec:
            if cur is not None:
                onsets.append(cur)
            cur = {"time": float(start), "pitches": [], "durations": [], "amps": []}
            cur_t = float(start)
        cur["pitches"].append(int(pitch))
        cur["durations"].append(float(end - start))
        cur["amps"].append(float(amp))
    if cur is not None:
        onsets.append(cur)
    return onsets


def build_pitch_to_fret(all_pitches: list[int]) -> dict[int, int]:
    """Map each pitch to a fret 0–4 via *pitch-class* (mod 12) bins.

    Why pitch-class: octave doublings are extremely common on guitar (e.g. a
    power chord plays root + fifth + octave-root). If we bin by absolute
    pitch, those collapse into different frets and inflate chord_fraction
    artificially / destroy real chord structure. By binning the song's
    pitch-class distribution into 5 buckets, octaves naturally co-locate.
    """
    if not all_pitches:
        return {}
    arr = np.asarray(all_pitches)
    pcs = arr % 12
    # Use pitch-class popularity to define ordering: rarest → highest fret.
    # Then within the song, low pitches go to low frets via PC ranking on
    # mean MIDI value of each PC (so root tonic stays on G/R).
    pc_to_mean = {pc: float(arr[pcs == pc].mean()) for pc in set(pcs.tolist())}
    sorted_pcs = sorted(pc_to_mean.keys(), key=lambda pc: pc_to_mean[pc])
    # Quintile-bin the sorted-by-mean PCs across 5 frets
    n = len(sorted_pcs)
    pc_to_fret = {}
    for i, pc in enumerate(sorted_pcs):
        pc_to_fret[pc] = min(4, int(i * 5 / max(1, n)))
    return {int(p): pc_to_fret[int(p) % 12] for p in set(arr.tolist())}


def classify_onset(pitches: list[int], amps: list[float]) -> tuple[int, str]:
    """Return (root_midi_pitch, kind) for an onset cluster.

    kind ∈ {"single", "power", "chord"}:
      - single: one PC (octaves allowed) → lead note
      - power : two PCs separated by 7 semitones (perfect 5th, any octave) → power chord
      - chord : 2+ PCs not forming a power chord → other polyphony
    Root is the lowest pitch in the cluster.
    """
    if not pitches:
        return 0, "single"
    root = min(pitches)
    pcs = {p % 12 for p in pitches}
    if len(pcs) == 1:
        return root, "single"
    root_pc = root % 12
    fifth_pc = (root_pc + 7) % 12
    if fifth_pc in pcs and len(pcs) <= 3:
        # Power chord (root + 5th, possibly + octave root). Allow up to 3 PCs
        # because the 5th's octave often appears too.
        return root, "power"
    return root, "chord"


def build_root_pc_to_fret(onsets: list[dict]) -> dict[int, int]:
    """Map root pitch-classes to frets 0–4.

    Strategy: among all onsets that classify as power/chord/single, take each
    cluster's lowest pitch as its root. Rank the *root pitch-classes* by their
    mean MIDI value across the song (lowest → fret 0, highest → fret 4). This
    keeps the bass-heavy "low strings" feel — drop-tuning rumble lands on G/R
    and lead notes drift toward B/O. Falls back gracefully when fewer than 5
    PCs are present.
    """
    from collections import defaultdict
    pc_pitches: dict[int, list[int]] = defaultdict(list)
    for ons in onsets:
        if not ons["pitches"]:
            continue
        root = min(ons["pitches"])
        pc_pitches[root % 12].append(root)
    if not pc_pitches:
        return {}
    pc_mean = {pc: float(np.mean(ps)) for pc, ps in pc_pitches.items()}
    # Frequency tiebreaker keeps rare single-occurrence PCs from grabbing fret 0
    pc_count = {pc: len(ps) for pc, ps in pc_pitches.items()}
    # Rank: primary by mean pitch ascending, secondary by count descending
    ranked = sorted(pc_mean.keys(), key=lambda pc: (pc_mean[pc], -pc_count[pc]))
    pc_to_fret = {}
    n = len(ranked)
    for i, pc in enumerate(ranked):
        pc_to_fret[pc] = min(4, int(i * 5 / max(1, n)))
    return pc_to_fret


def onsets_to_chart_powerchord(
    onsets: list[dict],
    root_pc_to_fret: dict[int, int],
    global_amp_pct: float = 25.0,
    all_amps: list[float] | None = None,
    chord_fret_gap: int = 1,
):
    """Power-chord-aware mapper.

    - single → 1 fret  (root_pc_to_fret[root_pc])
    - power  → 2 adjacent frets (f, f+1) capped at fret 4
    - chord  → 2 frets with `chord_fret_gap` separation (f, f+gap) capped
    """
    if all_amps:
        amp_floor = float(np.percentile(np.asarray(all_amps), global_amp_pct))
    else:
        amp_floor = 0.0
    out = []
    for ons in onsets:
        pitches = ons["pitches"]
        amps = ons["amps"]
        durs = ons["durations"]
        if not pitches:
            continue
        if max(amps) < amp_floor:
            continue
        root, kind = classify_onset(pitches, amps)
        rpc = root % 12
        if rpc not in root_pc_to_fret:
            # Rare PC — fall back to fret 4
            f0 = 4
        else:
            f0 = root_pc_to_fret[rpc]
        if kind == "single":
            frets = (f0,)
        elif kind == "power":
            f1 = min(4, f0 + 1)
            if f1 == f0:  # f0 was already 4; shift down
                f0, f1 = 3, 4
            frets = (f0, f1)
        else:  # chord
            f1 = min(4, f0 + chord_fret_gap)
            if f1 == f0:
                f0 = max(0, f0 - chord_fret_gap)
            frets = (f0, f1) if f0 != f1 else (f0,)
        max_dur = max(durs)
        out.append((float(ons["time"]), frets, float(max_dur)))
    return out


# --- Learned mapper inference ----------------------------------------------
def _featurize_onsets_for_learned(onsets: list[dict], context: int = 2) -> np.ndarray:
    """Mirror of scripts/build_mapper_dataset.py::featurize_onsets."""
    N = len(onsets)
    per = 19
    feats = np.zeros((N, per), dtype=np.float32)
    for i, ons in enumerate(onsets):
        pitches = ons["pitches"]
        amps = ons["amps"]
        durs = ons["durations"]
        if not pitches:
            continue
        pcs = np.array([p % 12 for p in pitches])
        max_per_pc = np.zeros(12, dtype=np.float32)
        for pc, a in zip(pcs, amps):
            if a > max_per_pc[pc]:
                max_per_pc[pc] = a
        norm = max_per_pc.max()
        if norm > 0:
            max_per_pc = max_per_pc / norm
        feats[i, :12] = max_per_pc
        feats[i, 12] = np.log1p(len(set(pitches)))
        root = min(pitches)
        feats[i, 13] = root / 127.0
        feats[i, 14] = (max(pitches) - min(pitches)) / 24.0
        feats[i, 15] = float(max(amps))
        feats[i, 16] = min(2.0, float(max(durs))) / 2.0
        pcset = set(pcs.tolist())
        is_power = ((root % 12 + 7) % 12) in pcset and 2 <= len(pcset) <= 3
        is_chord = len(pcset) >= 2 and not is_power
        feats[i, 17] = float(is_power)
        feats[i, 18] = float(is_chord)
    F = per * (1 + 2 * context)
    out = np.zeros((N, F), dtype=np.float32)
    for i in range(N):
        col = 0
        for off in range(-context, context + 1):
            j = i + off
            if 0 <= j < N:
                out[i, col:col + per] = feats[j]
            col += per
    return out


class _LearnedMapper:
    def __init__(self, ckpt_path: str):
        if not _HAS_TORCH:
            raise RuntimeError("torch required for --mapper learned")
        import torch as _t
        import torch.nn as _nn
        ckpt = _t.load(ckpt_path, map_location="cpu", weights_only=False)
        in_dim = int(ckpt["in_dim"])
        hidden = int(ckpt.get("hidden", 256))
        net = _nn.Sequential(
            _nn.Linear(in_dim, hidden), _nn.GELU(), _nn.Dropout(0.0),
            _nn.Linear(hidden, hidden), _nn.GELU(), _nn.Dropout(0.0),
            _nn.Linear(hidden, hidden), _nn.GELU(), _nn.Dropout(0.0),
            _nn.Linear(hidden, 5),
        )
        # Trained state_dict has keys like "net.0.weight"; wrap to load.
        wrapper = _nn.Module()
        wrapper.net = net
        wrapper.load_state_dict(ckpt["model_state"])
        self.net = net
        self.net.eval()
        self.mu = ckpt["feature_mean"]
        self.sd = ckpt["feature_std"]
        self.device = "cuda" if _t.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self._t = _t

    def predict_proba(self, onsets):
        if not onsets:
            return np.zeros((0, 5), dtype=np.float32)
        X = _featurize_onsets_for_learned(onsets)
        X = (X - self.mu) / self.sd
        with self._t.no_grad():
            x = self._t.from_numpy(X.astype(np.float32)).to(self.device)
            p = self._t.sigmoid(self.net(x)).cpu().numpy()
        return p

    def predict(self, onsets, threshold: float = 0.5):
        p = self.predict_proba(onsets)
        out = []
        for row in p:
            frets = tuple(i for i, v in enumerate(row) if v >= threshold)
            if not frets:
                frets = (int(row.argmax()),)
            out.append(frets)
        return out


def onsets_to_chart_learned(onsets, mapper: "_LearnedMapper", threshold: float = 0.5,
                            global_amp_pct: float = 25.0, all_amps=None,
                            use_viterbi: bool = False,
                            viterbi_kwargs: dict | None = None):
    if all_amps:
        amp_floor = float(np.percentile(np.asarray(all_amps), global_amp_pct))
    else:
        amp_floor = 0.0
    if use_viterbi:
        from viterbi_fret_decode import viterbi_decode
        P = mapper.predict_proba(onsets)
        times = np.array([o["time"] for o in onsets], dtype=np.float32)
        kw = viterbi_kwargs or {}
        preds = viterbi_decode(P, times=times, **kw)
    else:
        preds = mapper.predict(onsets, threshold=threshold)
    out = []
    for ons, frets in zip(onsets, preds):
        if not ons["pitches"]:
            continue
        if max(ons["amps"]) < amp_floor:
            continue
        if not frets:
            continue
        out.append((float(ons["time"]), tuple(sorted(frets)), float(max(ons["durations"]))))
    return out


def onsets_to_chart(
    onsets: list[dict],
    pitch_to_fret: dict[int, int],
    min_chord_amp_frac: float = 0.50,
    global_amp_pct: float = 25.0,
    all_amps: list[float] | None = None,
):
    """Convert grouped onsets into list of (time_sec, frets, sustain_sec).

    min_chord_amp_frac: a chord-companion pitch must be at least this fraction
        of the loudest pitch in the same onset to count.
    global_amp_pct: drop entire onsets whose loudest pitch is below this
        percentile of all amplitudes in the song (suppresses string-ringing
        and overtone artifacts).
    """
    if all_amps:
        amp_floor = float(np.percentile(np.asarray(all_amps), global_amp_pct))
    else:
        amp_floor = 0.0
    out = []
    for ons in onsets:
        pitches = ons["pitches"]
        amps = ons["amps"]
        durs = ons["durations"]
        if not pitches:
            continue
        loudest = max(amps)
        if loudest < amp_floor:
            continue
        thr = max(min_chord_amp_frac * loudest, amp_floor * 0.5)
        frets_set = set()
        max_dur = 0.0
        for p, a, d in zip(pitches, amps, durs):
            if p not in pitch_to_fret:
                continue
            if a < thr:
                continue
            frets_set.add(pitch_to_fret[p])
            if d > max_dur:
                max_dur = d
        if not frets_set:
            best_idx = int(np.argmax(amps))
            p = pitches[best_idx]
            if p in pitch_to_fret:
                frets_set.add(pitch_to_fret[p])
                max_dur = durs[best_idx]
        if not frets_set:
            continue
        frets = sorted(frets_set)
        if len(frets) > 3:
            frets = frets[:3]
        out.append((float(ons["time"]), tuple(frets), float(max_dur)))
    return out


# ─────────────────────────── MIDI writer ────────────────────────────────────
def write_chart_midi(
    events: list[tuple[float, tuple[int, ...], float]],
    output_path: Path,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    note_duration_sec: float = DEFAULT_NOTE_DUR_SEC,
    track_name: str = "PART GUITAR",
) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name))
    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us))

    tpb = mid.ticks_per_beat
    sec_to_tick = lambda s: int(round(s * tempo_bpm / 60.0 * tpb))

    msgs: list[tuple[int, bool, int]] = []
    for t, frets, _sus in events:
        on_t = sec_to_tick(t)
        off_t = sec_to_tick(t + note_duration_sec)
        for f in frets:
            note = FRET_TO_MIDI[f]
            msgs.append((on_t, True, note))
            msgs.append((off_t, False, note))
    msgs.sort(key=lambda x: (x[0], x[1]))  # off before on at same tick

    last_t = 0
    for tick, is_on, note in msgs:
        delta = max(0, tick - last_t)
        track.append(mido.Message(
            "note_on" if is_on else "note_off",
            note=note, velocity=100 if is_on else 0, time=delta,
        ))
        last_t = tick

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))


# ─────────────────────── Pipeline integration ───────────────────────────────
# Module-level cache so the learned mapper isn't reloaded per song when called
# repeatedly from batch_pipeline.
_GUITAR_MAPPER_CACHE: "dict[str, _LearnedMapper]" = {}


def transcribe_guitar_basicpitch(
    audio_path: "Path",
    tempo_bpm: float = 120.0,
    onset_threshold: float = 0.7,
    frame_threshold: float = 0.4,
    min_note_length: int = 16,
    global_amp_pct: float = 25.0,
    note_duration_ms: float = 100.0,
    learned_ckpt: str = "checkpoints/fret_mapper_v4.pt",
    use_viterbi: bool = True,
    viterbi_kwargs: dict | None = None,
):
    """Transcribe guitar to a `GuitarChart`.

    Drop-in for `batch_pipeline.transcribe_guitar` when
    `STRUM_GUITAR_BACKEND=basicpitch`. Uses Basic Pitch + the learned MLP
    fret mapper (`fret_mapper_v4.pt`) + Viterbi smoothing by default.
    """
    from src.inference.guitar_bass import GuitarChart, GuitarNote, GuitarChord

    if learned_ckpt not in _GUITAR_MAPPER_CACHE:
        _GUITAR_MAPPER_CACHE[learned_ckpt] = _LearnedMapper(learned_ckpt)
    mapper = _GUITAR_MAPPER_CACHE[learned_ckpt]

    _, _, notes = predict(
        str(audio_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
    )
    onsets = group_notes_by_onset(notes)
    all_amps = [a for o in onsets for a in o["amps"]]
    events = onsets_to_chart_learned(
        onsets, mapper,
        global_amp_pct=global_amp_pct,
        all_amps=all_amps,
        use_viterbi=use_viterbi,
        viterbi_kwargs=viterbi_kwargs or dict(
            fret_change_w=0.20, center_jump_w=0.03,
            chord_switch_w=0.10, same_state_bonus=0.10, time_decay_tau=0.8,
        ),
    )

    chart = GuitarChart(tempo_bpm=float(tempo_bpm), instrument="guitar")
    for t, frets, _sus in events:
        if len(frets) >= 2:
            chart.chords.append(GuitarChord(
                time_ms=t * 1000.0,
                frets=list(frets),
                duration_ms=note_duration_ms,
            ))
        else:
            chart.notes.append(GuitarNote(
                time_ms=t * 1000.0,
                fret=int(frets[0]),
                duration_ms=note_duration_ms,
            ))
    return chart


# ─────────────────────────── Per-song processing ────────────────────────────
def process_song(
    guitar_wav: Path,
    out_midi: Path,
    onset_threshold: float = 0.7,
    frame_threshold: float = 0.4,
    min_note_length: int = 16,  # ~85ms
    min_chord_amp_frac: float = 0.50,
    global_amp_pct: float = 25.0,
    track_name: str = "PART GUITAR",
    mapper: str = "power-chord",
    chord_fret_gap: int = 1,
    learned_ckpt: str | None = None,
    learned_threshold: float = 0.5,
    use_viterbi: bool = False,
    viterbi_kwargs: dict | None = None,
    _learned_mapper: "_LearnedMapper | None" = None,
):
    t0 = time.time()
    _, _, notes = predict(
        str(guitar_wav),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
    )
    bp_time = time.time() - t0

    onsets = group_notes_by_onset(notes)
    all_amps = [a for o in onsets for a in o["amps"]]
    if mapper == "learned":
        if _learned_mapper is None:
            if not learned_ckpt:
                raise ValueError("--learned-ckpt required for mapper=learned")
            _learned_mapper = _LearnedMapper(learned_ckpt)
        events = onsets_to_chart_learned(
            onsets, _learned_mapper,
            threshold=learned_threshold,
            global_amp_pct=global_amp_pct,
            all_amps=all_amps,
            use_viterbi=use_viterbi,
            viterbi_kwargs=viterbi_kwargs,
        )
    elif mapper == "power-chord":
        rpc2f = build_root_pc_to_fret(onsets)
        events = onsets_to_chart_powerchord(
            onsets, rpc2f,
            global_amp_pct=global_amp_pct,
            all_amps=all_amps,
            chord_fret_gap=chord_fret_gap,
        )
    else:
        all_pitches = [p for o in onsets for p in o["pitches"]]
        p2f = build_pitch_to_fret(all_pitches)
        events = onsets_to_chart(
            onsets, p2f,
            min_chord_amp_frac=min_chord_amp_frac,
            global_amp_pct=global_amp_pct,
            all_amps=all_amps,
        )

    write_chart_midi(events, out_midi, track_name=track_name)

    n_chord = sum(1 for _, fr, _ in events if len(fr) >= 2)
    log.info(
        f"  {guitar_wav.parent.parent.name}: bp={bp_time:.1f}s  "
        f"raw_notes={len(notes)}  onsets={len(events)}  "
        f"chord_frac={n_chord/max(1,len(events)):.2f}  →  {out_midi}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-root", required=True, type=Path,
                    help="Dir containing <song>/stems/guitar.wav subdirs")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--track", default="PART GUITAR",
                    choices=["PART GUITAR", "PART BASS"])
    ap.add_argument("--onset-threshold", type=float, default=0.7)
    ap.add_argument("--frame-threshold", type=float, default=0.4)
    ap.add_argument("--min-note-length", type=int, default=16)
    ap.add_argument("--min-chord-amp-frac", type=float, default=0.50)
    ap.add_argument("--global-amp-pct", type=float, default=25.0)
    ap.add_argument("--stem-name", default="guitar.wav")
    ap.add_argument("--mapper", default="power-chord",
                    choices=["power-chord", "pc-quintile", "learned"])
    ap.add_argument("--chord-fret-gap", type=int, default=1)
    ap.add_argument("--learned-ckpt", type=str, default=None)
    ap.add_argument("--learned-threshold", type=float, default=0.5)
    ap.add_argument("--use-viterbi", action="store_true", help="Apply Viterbi smoothing on learned mapper outputs")
    ap.add_argument("--viterbi-fret-change-w", type=float, default=0.20)
    ap.add_argument("--viterbi-center-jump-w", type=float, default=0.03)
    ap.add_argument("--viterbi-chord-switch-w", type=float, default=0.10)
    ap.add_argument("--viterbi-same-bonus", type=float, default=0.10)
    ap.add_argument("--viterbi-tau", type=float, default=0.8)
    args = ap.parse_args()

    songs = sorted([d for d in args.stems_root.iterdir() if d.is_dir()])
    log.info(f"Processing {len(songs)} songs from {args.stems_root}")

    # Load learned mapper once, reuse across songs
    learned_mapper = None
    if args.mapper == "learned":
        if not args.learned_ckpt:
            raise SystemExit("--learned-ckpt required for --mapper learned")
        log.info(f"Loading learned mapper from {args.learned_ckpt}")
        learned_mapper = _LearnedMapper(args.learned_ckpt)

    for song_dir in songs:
        guitar_wav = song_dir / "stems" / args.stem_name
        if not guitar_wav.exists():
            log.warning(f"  SKIP {song_dir.name}: missing {guitar_wav}")
            continue
        out_midi = args.out_dir / song_dir.name / "notes.mid"
        try:
            process_song(
                guitar_wav, out_midi,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                min_note_length=args.min_note_length,
                min_chord_amp_frac=args.min_chord_amp_frac,
                global_amp_pct=args.global_amp_pct,
                track_name=args.track,
                mapper=args.mapper,
                chord_fret_gap=args.chord_fret_gap,
                learned_ckpt=args.learned_ckpt,
                learned_threshold=args.learned_threshold,
                use_viterbi=args.use_viterbi,
                viterbi_kwargs=dict(
                    fret_change_w=args.viterbi_fret_change_w,
                    center_jump_w=args.viterbi_center_jump_w,
                    chord_switch_w=args.viterbi_chord_switch_w,
                    same_state_bonus=args.viterbi_same_bonus,
                    time_decay_tau=args.viterbi_tau,
                ),
                _learned_mapper=learned_mapper,
            )
        except Exception as e:
            log.error(f"  FAIL {song_dir.name}: {e}")


if __name__ == "__main__":
    main()
