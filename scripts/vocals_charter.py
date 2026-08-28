"""
Vocals charter for Clone Hero/YARG using Whisper for lyrics and Basic Pitch for pitch.

Clone Hero Vocals Format:
- MIDI notes 36-84 for pitch (C2-C6, maps to singable range)
- Lyric events as MIDI text events (lyrics meta messages)
- Phrase markers: note 105 = phrase start, note 106 = phrase end
- HARM1/HARM2 tracks for harmonies
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tempfile

import numpy as np
import torch
import librosa
import mido
import pyphen

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lyrics.fetcher import fetch_lyrics, extract_artist_title_from_path, LyricsResult, SyncedLyric

logger = logging.getLogger(__name__)

def _load_asr(model_name: str, device: str, compute_type: str = ""):
    """Load the speech model, preferring faster-whisper.

    faster-whisper runs the same weights through CTranslate2 at roughly a third
    of the memory and several times the speed. That matters here: this stage
    starts with PyTorch and TensorFlow already resident, and the run has been
    OOM-killed at exactly this point.

    Returns (backend_name, model). Falls back to openai-whisper when
    faster-whisper is not installed, so existing environments keep working.
    """
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        import whisper

        logger.info(f"Loading Whisper model (openai-whisper): {model_name}")
        return "openai-whisper", whisper.load_model(model_name, device=device)

    # float16 on GPU, int8 on CPU: the defaults faster-whisper is tuned for.
    ct = compute_type or ("float16" if dev == "cuda" else "int8")
    logger.info(f"Loading Whisper model (faster-whisper): {model_name} [{dev}/{ct}]")
    return "faster-whisper", WhisperModel(model_name, device=dev, compute_type=ct)


def _iter_words(backend: str, model, audio_path: str, language: str = "en"):
    """Yield (word, start, end) from either backend.

    The two APIs differ enough to be worth normalising once here rather than
    branching at every use: faster-whisper streams dataclass segments, while
    openai-whisper returns nested dicts.
    """
    lang = language or None
    if backend == "faster-whisper":
        segments, _info = model.transcribe(
            str(audio_path), word_timestamps=True, language=lang
        )
        for segment in segments:
            for w in (segment.words or []):
                yield w.word, float(w.start), float(w.end)
        return

    result = model.transcribe(str(audio_path), word_timestamps=True, language=lang)
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            yield (
                w.get("word", ""),
                float(w.get("start") or 0.0),
                float(w.get("end") or 0.0),
            )



@dataclass
class VocalNote:
    """A single vocal note with pitch, timing, and optional lyric."""
    start_time: float  # seconds
    end_time: float    # seconds
    midi_pitch: int    # MIDI note number (36-84 for vocals, 0 = pitchless)
    lyric: Optional[str] = None
    is_harmony: bool = False
    is_pitchless: bool = False  # Talky/rap/scream - no pitch matching required
    connects_to_next: bool = False  # Draw smooth line to next note


@dataclass 
class VocalPhrase:
    """A phrase of vocal notes (breathing break between phrases)."""
    start_time: float
    end_time: float
    notes: list[VocalNote]


class VocalsCharter:
    """
    Transcribes vocals from audio using Whisper for lyrics and Basic Pitch for pitch.
    """
    
    # Clone Hero vocal range: C2 (36) to C6 (84)
    VOCAL_MIDI_MIN = 36
    VOCAL_MIDI_MAX = 84
    PITCHLESS_NOTE = 0  # Special pitch for talky/rap sections
    
    # Phrase gap threshold (seconds) - if gap > this, start new phrase
    PHRASE_GAP_THRESHOLD = 1.5
    
    # Max words per phrase - force break even without gap (like reference charts)
    MAX_WORDS_PER_PHRASE = 8
    
    # Connection threshold - notes closer than this are connected (smooth transition)
    CONNECTION_THRESHOLD = 0.15  # seconds
    
    def __init__(
        self,
        whisper_model: str = os.environ.get("STRUM_WHISPER_MODEL", "medium"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        harmony_threshold: float = 0.3,  # Harmony volume relative to lead
        pitch_change_threshold: float = 5.0,  # Semitones to trigger syllable split (raised — only true big leaps split)
        fetch_lyrics_online: bool = True,  # Try web sources before Whisper
        timing_offset: float = 0.0,  # Static offset disabled — dynamic alignment handles latency
        dynamic_alignment: bool = True,  # Enable onset-based dynamic alignment
        alignment_tolerance: float = 0.06,
        compute_type: str = os.environ.get("STRUM_WHISPER_COMPUTE", ""),
        language: str = os.environ.get("STRUM_WHISPER_LANGUAGE", "en"),  # Max time to shift a word to reach onset (was 0.15 — caused mis-snaps to nearby breaths/harmonics)
    ):
        self.whisper_model_name = whisper_model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.harmony_threshold = harmony_threshold
        self.pitch_change_threshold = pitch_change_threshold
        self.fetch_lyrics_online = fetch_lyrics_online
        self.timing_offset = timing_offset  # Base offset (fine-tuned by dynamic alignment)
        self.dynamic_alignment = dynamic_alignment
        self.alignment_tolerance = alignment_tolerance
        
        self._whisper_model = None
        self._whisper_backend = ""
        self._hyphenator = pyphen.Pyphen(lang='en')
        self._cached_onsets = None  # Cache vocal onsets for reuse
        
        # Pitchless detection thresholds (very strict - only for truly unpitched)
        # Only mark as pitchless if essentially NO usable pitch detected
        self.pitchless_confidence_threshold = 0.08  # Extremely low confidence
        self.pitchless_variance_threshold = 8.0  # Very high variance (definite speech)
        self.pitchless_voiced_ratio_threshold = 0.15  # Less than 15% voiced frames
        
    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper_model is None:
            self._whisper_backend, self._whisper_model = _load_asr(
                self.whisper_model_name, self.device, self.compute_type
            )
        return self._whisper_model
    
    def transcribe_lyrics(self, audio_path: str) -> list[dict]:
        """
        Transcribe lyrics with word-level timestamps using Whisper.
        
        Returns list of word segments:
        [
            {
                'word': 'hello',
                'start': 0.5,
                'end': 0.8
            },
            ...
        ]
        """
        logger.info(f"Transcribing lyrics from: {audio_path}")
        
        model = self._load_whisper()
        
        # Transcribe with word timestamps
        raw_words = _iter_words(
            self._whisper_backend, model, audio_path, self.language
        )
        
        # Extract words with timestamps and apply timing offset
        words = []
        for word, w_start, w_end in raw_words:
            # Apply timing offset to compensate for detection latency; a
            # negative offset shifts words earlier, to match visual timing.
            start = max(0, w_start + self.timing_offset)
            end = max(start + 0.01, w_end + self.timing_offset)
            words.append({'word': word.strip(), 'start': start, 'end': end})
        
        logger.info(f"Transcribed {len(words)} words (timing offset: {self.timing_offset:+.3f}s)")
        
        # Filter out harmony/layer duplicates - same or similar words detected close together
        words = self._filter_harmony_duplicates(words)
        
        return words
    
    def _filter_harmony_duplicates(self, words: list[dict], time_threshold: float = 0.25) -> list[dict]:
        """
        Filter out duplicate words caused by harmonies/backing vocals.
        
        When Whisper hears both lead and backing vocals singing the same word
        at slightly different times, it may transcribe both. This keeps only
        the first occurrence of similar words within a time window.
        
        Args:
            words: List of word dicts
            time_threshold: Max time gap (seconds) to consider as duplicate (default 250ms)
            
        Returns:
            Filtered list with harmony duplicates removed
        """
        if len(words) <= 1:
            return words
        
        import re
        
        def clean_word(w):
            """Normalize word for comparison."""
            return re.sub(r'[^\w]', '', w.lower())
        
        def words_similar(w1, w2):
            """Check if two words are similar enough to be duplicates."""
            c1, c2 = clean_word(w1), clean_word(w2)
            if not c1 or not c2:
                return False
            # Exact match
            if c1 == c2:
                return True
            # One contains the other (e.g., "in" vs "in,")
            if c1 in c2 or c2 in c1:
                return True
            # First 3 chars match (handles slight Whisper mishearing)
            if len(c1) >= 3 and len(c2) >= 3 and c1[:3] == c2[:3]:
                return True
            return False
        
        filtered = [words[0]]
        removed_count = 0
        
        for i in range(1, len(words)):
            current = words[i]
            prev = filtered[-1]
            
            # Check if this word is a duplicate of the previous one
            time_gap = current['start'] - prev['start']
            
            if time_gap < time_threshold and words_similar(current['word'], prev['word']):
                # This looks like a harmony duplicate - skip it
                # Keep the earlier one (usually the lead vocal)
                removed_count += 1
                continue
            
            # Also check if current word overlaps significantly with previous
            # (backing vocal starting mid-word)
            if current['start'] < prev['end'] - 0.05:  # More than 50ms overlap
                if words_similar(current['word'], prev['word']):
                    removed_count += 1
                    continue
            
            filtered.append(current)
        
        if removed_count > 0:
            logger.info(f"Filtered {removed_count} harmony/layer duplicate words")
        
        return filtered
    
    def detect_vocal_onsets(self, audio_path: str) -> np.ndarray:
        """
        Detect vocal onset times in the audio.
        
        Uses a combination of spectral flux and RMS energy to find
        when vocals actually start - much more accurate than Whisper timing.
        
        Returns:
            Array of onset times in seconds
        """
        if self._cached_onsets is not None:
            return self._cached_onsets
        
        logger.info("Detecting vocal onsets for dynamic alignment...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Use onset detection optimized for vocals
        # Combine multiple onset detection methods for robustness
        
        # 1. Spectral flux (good for detecting new sounds)
        onset_env_sf = librosa.onset.onset_strength(
            y=y, sr=sr, 
            hop_length=512,
            aggregate=np.median,  # More robust to noise
            n_mels=128,
            fmin=80,   # Focus on vocal range
            fmax=4000  # Upper vocal harmonics
        )
        
        # 2. RMS energy (good for detecting amplitude changes)
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        # Compute derivative of RMS (energy onset)
        rms_diff = np.diff(rms, prepend=rms[0])
        rms_diff = np.maximum(0, rms_diff)  # Only positive changes (onsets, not offsets)
        
        # Normalize both
        onset_env_sf = onset_env_sf / (onset_env_sf.max() + 1e-8)
        rms_diff = rms_diff / (rms_diff.max() + 1e-8)
        
        # Combine (weight spectral flux more as it's better for vocals)
        combined_onset = 0.7 * onset_env_sf + 0.3 * rms_diff[:len(onset_env_sf)]
        
        # Peak picking with backtracking for precise onset times
        onsets_frames = librosa.onset.onset_detect(
            onset_envelope=combined_onset,
            sr=sr,
            hop_length=512,
            backtrack=True,  # Find the actual start, not the peak
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.07,  # Slightly higher threshold to avoid harmonics/doubles
            wait=6  # Minimum frames between onsets (~140ms) - filters layer doubles
        )
        
        # Convert to times
        onset_times = librosa.frames_to_time(onsets_frames, sr=sr, hop_length=512)
        
        # Additional filtering: remove onsets that are too close together
        # These are likely vocal layers/harmonies, not separate words
        min_onset_gap = 0.12  # 120ms minimum between onsets
        if len(onset_times) > 1:
            filtered_onsets = [onset_times[0]]
            for t in onset_times[1:]:
                if t - filtered_onsets[-1] >= min_onset_gap:
                    filtered_onsets.append(t)
            onset_times = np.array(filtered_onsets)
            logger.info(f"After layer filtering: {len(onset_times)} onsets (removed {len(onsets_frames) - len(onset_times)} layer duplicates)")
        
        logger.info(f"Detected {len(onset_times)} vocal onsets")
        self._cached_onsets = onset_times
        return onset_times
    
    def align_words_to_onsets(
        self,
        words: list[dict],
        onset_times: np.ndarray,
        tolerance: float = None
    ) -> list[dict]:
        """
        Dynamically align each word to the nearest detected vocal onset.
        
        This fixes variable Whisper latency by snapping each word's start time
        to when the vocal actually begins in the audio.
        
        Also enforces strict sequential ordering - words cannot overlap
        and each word's end time is capped at the next word's start.
        
        Args:
            words: List of word dicts with 'word', 'start', 'end'
            onset_times: Array of detected vocal onset times
            tolerance: Max time to shift a word (default: self.alignment_tolerance)
            
        Returns:
            Words with adjusted timing, strictly sequential
        """
        if tolerance is None:
            tolerance = self.alignment_tolerance
        
        if len(onset_times) == 0:
            logger.warning("No vocal onsets detected, skipping dynamic alignment")
            return words
        
        aligned_words = []
        shifts = []
        used_onsets = set()  # Track which onsets we've already assigned

        for i, word in enumerate(words):
            original_start = word['start']
            original_end = word['end']
            original_duration = original_end - original_start

            # Find nearest UNUSED onset within tolerance window
            search_start = original_start - tolerance
            search_end = original_start + tolerance

            # Get candidates that haven't been used
            candidates = []
            for j, t in enumerate(onset_times):
                if j not in used_onsets and search_start <= t <= search_end:
                    candidates.append((j, t, abs(t - original_start)))

            if candidates:
                # Sort by distance and pick closest
                candidates.sort(key=lambda x: x[2])
                best_idx, best_onset, best_dist = candidates[0]
                # SAFETY: only snap if the chosen onset is clearly isolated
                # (no other candidate within ~half the tolerance). When there
                # are multiple onsets nearby (breaths, harmonics, layers) we
                # have no way to pick the right one and snapping makes things
                # worse \u2014 trust Whisper instead.
                if len(candidates) > 1 and candidates[1][2] < tolerance * 0.6:
                    new_start = original_start
                    shifts.append(0)
                else:
                    used_onsets.add(best_idx)
                    new_start = best_onset
                    shifts.append(new_start - original_start)
            else:
                # No onset nearby - keep original timing
                new_start = original_start
                shifts.append(0)

            # Whisper often extends a word's end-time to fill the gap until
            # the next word, which historically caused drift. We protect
            # against word-to-word over-extension via the strict trim-prev
            # pass below (clips prev.end to next.start - min_gap). For
            # phrase-ending notes (no nearby next word), preserve Whisper's
            # duration up to a moderate cap so sustained held-out notes
            # actually ring out, but bounded so they don't trail forever.
            # 2.5s covers typical sung syllables (incl. long phrase ends);
            # earlier 4.0s was perceptibly too long on rapid songs.
            new_end = new_start + min(original_duration, 2.5)

            aligned_words.append({
                'word': word['word'],
                'start': new_start,
                'end': new_end,
            })

        # CRITICAL: enforce strict sequential ordering by ALWAYS TRIMMING
        # the previous word. Never push the current word forward — that's
        # what caused song-long drift. If a word's snapped start lands before
        # the previous word's end, the previous word was over-extended
        # (Whisper bug) and we should clip it.
        for i in range(1, len(aligned_words)):
            prev = aligned_words[i - 1]
            curr = aligned_words[i]
            min_gap = 0.02  # 20ms gap between words

            if prev['end'] > curr['start'] - min_gap:
                prev['end'] = curr['start'] - min_gap
                if prev['end'] - prev['start'] < 0.05:
                    # Prev start landed too close to curr start too — pull
                    # prev start back rather than pushing curr forward.
                    prev['start'] = max(0.0, curr['start'] - min_gap - 0.05)
                    prev['end'] = curr['start'] - min_gap
        
        # Log alignment statistics
        if shifts:
            nonzero_shifts = [s for s in shifts if s != 0]
            if nonzero_shifts:
                avg_shift = np.mean(np.abs(nonzero_shifts)) * 1000
                max_shift = np.max(np.abs(nonzero_shifts)) * 1000
            else:
                avg_shift = 0
                max_shift = 0
            aligned_count = len(nonzero_shifts)
            logger.info(f"Dynamic alignment: {aligned_count}/{len(words)} words adjusted")
            logger.info(f"  Avg shift: {avg_shift:.1f}ms, Max shift: {max_shift:.1f}ms")
        
        return aligned_words
    
    def detect_pitches(self, audio_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Detect vocal pitches using librosa pYIN.
        
        Returns:
            times: Array of time points (seconds)
            pitches: Array of MIDI pitches (0 if no pitch detected)
            confidences: Array of pitch confidence values
            note_events: List of note events (start, end, pitch, velocity)
        """
        logger.info(f"Detecting vocal pitches from: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Use pYIN for pitch detection (good for vocals)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C6'),  # ~1046 Hz
            sr=sr,
            frame_length=2048,
            hop_length=512
        )
        
        # Convert to time array
        hop_length = 512
        frame_duration = hop_length / sr
        n_frames = len(f0)
        times = np.arange(n_frames) * frame_duration
        
        # Convert Hz to MIDI, handle NaN
        pitches = np.zeros(n_frames)
        confidences = voiced_prob.copy()
        
        for i in range(n_frames):
            if voiced_flag[i] and not np.isnan(f0[i]) and f0[i] > 0:
                pitches[i] = librosa.hz_to_midi(f0[i])
        
        logger.info(f"Pitch detection: {np.sum(pitches > 0)}/{n_frames} voiced frames ({100*np.mean(pitches > 0):.1f}%)")
        
        # Convert frame-level pitches to note events
        # Group consecutive pitched frames into notes
        note_events = self._frames_to_notes(times, pitches, confidences)
        
        logger.info(f"Detected {len(note_events)} note events")
        
        return times, pitches, confidences, note_events
    
    def _frames_to_notes(
        self,
        times: np.ndarray,
        pitches: np.ndarray,
        confidences: np.ndarray,
        min_duration: float = 0.08  # Minimum note duration in seconds
    ) -> list:
        """Convert frame-level pitch to note events."""
        note_events = []
        
        i = 0
        while i < len(pitches):
            if pitches[i] > 0:
                # Start of a note
                start_time = times[i]
                start_idx = i
                
                # Find end of this note (pitch changes or becomes unvoiced)
                j = i + 1
                while j < len(pitches):
                    if pitches[j] == 0:
                        break
                    # Allow small pitch variation (within 1 semitone)
                    if abs(pitches[j] - pitches[i]) > 1.0:
                        break
                    j += 1
                
                end_time = times[j-1] if j > 0 else times[i]
                duration = end_time - start_time
                
                if duration >= min_duration:
                    # Average pitch over the note
                    avg_pitch = np.mean(pitches[start_idx:j])
                    avg_conf = np.mean(confidences[start_idx:j])
                    
                    note_events.append((
                        start_time,
                        end_time + 0.05,  # Add small buffer
                        avg_pitch,
                        int(avg_conf * 100),  # Velocity from confidence
                        avg_conf
                    ))
                
                i = j
            else:
                i += 1
        
        return note_events
    
    def align_lyrics_to_pitches(
        self,
        words: list[dict],
        note_events: list,
        times: np.ndarray,
        pitches: np.ndarray,
        confidences: np.ndarray
    ) -> list[VocalNote]:
        """
        Align transcribed lyrics to detected pitches.
        
        Creates VocalNote objects with both lyric and pitch information.
        Ensures ALL words get included - creates notes for words without detected pitch.
        """
        notes = []
        used_words = set()
        
        if not note_events:
            # Fallback to frame-based pitch if no note events
            logger.warning("No note events, falling back to frame-based alignment")
            return self._align_frame_based(words, times, pitches, confidences)
        
        # First pass: match note events to overlapping words
        for event in note_events:
            start_time, end_time, pitch, velocity = event[:4]
            confidence = event[4] if len(event) > 4 else 1.0

            # Cap pitch-segment duration so a single sustained pitch (e.g.
            # held vowel through a long phrase) doesn't ring on for many
            # seconds. Charter standard: held notes rarely exceed ~2.5s.
            if end_time - start_time > 2.5:
                end_time = start_time + 2.5

            # Clamp pitch to vocal range
            midi_pitch = int(np.clip(pitch, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX))
            
            # Find best overlapping word for this note
            best_word_idx = None
            best_overlap = 0
            
            for i, word in enumerate(words):
                if i in used_words:
                    continue
                    
                word_start = word['start']
                word_end = word['end']
                
                # Calculate overlap
                overlap_start = max(start_time, word_start)
                overlap_end = min(end_time, word_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_word_idx = i
                
                # Also check if word is very close (within 0.2s)
                if overlap == 0 and best_word_idx is None:
                    if abs(word_start - start_time) < 0.2 or abs(word_end - end_time) < 0.2:
                        best_word_idx = i
            
            lyric = None
            if best_word_idx is not None:
                lyric = words[best_word_idx]['word']
                used_words.add(best_word_idx)
            
            notes.append(VocalNote(
                start_time=start_time,
                end_time=end_time,
                midi_pitch=midi_pitch,
                lyric=lyric
            ))
        
        # Second pass: create notes for unmatched words (ensures ALL lyrics are included)
        for i, word in enumerate(words):
            if i in used_words:
                continue
            
            word_start = word['start']
            word_end = word['end']
            text = word['word']
            
            # Find pitch at this word's time using frame data
            mask = (times >= word_start) & (times <= word_end)
            if mask.any():
                word_pitches = pitches[mask]
                voiced = word_pitches > 0
                if voiced.any():
                    avg_pitch = np.mean(word_pitches[voiced])
                    midi_pitch = int(np.clip(avg_pitch, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX))
                else:
                    # No pitch detected - estimate from nearby notes
                    midi_pitch = self._estimate_pitch_from_context(word_start, notes)
            else:
                midi_pitch = self._estimate_pitch_from_context(word_start, notes)
            
            notes.append(VocalNote(
                start_time=word_start,
                end_time=word_end,
                midi_pitch=midi_pitch,
                lyric=text
            ))
        
        # Sort by start time
        notes.sort(key=lambda n: n.start_time)
        
        logger.info(f"Aligned {len(notes)} vocal notes ({sum(1 for n in notes if n.lyric)} with lyrics)")
        logger.info(f"  Words from Whisper: {len(words)}, matched: {len(used_words)}, added unmatched: {len(words) - len(used_words)}")
        return notes
    
    def _estimate_pitch_from_context(self, time: float, notes: list[VocalNote]) -> int:
        """Estimate pitch for a time point based on nearby notes."""
        if not notes:
            return 60  # Middle C fallback
        
        # Find closest note
        closest = min(notes, key=lambda n: min(abs(n.start_time - time), abs(n.end_time - time)))
        return closest.midi_pitch
    
    def _align_frame_based(
        self,
        words: list[dict],
        times: np.ndarray,
        pitches: np.ndarray,
        confidences: np.ndarray
    ) -> list[VocalNote]:
        """Fallback alignment using frame-based pitch detection."""
        notes = []
        
        for word in words:
            start_time = word['start']
            end_time = word['end']
            text = word['word']
            
            # Find average pitch during this word
            mask = (times >= start_time) & (times <= end_time)
            if mask.any():
                word_pitches = pitches[mask]
                word_confs = confidences[mask]
                
                # Weighted average of pitched frames
                voiced = word_pitches > 0
                if voiced.any():
                    avg_pitch = np.average(word_pitches[voiced], weights=word_confs[voiced])
                    midi_pitch = int(np.clip(avg_pitch, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX))
                else:
                    midi_pitch = 60  # Middle C as fallback
            else:
                midi_pitch = 60
            
            notes.append(VocalNote(
                start_time=start_time,
                end_time=end_time,
                midi_pitch=midi_pitch,
                lyric=text
            ))
        
        return notes
    
    def group_into_phrases(self, notes: list[VocalNote]) -> list[VocalPhrase]:
        """
        Group notes into phrases based on timing gaps AND word count.
        A phrase break occurs when:
        - Gap longer than PHRASE_GAP_THRESHOLD, OR
        - Reached MAX_WORDS_PER_PHRASE words
        This ensures lyrics display in manageable chunks like reference charts.
        """
        if not notes:
            return []
        
        phrases = []
        current_phrase_notes = [notes[0]]
        current_word_count = 1 if notes[0].lyric and not notes[0].lyric.startswith('+') else 0
        
        for i in range(1, len(notes)):
            gap = notes[i].start_time - notes[i-1].end_time
            is_new_word = notes[i].lyric and not notes[i].lyric.endswith('-') and not notes[i].lyric.startswith('+')
            
            # Break phrase if: big gap, OR too many words (and we're at a word boundary)
            should_break = (
                gap > self.PHRASE_GAP_THRESHOLD or
                (current_word_count >= self.MAX_WORDS_PER_PHRASE and is_new_word and gap > 0.1)
            )
            
            if should_break:
                # End current phrase and start new one
                phrases.append(VocalPhrase(
                    start_time=current_phrase_notes[0].start_time,
                    end_time=current_phrase_notes[-1].end_time,
                    notes=current_phrase_notes
                ))
                current_phrase_notes = [notes[i]]
                current_word_count = 1 if is_new_word else 0
            else:
                current_phrase_notes.append(notes[i])
                if is_new_word:
                    current_word_count += 1
        
        # Add final phrase
        if current_phrase_notes:
            phrases.append(VocalPhrase(
                start_time=current_phrase_notes[0].start_time,
                end_time=current_phrase_notes[-1].end_time,
                notes=current_phrase_notes
            ))
        
        logger.info(f"Grouped into {len(phrases)} phrases")
        return phrases
    
    def detect_harmonies(
        self,
        audio_path: str,
        lead_phrases: list[VocalPhrase]
    ) -> list[VocalPhrase]:
        """
        Detect harmony lines that differ from the lead vocal.
        
        Creates HARM1 track with:
        - Same phrase boundaries as lead vocals
        - Same lyrics as lead vocals (YARG requires this!)
        - Different pitches where harmony is detected
        - Lead pitch where no harmony detected (offset by interval)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        harmony_phrases = []
        
        for phrase in lead_phrases:
            phrase_harmony_notes = []
            
            for note in phrase.notes:
                start_sample = int(note.start_time * sr)
                end_sample = int(note.end_time * sr)
                
                # Default: harmony is lead pitch + 4 semitones (major 3rd above)
                # This is a common harmony interval
                default_harmony_pitch = min(self.VOCAL_MIDI_MAX, note.midi_pitch + 4)
                
                harm_pitch = default_harmony_pitch
                detected_harmony = False
                
                if end_sample > start_sample:
                    segment = y[start_sample:end_sample]
                    
                    if len(segment) >= 512:
                        # Try to detect actual harmony pitch
                        try:
                            S = np.abs(librosa.stft(segment))
                            freqs = librosa.fft_frequencies(sr=sr)
                            avg_spectrum = S.mean(axis=1)
                            
                            from scipy.signal import find_peaks
                            peaks, _ = find_peaks(avg_spectrum, height=avg_spectrum.max() * 0.2)
                            
                            if len(peaks) >= 2:
                                peak_mags = avg_spectrum[peaks]
                                sorted_idx = np.argsort(peak_mags)[::-1]
                                
                                lead_freq = freqs[peaks[sorted_idx[0]]]
                                harmony_freq = freqs[peaks[sorted_idx[1]]]
                                
                                if lead_freq > 0 and harmony_freq > 0:
                                    lead_midi = librosa.hz_to_midi(lead_freq)
                                    harm_midi = librosa.hz_to_midi(harmony_freq)
                                    
                                    # Check if it's actually different (>2 semitones)
                                    if abs(harm_midi - lead_midi) > 2:
                                        ratio = peak_mags[sorted_idx[1]] / peak_mags[sorted_idx[0]]
                                        if ratio >= self.harmony_threshold:
                                            harm_pitch = int(np.clip(harm_midi, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX))
                                            detected_harmony = True
                        except Exception:
                            pass  # Use default harmony
                
                # Always create harmony note with SAME lyric as lead
                phrase_harmony_notes.append(VocalNote(
                    start_time=note.start_time,
                    end_time=note.end_time,
                    midi_pitch=harm_pitch,
                    lyric=note.lyric,  # CRITICAL: same lyric as lead!
                    is_harmony=True,
                    is_pitchless=note.is_pitchless,
                    connects_to_next=note.connects_to_next
                ))
            
            # Create harmony phrase with same boundaries as lead
            if phrase_harmony_notes:
                harmony_phrases.append(VocalPhrase(
                    start_time=phrase.start_time,
                    end_time=phrase.end_time,
                    notes=phrase_harmony_notes
                ))
        
        total_harmony_notes = sum(len(p.notes) for p in harmony_phrases)
        logger.info(f"Created {total_harmony_notes} harmony notes in {len(harmony_phrases)} phrases (matching lead)")
        
        return harmony_phrases
    
    def transcribe(
        self,
        audio_path: str,
        artist: Optional[str] = None,
        title: Optional[str] = None
    ) -> tuple[list[VocalPhrase], list[VocalPhrase]]:
        """
        Full vocal transcription pipeline.
        
        Args:
            audio_path: Path to vocals audio (ideally isolated vocal stem)
            artist: Artist name for lyrics lookup (optional, will try to extract from path)
            title: Song title for lyrics lookup (optional, will try to extract from path)
            
        Returns:
            Tuple of (lead_phrases, harmony_phrases)
        """
        logger.info(f"Starting vocal transcription: {audio_path}")
        
        # Clear cached onsets for new song
        self._cached_onsets = None
        
        # Step 1: Try to fetch lyrics from web sources
        lyrics_result = None
        if self.fetch_lyrics_online:
            # Try to extract artist/title from path if not provided
            if not artist or not title:
                path_artist, path_title = extract_artist_title_from_path(audio_path)
                artist = artist or path_artist
                title = title or path_title
            
            if artist and title:
                logger.info(f"Searching for lyrics: {artist} - {title}")
                lyrics_result = fetch_lyrics(artist, title)
                if lyrics_result:
                    logger.info(f"Found lyrics from {lyrics_result.source}" +
                              (f" (synced: {len(lyrics_result.synced)} lines)" if lyrics_result.synced else ""))
        
        # Step 2: Detect pitches (frame-level for pitch lookup)
        times, pitches, confidences, _ = self.detect_pitches(audio_path)
        
        # Step 3: Always use Whisper for timing (detects actual audio)
        # Then optionally use LRCLIB lyrics to correct the text
        logger.info("Using Whisper for word-level timing detection")
        words = self.transcribe_lyrics(audio_path)
        
        # If we have LRCLIB lyrics, use them to correct Whisper's text
        # (Whisper hears the audio correctly timing-wise, but may mishear words)
        if lyrics_result and lyrics_result.text:
            words = self._align_lyrics_to_whisper(words, lyrics_result.text)
        
        # Step 3.5: Dynamic alignment - snap words to actual vocal onsets
        # This fixes variable Whisper latency throughout the song
        if self.dynamic_alignment:
            onset_times = self.detect_vocal_onsets(audio_path)
            words = self.align_words_to_onsets(words, onset_times)
        
        # Step 4: Create notes from lyrics with pitch lookup (lyrics-driven approach)
        notes = self._lyrics_to_notes(words, times, pitches, confidences)
        
        # Step 5: Filter out tiny spurious notes (under 80ms)
        notes = self._filter_tiny_notes(notes, min_duration=0.08)
        
        # Step 6: Group into phrases
        lead_phrases = self.group_into_phrases(notes)
        
        # Step 7: Smooth pitches to avoid unrealistic jumps
        lead_phrases = self._smooth_pitches(lead_phrases)
        
        # Step 8: Detect harmonies - copy lyrics from lead at matching times
        harmony_phrases = self.detect_harmonies(audio_path, lead_phrases)
        
        # Step 9: Smooth harmony pitches too
        harmony_phrases = self._smooth_pitches(harmony_phrases)
        
        logger.info(f"Transcription complete: {len(lead_phrases)} lead phrases, {len(harmony_phrases)} harmony phrases")
        
        return lead_phrases, harmony_phrases
    
    def _detect_vocal_segments(self, audio_path: str, search_start: float = 0.0) -> list[tuple[float, float]]:
        """
        Detect vocal segments (start, end) throughout the audio.
        
        Args:
            audio_path: Path to audio file
            search_start: Start searching from this time (seconds)
        
        Returns list of (start_time, end_time) for each detected vocal segment.
        """
        # Load full audio
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # Compute RMS energy in small windows
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Compute spectral centroid (vocals tend to have higher centroid)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Normalize features
        rms_norm = rms / (np.max(rms) + 1e-8)
        centroid_norm = centroid / (np.max(centroid) + 1e-8)
        
        # Combined vocal likelihood
        vocal_likelihood = rms_norm * 0.6 + centroid_norm * 0.4
        
        # Threshold for vocals present
        threshold = np.percentile(vocal_likelihood, 60)
        
        # Convert search_start to frame index
        start_frame = int(search_start * sr / hop_length)
        
        # Find vocal segments
        segments = []
        in_segment = False
        segment_start = None
        min_segment_frames = 10  # Minimum ~0.2s segment
        
        for i in range(start_frame, len(vocal_likelihood)):
            v = vocal_likelihood[i]
            if v > threshold and not in_segment:
                in_segment = True
                segment_start = i
            elif v <= threshold and in_segment:
                if i - segment_start >= min_segment_frames:
                    start_time = librosa.frames_to_time(segment_start, sr=sr, hop_length=hop_length)
                    end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                    segments.append((start_time, end_time))
                in_segment = False
        
        # Handle segment at end
        if in_segment and len(vocal_likelihood) - segment_start >= min_segment_frames:
            start_time = librosa.frames_to_time(segment_start, sr=sr, hop_length=hop_length)
            end_time = duration
            segments.append((start_time, end_time))
        
        logger.info(f"Detected {len(segments)} vocal segments in audio (from {search_start:.1f}s)")
        return segments
    
    def _calculate_time_warp(
        self,
        synced: list[SyncedLyric],
        audio_path: str
    ) -> tuple[float, float]:
        """
        Calculate time warp parameters (offset, scale) to align LRC to audio.
        
        Strategy: Trust LRC timing mostly - only calculate scale for tempo drift.
        Most YouTube lyric videos have same tempo as official release.
        
        Returns:
            (offset, scale) tuple where: actual_time = lrc_time * scale + offset
        """
        # Get LRC lyric line times
        lrc_times = []
        for lyric in synced:
            if lyric.text.strip():
                lrc_times.append(lyric.time)
        
        if not lrc_times:
            return (0.0, 1.0)
        
        first_lrc = lrc_times[0]
        last_lrc = lrc_times[-1]
        lrc_duration = last_lrc - first_lrc
        
        # For now, just trust the LRC timing directly
        # Most YouTube lyric videos match official release timing
        # Users can manually adjust with --audio-offset if needed
        offset = 0.0
        scale = 1.0
        
        logger.info(f"Using LRC timing directly (no auto-adjustment)")
        logger.info(f"  LRC range: {first_lrc:.1f}s - {last_lrc:.1f}s ({lrc_duration:.1f}s)")
        
        return (offset, scale)
    
    def _synced_lyrics_to_words(
        self,
        synced: list[SyncedLyric],
        audio_path: str
    ) -> list[dict]:
        """
        Convert synced LRC lyrics to word-level format with dynamic time warp.
        
        LRC lyrics are line-based, so we need to split into words and estimate timing.
        Uses both offset AND scale factor to handle tempo differences.
        """
        words = []
        
        # Calculate time warp parameters
        offset, scale = self._calculate_time_warp(synced, audio_path)
        
        def warp_time(lrc_time: float) -> float:
            """Convert LRC time to actual audio time."""
            return lrc_time * scale + offset
        
        for i, lyric in enumerate(synced):
            line_text = lyric.text.strip()
            if not line_text:
                continue
            
            # Apply time warp to timestamps
            line_start = warp_time(lyric.time)
            next_lrc_time = synced[i + 1].time if i + 1 < len(synced) else lyric.time + 3.0
            line_end = warp_time(next_lrc_time)
            
            # Skip if this would be before audio start (negative time)
            if line_end < 0:
                continue
            if line_start < 0:
                line_start = 0.0
            
            # Split into words
            line_words = line_text.split()
            if not line_words:
                continue
            
            # Estimate timing for each word (distribute evenly)
            word_duration = (line_end - line_start) / len(line_words)
            
            for j, word in enumerate(line_words):
                word_start = line_start + j * word_duration
                word_end = word_start + word_duration * 0.9  # 90% duration, 10% gap
                
                words.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end
                })
        
        logger.info(f"Converted {len(synced)} synced lines to {len(words)} words (offset: {offset:+.2f}s, scale: {scale:.4f})")
        return words
    
    def _align_lyrics_to_whisper(
        self,
        whisper_words: list[dict],
        lrclib_text: str
    ) -> list[dict]:
        """
        Align LRCLIB lyrics text to Whisper's word timing.
        
        Uses Whisper's timing (accurate for actual audio) but replaces
        text with LRCLIB's lyrics (more accurate spelling/words).
        
        Args:
            whisper_words: List of {'word': str, 'start': float, 'end': float} from Whisper
            lrclib_text: Plain text lyrics from LRCLIB
            
        Returns:
            Updated word list with LRCLIB text but Whisper timing
        """
        import re
        
        # Extract just the words from LRCLIB (remove punctuation for matching)
        lrc_words_raw = lrclib_text.split()
        lrc_words = []
        for w in lrc_words_raw:
            # Clean word for matching but keep original for output
            cleaned = re.sub(r'[^\w\']', '', w.lower())
            if cleaned:
                lrc_words.append({'original': w, 'cleaned': cleaned})
        
        if not lrc_words or not whisper_words:
            logger.warning("No words to align, using Whisper output directly")
            return whisper_words
        
        # Simple alignment: match words sequentially
        # Whisper and LRCLIB should have roughly same word count
        result = []
        lrc_idx = 0
        
        for whisper_word in whisper_words:
            whisper_cleaned = re.sub(r'[^\w\']', '', whisper_word['word'].lower())
            
            # Try to find a matching LRCLIB word nearby
            best_match_idx = None
            best_score = 0
            
            # Search within a window around current position
            search_start = max(0, lrc_idx - 5)
            search_end = min(len(lrc_words), lrc_idx + 10)
            
            for i in range(search_start, search_end):
                lrc_cleaned = lrc_words[i]['cleaned']
                
                # Calculate similarity (simple ratio of matching chars)
                if whisper_cleaned == lrc_cleaned:
                    score = 1.0
                elif whisper_cleaned in lrc_cleaned or lrc_cleaned in whisper_cleaned:
                    score = 0.8
                elif whisper_cleaned[:3] == lrc_cleaned[:3] if len(whisper_cleaned) >= 3 and len(lrc_cleaned) >= 3 else False:
                    score = 0.6
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_match_idx = i
            
            # Use LRCLIB word if good match, otherwise keep Whisper
            if best_match_idx is not None and best_score >= 0.6:
                result.append({
                    'word': lrc_words[best_match_idx]['original'],
                    'start': whisper_word['start'],
                    'end': whisper_word['end']
                })
                # Advance LRC index past the matched word
                lrc_idx = best_match_idx + 1
            else:
                # No good match, keep Whisper's word
                result.append(whisper_word)
        
        matched = sum(1 for r in result if any(r['word'] == lw['original'] for lw in lrc_words))
        logger.info(f"Aligned {len(result)} words ({matched} matched with LRCLIB, {len(result)-matched} from Whisper)")
        
        return result
    
    def _smooth_pitches(
        self,
        phrases: list[VocalPhrase],
        max_jump: int = 3  # Maximum semitone jump allowed (default 3 = minor 3rd)
    ) -> list[VocalPhrase]:
        """
        Smooth pitch transitions to avoid unrealistic jumps.
        
        Rock Band charts typically don't have huge pitch jumps between consecutive
        notes in a phrase. This makes the chart feel more natural and playable.
        
        Args:
            phrases: List of vocal phrases
            max_jump: Maximum allowed semitone jump (default 3 = minor 3rd)
            
        Returns:
            Phrases with smoothed pitches
        """
        for phrase in phrases:
            if len(phrase.notes) < 2:
                continue
            
            # First pass: apply median filter to remove outliers
            pitches = [n.midi_pitch for n in phrase.notes]
            smoothed_pitches = self._median_filter(pitches, window=3)
            
            for i, note in enumerate(phrase.notes):
                note.midi_pitch = smoothed_pitches[i]
            
            # Second pass: cap maximum jumps
            for i in range(1, len(phrase.notes)):
                prev_pitch = phrase.notes[i-1].midi_pitch
                curr_pitch = phrase.notes[i].midi_pitch
                
                jump = curr_pitch - prev_pitch
                
                if abs(jump) > max_jump:
                    if jump > 0:
                        phrase.notes[i].midi_pitch = prev_pitch + max_jump
                    else:
                        phrase.notes[i].midi_pitch = prev_pitch - max_jump
                    
                    phrase.notes[i].midi_pitch = max(
                        self.VOCAL_MIDI_MIN,
                        min(self.VOCAL_MIDI_MAX, phrase.notes[i].midi_pitch)
                    )
        
        return phrases
    
    def _median_filter(self, values: list[int], window: int = 3) -> list[int]:
        """Apply median filter to smooth pitch outliers."""
        result = values.copy()
        half = window // 2
        
        for i in range(len(values)):
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            neighborhood = sorted(values[start:end])
            result[i] = neighborhood[len(neighborhood) // 2]
        
        return result
    
    def _filter_tiny_notes(
        self,
        notes: list[VocalNote],
        min_duration: float = 0.08  # Minimum 80ms
    ) -> list[VocalNote]:
        """
        Filter out spurious tiny notes while preserving ALL lyrics.
        
        CRITICAL: Never drop a note that has a lyric - just extend its duration.
        Only merge notes that have no lyric (continuation markers).
        """
        if not notes:
            return notes
        
        filtered = []
        
        for note in notes:
            duration = note.end_time - note.start_time
            
            if duration >= min_duration:
                # Note is long enough, keep as-is
                filtered.append(note)
            elif note.lyric and note.lyric.strip():
                # Has a lyric - MUST keep this note, just extend duration
                note.end_time = note.start_time + min_duration
                filtered.append(note)
            elif filtered:
                # No lyric, too short - extend previous note's duration
                filtered[-1].end_time = max(filtered[-1].end_time, note.end_time)
            else:
                # First note is tiny with no lyric - extend it
                note.end_time = note.start_time + min_duration
                filtered.append(note)
        
        # CRITICAL: Fix any overlaps without dropping notes
        # Sort by start time first
        filtered.sort(key=lambda n: n.start_time)
        
        # Make multiple passes until no overlaps remain
        for _ in range(3):  # Up to 3 passes
            had_overlap = False
            for i in range(len(filtered) - 1):
                current = filtered[i]
                next_note = filtered[i + 1]

                # Connected notes (slides/melismas): allow them to TOUCH —
                # YARG/CH draws a smooth slide when end == next.start.
                # Forcing a 10ms gap on these caused the stair-step look.
                if current.connects_to_next:
                    if current.end_time != next_note.start_time:
                        current.end_time = next_note.start_time
                    if current.end_time - current.start_time < 0.03:
                        current.end_time = current.start_time + 0.03
                    continue

                min_gap = 0.01  # 10ms minimum gap (only for unconnected notes)

                # If current extends past next's start, truncate current
                if current.end_time > next_note.start_time - min_gap:
                    current.end_time = next_note.start_time - min_gap
                    had_overlap = True

                # If still overlapping after truncation (start times too close),
                # pull current's START back rather than shifting next forward
                # — shifting next is what caused song-long timing drift.
                if current.end_time > next_note.start_time - min_gap:
                    new_end = next_note.start_time - min_gap
                    current.start_time = max(0.0, new_end - 0.05)
                    current.end_time = new_end
                    had_overlap = True

                # Ensure current still has valid duration (at least 30ms)
                if current.end_time - current.start_time < 0.03:
                    current.end_time = current.start_time + 0.03
            
            if not had_overlap:
                break
        
        # Count lyrics preserved
        input_lyrics = sum(1 for n in notes if n.lyric and n.lyric.strip())
        output_lyrics = sum(1 for n in filtered if n.lyric and n.lyric.strip())
        
        logger.info(f"Filtered {len(notes)} notes -> {len(filtered)} notes")
        logger.info(f"  Lyrics preserved: {output_lyrics}/{input_lyrics}")
        
        return filtered
    
    def _lyrics_to_notes(
        self,
        words: list[dict],
        times: np.ndarray,
        pitches: np.ndarray,
        confidences: np.ndarray,
    ) -> list[VocalNote]:
        """
        Create vocal notes from lyrics, detecting pitch changes within words.
        
        When pitch shifts significantly during a word (melisma/pitch bend),
        creates multiple notes: first gets the lyric, continuations get syllables.
        
        CRITICAL: Each word MUST produce at least one note with its lyric.
        Syllable splits are additional notes, not replacements.
        """
        notes = []
        
        for word_idx, word in enumerate(words):
            start_time = word['start']
            end_time = word['end']
            text = word['word'].strip()
            
            if not text:
                continue
            
            # Ensure minimum duration
            if end_time <= start_time:
                end_time = start_time + 0.1
            
            # Get all frames during this word
            mask = (times >= start_time) & (times <= end_time)
            word_times = times[mask]
            word_pitches = pitches[mask]
            word_confs = confidences[mask]
            
            if len(word_times) == 0:
                # No frames - create single note with word timing
                notes.append(VocalNote(
                    start_time=start_time,
                    end_time=end_time,
                    midi_pitch=60,
                    lyric=text,
                    is_pitchless=True
                ))
                continue
            
            # Check for pitchless section
            is_pitchless = self._is_pitchless_section(word_pitches, word_confs)
            
            if is_pitchless:
                notes.append(VocalNote(
                    start_time=start_time,
                    end_time=end_time,
                    midi_pitch=60,
                    lyric=text,
                    is_pitchless=True
                ))
                continue
            
            # Find pitch segments within this word
            segments = self._find_pitch_segments(
                word_times, word_pitches, word_confs, 
                self.pitch_change_threshold
            )
            
            if not segments:
                # No voiced segments - create single note
                voiced = word_pitches > 0
                if voiced.any():
                    avg_pitch = np.mean(word_pitches[voiced])
                    midi_pitch = int(np.clip(avg_pitch, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX))
                else:
                    midi_pitch = 60
                notes.append(VocalNote(
                    start_time=start_time,
                    end_time=end_time,
                    midi_pitch=midi_pitch,
                    lyric=text
                ))
                continue
            
            # CRITICAL: Clamp segments to word boundaries
            # This ensures notes don't extend past the word's timing
            clamped_segments = []
            for seg_start, seg_end, seg_pitch in segments:
                seg_start = max(seg_start, start_time)
                seg_end = min(seg_end, end_time)
                if seg_end > seg_start + 0.02:
                    clamped_segments.append((seg_start, seg_end, seg_pitch))
            
            if not clamped_segments:
                # All segments outside word - use word timing with avg pitch
                voiced = word_pitches > 0
                avg_pitch = np.mean(word_pitches[voiced]) if voiced.any() else 60
                notes.append(VocalNote(
                    start_time=start_time,
                    end_time=end_time,
                    midi_pitch=int(np.clip(avg_pitch, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX)),
                    lyric=text
                ))
                continue
            
            # CRITICAL FIX: First segment MUST start at word start time
            # This ensures the lyric appears at the right time
            first_seg = clamped_segments[0]
            if first_seg[0] > start_time + 0.05:  # More than 50ms late
                # Insert a note at word start to carry the lyric
                clamped_segments.insert(0, (start_time, first_seg[0], first_seg[2]))
            else:
                # Adjust first segment to start at word start
                clamped_segments[0] = (start_time, first_seg[1], first_seg[2])
            
            # Split word into syllables for pitch changes
            syllables = self._split_into_syllables(text, len(clamped_segments))

            # Create notes for each segment.
            #
            # Slide encoding (Rock Band / YARG / Clone Hero spec):
            #   * Within a single sung syllable, additional notes on different
            #     pitches use the lyric '+' (literal plus sign). YARG renders
            #     these as a smooth slide line between the notes.
            #   * '-' suffix means "continuation to the next syllable of the
            #     same word" — NOT a slide. So if pyphen returned 2 syllables
            #     for 2 segments, both get their own syllable text. If we have
            #     MORE segments than syllables, the extras get '+'.
            for i, (seg_start, seg_end, seg_pitch) in enumerate(clamped_segments):
                if i < len(syllables) and syllables[i]:
                    lyric = syllables[i]
                else:
                    lyric = '+'  # melisma continuation — triggers slide rendering
                connects = i < len(clamped_segments) - 1

                notes.append(VocalNote(
                    start_time=seg_start,
                    end_time=seg_end,
                    midi_pitch=int(np.clip(seg_pitch, self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX)),
                    lyric=lyric,
                    connects_to_next=connects,
                ))

        # NOTE: previously called _mark_connections() here to slide between
        # adjacent WORDS — but in RB/YARG that just produces two separate
        # notes drawn close, not a slide line. Slides are an in-syllable
        # concept (handled per-word above with the '+' lyric).
        
        total_words = len(words)
        total_notes = len(notes)
        pitchless_count = sum(1 for n in notes if n.is_pitchless)
        logger.info(f"Created {total_notes} vocal notes from {total_words} words ({pitchless_count} pitchless)")
        return notes
    
    def _is_pitchless_section(self, pitches: np.ndarray, confidences: np.ndarray) -> bool:
        """
        Detect if a section should be marked as pitchless (talky/rap/scream).
        
        Very strict - only returns True if there's essentially NO usable pitch:
        - Extremely low voiced ratio (almost no pitch detected)
        - AND extremely low confidence OR extremely high variance
        """
        voiced_mask = pitches > 0
        voice_ratio = voiced_mask.mean()
        
        # Need VERY few voiced frames to even consider pitchless
        if voice_ratio >= self.pitchless_voiced_ratio_threshold:
            return False  # Enough pitch detected, not pitchless
        
        # Almost no voiced frames - check if it's truly unpitched
        voiced_pitches = pitches[voiced_mask]
        voiced_confs = confidences[voiced_mask]
        
        if len(voiced_pitches) == 0:
            return True  # No pitch at all
        
        # Need BOTH low ratio AND (low confidence OR high variance)
        avg_confidence = voiced_confs.mean()
        pitch_variance = np.std(voiced_pitches) if len(voiced_pitches) > 1 else 0
        
        if avg_confidence < self.pitchless_confidence_threshold:
            return True
        if pitch_variance > self.pitchless_variance_threshold:
            return True
        
        return False
    
    def _mark_connections(self, notes: list[VocalNote]) -> None:
        """
        Mark notes that should be drawn connected (smooth line to next note).
        
        Notes are connected if:
        - Close in time (gap < CONNECTION_THRESHOLD)
        - Both are pitched (not pitchless)
        - In the same phrase (didn't cross a big gap)
        """
        for i in range(len(notes) - 1):
            if notes[i].is_pitchless or notes[i + 1].is_pitchless:
                continue
            
            gap = notes[i + 1].start_time - notes[i].end_time
            
            # Connect if very close (within same syllable or smooth phrase)
            if gap < self.CONNECTION_THRESHOLD:
                notes[i].connects_to_next = True
    
    def _split_into_syllables(self, word: str, num_parts: int) -> list[str]:
        """
        Split a word into syllables for multi-note display.
        
        Uses pyphen for hyphenation. Clone Hero format:
        - First syllables end with '-' to indicate continuation
        - Last syllable has no hyphen
        - If word has fewer syllables than notes, last syllable spans remaining notes
        """
        if num_parts <= 1:
            return [word]
        
        # Clean the word for hyphenation
        clean_word = word.strip()
        
        # Get hyphenation points
        hyphenated = self._hyphenator.inserted(clean_word.lower())
        syllables = hyphenated.split('-')
        
        # Build result with proper hyphenation
        result = []
        used_extended = False  # Only add one '=' for extended notes
        for i in range(num_parts):
            if i < len(syllables):
                syl = syllables[i]
                # Preserve original case for first syllable
                if i == 0 and clean_word[0].isupper():
                    syl = syl.capitalize()
                # Add hyphen suffix if not the last part AND there are more syllables/notes coming
                if i < len(syllables) - 1 or (i == len(syllables) - 1 and num_parts > len(syllables)):
                    syl = syl + '-'
                result.append(syl)
            else:
                # More notes than syllables - only first extended note gets marker
                # Subsequent notes in same word are silent (empty string)
                # This matches Rock Band convention where held notes just extend
                if not used_extended:
                    result.append('')  # First extension has no text (note just extends)
                    used_extended = True
                else:
                    result.append('')  # Subsequent extensions also silent
        
        return result
    
    def _find_pitch_segments(
        self,
        times: np.ndarray,
        pitches: np.ndarray,
        confidences: np.ndarray,
        threshold: float,
        min_segment_dur: float = 0.20,  # 200ms — segments shorter than this get merged into neighbour
    ) -> list[tuple[float, float, float]]:
        """Find sustained pitch segments within a word's time range.

        Two-pass:
          1. Median-filter the pitch contour (window=5 frames, ~115ms) to
             squash pYIN jitter and vibrato.
          2. Walk the smoothed contour and only emit a NEW segment when the
             pitch differs from the running median by > threshold AND that
             new pitch is held for at least min_segment_dur.

        This keeps held notes as ONE note instead of stair-stepping every
        time pYIN wobbles. Real melismas (>5 semitones, sustained) still split.
        """
        voiced_mask = pitches > 0
        if not voiced_mask.any():
            return []
        vt = times[voiced_mask]
        vp = pitches[voiced_mask]
        vc = confidences[voiced_mask]
        if len(vt) < 3:
            avg = float(np.average(vp, weights=vc) if vc.sum() > 0 else np.mean(vp))
            return [(float(vt[0]), float(vt[-1] + 0.02), avg)]

        # 1. Median filter (5-frame window) to remove pitch jitter
        win = 5
        half = win // 2
        smoothed = vp.copy()
        for i in range(len(vp)):
            lo = max(0, i - half)
            hi = min(len(vp), i + half + 1)
            smoothed[i] = float(np.median(vp[lo:hi]))

        # 2. Greedy segmentation with hold-duration requirement
        segments: list[tuple[float, float, float]] = []
        seg_start_idx = 0
        seg_pitches = [smoothed[0]]
        seg_confs = [vc[0]]

        i = 1
        while i < len(smoothed):
            running_avg = float(np.average(seg_pitches, weights=seg_confs)
                                if sum(seg_confs) > 0 else np.mean(seg_pitches))
            if abs(smoothed[i] - running_avg) > threshold:
                # Look ahead: is the new pitch SUSTAINED for >= min_segment_dur?
                target_pitch = smoothed[i]
                lookahead_end = i
                while (lookahead_end < len(smoothed)
                       and abs(smoothed[lookahead_end] - target_pitch) <= threshold):
                    lookahead_end += 1
                hold_dur = float(vt[min(lookahead_end - 1, len(vt) - 1)] - vt[i])
                if hold_dur >= min_segment_dur:
                    # Commit current segment, start new
                    segments.append((
                        float(vt[seg_start_idx]),
                        float(vt[i - 1] + 0.02),
                        running_avg,
                    ))
                    seg_start_idx = i
                    seg_pitches = [smoothed[i]]
                    seg_confs = [vc[i]]
                else:
                    # Brief excursion (vibrato/scoop) — keep in current segment
                    seg_pitches.append(smoothed[i])
                    seg_confs.append(vc[i])
            else:
                seg_pitches.append(smoothed[i])
                seg_confs.append(vc[i])
            i += 1

        # Final segment
        if seg_pitches:
            avg = float(np.average(seg_pitches, weights=seg_confs)
                        if sum(seg_confs) > 0 else np.mean(seg_pitches))
            segments.append((float(vt[seg_start_idx]), float(vt[-1] + 0.02), avg))

        # Drop any segment shorter than min_segment_dur by merging into neighbour
        if len(segments) > 1:
            cleaned: list[tuple[float, float, float]] = []
            for seg in segments:
                s, e, p = seg
                if e - s < min_segment_dur and cleaned:
                    # Extend previous segment to absorb this short one
                    ps, pe, pp = cleaned[-1]
                    cleaned[-1] = (ps, e, pp)
                else:
                    cleaned.append(seg)
            segments = cleaned

        return segments
    
    def _find_nearby_pitch(self, target_time: float, times: np.ndarray, pitches: np.ndarray) -> int:
        """Find the nearest valid pitch to a target time."""
        # Find closest time index
        idx = np.abs(times - target_time).argmin()
        
        # Search outward for a voiced frame
        for offset in range(50):
            for i in [idx + offset, idx - offset]:
                if 0 <= i < len(pitches) and pitches[i] > 0:
                    return int(np.clip(pitches[i], self.VOCAL_MIDI_MIN, self.VOCAL_MIDI_MAX))
        
        return 60  # Middle C fallback
    
    def export_midi(
        self,
        lead_phrases: list[VocalPhrase],
        harmony_phrases: list[VocalPhrase],
        output_path: str,
        tempo_bpm: float = 120.0,
        ticks_per_beat: int = 480
    ) -> None:
        """
        Export vocal phrases to Clone Hero format MIDI.
        
        Creates:
        - PART VOCALS track with lead vocals + phrase markers + lyrics
        - HARM1 track with first harmony line (if any)
        
        Special notes:
        - Pitchless/talky notes use MIDI note 96 (no pitch matching required)
        - Connected notes have overlapping end/start times
        """
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        
        # Tempo track
        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage('track_name', name='Tempo', time=0))
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm), time=0))
        mid.tracks.append(tempo_track)
        
        PITCHLESS_MIDI_NOTE = 96  # Clone Hero pitchless/talky marker
        
        def seconds_to_ticks(seconds: float) -> int:
            return int(seconds * tempo_bpm / 60 * ticks_per_beat)
        
        # Lead vocals track
        vocals_track = mido.MidiTrack()
        vocals_track.append(mido.MetaMessage('track_name', name='PART VOCALS', time=0))
        
        events = []
        
        for phrase in lead_phrases:
            # Phrase start marker (note 105)
            phrase_start_tick = seconds_to_ticks(phrase.start_time)
            events.append(('phrase_start', phrase_start_tick))
            
            for note in phrase.notes:
                start_tick = seconds_to_ticks(note.start_time)
                end_tick = seconds_to_ticks(note.end_time)
                duration = max(1, end_tick - start_tick)
                
                # Lyric event (before note) - EVERY word needs a lyric
                if note.lyric:
                    events.append(('lyric', start_tick, note.lyric))
                
                # Always use detected pitch clamped to vocal range (no pitchless note 96)
                midi_note = max(36, min(84, note.midi_pitch))
                events.append(('note_on', start_tick, midi_note, note.connects_to_next))
                events.append(('note_off', start_tick + duration, midi_note, False))
            
            # Phrase end marker (note 106)
            phrase_end_tick = seconds_to_ticks(phrase.end_time)
            events.append(('phrase_end', phrase_end_tick))
        
        # Sort events by time
        events.sort(key=lambda x: x[1])
        
        # Convert to MIDI messages with delta times
        current_tick = 0
        for event in events:
            event_type = event[0]
            event_tick = event[1]
            delta = max(0, event_tick - current_tick)  # Ensure non-negative
            
            if event_type == 'phrase_start':
                vocals_track.append(mido.Message('note_on', note=105, velocity=100, time=delta))
                vocals_track.append(mido.Message('note_off', note=105, velocity=0, time=1))
                current_tick = event_tick + 1
            elif event_type == 'phrase_end':
                vocals_track.append(mido.Message('note_on', note=106, velocity=100, time=delta))
                vocals_track.append(mido.Message('note_off', note=106, velocity=0, time=1))
                current_tick = event_tick + 1
            elif event_type == 'lyric':
                lyric_text = event[2]
                vocals_track.append(mido.MetaMessage('lyrics', text=lyric_text, time=delta))
                current_tick = event_tick
            elif event_type == 'note_on':
                pitch = event[2]
                vocals_track.append(mido.Message('note_on', note=pitch, velocity=100, time=delta))
                current_tick = event_tick
            elif event_type == 'note_off':
                pitch = event[2]
                vocals_track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))
                current_tick = event_tick
        
        mid.tracks.append(vocals_track)
        
        # Harmony track (if any) - includes lyrics like lead vocals
        if harmony_phrases:
            harm_track = mido.MidiTrack()
            harm_track.append(mido.MetaMessage('track_name', name='HARM1', time=0))
            
            harm_events = []
            for phrase in harmony_phrases:
                # Phrase markers for harmony too
                phrase_start_tick = seconds_to_ticks(phrase.start_time)
                phrase_end_tick = seconds_to_ticks(phrase.end_time)
                harm_events.append(('phrase_start', phrase_start_tick))
                
                for note in phrase.notes:
                    start_tick = seconds_to_ticks(note.start_time)
                    end_tick = seconds_to_ticks(note.end_time)
                    duration = max(1, end_tick - start_tick)
                    
                    # Include lyrics for harmony notes
                    if note.lyric:
                        harm_events.append(('lyric', start_tick, note.lyric))
                    
                    # Always use pitch clamped to vocal range
                    midi_note = max(36, min(84, note.midi_pitch))
                    harm_events.append(('note_on', start_tick, midi_note))
                    harm_events.append(('note_off', start_tick + duration, midi_note))
                
                harm_events.append(('phrase_end', phrase_end_tick))
            
            harm_events.sort(key=lambda x: x[1])
            
            current_tick = 0
            for event in harm_events:
                event_type = event[0]
                event_tick = event[1]
                delta = max(0, event_tick - current_tick)  # Ensure non-negative
                
                if event_type == 'phrase_start':
                    harm_track.append(mido.Message('note_on', note=105, velocity=100, time=delta))
                    harm_track.append(mido.Message('note_off', note=105, velocity=0, time=1))
                    current_tick = event_tick + 1
                elif event_type == 'phrase_end':
                    harm_track.append(mido.Message('note_on', note=106, velocity=100, time=delta))
                    harm_track.append(mido.Message('note_off', note=106, velocity=0, time=1))
                    current_tick = event_tick + 1
                elif event_type == 'lyric':
                    lyric_text = event[2]
                    harm_track.append(mido.MetaMessage('lyrics', text=lyric_text, time=delta))
                    current_tick = event_tick
                elif event_type == 'note_on':
                    pitch = event[2]
                    harm_track.append(mido.Message('note_on', note=pitch, velocity=100, time=delta))
                    current_tick = event_tick
                elif event_type == 'note_off':
                    pitch = event[2]
                    harm_track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))
                    current_tick = event_tick
            
            mid.tracks.append(harm_track)
        
        mid.save(output_path)
        logger.info(f"Exported vocals MIDI to {output_path}")


def main():
    """Test the vocals charter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe vocals for Clone Hero")
    parser.add_argument("input", help="Input audio file (vocals stem preferred)")
    parser.add_argument("-o", "--output", default="vocals.mid", help="Output MIDI file")
    parser.add_argument("--model", default="medium", help="Whisper model size")
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo in BPM")
    parser.add_argument("--artist", help="Artist name for lyrics lookup")
    parser.add_argument("--title", help="Song title for lyrics lookup")
    parser.add_argument("--no-fetch", action="store_true", help="Skip online lyrics fetch, use Whisper only")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    charter = VocalsCharter(whisper_model=args.model, fetch_lyrics_online=not args.no_fetch)
    lead_phrases, harmony_phrases = charter.transcribe(
        args.input,
        artist=args.artist,
        title=args.title
    )
    
    print(f"\nResults:")
    print(f"  Lead phrases: {len(lead_phrases)}")
    print(f"  Lead notes: {sum(len(p.notes) for p in lead_phrases)}")
    print(f"  Harmony phrases: {len(harmony_phrases)}")
    print(f"  Harmony notes: {sum(len(p.notes) for p in harmony_phrases)}")
    
    charter.export_midi(lead_phrases, harmony_phrases, args.output, tempo_bpm=args.tempo)
    print(f"\nExported to {args.output}")


if __name__ == "__main__":
    main()
