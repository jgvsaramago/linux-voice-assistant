"""From an utterance's audio to "who said it", or "unknown".

Scores are cosine similarities between the utterance's embedding and each
enrolled speaker's centroid. They are not probabilities: a speaker is only
named when the best score clears ``threshold`` *and* beats the runner-up by
``margin``; anything less is unknown, so a generic answer is preferred to a
wrong personal one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .backend import SAMPLE_RATE, EmbeddingBackend
from .profiles import ProfileStore

# Energy-based speech trimming: 30 ms frames above the noise floor.
_FRAME = int(SAMPLE_RATE * 0.03)


@dataclass
class SpeakerResult:
    speaker_id: Optional[str]
    #: Best cosine similarity (raw, uncalibrated), or None if nothing was scored.
    score: Optional[float]
    identified: bool
    source: str
    #: Why it's unknown: "no_speech", "too_short", "no_profiles", "below_threshold", "ambiguous", "error".
    reason: Optional[str] = None
    speech_seconds: float = 0.0
    timings_ms: dict = field(default_factory=dict)

    @classmethod
    def unknown(cls, source: str, reason: str, **kw) -> "SpeakerResult":
        return cls(None, kw.pop("score", None), False, source, reason, **kw)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def speech_only(samples: np.ndarray) -> np.ndarray:
    """The frames louder than the quiet ones: drops leading/trailing and
    in-between silence, which carries no speaker information."""
    n = len(samples) // _FRAME
    if n == 0:
        return samples[:0]
    frames = samples[: n * _FRAME].reshape(n, _FRAME)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    # Louder than 3x the quietest fifth — but never above half the loud
    # level, so continuous speech (no pauses to measure a floor from) is kept.
    floor, loud = np.percentile(rms, 20), np.percentile(rms, 80)
    keep = rms > max(min(floor * 3.0, loud * 0.5), 0.004)
    return frames[keep].reshape(-1)


class SpeakerIdentifier:
    def __init__(
        self,
        backend: EmbeddingBackend,
        store: ProfileStore,
        threshold: float = 0.45,
        margin: float = 0.10,
        min_speech_seconds: float = 1.0,
        max_seconds: float = 8.0,
    ) -> None:
        self.backend = backend
        self.store = store
        self.threshold = threshold
        self.margin = margin
        self.min_speech_seconds = min_speech_seconds
        self.max_seconds = max_seconds

    @property
    def source(self) -> str:
        return f"{self.backend.name}:{self.backend.model_id}"

    def embedding(self, samples: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        """Embedding of the speech in ``samples`` (16 kHz float), its length
        in seconds, and why there's none if so."""
        speech = speech_only(samples)
        seconds = len(speech) / SAMPLE_RATE
        if seconds == 0:
            return None, 0.0, "no_speech"
        if seconds < self.min_speech_seconds:
            return None, seconds, "too_short"
        speech = speech[-int(self.max_seconds * SAMPLE_RATE) :]
        return self.backend.embed(speech), seconds, None

    def scores(self, embedding: np.ndarray) -> List[Tuple[float, str]]:
        unit = embedding / (np.linalg.norm(embedding) or 1.0)
        return sorted(((float(np.dot(unit, s.centroid)), sid) for sid, s in self.store.speakers.items()), reverse=True)

    def identify(self, samples: np.ndarray) -> SpeakerResult:
        t0 = time.perf_counter()
        if self.store.incompatible or not self.store.speakers:
            return SpeakerResult.unknown(self.source, "no_profiles")
        try:
            emb, seconds, why = self.embedding(samples)
        except Exception:  # noqa: BLE001 - inference failing must never break the pipeline
            return SpeakerResult.unknown(self.source, "error")
        timings = {"embed": round((time.perf_counter() - t0) * 1000, 1)}
        if emb is None:
            return SpeakerResult.unknown(self.source, why or "no_speech", speech_seconds=seconds, timings_ms=timings)
        ranked = self.scores(emb)
        best, who = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else -1.0
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        common = dict(score=round(best, 3), speech_seconds=round(seconds, 2), timings_ms=timings)
        if best < self.threshold:
            return SpeakerResult.unknown(self.source, "below_threshold", **common)
        if best - second < self.margin:
            return SpeakerResult.unknown(self.source, "ambiguous", **common)
        return SpeakerResult(who, common["score"], True, self.source, None, common["speech_seconds"], timings)
