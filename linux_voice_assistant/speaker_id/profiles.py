"""Enrolled speakers, stored locally as embeddings (never audio).

One JSON file, readable only by its owner. Every profile set records the
model (and preprocessing) that made its embeddings; under any other model
it is refused rather than compared, and the speakers must be re-enrolled.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

_VERSION = 1
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class ProfileError(ValueError):
    """A profile operation that can't be done (bad id, unknown speaker...)."""


@dataclass
class Speaker:
    speaker_id: str
    name: str
    embeddings: List[np.ndarray]
    updated: float

    @property
    def centroid(self) -> np.ndarray:
        """Mean of the unit-length sample embeddings, itself unit length."""
        unit = [e / (np.linalg.norm(e) or 1.0) for e in self.embeddings]
        c = np.mean(unit, axis=0)
        return c / (np.linalg.norm(c) or 1.0)


class ProfileStore:
    def __init__(self, path: Path, model_id: str, dim: int) -> None:
        self.path = path
        self.model_id = model_id
        self.dim = dim
        #: Set when the stored profiles came from another model: identifying
        #: is off until they're re-enrolled (or the file is removed).
        self.incompatible: Optional[str] = None
        self.speakers: Dict[str, Speaker] = {}
        self._load()

    # --- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("version") != _VERSION:
                raise ValueError(f"unknown profile file version {raw.get('version')}")
            if raw.get("model_id") != self.model_id or raw.get("dim") != self.dim:
                self.incompatible = f"profiles were enrolled with {raw.get('model_id')} ({raw.get('dim')}-dim), not {self.model_id} ({self.dim}-dim)"
                _LOGGER.warning("Speaker id: %s; re-enrol the speakers", self.incompatible)
                return
            for sid, s in raw.get("speakers", {}).items():
                embs = [np.asarray(e, dtype=np.float32) for e in s["embeddings"]]
                if not embs or any(e.shape != (self.dim,) for e in embs):
                    raise ValueError(f"speaker {sid} has malformed embeddings")
                self.speakers[sid] = Speaker(sid, s.get("name") or sid, embs, float(s.get("updated", 0)))
        except (OSError, ValueError, KeyError, TypeError) as err:
            # Corrupted or foreign: keep it aside rather than lose or trust it.
            backup = self.path.with_suffix(f".broken-{int(time.time())}.json")
            _LOGGER.error("Speaker id: can't read %s (%s); moved to %s, no speakers enrolled", self.path, err, backup)
            try:
                self.path.replace(backup)
            except OSError:
                pass
            self.speakers = {}

    def save(self) -> None:
        if self.incompatible:
            raise ProfileError(f"{self.incompatible}; delete them first (python -m linux_voice_assistant.speaker_id reset)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        data = {
            "version": _VERSION,
            "model_id": self.model_id,
            "dim": self.dim,
            "speakers": {sid: {"name": s.name, "updated": s.updated, "embeddings": [e.round(6).tolist() for e in s.embeddings]} for sid, s in self.speakers.items()},
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".profiles-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- management -------------------------------------------------------

    @staticmethod
    def check_id(speaker_id: str) -> str:
        if not _ID.match(speaker_id):
            raise ProfileError("speaker id must be lowercase letters, digits, '-' or '_' (e.g. joao, person_2)")
        return speaker_id

    def enroll(self, speaker_id: str, embeddings: List[np.ndarray], name: Optional[str] = None, replace: bool = False) -> Speaker:
        """New speaker (or, with ``replace``, a fresh profile for one)."""
        self.check_id(speaker_id)
        if speaker_id in self.speakers and not replace:
            raise ProfileError(f"{speaker_id} is already enrolled (use add, or enroll --replace)")
        if not embeddings:
            raise ProfileError("no usable samples")
        self.speakers[speaker_id] = Speaker(speaker_id, name or speaker_id, list(embeddings), time.time())
        self.save()
        return self.speakers[speaker_id]

    def add(self, speaker_id: str, embeddings: List[np.ndarray]) -> Speaker:
        speaker = self.get(speaker_id)
        speaker.embeddings.extend(embeddings)
        speaker.updated = time.time()
        self.save()
        return speaker

    def delete(self, speaker_id: str) -> None:
        self.get(speaker_id)
        del self.speakers[speaker_id]
        self.save()

    def reset(self) -> None:
        """Forget everyone, including profiles from another model."""
        self.speakers = {}
        self.incompatible = None
        self.path.unlink(missing_ok=True)

    def get(self, speaker_id: str) -> Speaker:
        if speaker_id not in self.speakers:
            raise ProfileError(f"no speaker {speaker_id!r} enrolled")
        return self.speakers[speaker_id]
