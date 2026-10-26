"""Speaker identification within the voice pipeline's run lifecycle.

The audio thread hands over the chunks it already streams to Home
Assistant (``feed``: an append, nothing else). At end of speech
(``finish``) one worker thread computes the result — concurrently with HA's
speech-to-text — and hands it to the event loop. Each run is numbered: a
result is only delivered for the run that is still current, and the run's
audio is dropped as soon as it has been used.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from .backend import SAMPLE_RATE
from .identifier import SpeakerIdentifier, SpeakerResult, pcm16_to_float

_LOGGER = logging.getLogger(__name__)

ResultCallback = Callable[[int, SpeakerResult], None]


class SpeakerIdService:
    def __init__(self, identifier: SpeakerIdentifier, loop: asyncio.AbstractEventLoop, max_seconds: float = 12.0) -> None:
        self.identifier = identifier
        self._loop = loop
        self._max_bytes = int(max_seconds * SAMPLE_RATE * 2)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker-id")
        self._run = 0
        self._chunks: List[bytes] = []
        self._bytes = 0
        self._collecting = False

    @property
    def source(self) -> str:
        return self.identifier.source

    def begin_run(self) -> int:
        """A new interaction: forget anything from the previous one."""
        with self._lock:
            self._run += 1
            self._chunks = []
            self._bytes = 0
            self._collecting = True
            return self._run

    def feed(self, chunk: bytes) -> None:
        """Called from the audio thread with each 16 kHz mono PCM16 chunk."""
        with self._lock:
            if self._collecting and self._bytes < self._max_bytes:
                self._chunks.append(chunk)
                self._bytes += len(chunk)

    def finish(self, callback: ResultCallback) -> Optional[int]:
        """End of speech: identify this run's audio in the background.
        Only once per run; ``callback(run, result)`` runs on the event loop."""
        with self._lock:
            if not self._collecting:
                return None
            self._collecting = False
            audio, run = b"".join(self._chunks), self._run
            self._chunks, self._bytes = [], 0
        self._executor.submit(self._work, run, audio, callback)
        return run

    def end_run(self) -> None:
        """The interaction ended, was cancelled or failed: drop its audio
        and any result still being computed for it."""
        with self._lock:
            self._run += 1
            self._chunks, self._bytes = [], 0
            self._collecting = False

    def is_current(self, run: int) -> bool:
        with self._lock:
            return run == self._run

    def _work(self, run: int, audio: bytes, callback: ResultCallback) -> None:
        try:
            result = self.identifier.identify(pcm16_to_float(audio))
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Speaker id failed")
            result = SpeakerResult.unknown(self.source, "error")
        del audio
        if self.is_current(run):
            self._loop.call_soon_threadsafe(callback, run, result)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
