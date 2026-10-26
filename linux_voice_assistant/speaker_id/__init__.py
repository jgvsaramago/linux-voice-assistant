"""Local speaker identification: who is speaking, not what they say.

Self-contained: nothing here knows about Home Assistant or the voice
pipeline. ``SpeakerIdService`` is the only piece the satellite uses; the
inference runtime stays behind ``EmbeddingBackend`` (see backend.py).
"""

from .backend import BackendUnavailable, EmbeddingBackend, create_backend
from .identifier import SpeakerIdentifier, SpeakerResult
from .profiles import ProfileError, ProfileStore
from .service import SpeakerIdService

__all__ = [
    "BackendUnavailable",
    "EmbeddingBackend",
    "ProfileError",
    "ProfileStore",
    "SpeakerIdService",
    "SpeakerIdentifier",
    "SpeakerResult",
    "create_backend",
]
