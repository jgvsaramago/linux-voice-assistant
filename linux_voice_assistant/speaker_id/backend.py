"""Speaker-embedding backends behind one interface.

The rest of the application only sees ``EmbeddingBackend``; which runtime
computes the embedding (sherpa-onnx on the CPU today) is chosen here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class BackendUnavailable(RuntimeError):
    """The requested backend can't run here (missing runtime, model, driver)."""


class EmbeddingBackend(Protocol):
    """Turns speech into a fixed-size speaker embedding."""

    #: Short name for logs and results, e.g. ``"cpu"``.
    name: str
    #: Identifies the model *and* preprocessing that produce the embeddings;
    #: profiles store it and are only compared under the same value.
    model_id: str
    dim: int

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Embedding of mono float32 audio at 16 kHz in [-1, 1]."""
        ...


def create_backend(kind: str, model_path: Path, threads: int = 2) -> EmbeddingBackend:
    """A loaded, verified backend: ``cpu``, ``rknn`` or ``auto``.

    ``auto`` tries the NPU first and falls back to the CPU. A backend is only
    returned once it has produced an embedding of the expected size.
    """
    kind = kind.lower()
    if kind not in ("cpu", "rknn", "auto"):
        raise ValueError(f"unknown speaker-id backend {kind!r} (cpu, rknn or auto)")
    if kind in ("rknn", "auto"):
        try:
            return _verified(_rknn_backend(model_path, threads))
        except BackendUnavailable as err:
            if kind == "rknn":
                raise
            _LOGGER.info("Speaker id: NPU backend unavailable (%s); using the CPU", err)
    from .sherpa_backend import SherpaOnnxBackend  # pylint: disable=import-outside-toplevel

    return _verified(SherpaOnnxBackend(model_path, threads=threads))


def _rknn_backend(model_path: Path, threads: int) -> EmbeddingBackend:
    """The RK3588 NPU through Rockchip's RKNN runtime.

    Not implemented: see docs/speaker_identification.md ("NPU"). The RKNN
    runtime (librknnrt) needs Rockchip's vendor ``rknpu`` driver; mainline
    kernels drive the NPU with ``rocket`` instead, reachable only through
    Mesa's Teflon TFLite delegate (Mesa 25.3+). And at ~26 ms of CPU per
    second of speech there is little left to win.
    """
    del model_path, threads
    if not Path("/dev/rknpu").exists() and not any(Path("/sys/kernel/debug").glob("rknpu*")):
        raise BackendUnavailable("no Rockchip rknpu driver (mainline kernels use 'rocket', which librknnrt can't drive)")
    raise BackendUnavailable("RKNN backend not implemented; the CPU backend meets the latency target")


def _verified(backend: EmbeddingBackend) -> EmbeddingBackend:
    probe = backend.embed(np.zeros(SAMPLE_RATE, dtype=np.float32) + 1e-4 * np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32))
    if probe.shape != (backend.dim,) or not np.all(np.isfinite(probe)):
        raise BackendUnavailable(f"{backend.name} produced a bad embedding {probe.shape}")
    return backend
