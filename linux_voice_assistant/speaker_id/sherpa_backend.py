"""CPU backend: an ONNX speaker-embedding model run by sherpa-onnx.

sherpa-onnx computes the model's own features (fbank) inside, so the
application only hands it 16 kHz audio. Imported only when this backend is
selected, keeping it an optional dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import SAMPLE_RATE, BackendUnavailable


class SherpaOnnxBackend:
    name = "cpu"

    def __init__(self, model_path: Path, threads: int = 2) -> None:
        if not model_path.is_file():
            raise BackendUnavailable(f"speaker model not found: {model_path} (run: python -m linux_voice_assistant.speaker_id download)")
        try:
            import sherpa_onnx  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise BackendUnavailable("sherpa-onnx is not installed (pip install sherpa-onnx)") from err
        try:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=threads, provider="cpu")
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except Exception as err:  # noqa: BLE001 - any load failure means "unavailable"
            raise BackendUnavailable(f"could not load {model_path.name}: {err}") from err
        self.dim = int(self._extractor.dim)
        # The model file plus sherpa-onnx's feature pipeline, which is fixed
        # per model: a different model or front end makes embeddings
        # incomparable.
        self.model_id = f"sherpa-onnx/{model_path.name}/fbank16k"

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
        stream.input_finished()
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)
