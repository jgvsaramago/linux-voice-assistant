"""Manage enrolled speakers and the model: ``python -m linux_voice_assistant.speaker_id``.

  download                     fetch the speaker-embedding model (setup time)
  enroll ID [--name N] (--record K | WAV...) [--replace]
  add ID (--record K | WAV...)  more samples for an enrolled speaker
  list | delete ID | reset
  identify WAV...              who is speaking in each file, with scores
  benchmark WAV...             model load and per-utterance timings

Samples: 16-bit PCM WAV (any rate; resampled to 16 kHz) or recordings
from the microphone, 3 to 8 s of natural speech each, from where people
usually stand. Only embeddings are stored, never the audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import List

import numpy as np

from .backend import SAMPLE_RATE, BackendUnavailable, create_backend
from .identifier import SpeakerIdentifier
from .profiles import ProfileError, ProfileStore

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _REPO / "local" / "speaker_models" / "nemo_en_titanet_small.onnx"
DEFAULT_PROFILES = _REPO / "local" / "speaker_profiles.json"
_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"
_MODEL_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"


def load_wav(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz, or a clear error for anything unusable."""
    try:
        with wave.open(str(path)) as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"{path.name}: only 16-bit PCM WAV is supported")
            rate, channels = w.getframerate(), w.getnchannels()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, EOFError) as err:
        raise ValueError(f"{path.name}: not a readable WAV file ({err})") from err
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    if rate != SAMPLE_RATE and len(audio):
        n = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def record(seconds: float, device: str | None) -> np.ndarray:
    import soundcard  # pylint: disable=import-outside-toplevel

    mic = soundcard.get_microphone(device) if device else soundcard.default_microphone()
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
        return rec.record(int(seconds * SAMPLE_RATE)).reshape(-1).astype(np.float32)


def _out(args, text: str, **data) -> None:
    """Human text, or one JSON object per command with --json (for the
    screen's enrolment, which runs this CLI)."""
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        print(text)


def _samples(args, identifier: SpeakerIdentifier, report: list | None = None) -> List[np.ndarray]:
    sources = []
    if args.record:
        for i in range(args.record):
            input(f"Sample {i + 1}/{args.record}: press Enter, then speak naturally for {args.seconds:.0f} s… ")
            sources.append((f"recording {i + 1}", record(args.seconds, args.device)))
    for path in args.wav or []:
        sources.append((Path(path).name, load_wav(Path(path))))
    if not sources:
        raise ProfileError("give WAV files or --record N")
    embeddings = []
    for label, audio in sources:
        emb, seconds, why = identifier.embedding(audio)
        if report is not None:
            report.append({"sample": label, "ok": emb is not None, "reason": why, "speech_seconds": round(seconds, 2)})
        if emb is None:
            if not args.json:
                print(f"  ✗ {label}: rejected ({why}, {seconds:.1f} s of speech; need {identifier.min_speech_seconds:.1f} s)")
            continue
        if not args.json:
            print(f"  ✓ {label}: {seconds:.1f} s of speech")
        embeddings.append(emb)
    if not embeddings:
        raise ProfileError("none of the samples had enough speech")
    return embeddings


def _download(path: Path) -> None:
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == _MODEL_SHA256:
        print(f"Model already present: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading {_MODEL_URL}")
    urllib.request.urlretrieve(_MODEL_URL, tmp)  # noqa: S310 - fixed https URL
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != _MODEL_SHA256:
        tmp.unlink()
        raise SystemExit(f"Checksum mismatch for the downloaded model ({digest}); not installed")
    tmp.replace(path)
    print(f"Model installed: {path}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m linux_voice_assistant.speaker_id", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    p.add_argument("--backend", default="auto", choices=["cpu", "rknn", "auto"])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--min-speech", type=float, default=1.0)
    p.add_argument("--json", action="store_true", help="machine-readable output (one JSON object)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    for name in ("enroll", "add"):
        s = sub.add_parser(name)
        s.add_argument("speaker_id")
        s.add_argument("wav", nargs="*")
        s.add_argument("--record", type=int, default=0, metavar="N", help="record N samples from the microphone")
        s.add_argument("--seconds", type=float, default=5.0)
        s.add_argument("--device", help="microphone name (default: the system default)")
        if name == "enroll":
            s.add_argument("--name", help="display name")
            s.add_argument("--replace", action="store_true", help="replace an existing profile")
    sub.add_parser("list")
    sub.add_parser("delete").add_argument("speaker_id")
    sub.add_parser("reset")
    sub.add_parser("identify").add_argument("wav", nargs="+")
    sub.add_parser("benchmark").add_argument("wav", nargs="+")
    args = p.parse_args(argv)

    if args.cmd == "download":
        _download(args.model)
        return 0
    t0 = time.perf_counter()
    try:
        backend = create_backend(args.backend, args.model, args.threads)
    except BackendUnavailable as err:
        if args.json:
            print(json.dumps({"ok": False, "error": f"model unavailable: {err}"}))
        else:
            print(f"Speaker model unavailable: {err}", file=sys.stderr)
        return 2
    load_ms = (time.perf_counter() - t0) * 1000
    store = ProfileStore(args.profiles, backend.model_id, backend.dim)
    ident = SpeakerIdentifier(backend, store, args.threshold, args.margin, args.min_speech)
    try:
        if args.cmd == "list":
            if args.json:
                speakers = [{"id": s.speaker_id, "name": s.name, "samples": len(s.embeddings), "updated": s.updated} for s in store.speakers.values()]
                print(json.dumps({"ok": True, "incompatible": store.incompatible, "speakers": speakers}, ensure_ascii=False))
                return 0
            if store.incompatible:
                print(f"Profiles unusable: {store.incompatible}. Re-enrol, or 'reset' to delete them.")
            for s in store.speakers.values():
                print(f"{s.speaker_id:16s} {s.name:20s} {len(s.embeddings)} samples, updated {time.strftime('%Y-%m-%d %H:%M', time.localtime(s.updated))}")
            if not store.speakers and not store.incompatible:
                print("No speakers enrolled.")
        elif args.cmd in ("enroll", "add"):
            report: list = []
            if args.cmd == "add":
                store.get(args.speaker_id)
            embs = _samples(args, ident, report)
            s = store.enroll(args.speaker_id, embs, args.name, args.replace) if args.cmd == "enroll" else store.add(args.speaker_id, embs)
            _out(args, f"{s.speaker_id}: {len(s.embeddings)} samples.", ok=True, id=s.speaker_id, samples=len(s.embeddings), report=report)
        elif args.cmd == "delete":
            store.delete(args.speaker_id)
            _out(args, f"Deleted {args.speaker_id} and all its stored data.", ok=True, id=args.speaker_id)
        elif args.cmd == "reset":
            store.reset()
            print("All speaker profiles deleted.")
        elif args.cmd == "identify":
            for path in args.wav:
                audio = load_wav(Path(path))
                r = ident.identify(audio)
                emb, _, _ = ident.embedding(audio)
                ranked = ", ".join(f"{sid} {sc:.2f}" for sc, sid in ident.scores(emb)) if emb is not None and store.speakers else "-"
                print(f"{Path(path).name}: {r.speaker_id or 'unknown'} ({r.reason or 'identified'}; scores: {ranked}; {r.timings_ms.get('total', r.timings_ms.get('embed', 0))} ms)")
        elif args.cmd == "benchmark":
            print(f"backend={backend.name} model={backend.model_id} dim={backend.dim} load={load_ms:.0f} ms threads={args.threads}")
            for path in args.wav:
                audio = load_wav(Path(path))
                times = []
                for _ in range(5):
                    t = time.perf_counter()
                    ident.embedding(audio)
                    times.append((time.perf_counter() - t) * 1000)
                print(f"{Path(path).name}: {len(audio) / SAMPLE_RATE:.1f} s audio, embedding {np.median(times):.0f} ms (median of 5)")
    except (ProfileError, ValueError) as err:
        if args.json:
            print(json.dumps({"ok": False, "error": str(err)}, ensure_ascii=False))
        else:
            print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
