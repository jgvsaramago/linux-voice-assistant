"""Speaker identification: engine, profiles, run lifecycle, satellite wiring.

No model or hardware needed: a fake backend's "embedding" is the audio's
spectrum in a few bands, and test speakers are tones at different pitches.
"""

import asyncio
import json
import threading
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

from linux_voice_assistant.speaker_id import (
    BackendUnavailable,
    ProfileError,
    ProfileStore,
    SpeakerIdentifier,
    SpeakerIdService,
    create_backend,
)
from linux_voice_assistant.speaker_id.identifier import SpeakerResult, speech_only
from tests.unit.conftest import make_satellite

SR = 16000


class FakeBackend:
    name = "fake"
    model_id = "fake/bands-v1"
    dim = 16

    def __init__(self):
        self.calls = 0

    def embed(self, samples):
        self.calls += 1
        spectrum = np.abs(np.fft.rfft(samples[:SR]))
        bands = np.array([b.sum() for b in np.array_split(spectrum[: SR // 4], self.dim)])
        return (bands / (np.linalg.norm(bands) or 1)).astype(np.float32)


def voice(freq, seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def silence(seconds=2.0):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def pcm16(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def store(tmp_path, backend):
    return ProfileStore(tmp_path / "profiles.json", backend.model_id, backend.dim)


def enrolled(store, backend):
    ident = SpeakerIdentifier(backend, store, threshold=0.8, margin=0.05)
    for sid, freq in (("ana", 300), ("joao", 1200)):
        store.enroll(sid, [ident.embedding(voice(freq, seed=s))[0] for s in range(3)])
    return ident


# --- engine -----------------------------------------------------------------


class TestIdentify:
    def test_identifies_each_enrolled_speaker(self, store, backend):
        ident = enrolled(store, backend)
        assert ident.identify(voice(300, seed=9)).speaker_id == "ana"
        assert ident.identify(voice(1200, seed=9)).speaker_id == "joao"

    def test_unknown_speaker_is_not_guessed(self, store, backend):
        r = enrolled(store, backend).identify(voice(3000, seed=9))
        assert (r.speaker_id, r.identified, r.reason) == (None, False, "below_threshold")
        assert r.score is not None

    def test_ambiguous_when_two_profiles_match_alike(self, store, backend):
        ident = SpeakerIdentifier(backend, store, threshold=0.5, margin=0.05)
        e = ident.embedding(voice(500))[0]
        store.enroll("a", [e])
        store.enroll("b", [e])
        assert ident.identify(voice(500, seed=3)).reason == "ambiguous"

    def test_silence_and_short_speech(self, store, backend):
        ident = enrolled(store, backend)
        assert ident.identify(silence()).reason == "no_speech"
        assert ident.identify(voice(300, seconds=0.4)).reason == "too_short"
        assert ident.identify(np.zeros(0, dtype=np.float32)).reason == "no_speech"

    def test_no_profiles(self, store, backend):
        r = SpeakerIdentifier(backend, store).identify(voice(300))
        assert r.reason == "no_profiles" and backend.calls == 0

    def test_inference_failure_is_unknown_not_a_crash(self, store, backend):
        ident = enrolled(store, backend)
        backend.embed = MagicMock(side_effect=RuntimeError("boom"))
        assert ident.identify(voice(300)).reason == "error"

    def test_speech_only_drops_silence(self):
        audio = np.concatenate([silence(1), voice(300, 1), silence(1)])
        assert abs(len(speech_only(audio)) / SR - 1.0) < 0.1


# --- profiles ----------------------------------------------------------------


class TestProfiles:
    def test_persist_and_reload(self, store, backend, tmp_path):
        enrolled(store, backend)
        again = ProfileStore(store.path, backend.model_id, backend.dim)
        assert sorted(again.speakers) == ["ana", "joao"]
        assert len(again.speakers["ana"].embeddings) == 3
        assert (store.path.stat().st_mode & 0o777) == 0o600

    def test_add_replace_delete(self, store, backend):
        ident = enrolled(store, backend)
        store.add("ana", [ident.embedding(voice(300, seed=7))[0]])
        assert len(store.get("ana").embeddings) == 4
        with pytest.raises(ProfileError):
            store.enroll("ana", [ident.embedding(voice(300))[0]])
        store.enroll("ana", [ident.embedding(voice(300))[0]], replace=True)
        assert len(store.get("ana").embeddings) == 1
        store.delete("ana")
        assert "ana" not in ProfileStore(store.path, backend.model_id, backend.dim).speakers

    def test_bad_ids_rejected(self, store):
        for bad in ("", "Ana", "a b", "x" * 40, "../etc"):
            with pytest.raises(ProfileError):
                store.check_id(bad)

    def test_no_audio_is_stored(self, store, backend):
        enrolled(store, backend)
        raw = json.loads(store.path.read_text())
        assert set(raw["speakers"]["ana"]) == {"name", "updated", "embeddings"}

    def test_other_model_profiles_are_refused_not_compared(self, store, backend):
        enrolled(store, backend)
        other = ProfileStore(store.path, "another/model", backend.dim)
        assert other.incompatible and not other.speakers
        assert SpeakerIdentifier(backend, other).identify(voice(300)).reason == "no_profiles"
        with pytest.raises(ProfileError):
            other.enroll("ana", [np.ones(backend.dim, dtype=np.float32)])
        other.reset()
        assert not other.incompatible

    def test_corrupted_file_is_set_aside(self, tmp_path, backend):
        path = tmp_path / "profiles.json"
        path.write_text("{not json")
        s = ProfileStore(path, backend.model_id, backend.dim)
        assert s.speakers == {} and not path.exists()
        assert list(tmp_path.glob("profiles.broken-*.json"))

    def test_missing_file_is_empty(self, tmp_path, backend):
        assert ProfileStore(tmp_path / "none.json", backend.model_id, backend.dim).speakers == {}


# --- backends ----------------------------------------------------------------


class TestBackends:
    def test_rknn_reports_unavailable(self, tmp_path):
        with pytest.raises(BackendUnavailable):
            create_backend("rknn", tmp_path / "m.onnx")

    def test_auto_falls_back_to_cpu(self, tmp_path, monkeypatch):
        import linux_voice_assistant.speaker_id.sherpa_backend as sb

        monkeypatch.setattr(sb, "SherpaOnnxBackend", lambda path, threads=2: FakeBackend())
        assert create_backend("auto", tmp_path / "m.onnx").name == "fake"

    def test_missing_model_is_unavailable(self, tmp_path):
        with pytest.raises(BackendUnavailable):
            create_backend("cpu", tmp_path / "missing.onnx")

    def test_unknown_backend_name(self, tmp_path):
        with pytest.raises(ValueError):
            create_backend("gpu", tmp_path / "m.onnx")


# --- run lifecycle (service) ---------------------------------------------------


class TestService:
    def _run(self, ident, steps):
        async def go():
            loop = asyncio.get_running_loop()
            svc = SpeakerIdService(ident, loop)
            got = loop.create_future()
            await steps(svc, lambda run, r: got.done() or got.set_result((run, r)), got)
            svc.close()

        asyncio.run(go())

    def test_result_for_the_run_on_the_loop(self, store, backend):
        ident = enrolled(store, backend)

        async def steps(svc, cb, got):
            run = svc.begin_run()
            for i in range(0, 2 * SR * 2, 2048):
                svc.feed(pcm16(voice(300, seed=4))[i : i + 2048])
            assert svc.finish(cb) == run
            got_run, result = await asyncio.wait_for(got, 5)
            assert got_run == run and result.speaker_id == "ana"
            assert threading.current_thread() is threading.main_thread()

        self._run(ident, steps)

    def test_stale_run_is_dropped(self, store, backend):
        ident = enrolled(store, backend)
        gate = threading.Event()
        real = backend.embed
        backend.embed = lambda s: (gate.wait(5), real(s))[1]

        async def steps(svc, cb, got):
            svc.begin_run()
            svc.feed(pcm16(voice(300)))
            svc.finish(cb)
            svc.end_run()  # interaction cancelled while identifying
            gate.set()
            await asyncio.sleep(0.3)
            assert not got.done()

        self._run(ident, steps)

    def test_only_once_per_run_and_not_before_begin(self, store, backend):
        ident = enrolled(store, backend)

        async def steps(svc, cb, got):
            svc.feed(pcm16(voice(300)))  # before any run: ignored
            assert svc.finish(cb) is None
            svc.begin_run()
            svc.feed(pcm16(voice(300)))
            assert svc.finish(cb) is not None
            assert svc.finish(cb) is None  # STT_END after VAD_END
            await asyncio.wait_for(got, 5)

        self._run(ident, steps)

    def test_new_run_forgets_previous_audio(self, store, backend):
        ident = enrolled(store, backend)

        async def steps(svc, cb, got):
            svc.begin_run()
            svc.feed(pcm16(voice(300)))
            svc.begin_run()  # next interaction
            svc.feed(pcm16(voice(1200)))
            svc.finish(cb)
            _, result = await asyncio.wait_for(got, 5)
            assert result.speaker_id == "joao"

        self._run(ident, steps)

    def test_no_microphone_of_its_own(self):
        import inspect

        import linux_voice_assistant.speaker_id.identifier as ident_mod
        import linux_voice_assistant.speaker_id.service as service_mod

        for mod in (service_mod, ident_mod):
            assert "soundcard" not in inspect.getsource(mod)


# --- satellite wiring ----------------------------------------------------------


def _events(sat):
    out = []
    for call in sat.send_messages.call_args_list:
        for msg in call.args[0]:
            if type(msg).__name__ == "HomeassistantActionRequest":
                out.append((msg.service, msg.is_event, {m.key: m.value for m in msg.data}))
    return out


class TestSatellite:
    def _sat(self, tmp_path):
        svc = MagicMock()
        svc.begin_run.return_value = 7
        svc.source = "fake:model"
        sat = make_satellite(tmp_path, state_overrides={"speaker_id": svc})
        sat.send_messages = MagicMock()
        return sat, svc

    def test_run_reports_pending_then_result(self, tmp_path):
        from aioesphomeapi import VoiceAssistantEventType as E

        sat, svc = self._sat(tmp_path)
        sat.handle_voice_event(E.VOICE_ASSISTANT_RUN_START, {})
        assert _events(sat)[-1] == ("esphome.lva_speaker", True, {"run": "7", "status": "pending", "source": "fake:model"})
        sat.handle_voice_event(E.VOICE_ASSISTANT_STT_VAD_END, {})
        svc.finish.assert_called_once()
        callback = svc.finish.call_args.args[0]
        callback(7, SpeakerResult("ana", 0.81, True, "fake:model", None, 2.4, {"total": 80}))
        data = _events(sat)[-1][2]
        assert data["status"] == "done" and data["speaker_id"] == "ana" and data["identified"] == "true"

    def test_stale_result_not_reported(self, tmp_path):
        from aioesphomeapi import VoiceAssistantEventType as E

        sat, svc = self._sat(tmp_path)
        sat.handle_voice_event(E.VOICE_ASSISTANT_RUN_START, {})
        sat.send_messages.reset_mock()
        sat._speaker_result(6, SpeakerResult.unknown("x", "error"))
        assert _events(sat) == []

    def test_unknown_result_reported_as_unknown(self, tmp_path):
        from aioesphomeapi import VoiceAssistantEventType as E

        sat, svc = self._sat(tmp_path)
        sat.handle_voice_event(E.VOICE_ASSISTANT_RUN_START, {})
        sat._speaker_result(7, SpeakerResult.unknown("fake", "below_threshold", score=0.3))
        data = _events(sat)[-1][2]
        assert data["speaker_id"] == "" and data["identified"] == "false" and data["reason"] == "below_threshold"

    def test_run_end_and_error_clear_the_run(self, tmp_path):
        from aioesphomeapi import VoiceAssistantEventType as E

        sat, svc = self._sat(tmp_path)
        sat.handle_voice_event(E.VOICE_ASSISTANT_RUN_END, {})
        sat.handle_voice_event(E.VOICE_ASSISTANT_ERROR, {})
        assert svc.end_run.call_count == 2

    def test_previous_runs_end_does_not_clear_the_new_run(self, tmp_path):
        from aioesphomeapi import VoiceAssistantEventType as E

        sat, svc = self._sat(tmp_path)
        sat._awaiting_run_start = True
        sat.handle_voice_event(E.VOICE_ASSISTANT_RUN_END, {})
        svc.end_run.assert_not_called()

    def test_streamed_audio_is_fed_only_while_streaming(self, tmp_path):
        sat, svc = self._sat(tmp_path)
        sat.handle_audio(b"\x00\x01" * 512)
        svc.feed.assert_not_called()
        sat._is_streaming_audio = True
        sat.handle_audio(b"\x00\x01" * 512)
        svc.feed.assert_called_once()

    def test_disabled_sends_nothing(self, tmp_path):
        from aioesphomeapi import VoiceAssistantEventType as E

        sat = make_satellite(tmp_path)
        sat.send_messages = MagicMock()
        sat.handle_voice_event(E.VOICE_ASSISTANT_RUN_START, {})
        sat.handle_voice_event(E.VOICE_ASSISTANT_STT_VAD_END, {})
        assert _events(sat) == []


# --- CLI audio loading -----------------------------------------------------------


class TestLoadWav:
    def test_resamples_and_downmixes(self, tmp_path):
        from linux_voice_assistant.speaker_id.__main__ import load_wav

        path = tmp_path / "s.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(np.zeros((48000, 2), dtype="<i2").tobytes())
        assert len(load_wav(path)) == SR

    def test_rejects_non_wav_and_8bit(self, tmp_path):
        from linux_voice_assistant.speaker_id.__main__ import load_wav

        bad = tmp_path / "x.wav"
        bad.write_bytes(b"not a wav")
        with pytest.raises(ValueError):
            load_wav(bad)
        eight = tmp_path / "e.wav"
        with wave.open(str(eight), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(16000)
            w.writeframes(b"\x80" * 16000)
        with pytest.raises(ValueError):
            load_wav(eight)
