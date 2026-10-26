# Speaker identification

Optional, fully local identification of **who** is speaking (not what is
said), so Home Assistant can personalise answers: "o meu calendário" means
the speaker's calendar. Off by default (`--speaker-id`).

Voice similarity is probabilistic. This is for convenience, **not
authentication**: never use it to unlock doors, disarm alarms or reveal
anything sensitive.

## How it works

```
mic ─► process_audio (16 kHz mono PCM16 chunks) ─► satellite.handle_audio ─► HA (VoiceAssistantAudio)
                                                         │ same chunk
                                                         ▼
RUN_START ──► SpeakerIdService.begin_run()   ─► event esphome.lva_speaker {status: pending}
STT_VAD_END ► finish(): worker thread embeds + scores (≈80–140 ms) while HA transcribes
             └► event esphome.lva_speaker {status: done, speaker_id, identified, score, …}
RUN_END / error ► end_run(): audio and any late result for the run dropped
```

* **No second microphone stream.** The service only sees the chunks the
  satellite already streams to Home Assistant, i.e. the XVF3800's processed
  (echo-cancelled, beamformed) channel at 16 kHz mono, between `RUN_START`
  and end of speech. The audio thread only appends; inference runs on one
  worker thread; results go back through the event loop.
* **Per run.** Every run is numbered; a result is only reported for the run
  that is still current, and a run's audio is freed once it has been
  embedded. Nothing is kept between interactions.
* **To Home Assistant** as the event `esphome.lva_speaker`, over the existing
  ESPHome API connection (`HomeassistantActionRequest` with `is_event`).
  ESPHome devices may fire `esphome.*` events without the "allow actions"
  option, and HA adds the satellite's `device_id`, so each satellite's
  result is its own.

Event data (all strings, as ESPHome sends them):

| key | meaning |
|---|---|
| `run` | run number on this satellite (correlates `pending` and `done`) |
| `status` | `pending` (at run start) or `done` |
| `speaker_id` | enrolled id, or empty when unknown |
| `identified` | `true` / `false` |
| `score` | best raw cosine similarity (not a probability), or empty |
| `reason` | why unknown: `no_speech`, `too_short`, `no_profiles`, `below_threshold`, `ambiguous`, `error` |
| `source` | backend and model, e.g. `cpu:sherpa-onnx/nemo_en_titanet_small.onnx/fbank16k` |
| `elapsed_ms` | identification time for this utterance |

How a conversation agent uses it is up to Home Assistant. The ByteHomeDash
integration (`custom_components/bytehomedash` in jgvsaramago/bytehomedash)
keeps the latest result per satellite device, waits up to 2 s for a
`pending` one when the agent prepares its turn, and adds the speaker — and
the calendar and to-do list configured for that person — to that turn's
instructions only.

## Model

[NVIDIA TitaNet-small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)
(192-dim embeddings, VoxCeleb-trained), as ONNX, run by
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), which computes the
model's fbank features itself. sherpa-onnx is an optional dependency
(`pip install .[speaker-id]`, ~45 MB); nothing else imports it.

SpeechBrain's ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) was the
suggested starting point. It needs PyTorch (hundreds of MB on aarch64) at
runtime; TitaNet-small is the same family (1-D convolutional speaker
embeddings, VoxCeleb), ships as ONNX, and measured best of the candidates
on the Rock 5B (RK3588, Python 3.13, sherpa-onnx 1.13.8, 2 threads; 3
enrolled speakers × 7 test utterances + 2 never-enrolled speakers from
csukuangfj/sr-data, ~5.5 s each):

| model | per utterance | load | correct | min. same-speaker | max. other-speaker | max. never-enrolled |
|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (VoxCeleb) | 175 ms | 0.6 s | 6/7 | 0.36 | 0.94 | 0.87 |
| WeSpeaker CAM++ | 177 ms | 1.6 s | 6/7 | 0.59 | 0.86 | 0.72 |
| WeSpeaker ResNet34-LM | 505 ms | 0.2 s | 7/7 | 0.83 | 0.88 | 0.73 |
| **NeMo TitaNet-small** | **139 ms** | 0.2 s | **7/7** | **0.72** | **0.34** | **0.10** |

That is a small set: it ranks the models, it doesn't establish an error
rate. Calibrate on your own household (below). ~26 ms of CPU per second of
speech: a 3 s request takes about 80 ms, well inside the ~0.7 s Home
Assistant spends transcribing after end of speech.

## NPU (RK3588)

Evaluated, not used:

* Rockchip's RKNN runtime (`librknnrt`) needs the vendor `rknpu` kernel
  driver. Mainline kernels (this board: Armbian 6.18) drive the NPU with
  the `rocket` driver instead (`/dev/accel/accel0`), which RKNN can't use;
  its user space is Mesa's Teflon TFLite delegate, in Mesa 25.3+ (Debian 13
  ships 25.0). Using it would mean a different kernel or a newer Mesa than
  the rest of the system — not justified for this.
* There is little to win: inference is ~80 ms per request on the CPU, and a
  model this small would spend a good part of an NPU call on data transfer.

`--speaker-id-backend rknn` therefore reports "unavailable" (with the
reason) and `auto` uses the CPU. The backend interface (`speaker_id/backend.py`)
is where an NPU backend would go; it must pass the same embedding check
before it's used.

## Setup

```sh
.venv/bin/pip install "sherpa-onnx>=1.12,<2"          # or: pip install .[speaker-id]
.venv/bin/python -m linux_voice_assistant.speaker_id download   # model, checksum-verified
```

Enable it with `--speaker-id` (or `SPEAKER_ID=1` for the Docker entrypoint)
and restart. The log says `Speaker id on: …, N speaker(s) enrolled`, then
one `Speaker: <id|unknown> (…)` line per request.

| flag / env | default | |
|---|---|---|
| `--speaker-id` / `SPEAKER_ID=1` | off | enable |
| `--speaker-id-backend` / `SPEAKER_ID_BACKEND` | `auto` | `cpu`, `rknn`, `auto` |
| `--speaker-id-model` / `SPEAKER_ID_MODEL` | `local/speaker_models/nemo_en_titanet_small.onnx` | |
| `--speaker-id-profiles` / `SPEAKER_ID_PROFILES` | `local/speaker_profiles.json` | |
| `--speaker-id-threshold` / `SPEAKER_ID_THRESHOLD` | `0.45` | minimum score to name anyone |
| `--speaker-id-margin` / `SPEAKER_ID_MARGIN` | `0.10` | minimum lead over the runner-up |
| `--speaker-id-min-speech` / `SPEAKER_ID_MIN_SPEECH` | `1.0` | seconds of speech needed to try |
| `--speaker-id-threads` / `SPEAKER_ID_THREADS` | `2` | CPU threads |

The voice assistant runs the same if speaker id is off, nobody is enrolled,
the model or sherpa-onnx is missing, inference fails, the profile file is
missing or corrupted, or the utterance is too short: the result is
"unknown" (or no event at all when off) and answers are generic.

## Enrolling people

Stop the service first only if you record from the same microphone and
want it quiet; PipeWire shares the device either way.

```sh
cd ~/linux-voice-assistant
P=".venv/bin/python -m linux_voice_assistant.speaker_id"
$P enroll joao --name "João" --record 5        # 5 recordings of 5 s, press Enter before each
$P enroll ana --name "Ana" ana1.wav ana2.wav   # or WAV files (16-bit PCM, any rate)
$P add joao --record 3                         # more samples later (other distance, tired voice…)
$P enroll joao --record 5 --replace            # start João's profile again
$P list
$P delete ana                                   # removes Ana and all her stored data
$P reset                                        # removes everyone
```

The running assistant notices changes to the profile file (from this CLI or
the ByteHomeDash screen's Definições → Vozes reconhecidas) before the next
request; no restart needed. `--json` makes `list`, `enroll`, `add` and
`delete` print one JSON object, for programs driving the CLI.

Record the samples where people actually talk to the screen — near and a
few metres away, normal and quieter voice. Each needs at least 1 s of
speech (silence is trimmed); unusable samples are rejected with the reason.

The id (`joao`) is what Home Assistant receives. In the ByteHomeDash panel
(Pessoas), put it in that person's **Identificação por voz** field to link
it to their calendar and to-do list.

## Tuning the thresholds

Scores are cosine similarities, not probabilities. Record a few utterances
per person in the room (WAV, as the assistant hears them), plus some from
people who aren't enrolled, then:

```sh
$P identify samples/*.wav
# joao-3.wav: joao (identified; scores: joao 0.71, ana 0.18; 84 ms)
# visita-1.wav: unknown (below_threshold; scores: joao 0.31, ana 0.22; 80 ms)
```

Set `--speaker-id-threshold` between the lowest score of the right person
and the highest score of anyone else, closer to the right person's side if
a wrong name is worse than "unknown" (it is). Raise `--speaker-id-margin`
if two enrolled people score close together. Retest after changing the
room, the microphone or its gain.

## Privacy

* Everything is local; no audio or embedding leaves the device.
* Only embeddings and a display name are stored (`local/speaker_profiles.json`,
  mode 0600 in a 0700 directory); enrolment audio isn't kept. A run's audio
  lives in memory until it's embedded.
* Logs show the id, score and timing — never audio or embeddings.
* Each profile file records the model and preprocessing that produced it.
  Profiles from a different model are refused (never compared) until the
  speakers are re-enrolled or the file is reset.
* Back up: copy `local/speaker_profiles.json` (it's biometric data — keep it
  private). Delete: `delete <id>` or `reset`.

## Tests

`tests/unit/test_speaker_id.py` (no model or hardware: a fake backend):
identification, unknown and ambiguous speakers, silence and short speech,
inference failure, persistence and permissions, incompatible and corrupted
profiles, backend fallback, the per-run service (results on the event loop,
stale runs dropped, one identification per run, no microphone of its own),
the satellite's events, and WAV validation.

Manual end-to-end check on the device: enrol yourself, say the wake word
and ask something, and look for `Speaker: <you> (identified …)` in
`journalctl -u linux-voice-assistant` and the `esphome.lva_speaker` event
in Home Assistant (Developer tools → Events → listen to
`esphome.lva_speaker`).
