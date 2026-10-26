# linux-voice-assistant for ByteHomeDash

This fork's `main` is upstream [linux-voice-assistant](https://github.com/OHF-Voice/linux-voice-assistant)
`main` plus the fixes the [ByteHomeDash](https://github.com/jgvsaramago/bytehomedash)
kiosk needs (it runs LVA with a reSpeaker XVF3800 and drives it from its UI
through the peripheral API). One commit per change, each explaining why:

| Commit | What | Why |
|---|---|---|
| Use --name as the ESPHome device name | the ESPHome node name follows `--name` (slugged) instead of always `lva-<mac>` | HA should discover the satellite under the screen's name |
| Fix the satellite getting stuck in "listening" | `stop_pipeline` ends HA's run; a late `RUN_END` no longer cuts the next run's audio; a reply stopped before it starts isn't played | HA's Assist satellite stayed in "listening" until LVA restarted |
| Local speaker identification | `speaker_id/` (TitaNet-small via sherpa-onnx, CPU), enrol/manage CLI, per-run identity sent to HA as the `esphome.lva_speaker` event; opt-in `--speaker-id` (docs/speaker_identification.md) | Home Assistant personalises answers for whoever is speaking |

Install as upstream describes (`script/setup`), from this branch:

```sh
git clone https://github.com/jgvsaramago/linux-voice-assistant.git
cd linux-voice-assistant && script/setup
```

## Updating from upstream

Rebase these commits onto upstream's `main` (`git pull --rebase
https://github.com/OHF-Voice/linux-voice-assistant.git main`), check the
satellite still works with Home Assistant, then push.
