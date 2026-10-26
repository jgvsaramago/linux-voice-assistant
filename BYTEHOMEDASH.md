# linux-voice-assistant for ByteHomeDash

This fork's `bytehomedash` branch is [linux-voice-assistant](https://github.com/OHF-Voice/linux-voice-assistant)
`v1.1.16` plus the fixes the [ByteHomeDash](https://github.com/jgvsaramago/bytehomedash)
kiosk needs (it runs LVA with a reSpeaker XVF3800 and drives it from its UI
through the peripheral API). One commit per change, each explaining why:

| Commit | What | Why |
|---|---|---|
| Use --name as the ESPHome device name | the ESPHome node name follows `--name` (slugged) instead of always `lva-<mac>` | HA should discover the satellite under the screen's name |
| Fix the satellite getting stuck in "listening" | `stop_pipeline` ends HA's run; a late `RUN_END` no longer cuts the next run's audio; a reply stopped before it starts isn't played | HA's Assist satellite stayed in "listening" until LVA restarted |

Install as upstream describes (`script/setup`), from this branch:

```sh
git clone -b bytehomedash https://github.com/jgvsaramago/linux-voice-assistant.git
cd linux-voice-assistant && script/setup
```
