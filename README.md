uv run microduck-sounds audition --seed 100

## Parrot mode

An always-on ear that learns short phrases it hears often (3 times by
default) and squawks them back in a parrot voice. Pure numpy — energy VAD,
mel-cepstra + DTW matching, pitch-up warble playback. No models, no cloud.

```bash
# listen on the default mic, learn, squawk
uv run microduck-sounds parrot

# hear the parrot voice on any wav
uv run microduck-sounds parrot-say some_recording.wav
```

Learned phrases persist in `~/.microduck/parrot/` as plain wavs. The `d=`
values in the logs are match distances — tune `--threshold` (default 1.5,
lower = stricter) if it merges different phrases or misses repeats.

On the robot, the mic is held by the runtime's pet worker, so feed the
parrot the same raw stream instead of opening the device twice
(same contract as `pet_detect`: S16LE 16 kHz mono on stdin):

```bash
arecord -D plughw:aic3104,0 -f S16_LE -r 16000 -c 1 -t raw \
  | microduck-sounds parrot --stdin
```

Playback is resampled to 48 kHz by default (the Radxa's I2S clock is pinned
to the 48k family; harmless on the Pi).
