#!/bin/sh
# Regenerates the decode fixtures. Requires ffmpeg with libmp3lame,
# libvorbis, and libopus. The audio is synthetic (two sine tones), so the
# files carry no licensing restrictions.
#
# Left channel: 1 kHz, right channel: 500 Hz, peak 0.0625, 0.5 s, 44.1 kHz stereo.
set -eu
cd "$(dirname "$0")"

q="-hide_banner -loglevel error -y"
ffmpeg $q \
    -f lavfi -i "sine=frequency=1000:sample_rate=44100:duration=0.5" \
    -f lavfi -i "sine=frequency=500:sample_rate=44100:duration=0.5" \
    -filter_complex "[0][1]amerge=inputs=2,volume=0.5" -ac 2 tone.flac
ffmpeg $q -i tone.flac -c:a libmp3lame -b:a 64k tone.mp3
ffmpeg $q -i tone.flac -c:a aac -b:a 64k tone.m4a
ffmpeg $q -i tone.flac -c:a libvorbis -q:a 2 tone.ogg
ffmpeg $q -i tone.flac -c:a libopus -b:a 32k tone.opus

# Different layouts, appended to the files above to test mid-stream changes.
ffmpeg $q -i tone.flac -ac 1 -c:a libmp3lame -b:a 32k tone-mono.mp3
ffmpeg $q -i tone.flac -ar 22050 -c:a libvorbis -q:a 0 tone-22k.ogg

# Formats added with Symphonia 0.6: 0.2 s, mono, 16 kHz to keep them small.
small="-t 0.2 -ac 1 -ar 16000"
ffmpeg $q -i tone.flac $small -c:a alac tone-alac.m4a
ffmpeg $q -i tone.flac $small tone.aiff
ffmpeg $q -i tone.flac $small -c:a adpcm_ima_wav tone-adpcm.wav
ffmpeg $q -i tone.flac $small -c:a pcm_s16le tone.caf
ffmpeg $q -i tone.flac $small -c:a libopus -b:a 24k tone-opus.webm

# Safety fixtures: a FLAC cut mid-stream (truncated), and a decompression
# bomb (minutes of silence that decode past the sample cap but compress tiny).
head -c 16000 tone.flac > tone-truncated.flac
ffmpeg $q -f lavfi -i "anullsrc=r=48000:cl=stereo" -t 780 -c:a flac tone-bomb.flac
