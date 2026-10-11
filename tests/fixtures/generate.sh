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
