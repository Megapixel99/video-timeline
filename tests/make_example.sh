#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seth Wheeler
#
# Rebuild the worked example in docs/example/ from its sources.
#
# Fetches four CC0 clips from pexels.com and joins them end to end with hard
# cuts, producing a 40-second montage. A montage rather than one clip because
# most of what the format has to say is about structure — where the cuts are,
# which shots resemble which, how the camera behaves in each.
#
# The clips are not committed to the repo: CC0 or not, 40 seconds of video is
# several MB and the whole repo is smaller than that.
#
#   0–10s  360 black and white urban panorama     pexels 35803522
#  10–20s  pan shot of a grass field              pexels 4085319
#  20–30s  metro cdmx                             pexels 19595834
#  30–40s  patterns and lines on white background  pexels 10922866
set -euo pipefail

OUT="${1:-/tmp/vtl-example}"
mkdir -p "$OUT"
cd "$OUT"

fetch() {  # name url
  if [ -s "$1.mp4" ]; then echo "  have $1.mp4"; return; fi
  echo "  fetching $1..."
  curl -fsSL --max-time 180 -A "Mozilla/5.0" "$2" -o "$1.mp4"
}

fetch urban "https://videos.pexels.com/video-files/35803522/15179357_640_360_60fps.mp4"
fetch grass "https://videos.pexels.com/video-files/4085319/4085319-sd_640_360_30fps.mp4"
fetch metro "https://videos.pexels.com/video-files/19595834/19595834-sd_640_360_30fps.mp4"
fetch lines "https://videos.pexels.com/video-files/10922866/10922866-sd_640_360_24fps.mp4"

# normalise everything to one geometry and frame rate so concat is a clean copy,
# and so the only discontinuities are the intended cuts
V="-c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 21"
VF="scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"

cut_seg() {  # index input start
  ffmpeg -v error -y -ss "$3" -t 10 -i "$2.mp4" -an -vf "$VF" $V "seg$1.mp4"
}
cut_seg 1 urban 2
cut_seg 2 grass 1
cut_seg 3 metro 20
cut_seg 4 lines 30

: > list.txt
for n in 1 2 3 4; do echo "file '$OUT/seg$n.mp4'" >> list.txt; done
ffmpeg -v error -y -f concat -safe 0 -i list.txt -c copy example.mp4

echo
echo "built $OUT/example.mp4 ($(du -h example.mp4 | cut -f1), 40s, cuts at 10/20/30s)"
echo "now:  python3 vtl_convert.py $OUT/example.mp4"
