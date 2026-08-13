#!/bin/bash
# Build a synthetic test video with known ground truth, so the converter's
# output can be checked against what is actually in the file.
#
# Ground truth (shot: time — content — expected label):
#   1: 0.0– 4.0  dark blue title card "CHAPTER ONE", fades in   -> static, fade_in
#   2: 4.0– 9.0  camera pans right across a test pattern        -> pan_right
#   3: 9.0–13.0  dark blue title card "CHAPTER TWO"             -> static, resembles shot 1
#   4: 13.0–17.0 testsrc2, heavy motion                         -> high activity
#   5: 17.0–21.0 camera zooms in (hue-shifted, so the cut is visible) -> zoom_in
#   6: 21.0–25.0 completely frozen frame                        -> frozen span
# Audio: silence 0–2, 440 Hz tone 2–8, silence 8–11, noise 11–25.
set -euo pipefail
cd "$(dirname "$0")"
F=/System/Library/Fonts/Supplemental/Arial.ttf
[ -f "$F" ] || F=/System/Library/Fonts/Helvetica.ttc
S=1280x720
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
V="-c:v libx264 -pix_fmt yuv420p -r 30 -preset veryfast"

ffmpeg -v error -y -f lavfi -i "color=c=0x102040:s=$S:d=4" \
  -vf "drawtext=fontfile=$F:text='CHAPTER ONE':fontcolor=white:fontsize=80:x=(w-tw)/2:y=(h-th)/2,drawtext=fontfile=$F:text='a quiet beginning':fontcolor=0xbbbbbb:fontsize=34:x=(w-tw)/2:y=(h-th)/2+90,fade=in:st=0:d=1" $V "$TMP/1.mp4"

ffmpeg -v error -y -f lavfi -i "testsrc2=s=2560x720:d=5:r=30" \
  -vf "crop=1280:720:x='min(t*250,1280)':y=0" $V "$TMP/2.mp4"

ffmpeg -v error -y -f lavfi -i "color=c=0x102040:s=$S:d=4" \
  -vf "drawtext=fontfile=$F:text='CHAPTER TWO':fontcolor=white:fontsize=80:x=(w-tw)/2:y=(h-th)/2,drawtext=fontfile=$F:text='the return':fontcolor=0xbbbbbb:fontsize=34:x=(w-tw)/2:y=(h-th)/2+90" $V "$TMP/3.mp4"

ffmpeg -v error -y -f lavfi -i "testsrc2=s=$S:d=4:r=30" $V "$TMP/4.mp4"

ffmpeg -v error -y -f lavfi -i "testsrc2=s=$S:d=4:r=30" \
  -vf "hue=h=140:s=1.4,zoompan=z='min(1+0.012*in,1.8)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=$S:fps=30" \
  $V "$TMP/5.mp4"

ffmpeg -v error -y -f lavfi -i "color=c=0x603010:s=$S:d=4" \
  -vf "drawtext=fontfile=$F:text='END OF TAPE':fontcolor=white:fontsize=64:x=(w-tw)/2:y=(h-th)/2" $V "$TMP/6.mp4"

for i in 1 2 3 4 5 6; do echo "file '$TMP/$i.mp4'"; done > "$TMP/list.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$TMP/list.txt" -c copy "$TMP/video.mp4"

ffmpeg -v error -y \
  -f lavfi -i "sine=f=440:d=25" -f lavfi -i "anoisesrc=d=25:c=pink:a=0.3" \
  -filter_complex "[0:a]volume='between(t,2,8)':eval=frame[a];[1:a]volume='between(t,11,25)*0.6':eval=frame[b];[a][b]amix=inputs=2:normalize=0[out]" \
  -map "[out]" -c:a aac -b:a 128k "$TMP/audio.m4a"

ffmpeg -v error -y -i "$TMP/video.mp4" -i "$TMP/audio.m4a" -c copy -shortest fixture.mp4
ffmpeg -v error -y -i fixture.mp4 -c copy -f mov fixture.mov
echo "wrote $(pwd)/fixture.mp4 and fixture.mov"
