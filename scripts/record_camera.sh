#!/usr/bin/env bash
# Record the robot's camera on this computer, raw and IMU-stabilized at the same time, until Enter; then a
# side-by-side video of the two (raw left). The dashboard must be running with the camera calibration
# (JUAN_CAMERA_CALIBRATION=..., docs/camera-stabilization.md). Videos: out/camera/<name>_*.mp4.
#
#   ./scripts/record_camera.sh                    # named by the time
#   ./scripts/record_camera.sh walk_v7_24         # named
#   HOST=192.168.1.42 ./scripts/record_camera.sh  # if juanpi.local doesn't resolve
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-juanpi.local}
NAME=${1:-$(date +%Y%m%d_%H%M%S)}
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=out/camera/$NAME
mkdir -p out/camera

pids=()
for stream in camera camera-stabilized; do
    "$FF" -nostdin -loglevel error -y -use_wallclock_as_timestamps 1 -i "http://$HOST:8000/$stream.mjpg" \
        -c:v libx264 -pix_fmt yuv420p "${OUT}_$stream.mp4" &
    pids+=($!)
done
read -rp "Recording raw and stabilized from $HOST: Enter stops. "
kill -INT "${pids[@]}" 2>/dev/null || true   # ffmpeg finishes the files cleanly on SIGINT
wait || true

"$FF" -nostdin -loglevel error -y -i "${OUT}_camera.mp4" -i "${OUT}_camera-stabilized.mp4" \
    -filter_complex "[0:v][1:v]hstack=shortest=1" -c:v libx264 -pix_fmt yuv420p "${OUT}_side_by_side.mp4"
echo "Saved ${OUT}_camera.mp4, ${OUT}_camera-stabilized.mp4 and ${OUT}_side_by_side.mp4"
