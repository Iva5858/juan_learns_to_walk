#!/usr/bin/env bash
# Bring the robot's Raspberry Pi up to date with this computer, in one go:
#   1. the code: the Pi checks out the same branch and pulls it from GitHub (push it first);
#      the Pi's own robot/calibration.json is kept whatever the branch holds;
#   2. what git doesn't carry: the speech and face models (robot/models/) and every exported policy
#      (robot/policies/*.npz), including experimental ones that aren't committed;
#   3. with --install, the Pi packages for voice and faces (OpenCV, Vosk, sounddevice; asks for sudo).
# It doesn't start anything: run python3 -m robot.dashboard on the Pi afterwards.
#
#   ./scripts/deploy_pi.sh                    # code + models + policies
#   ./scripts/deploy_pi.sh --install          # + install the voice and face packages on the Pi
#   PI=iva585@192.168.1.42 ./scripts/deploy_pi.sh     # if juanpi.local doesn't resolve
set -euo pipefail
cd "$(dirname "$0")/.."
PI=${PI:-iva585@juanpi.local}
DIR=${DIR:-juan_learns_to_walk}
BRANCH=${BRANCH:-$(git branch --show-current)}

if [[ -n $(git log "origin/$BRANCH..$BRANCH" 2>/dev/null) ]]; then
    echo "!! $BRANCH has commits that aren't on GitHub: push them first (git push origin $BRANCH)." >&2
    exit 1
fi
[[ -n $(git status --porcelain --untracked-files=no) ]] && echo "note: uncommitted changes here won't reach the Pi (commit and push them)."

echo "=== 1/3 code: $BRANCH on $PI:$DIR"
ssh "$PI" "set -e; cd $DIR
    cp robot/calibration.json /tmp/juan_calibration.json 2>/dev/null || true   # the Pi's own, put back below
    git checkout -q -- robot/calibration.json 2>/dev/null || rm -f robot/calibration.json   # out of git's way
    git fetch --prune -q
    git checkout -q $BRANCH 2>/dev/null || git checkout -q -b $BRANCH origin/$BRANCH
    git pull -q --ff-only
    [ -f /tmp/juan_calibration.json ] && cp /tmp/juan_calibration.json robot/calibration.json
    echo \"   now at \$(git log --oneline -1)\"
    git status --short robot/calibration.json | sed 's/^/   (the Pi keeps its own calibration) /'"

echo "=== 2/3 models and policies"
rsync -az robot/models/ "$PI:$DIR/robot/models/"
rsync -az robot/policies/*.npz "$PI:$DIR/robot/policies/"
ssh "$PI" "ls $DIR/robot/policies/ | sed 's/^/   /'"

if [[ ${1:-} == --install ]]; then
    echo "=== 3/3 packages for voice and faces"
    ssh -t "$PI" "sudo apt-get install -y python3-opencv libportaudio2 &&pip3 install --user --break-system-packages 'vosk==0.3.44' sounddevice"
else
    echo "=== 3/3 packages: skipped (add --install the first time)"
fi
echo "=== done. On the Pi: cd $DIR && python3 -m robot.dashboard"
