# Pi SD-card recovery: paused handoff

**Status: paused at Isaac's request on November 4, 2026.** The repaired image is
ready, but it has **not been written back to the SD card**. Resume only when Isaac
asks to continue. No additional repair, card writes, deploys or dashboard starts
are part of the paused work.

This document is the durable handoff for a later session. Start here when asked
to "resume Pi recovery"; inspect the saved artifacts rather than relying on chat
history. The original card remains damaged until write-back and verification finish.

## What happened

Deploy initially failed on seven corrupt Git objects on `iva585@juanpi.local`.
Those objects were restored from verified copies on the Mac using
[`scripts/repair_pi_git.py`](../scripts/repair_pi_git.py). Deployment now checks
Git integrity before changing calibration or cleaning policy files.

After a Pi restart, Python failed during `import argparse` because `re.compile`
was missing. An isolated interpreter failed too. Package verification showed
changed Python files and unreadable directory entries. Kernel logs confirmed
ext4 directory checksum corruption on `mmcblk0p2` and requested `e2fsck -D`.

The cause is still unknown. `throttled=0x0` and the supplied boot log did not show
undervoltage, but they cannot rule out an earlier power interruption or failing
storage. "No space for directory leaf checksum" concerns directory structure;
it does not establish that the card is full.

## Saved artifacts — keep these

All paths below are on Isaac's Mac. `out/` is ignored by Git: these files stay on
this Mac and are not included in a commit or recovered by cloning the repository.
Do not clean or delete the recovery directory while this task is paused.

| Artifact | Location |
|---|---|
| Recovery directory | `/Users/isaacvelez1/Projects/juan_learns_to_walk/out/pi-sd-recovery.RKCebF/` |
| Original whole-card image | `juanpi-sd.img` in that directory; leave unchanged |
| Repaired Linux-partition image to write back | `rootfs-working.img` in that directory |
| Intermediate metadata-only repaired copy | `rootfs-metadata-only.img` in that directory |
| Geometry, identifiers, status | `device.json` in that directory |
| Checksum files | `juanpi-sd.img.sha256`, `rootfs-working.img.sha256` |
| Filesystem check logs | `fsck-before.log`, `fsck-repair.log`, `fsck-final.log` |
| Python verification report | `python-verification-final.json` |
| Pi Git verification log | `project-audit/git-fsck.log` (empty; check exited 0) |
| Separate robot-file backup | `/Users/isaacvelez1/Desktop/juanpi-recovery.mGRvLu/robot/` |

The separate robot backup contains valid calibration JSON, 125 log files and 42
policy files. Counts and JSON validity do not establish every file's integrity.

Original whole-card image: **31,266,439,168 bytes**; SHA-256:

```text
9f43e52a146f90315d55a313a7582f49e13faa6e7c3ded02f9c598ab443a014a
```

Repaired Linux-partition image: **30,721,179,648 bytes**; SHA-256:

```text
ae3988aeaee936a07b0eacbe6d3369e44469440d026834af4b676772b8938b67
```

## Completed work

1. Installed Homebrew `e2fsprogs` 1.47.4. The repair executable is
   `/opt/homebrew/opt/e2fsprogs/sbin/e2fsck`.
2. Identified the card in the built-in SD reader. At that time it was `/dev/disk4`,
   with FAT32 `bootfs` at `disk4s1` and Linux at `disk4s2`. The built-in reader is
   classified as **internal**, so `diskutil list external physical` omitted it.
3. Unmounted the card. Isaac created the complete original image with `sudo dd`.
   Its size and filesystem UUID matched the Pi. Rechecked the original image's
   SHA-256 after the repair work; it was unchanged.
4. Extracted the Linux partition to a working image. Partition offset:
   **545,259,520 bytes**. Filesystem UUID:
   **`02f61d5e-92cb-4f9c-a781-099d97795020`**.
5. Ran interactive `e2fsck -f -D` on the working image. Repaired directory metadata
   and preserved unattached files/directories in `lost+found`.
6. Restored **44 missing or altered Python library files** from Debian's exact
   installed arm64 package version, **3.13.5-2+deb13u5**. Checked replacement
   contents against the Pi's recorded package checksums before using them.
7. Removed **590 cached Python bytecode files** so the Pi regenerates them.
8. Verified **all 659 recorded files** from `libpython3.13-minimal` and
   `libpython3.13-stdlib`: no missing files or checksum mismatches.
9. Ran a full read-only filesystem check on the repaired image: all five passes
   completed, **exit 0**. Extracted the Pi's Git database and ran `git fsck --full`:
   **exit 0**.

These checks cover filesystem structure, the two Python library packages, and
Git. Other OS packages and the physical card's reliability remain unverified.
No boot test of the repaired filesystem has happened yet.

## Exact plan when resuming

1. **Reconnect and identify the card again.** Run `diskutil list` and
   `diskutil info` for the candidate disk and Linux partition. Match the Secure
   Digital device, whole-card size and partition layout above. **Do not reuse
   `disk4` without checking: device numbers can change.**
2. **Verify the saved repaired image.** Confirm its size and SHA-256 above. Leave
   the original whole-card backup unchanged; no need to repeat the completed
   image repairs if the saved image still matches.
3. **Unmount the whole SD card.** Confirm the Linux partition is unmounted.
4. **Write the repaired Linux-partition image to the verified Linux partition.**
   This is the first pending disk write. It preserves the boot partition and
   restores the Pi's project/calibration/policies/logs from the repaired copy.
   Raw-device access needs the **Mac administrator password in Mac Terminal**;
   the assistant could not obtain passwordless sudo. Never request the password
   in chat.
5. **Verify the physical card after writing.** Run `sync`, a full read-only
   `e2fsck`, then SHA-256 of the physical Linux partition. Require exit 0 and
   exactly the repaired-image hash above. Stop on I/O errors or a hash mismatch.
6. **Eject and boot.** Eject the card, insert it into the powered-off Pi, and boot
   using reliable power.
7. **Check the Pi before starting the dashboard.** Verify Python packages, the
   isolated Python imports, Git integrity and kernel logs using the commands
   below. Review `sudo dpkg -V` for other package damage; configuration-file
   changes can be intentional. Reinstall only what the results justify.
8. **Resume dashboard bring-up only after checks pass.** Start
   `python3 -m robot.dashboard`. If corruption recurs or other OS damage is
   extensive, preserve this card and rebuild on a known-good replacement.

After identifying the device, substitute its current identifier for **diskN**.
The following is a template, not a command to run while the task is paused:

```bash
PI_DISK=diskN
PI_ROOT="${PI_DISK}s2"  # verify Linux is partition 2 before using this
PI_IMAGE="/Users/isaacvelez1/Projects/juan_learns_to_walk/out/pi-sd-recovery.RKCebF/rootfs-working.img"

# First compare this output with the repaired-image hash above:
shasum -a 256 "$PI_IMAGE"

# Only after matching the image and identifying the destination:
diskutil unmountDisk "/dev/$PI_DISK" &&
sudo dd if="$PI_IMAGE" of="/dev/r$PI_ROOT" bs=4m &&
sync &&
sudo /opt/homebrew/opt/e2fsprogs/sbin/e2fsck -f -n "/dev/r$PI_ROOT" &&
sudo shasum -a 256 "/dev/r$PI_ROOT"

# Only after the physical partition's hash also matches:
diskutil eject "/dev/$PI_DISK"
```

On the Pi after boot:

```bash
dpkg -V libpython3.13-minimal libpython3.13-stdlib
python3 -I -S -c 'import re, argparse; print(re.__file__); print(re.compile("test")); print("Standard library OK")'
cd ~/juan_learns_to_walk && git fsck --full
sudo journalctl -k -b --no-pager | grep -Ei 'I/O error|EXT4-fs error|corrupt|under.?voltage'
sudo dpkg -V
```

The targeted Python package check should report no discrepancies. No matches from
the filtered kernel log is a good result; grep then exits 1 because nothing matched.
No blanket Python reinstall is needed for the 44 files already restored offline.

## Pausing and prevention

If no write/check is running, safely eject the currently identified card before
removing it. Do not unplug during `dd`, filesystem repair or read-back verification.
The physical card's connection/ejection state must be checked at resume time.

The dashboard's **Shut down dashboard** button stops the application; it does not
shut down Linux. Before disconnecting Pi power, stop the dashboard and run
`sudo poweroff`, then wait for shutdown to finish.

References: [e2fsck manual](https://manpages.debian.org/trixie/e2fsprogs/e2fsck.8.en.html),
[Debian Python package source](https://deb.debian.org/debian/pool/main/p/python3.13/).
