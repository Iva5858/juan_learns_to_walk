"""Restore corrupt loose Git objects on the Pi from this computer's verified copies.

    python3 scripts/repair_pi_git.py <full object ID>
    python3 scripts/repair_pi_git.py <object ID> <another object ID> ...
    PI=iva585@192.168.1.42 python3 scripts/repair_pi_git.py <full object ID>

Only Git's object database is changed. The old object is backed up inside .git;
calibration, policies, logs, local edits, branches and the index are left alone.
Uses Python's standard library and SSH; does not require a commit or push first.
"""
import argparse
import base64
import hashlib
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import zlib
from pathlib import Path


def object_payload(repo, oid):
    kind = subprocess.check_output(["git", "cat-file", "-t", oid], cwd=repo).strip()
    body = subprocess.check_output(["git", "cat-file", kind.decode(), oid], cwd=repo)
    raw = kind + b" " + str(len(body)).encode() + b"\0" + body
    if hashlib.sha1(raw).hexdigest() != oid:
        raise ValueError("the local object failed verification; the Pi has not been touched")
    return zlib.compress(raw)


def verify_payload(oid, payload):
    import hashlib
    import re
    import zlib

    raw = zlib.decompress(payload)
    header, body = raw.split(b"\0", 1)
    kind, size = header.split(b" ", 1)
    if (not re.fullmatch(r"[0-9a-f]{40}", oid) or kind not in (b"blob", b"tree", b"commit", b"tag")
            or int(size) != len(body) or hashlib.sha1(raw).hexdigest() != oid):
        raise ValueError("replacement failed verification; nothing changed")
    return raw


def repair_object(repo, oid, payload, check=True):
    # Self-contained: this function is sent over SSH, so the Pi need not have this script.
    import os
    import shutil
    import subprocess
    import tempfile
    import zlib
    from pathlib import Path

    raw = verify_payload(oid, payload)
    repo = Path(repo).expanduser().resolve()
    git_dir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=repo, text=True).strip())
    objects = Path(subprocess.check_output(["git", "rev-parse", "--git-path", "objects"], cwd=repo, text=True).strip())
    if not objects.is_absolute():
        objects = repo / objects
    target = objects / oid[:2] / oid[2:]
    healthy = False
    if target.exists():
        try:
            healthy = zlib.decompress(target.read_bytes()) == raw
        except zlib.error:
            pass
    if healthy:
        print(f"Object {oid} is already healthy.", flush=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(prefix="repair-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            if target.exists():
                backup = Path(tempfile.mkdtemp(prefix="object-repair-", dir=git_dir)) / oid
                shutil.copy2(target, backup)
                print(f"Saved corrupt object to {backup}", flush=True)
            os.chmod(staged, 0o444)
            os.replace(staged, target)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
        print(f"Restored {oid}; working files and calibration were not changed.", flush=True)
    if not check:
        return 0
    print("Checking the whole Pi repository...", flush=True)
    result = subprocess.run(["git", "fsck", "--full"], cwd=repo)
    if result.returncode:
        print("More Git errors remain. Stop here and inspect the output before deploying.", flush=True)
    return result.returncode


def repair_objects(repo, payloads):
    # Verify the entire batch before changing any Pi objects.
    if not payloads:
        raise ValueError("no replacement objects supplied; nothing changed")
    for oid, payload in payloads:
        verify_payload(oid, payload)
    for i, (oid, payload) in enumerate(payloads):
        result = repair_object(repo, oid, payload, check=(i == len(payloads) - 1))
    return result


def receiver_source():
    return ("\n".join(inspect.getsource(f) for f in (verify_payload, repair_object, repair_objects))
            + "\nimport base64, json, sys\n"
            + "items = [(oid, base64.b64decode(data, validate=True)) for oid, data in json.load(sys.stdin)]\n"
            + "sys.exit(repair_objects(sys.argv[1], items))\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("object_id", nargs="+", help="one or more full 40-character SHA-1s from the corruption errors")
    args = parser.parse_args()
    oids = list(dict.fromkeys(oid.lower() for oid in args.object_id))
    if any(not re.fullmatch(r"[0-9a-f]{40}", oid) for oid in oids):
        parser.error("provide a full 40-character hexadecimal object ID")
    pi = os.environ.get("PI", "iva585@juanpi.local")
    repo = os.environ.get("DIR", "juan_learns_to_walk")
    local = Path(__file__).resolve().parent.parent
    # Read and verify every local copy before opening SSH.
    payload = json.dumps([(oid, base64.b64encode(object_payload(local, oid)).decode()) for oid in oids]).encode()
    print(f"Verified {len(oids)} local object(s); repairing {pi}:{repo}", flush=True)
    result = subprocess.run(["ssh", "--", pi, shlex.join(["python3", "-c", receiver_source(), repo])], input=payload)
    if result.returncode == 0:
        print("Git integrity check passed. You can rerun ./scripts/deploy_pi.sh.")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
