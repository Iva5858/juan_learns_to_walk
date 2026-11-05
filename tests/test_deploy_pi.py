"""Exercise Git corruption recovery on disposable repos, with no Pi or network."""
import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
import zlib

import pytest

from tests.conftest import load_script

repair = load_script("repair_pi_git.py")


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL).strip()


@pytest.fixture
def repos(tmp_path):
    source = tmp_path / "Mac"
    source.mkdir()
    git(source, "init", "-b", "dev")
    (source / "robot").mkdir()
    (source / "robot/calibration.json").write_text('{"zero": 1}\n')
    git(source, "add", ".")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Initial")
    # The payload also has to work when the healthy source object is packed.
    git(source, "gc")
    target = tmp_path / "Pi's checkout with spaces"
    git(tmp_path, "clone", "--no-hardlinks", str(source), str(target))
    oid = git(source, "rev-parse", "HEAD").decode()
    # Make a loose corrupt object shadow the healthy packed copy.
    obj = target / ".git/objects" / oid[:2] / oid[2:]
    obj.parent.mkdir(exist_ok=True)
    obj.write_bytes(b"corrupt object from interrupted storage write")
    return source, target, oid, obj


def test_repair_over_remote_command_preserves_pi_files_and_index(repos):
    source, target, oid, obj = repos
    (target / "robot/calibration.json").write_text('{"zero": 99}\n')
    (target / "robot/policies/getup").mkdir(parents=True)
    (target / "robot/policies/getup/local.npz").write_bytes(b"Pi-only policy")
    (target / "robot/logs").mkdir()
    (target / "robot/logs/local.npz").write_bytes(b"Pi-only log")
    before = {p.relative_to(target): p.read_bytes() for p in (target / "robot").rglob("*") if p.is_file()}
    index = (target / ".git/index").read_bytes()
    corrupt = obj.read_bytes()
    # Exercise the same Python receiver and shell quoting that SSH runs.
    payload = json.dumps([(oid, base64.b64encode(repair.object_payload(source, oid)).decode())]).encode()
    result = subprocess.run(["sh", "-c", shlex.join([sys.executable, "-c", repair.receiver_source(), str(target)])],
                            input=payload, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert hashlib.sha1(zlib.decompress(obj.read_bytes())).hexdigest() == oid
    assert [p.read_bytes() for p in (target / ".git").glob(f"object-repair-*/{oid}")] == [corrupt]
    assert (target / ".git/index").read_bytes() == index
    assert all((target / p).read_bytes() == data for p, data in before.items())
    assert git(target, "fsck", "--full") == b""
    assert repair.repair_object(target, oid, repair.object_payload(source, oid)) == 0
    assert len(list((target / ".git").glob("object-repair-*"))) == 1


def test_bad_replacement_does_not_touch_corrupt_object(repos):
    _, target, oid, obj = repos
    corrupt = obj.read_bytes()
    with pytest.raises(ValueError, match="verification"):
        repair.repair_object(target, oid, zlib.compress(b"blob 5\0wrong"))
    assert obj.read_bytes() == corrupt
    assert not list((target / ".git").glob("object-repair-*"))


def test_additional_corruption_returns_failure(repos):
    source, target, oid, _ = repos
    blob = git(source, "rev-parse", "HEAD:robot/calibration.json").decode()
    other = target / ".git/objects" / blob[:2] / blob[2:]
    other.parent.mkdir(exist_ok=True)
    other.write_bytes(b"another corrupt object")
    assert repair.repair_object(target, oid, repair.object_payload(source, oid)) != 0


def test_batch_repairs_commit_tree_and_blob_then_checks_once(repos):
    source, target, oid, obj = repos
    oids = [oid, git(source, "rev-parse", "HEAD^{tree}").decode(),
            git(source, "rev-parse", "HEAD:robot/calibration.json").decode()]
    for other in oids[1:]:
        path = target / ".git/objects" / other[:2] / other[2:]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"another corrupt object")
    payloads = [(other, repair.object_payload(source, other)) for other in oids]
    # An invalid later replacement must prevent every write in the batch.
    before = obj.read_bytes()
    with pytest.raises(ValueError, match="verification"):
        repair.repair_objects(target, [payloads[0], (oids[1], zlib.compress(b"blob 5\0wrong"))])
    assert obj.read_bytes() == before
    assert not list((target / ".git").glob("object-repair-*"))
    wire = json.dumps([(other, base64.b64encode(data).decode()) for other, data in payloads]).encode()
    result = subprocess.run([sys.executable, "-c", repair.receiver_source(), str(target)], input=wire, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.count(b"Checking the whole Pi repository") == 1
    assert len(list((target / ".git").glob("object-repair-*"))) == 3
    assert git(target, "fsck", "--full") == b""


def test_deploy_stops_before_cleanup_on_corruption(repos, tmp_path):
    _, target, _, _ = repos
    cal = target / "robot/calibration.json"
    cal.write_text('{"zero": 99}\n')
    policies = target / "robot/policies"
    policies.mkdir()
    policy = policies / "local.npz"
    policy.write_bytes(b"only copy")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text('#!/bin/sh\nexec /bin/bash -c "$2"\n')
    ssh.chmod(0o755)
    # The deploy script currently expects a shell-safe DIR.
    alias = tmp_path / "pi"
    alias.symlink_to(target, target_is_directory=True)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DIR=str(alias), BRANCH="dev", HOME=str(tmp_path))
    script = repair.Path(repair.__file__).with_name("deploy_pi.sh")
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Pi Git integrity check failed" in result.stderr
    assert "=== 2/3" not in result.stdout
    assert cal.read_text() == '{"zero": 99}\n'
    assert policy.read_bytes() == b"only copy"
    assert not (tmp_path / "juan_calibration").exists()
