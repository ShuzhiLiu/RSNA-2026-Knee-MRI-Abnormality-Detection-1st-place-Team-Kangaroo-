"""Durable local records and content fingerprints."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def dump(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_files(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise RuntimeError(f'Fingerprint mismatch: {path}')


def sealed(path, payload):
    """A receipt binds a completed operation to its exact result and dependencies."""
    dump(path, payload)
    dump(Path(path).with_suffix('.receipt.json'), {'sha256': sha(path)})


def completed(path):
    path = Path(path)
    receipt = path.with_suffix('.receipt.json')
    if not receipt.exists():
        return None
    if sha(path) != read(receipt)['sha256']:
        raise RuntimeError(f'Completed result changed: {path}')
    value = read(path)
    verify_files(value.get('files', {}))
    return value
