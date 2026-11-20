"""Save one completed molecular decision at a time and resume it under the same settings."""

import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from ._molecular_cli import _write_outputs
from ._molecular_recording import digest


class BatchJournal:
    def __init__(self, root, settings, paths, resume):
        self.root = root
        self.settings = settings
        self.paths = paths
        self.resume = resume
        self.connection = None

    def begin(self):
        manifest = self.root / "manifest.json"
        if self.resume and not manifest.is_file():
            raise ValueError("No saved molecular batch at this output location.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.root / "batch.lock", timeout=0)
        try:
            self.connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise ValueError("Another process is working on this molecular batch.") from exc

        if manifest.exists():
            if not self.resume:
                raise ValueError("This batch already exists. Use --resume or choose a fresh output.")
            if json.loads(manifest.read_text(encoding="utf-8")) != self.settings:
                raise ValueError("Cannot resume because the input, model, versions, settings or destinations changed.")
        else:
            _write_outputs({manifest: json.dumps(self.settings, ensure_ascii=False)})

        seal = self.root / "final-files.json"
        self.owned = json.loads(seal.read_text(encoding="utf-8")) if self.resume and seal.exists() else {}
        for path in self.paths:
            if path.exists() and self.owned.get(str(path.resolve())) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Output already exists at {path}. Choose a new path.")

    def load(self, ordinal, identifier, smiles):
        path = self.root / f"compound-{ordinal + 1:05d}.json"
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        record = entry["record"]
        saved = record["result"]
        if entry["sha256"] != digest(record) or (saved["identifier"], saved["smiles"]) != (identifier, smiles):
            raise ValueError(f"Checkpoint for {identifier} differs from the saved molecule.")
        return record

    def save(self, ordinal, result, calls):
        record = {"result": asdict(result), "calls": calls}
        entry = {"sha256": digest(record), "record": record}
        path = self.root / f"compound-{ordinal + 1:05d}.json"
        _write_outputs({path: json.dumps(entry, indent=2, ensure_ascii=False, allow_nan=False)})

    def finish(self, outputs):
        # Seal the exact bytes before publishing. After a process interruption a
        # matching final file may be reused while remaining files are installed.
        hashes = {str(path.resolve()): hashlib.sha256(content.encode()).hexdigest()
                  for path, content in outputs.items()}
        _write_outputs({self.root / "final-files.json": json.dumps(hashes)})
        with ExitStack() as rollback:
            for path, content in outputs.items():
                key = str(path.resolve())
                if path.exists():
                    if self.owned.get(key) != hashes[key] or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key]:
                        raise ValueError(f"Output already exists at {path}.")
                    continue
                with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        stream.write(content.encode())
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                try:
                    # Exclusive creation protects other runs sharing a recording path.
                    os.link(temporary, path)
                    rollback.callback(path.unlink, missing_ok=True)
                finally:
                    temporary.unlink(missing_ok=True)
            rollback.pop_all()

    def close(self):
        if self.connection is not None:
            self.connection.close()
