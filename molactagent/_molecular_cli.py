"""Command-line entry point for the SMILES classifier."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import MOLECULAR_MAX_TOKENS, build_llm_client, load_llm_config
from .io import load_background, load_table


def add_molecular_parser(commands):
    parser = commands.add_parser(
        "molecular", help="Classify SMILES against labelled molecular references."
    )
    parser.add_argument("--reference", required=True, help="Labelled reference table.")
    parser.add_argument("--test", required=True, help="Query table with SMILES.")
    parser.add_argument("--id-column", help="Optional unique ID column in both tables.")
    parser.add_argument("--name-column", help="Display-name column. Uses name where available.")
    parser.add_argument("--smiles-column", help="SMILES column (default: smiles).")
    parser.add_argument("--label-column", help="Reference label column (default: label).")
    parser.add_argument("--pchembl-column", help="Optional measured pChEMBL column.")
    parser.add_argument("--pchembl-threshold", type=float, help="Assay pChEMBL cutoff, if applicable.")
    parser.add_argument("--labels-from-pchembl", action="store_true", default=None,
                        help="Check reference labels against active at or above the cutoff.")
    parser.add_argument("--endpoint", help="Assay and label definition. Use @file to read text.")
    parser.add_argument("--series", help="Optional reference compound series.")
    parser.add_argument("--molecular-profile", choices=["panel", "focused"],
                        help="Comparison mode (default: panel).")
    parser.add_argument("--neighbors", type=int, help="Maximum references per class (default: 8).")
    parser.add_argument("--repeats", type=int, help="Main-panel judgments per query (default: 5).")
    parser.add_argument("--workers", type=int, help="Parallel model calls (default: 4).")
    parser.add_argument("--plain-json", action="store_true", default=None, help="Omit the API JSON-mode parameter.")
    parser.add_argument("--output-dir", "-o", type=Path, help="Save predictions.json and report.html without overwriting.")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--replay", type=Path, help="Replay responses and settings with no API connection.")
    recording.add_argument("--record", type=Path, help="Save model responses and run settings.")
    return parser


def _display_names(table, column):
    if column is not None and column not in table.columns:
        raise ValueError(f"Missing name column {column!r}.")
    column = column or "name"
    if column not in table.columns:
        return []
    return table[column].astype("string").fillna("").str.strip().tolist()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _settings(args, replay):
    settings = {
        "endpoint": load_background(args.endpoint),
        "series": args.series,
        "profile": args.molecular_profile,
        "workers": args.workers,
        "neighbors": args.neighbors,
        "repeats": args.repeats,
        "pchembl_threshold": args.pchembl_threshold,
        "labels_from_pchembl": args.labels_from_pchembl,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": False if args.plain_json else None,
    }
    columns = {name: getattr(args, name) for name in (
        "smiles_column", "label_column", "id_column", "name_column", "pchembl_column",
    )}
    files = {name: _file_digest(path) for name, path in (
        ("reference", args.reference), ("test", args.test),
    )}
    if replay:
        if any(value is not None for value in (
            args.llm_config, args.profile, args.api_key, args.base_url, args.model_name,
            args.timeout, args.max_retries,
        )):
            raise ValueError("API connection options apply only to live runs.")
        for name, value in settings.items():
            if value is not None and value != replay.settings.get(name):
                raise ValueError(f"Recording setting differs for {name!r}.")
        settings = replay.settings.copy()
        transport = {}
        if replay.command:
            for name, actual in files.items():
                if actual != replay.command["files"].get(name):
                    raise ValueError(f"The {name} file differs from the recording (SHA-256 mismatch).")
            for name, value in columns.items():
                if value is not None and value != replay.command["columns"].get(name):
                    raise ValueError(f"Recording column differs for {name!r}.")
            columns.update(replay.command["columns"])
            transport = replay.command["transport"]
        client = replay
    else:
        config = load_llm_config(
            config_path=args.llm_config, profile=args.profile, api_key=args.api_key,
            base_url=args.base_url, model_name=args.model_name, temperature=args.temperature,
            max_tokens=args.max_tokens, timeout=args.timeout, max_retries=args.max_retries,
            default_max_tokens=MOLECULAR_MAX_TOKENS,
        )
        settings.update(temperature=config.temperature, max_tokens=config.max_tokens)
        transport = {"timeout_seconds": config.timeout, "max_retries": config.max_retries}
        client = build_llm_client(config)
    settings = {name: value for name, value in settings.items() if value is not None}
    columns["smiles_column"] = columns.get("smiles_column") or "smiles"
    columns["label_column"] = columns.get("label_column") or "label"
    return settings, {"files": files, "columns": columns, "transport": transport}, client


@contextmanager
def _reserve_outputs(paths):
    """Reserve names before a paid call and remove files on an incomplete write."""
    with ExitStack() as rollback:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x", encoding="utf-8"):
                    pass
            except FileExistsError as exc:
                raise ValueError(f"Output already exists at {path}. Choose a new path.") from exc
            rollback.callback(path.unlink, missing_ok=True)
        yield
        rollback.pop_all()


def _write_outputs(outputs):
    with ExitStack() as cleanup:
        staged = {}
        for path, content in outputs.items():
            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                    dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                cleanup.callback(temporary.unlink, missing_ok=True)
                output.write(content)
                staged[path] = temporary
        for path, temporary in staged.items():
            temporary.replace(path)


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def run_molecular(args):
    from ._molecular_recording import ReplayClient, runtime_versions
    from ._molecular_report import render_molecular_report
    from .molecular import MolecularClassifier

    replay = ReplayClient(args.replay) if args.replay else None
    settings, command, client = _settings(args, replay)
    columns = command["columns"]
    reference = load_table(args.reference, preserve_text=True)
    test = load_table(args.test, preserve_text=True)
    for name, path in (("reference", args.reference), ("test", args.test)):
        if _file_digest(path) != command["files"][name]:
            raise ValueError(f"The {name} file changed while it was being read. Try again with a stable copy.")
    reference_names = _display_names(reference, columns["name_column"])
    query_names = _display_names(test, columns["name_column"])
    try:
        classifier = MolecularClassifier(client, **settings)
    except TypeError as exc:
        if not replay:
            raise
        raise ValueError("The recording has invalid classifier settings.") from exc
    classifier.fit(reference, smiles_column=columns["smiles_column"], label_column=columns["label_column"],
                   id_column=columns["id_column"], pchembl_column=columns["pchembl_column"])

    paths = []
    output_dir = args.output_dir.expanduser() if args.output_dir else None
    if output_dir:
        paths.extend(output_dir / name for name in ("predictions.json", "report.html"))
    record_path = args.record.expanduser() if args.record else None
    if record_path:
        paths.append(record_path)

    with _reserve_outputs(paths):
        results = classifier.predict(
            test, smiles_column=columns["smiles_column"], id_column=columns["id_column"],
            record=record_path is not None,
        )
        metadata = {
            **classifier.settings,
            "mode": "replay" if replay else "live",
            "model": client.model_name,
            "versions": runtime_versions(),
            **command,
        }
        payload = _json({"run": metadata, "predictions": [asdict(result) for result in results]})
        outputs = {}
        if record_path:
            classifier.recording["command"] = command
            outputs[record_path] = _json(classifier.recording)
        if output_dir:
            outputs[output_dir / "predictions.json"] = payload
            outputs[output_dir / "report.html"] = render_molecular_report(
                results, endpoint=metadata["endpoint"], run=metadata, series=metadata["series"],
                reference_names=dict(enumerate(reference_names)), query_names=query_names,
            )
        _write_outputs(outputs)

    if output_dir:
        for result in results:
            print(f"{result.identifier}\t{result.label}")
        print(output_dir / "report.html")
    else:
        sys.stdout.write(payload)
    return 4 if any(result.label == "Inconclusive" for result in results) else 0
