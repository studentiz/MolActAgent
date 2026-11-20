"""CLI plumbing for the optional SMILES classifier."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import build_llm_client, load_llm_config
from .io import load_background, load_table


def add_molecular_parser(commands):
    parser = commands.add_parser(
        "molecular", help="Classify SMILES against labelled molecular references."
    )
    parser.add_argument("--reference", required=True, help="Labelled reference table.")
    parser.add_argument("--test", required=True, help="Query table with SMILES.")
    parser.add_argument("--id-column", help="Optional unique ID column in both tables.")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--pchembl-column", help="Optional measured pChEMBL column in reference data.")
    parser.add_argument("--pchembl-threshold", type=float, default=7.0)
    parser.add_argument("--endpoint", help="Assay and label definition. Use @file to read text.")
    parser.add_argument("--series", default="", help="Optional reference compound series.")
    parser.add_argument("--molecular-profile", choices=["panel", "focused"], default="panel")
    parser.add_argument("--neighbors", type=int, default=8, help="Maximum references per class.")
    parser.add_argument("--repeats", type=int, default=5, help="Main-panel judgments per query.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plain-json", action="store_true", help="Omit the API JSON-mode parameter.")
    parser.add_argument("--output-dir", "-o", type=Path, help="Write predictions.json and report.html.")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--replay", type=Path, help="Replay responses with no API connection.")
    recording.add_argument("--record", type=Path, help="Save model responses from this run.")
    return parser


def run_molecular(args):
    from ._molecular_recording import ReplayClient
    from .molecular import MolecularClassifier
    from ._molecular_report import render_molecular_report

    reference = load_table(args.reference)
    test = load_table(args.test)
    endpoint = load_background(args.endpoint) or "Binary activity classification."
    if args.replay:
        if any((args.llm_config, args.profile, args.api_key, args.base_url,
                args.model_name, args.timeout is not None, args.max_retries is not None)):
            raise ValueError("API connection options apply only to live runs.")
        client = ReplayClient(args.replay)
        temperature = client.settings.get("temperature", 0.0) if args.temperature is None else args.temperature
        max_tokens = client.settings.get("max_tokens", 16384) if args.max_tokens is None else args.max_tokens
        response_format = client.settings.get("response_format", True) and not args.plain_json
    else:
        config = load_llm_config(
            config_path=args.llm_config, profile=args.profile, api_key=args.api_key,
            base_url=args.base_url, model_name=args.model_name, temperature=args.temperature,
            max_tokens=args.max_tokens, timeout=args.timeout, max_retries=args.max_retries,
        )
        temperature = config.temperature
        max_tokens = config.max_tokens or 16384
        response_format = not args.plain_json
        client = build_llm_client(config)
    classifier = MolecularClassifier(
        client, endpoint=endpoint, series=args.series, profile=args.molecular_profile,
        workers=args.workers, neighbors=args.neighbors, repeats=args.repeats,
        pchembl_threshold=args.pchembl_threshold, temperature=temperature,
        max_tokens=max_tokens, response_format=response_format,
    ).fit(
        reference, smiles_column=args.smiles_column, label_column=args.label_column,
        id_column=args.id_column, pchembl_column=args.pchembl_column,
    )
    results = classifier.predict(
        test, smiles_column=args.smiles_column, id_column=args.id_column,
        record=args.record is not None,
    )
    metadata = {
        "mode": "replay" if args.replay else "live",
        "model": client.model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "profile": args.molecular_profile,
    }
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with args.record.open("x", encoding="utf-8") as output:
            json.dump(classifier.recording, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
    payload = json.dumps(
        {"run": metadata, "predictions": [asdict(result) for result in results]},
        indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    if args.output_dir is None:
        sys.stdout.write(payload)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "predictions.json").write_text(payload, encoding="utf-8")
        (args.output_dir / "report.html").write_text(
            render_molecular_report(results, endpoint=endpoint, run=metadata),
            encoding="utf-8",
        )
        for result in results:
            print(f"{result.identifier}\t{result.label}")
        print(args.output_dir / "report.html")
    return 4 if any(result.label == "Inconclusive" for result in results) else 0
