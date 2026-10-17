"""Export an explicitly illustrative evidence report without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molactagent.report import render_html_report


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=directory / "outputs" / "evidence_report.html")
    args = parser.parse_args()
    record = json.loads((directory / "illustrative_record.json").read_text(encoding="utf-8"))
    rendered = render_html_report(
        record,
        title="Illustrative chemical evidence report — MolActAgent",
        explanation="Illustrative example only. No activity experiment or LLM inference was performed.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Illustrative report (no API call): {args.output.resolve()}")


if __name__ == "__main__":
    main()
