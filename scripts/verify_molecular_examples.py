"""Replay the published examples and check the decisions and standalone reports."""

import base64
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PNG = re.compile(rb'(?<=src="data:image/png;base64,)[A-Za-z0-9+/=]+')


def check(name, report, expected_exit, nimages):
    inputs = ROOT / "examples" / name
    with TemporaryDirectory() as tmp:
        output = Path(tmp)
        command = [
            sys.executable, "-m", "molactagent", "molecular",
            "--reference", str(inputs / "reference.csv"), "--test", str(inputs / "query.csv"),
            "--replay", str(inputs / "responses.json"), "--output-dir", str(output),
        ]
        run = subprocess.run(command, cwd=ROOT, check=False)
        assert run.returncode == expected_exit, (name, run.returncode)
        predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))["predictions"]
        if name == "cdk2":
            live = json.loads((inputs / "predictions.json").read_text(encoding="utf-8"))["predictions"]
            assert predictions == live
        else:
            row, = predictions
            assert (row["identifier"], row["label"]) == ("Q1", "active")
            assert [r["reference"] for r in row["references"]] == ["A2", "A1", "A3", "I3", "I2", "I1"]
            assert [v["label"] for v in row["judgments"]] == ["active"] * 5
            assert (row["decision"]["decisive"], row["decision"]["required"]) == (5, 3)
            shown = {r["position"] for r in row["evidence_references"]}
            assert all(set(v["evidence"]["reference_ids"]) <= shown for v in row["judgments"])
        html = (output / "report.html").read_bytes()
        images = PNG.findall(html)
        assert len(images) == nimages
        assert all(base64.b64decode(img, validate=True).startswith(b"\x89PNG\r\n\x1a\n") for img in images)
        expected = (ROOT / "docs" / report).read_bytes()
        # Platform-specific Cairo encoders can produce different PNG bytes.
        assert PNG.sub(b"[PNG]", html) == PNG.sub(b"[PNG]", expected)
        print(f"Verified {name} decisions, structures and report.")


if __name__ == "__main__":
    check("molecular", "molecular-example.html", 0, 9)
    check("cdk2", "cdk2-report.html", 4, 7)
