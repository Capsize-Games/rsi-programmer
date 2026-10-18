"""Regenerate the frozen corpus in a temporary checkout and compare bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/generate_benchmark_pre_eval_corpus.js"
CORPUS = ROOT / "fixtures/benchmark_pre_eval_corpus_v1.json"


def main() -> None:
    """Leave the published fixtures untouched while proving reproduction."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        (temporary / "scripts").mkdir()
        (temporary / "fixtures").mkdir()
        script = temporary / "scripts" / SOURCE.name
        shutil.copyfile(SOURCE, script)
        subprocess.run(["node", str(script)], check=True)
        generated = (temporary / "fixtures" / CORPUS.name).read_bytes()
    reproduced = generated == CORPUS.read_bytes()
    report = {
        "node_version": subprocess.check_output(
            ["node", "--version"], text=True).strip(),
        "generator_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(generated).hexdigest(),
        "corpus_bytes_reproduce": reproduced,
    }
    print(json.dumps(report, indent=2))
    if not reproduced:
        raise RuntimeError("generated corpus differs from frozen bytes")


if __name__ == "__main__":
    main()
