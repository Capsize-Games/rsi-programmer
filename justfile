set shell := ["bash", "-cu"]

default:
    @just --list

setup SPIKEFORGE_DIR="$HOME/Projects/spikeforge/spikeforge":
    uv venv .venv
    uv pip install --python .venv/bin/python -r requirements.txt
    SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])"); echo "{{SPIKEFORGE_DIR}}" > "$SITE/spikeforge-src.pth"
    @echo "spikeforge source pinned via spikeforge-src.pth"

test:
    .venv/bin/python -m unittest discover -s tests -t . -v

# Dev loop: fast files only (no scale/evolve/transfer/beam/author/
# operands/retry budgets; retry's distractor search alone takes ~15
# minutes). The full `just test` stays the release gate.
test-fast:
    .venv/bin/python -m unittest tests.test_rsi_actions tests.test_rsi_sandbox tests.test_rsi_headless tests.test_proxy tests.test_rsi_hard tests.test_typed_dag_programming_probe tests.test_rsi_decoder tests.test_modular tests.test_rsi_tasks tests.test_edits tests.test_self_gate -v

train:
    .venv/bin/python scripts/train.py --seed 0 --epochs 50
