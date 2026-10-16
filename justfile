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

# Lanes (measured code-v18 era, Ryzen 5800X): test-fast ~4 min;
# slow files seconds to minutes each, except the retry
# distractor search at ~58 min (solved, tried=830954). Full
# `just test` needs ~2.5 h wall; every file stays inside the
# default discover gate, retry included (proven green, not
# quarantined). Dev loop: fast files only (no scale/evolve/
# transfer/beam/author/operands/retry budgets).
test-fast:
    .venv/bin/python -m unittest tests.test_rsi_actions tests.test_rsi_sandbox tests.test_rsi_headless tests.test_proxy tests.test_rsi_hard tests.test_typed_dag_programming_probe tests.test_rsi_decoder tests.test_modular tests.test_rsi_tasks tests.test_edits tests.test_self_gate tests.test_rsi_loop tests.test_autonomy tests.test_rsi_ood tests.test_gains tests.test_rsi_words tests.test_rsi_lattice tests.test_rsi_lattice_train tests.test_rsi_lattice_author -v

train:
    .venv/bin/python scripts/train.py --seed 0 --epochs 50
