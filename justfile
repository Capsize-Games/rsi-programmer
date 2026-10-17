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

# Lanes (measured code-v34 era, Ryzen 5800X, CPU-only torch):
# test-fast ~5 min (145 tests); test-slow ~2 h 11 min (51
# tests: hub/author/baseline/beam/evolve/operands/scale/
# transfer); test-retry ~61 min (distractor solved,
# tried=830954). Full `just test` ~3.7 h wall. One builder
# at a time: never run lanes in parallel and never train in
# the background during a build (freeze lesson). A red lane
# stays red and blocking; never split a lane to pass.
test-fast:
    .venv/bin/python -m unittest tests.test_rsi_actions tests.test_rsi_sandbox tests.test_rsi_headless tests.test_proxy tests.test_rsi_hard tests.test_typed_dag_programming_probe tests.test_rsi_decoder tests.test_modular tests.test_rsi_tasks tests.test_edits tests.test_self_gate tests.test_rsi_loop tests.test_autonomy tests.test_rsi_ood tests.test_gains tests.test_rsi_words tests.test_rsi_lattice tests.test_rsi_lattice_train tests.test_rsi_lattice_author tests.test_rsi_bytecode -v

test-slow:
    .venv/bin/python -m unittest tests.test_hub_verify tests.test_rsi_author tests.test_rsi_baseline tests.test_rsi_beam tests.test_rsi_evolve tests.test_rsi_operands tests.test_rsi_scale tests.test_rsi_transfer -v

test-retry:
    .venv/bin/python -m unittest tests.test_rsi_retry -v

train:
    .venv/bin/python scripts/train.py --seed 0 --epochs 50
