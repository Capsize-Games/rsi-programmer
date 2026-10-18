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
# test-fast ~5 min (147 tests); test-slow ~2 h 11 min (60
# tests: hub/author/baseline/beam/evolve/loop/operands/
# scale/transfer); test-retry ~61 min (distractor solved,
# tried=830954). Full `just test` ~3.7 h wall. One builder
# at a time: never run lanes in parallel and never train in
# the background during a build (freeze lesson). A red lane
# stays red and blocking; never split a lane to pass.
# Release gating (code-v39 onward): a release needs
# test-fast plus every lane its diff touches green; the full
# discover runs on main push via CI full (once #44 lands)
# and locally per milestone, not per release. Cross-file
# interference is covered by those full runs, not by
# re-running 3.7 h per release.
test-fast:
    .venv/bin/python -m unittest tests.test_online_plastic_encoder \
        tests.test_online_plastic_benchmark tests.test_audit_plastic_encoder \
        tests.test_plastic_control_comparison \
        tests.test_online_observed_context \
        tests.test_online_context_benchmark tests.test_audit_observed_context \
        tests.test_counterbalanced_protocol \
        tests.test_online_diagnostics tests.test_counterbalanced_audit \
        tests.test_online_spiking \
        tests.test_online_graph tests.test_online_corpus \
        tests.test_online_audit tests.test_online_benchmark \
        tests.test_pre_eval \
        tests.test_pre_eval_contract tests.test_benchmark_policy \
        tests.test_pre_eval_features tests.test_pre_eval_feature_contract \
        tests.test_benchmark_total_work tests.test_total_work_contract \
        tests.test_bounded_pre_eval tests.test_benchmark_bounded \
        tests.test_shortlist_pre_eval tests.test_shortlist_contract \
        tests.test_benchmark_shortlist \
        tests.test_benchmark_fit_work tests.test_frozen_benchmark \
        tests.test_frozen_benchmark_eval \
        tests.test_benchmark_v2_corpus tests.test_benchmark_v2_runner \
        tests.test_benchmark_v2_runner_training \
        tests.test_benchmark_v2_runner_acceptance \
        tests.test_benchmark_v2_runner_provenance \
        tests.test_hub_release_lookup tests.test_rich_amendments tests.test_scale_amendments tests.test_measurement_provenance tests.test_search_amendments tests.test_objective_amendments tests.test_supervision_amendments tests.test_generator_amendments tests.test_review_search tests.test_review_search_edges tests.test_review_acceptance tests.test_review_integrity tests.test_review_lattice tests.test_review_protocol tests.test_rsi_actions tests.test_rsi_sandbox tests.test_rsi_headless tests.test_proxy tests.test_rsi_hard tests.test_typed_dag_programming_probe tests.test_rsi_decoder tests.test_modular tests.test_rsi_tasks tests.test_edits tests.test_self_gate tests.test_autonomy tests.test_rsi_ood tests.test_gains tests.test_rsi_words tests.test_rsi_lattice tests.test_rsi_lattice_train tests.test_rsi_lattice_author tests.test_rsi_bytecode tests.test_rsi_selectivity tests.test_rsi_proposer -v

test-slow:
    .venv/bin/python -m unittest tests.test_hub_verify tests.test_rsi_author tests.test_rsi_baseline tests.test_rsi_beam tests.test_rsi_evolve tests.test_rsi_loop tests.test_rsi_operands tests.test_rsi_scale tests.test_rsi_transfer -v

test-retry:
    .venv/bin/python -m unittest tests.test_rsi_retry -v

# Parallel lanes for local iteration only (measured: 8268s
# wall vs 13530s sequential = 1.64x; fast 147, slow 60,
# retry 3 all green, memory under 70%). Lane granularity
# cannot reach 2x since slow dominates. Release gates and
# CI stay sequential for deterministic logs. One builder at
# a time still applies: never train in the background
# during a par run.
test-par:
    #!/usr/bin/env bash
    set -u
    log=$(mktemp -d /tmp/rsi-par-XXXXXX)
    echo "lane logs in $log"
    just test-fast > $log/fast.log 2>&1 & p_fast=$!
    just test-slow > $log/slow.log 2>&1 & p_slow=$!
    just test-retry > $log/retry.log 2>&1 & p_retry=$!
    code=0
    wait $p_fast || code=1
    wait $p_slow || code=1
    wait $p_retry || code=1
    grep -H -E "^(OK|FAILED|Ran)" $log/fast.log $log/slow.log $log/retry.log
    exit $code

train:
    .venv/bin/python scripts/train.py --seed 0 --epochs 50
