"""Learn array edit values from bounded descendant execution outcomes."""
from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn

from rsi_programmer.array_partial import complete, outputs
from rsi_programmer.array_syntax import MAX_TOKENS, TOKENS, analyze

CorpusRow = Mapping[str, object]
Pairs = Sequence[tuple[list[int], list[int]]]
_TOKEN_INDEX = {token: index for index, token in enumerate(TOKENS)}


def _state_features(prefix: tuple[str, ...], action: str,
                    pairs: Pairs) -> list[float]:
    """Encode only current-prefix execution and syntax transition data."""
    child = (*prefix, action)
    inputs = [pair[0] for pair in pairs]
    before, before_valid = outputs(prefix, inputs)
    after, after_valid = outputs(child, inputs)
    features = [0.0] * (len(TOKENS) * 5 + 15)
    features[_TOKEN_INDEX[action]] = 1.0
    for offset, token in enumerate(reversed(prefix[-4:]), start=1):
        features[offset * len(TOKENS) + _TOKEN_INDEX[token]] = 1.0
    base = len(TOKENS) * 5
    features[base] = len(child) / MAX_TOKENS
    try:
        _, allowed = analyze(child)
        features[base + 1] = float(
            bool(allowed) or complete(child) is not None)
    except ValueError:
        features[base + 1] = 0.0
    features[base + 2] = sum(after_valid) / max(1, len(after_valid))
    features[base + 3] = sum(before_valid) / max(1, len(before_valid))
    lengths = [len(row) for row in after]
    sums = [sum(row) for row in after]
    features[base + 4] = sum(lengths) / max(1, len(lengths)) / 512.0
    features[base + 5] = sum(sums) / max(1, len(sums)) / 65536.0
    features[base + 6] = (sum(len(row) for row in after) -
                          sum(len(row) for row in before)) / max(1, len(after))
    features[base + 7] = float(action in ("if", "append", "end"))
    target_rows = [pair[1] for pair in pairs]
    target_lengths = [len(row) for row in target_rows]
    target_sums = [sum(row) for row in target_rows]
    length_errors = [abs(len(got) - len(want))
                     for got, want in zip(after, target_rows)]
    value_errors = []
    exact_rows = []
    for got, want, valid in zip(after, target_rows, after_valid):
        span = max(1, len(got), len(want))
        delta = abs(len(got) - len(want))
        delta += sum(a != b for a, b in zip(got, want))
        value_errors.append(min(1.0, delta / span) if valid else 1.0)
        exact_rows.append(float(valid and got == want))
    features[base + 8] = sum(target_lengths) / max(1, len(target_lengths))
    features[base + 9] = sum(target_sums) / max(1, len(target_sums)) / 65536.0
    features[base + 10] = sum(length_errors) / max(1, len(length_errors))
    features[base + 11] = sum(value_errors) / max(1, len(value_errors))
    features[base + 12] = sum(exact_rows) / max(1, len(exact_rows))
    features[base + 13] = sum(
        abs(value) for row in target_rows for value in row
    ) / max(1, sum(target_lengths)) / 65536.0
    features[base + 14] = float(any(target_rows))
    return features


def _is_success(program: tuple[str, ...], pairs: Pairs) -> bool:
    actual, valid = outputs(program, [pair[0] for pair in pairs])
    return bool(all(valid) and actual == [pair[1] for pair in pairs])


def _walk_rollouts(prefix: tuple[str, ...], first_action: str,
                   pairs: Pairs, rng: random.Random, count: int,
                   max_steps: int) -> tuple[list[int], int, int]:
    """Return binary completed outcomes, invalids, and censored rollouts."""
    outcomes_seen: list[int] = []
    invalid = 0
    censored = 0
    for _ in range(count):
        state = (*prefix, first_action)
        for _step in range(max_steps):
            try:
                body, allowed = analyze(state)
            except ValueError:
                invalid += 1
                break
            if body is not None:
                outcomes_seen.append(int(_is_success(state, pairs)))
                break
            if not allowed:
                censored += 1
                break
            action = rng.choice(allowed)
            state = (*state, action)
        else:
            censored += 1
    return outcomes_seen, invalid, censored


def _training_prefixes(programs: Iterable[Sequence[str]], seed: int
                       ) -> list[tuple[str, ...]]:
    prefixes: set[tuple[str, ...]] = {()}
    rng = random.Random(seed)
    for raw in programs:
        program = tuple(raw)
        for index in range(len(program)):
            state = program[:index]
            try:
                _, allowed = analyze(state)
            except ValueError:
                continue
            if allowed:
                prefixes.add(state)
    # Sample valid off-teacher paths so traces include failed search branches.
    frontier = [()]
    while frontier and len(prefixes) < 48:
        state = frontier.pop(0)
        try:
            _, allowed = analyze(state)
        except ValueError:
            continue
        if not allowed or len(state) >= 18:
            continue
        choices = list(allowed)
        rng.shuffle(choices)
        for action in choices[:2]:
            child = (*state, action)
            if child not in prefixes:
                prefixes.add(child)
                frontier.append(child)
    return sorted(prefixes, key=lambda value: (len(value), value))


def _balanced_transitions(
    prefixes: Sequence[tuple[str, ...]], programs: Sequence[Sequence[str]],
    limit: int, rng: random.Random
) -> list[tuple[tuple[str, ...], str]]:
    """Sample action states across early, middle, and late prefix depths."""
    max_depth = max((len(program) for program in programs), default=1)
    strata: list[list[tuple[tuple[str, ...], str]]] = [[], [], []]
    for prefix in prefixes:
        try:
            _, allowed = analyze(prefix)
        except ValueError:
            continue
        depth_ratio = len(prefix) / max(1, max_depth)
        stratum = min(2, int(depth_ratio * 3))
        strata[stratum].extend((prefix, action) for action in allowed)
    for values in strata:
        rng.shuffle(values)
    selected: list[tuple[tuple[str, ...], str]] = []
    while len(selected) < limit and any(strata):
        for stratum in strata:
            if stratum and len(selected) < limit:
                selected.append(stratum.pop())
    return selected


class EditValueRanker:
    """MLP action-value scorer trained from successful and failed rollouts.

    Corpus rows contain ``signature``, visible ``pairs``, and successful
    engine-generated ``programs``. Future rollout results are labels only;
    scorer inputs contain current-prefix state and typed token features.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(len(TOKENS) * 5 + 15, 64), nn.Tanh(),
                nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1),
                nn.Sigmoid())
        self.traces: list[dict[str, object]] = []
        self.fit_seconds = 0.0
        self.fitted_signatures: tuple[str, ...] = ()

    def fit(self, corpus: Iterable[CorpusRow], *,
            excluded_signatures: Iterable[str] = (), epochs: int = 30,
            rollout_count: int = 2, rollout_steps: int = 32,
            max_transitions: int = 2048, max_per_task: int = 32,
            max_seconds: float = 120.0, max_tasks: int = 64
            ) -> dict[str, object]:
        corpus = list(corpus)
        started = time.process_time()
        excluded = set(excluded_signatures)
        by_signature = {
            str(row["signature"]): row for row in corpus
            if str(row["signature"]) not in excluded
        }
        rows = list(by_signature.values())
        if not rows:
            raise ValueError("training corpus is empty after exclusions")
        eligible_task_count = len(rows)
        rng = random.Random(self.seed)
        rng.shuffle(rows)
        rows = rows[:max_tasks]
        self.traces = []
        vectors: list[list[float]] = []
        labels: list[float] = []
        timed_out = False
        for row in rows:
            pairs = row["pairs"]
            programs = row.get("programs", ())
            if not pairs:
                raise ValueError("each corpus task needs visible pairs")
            programs = [tuple(program) for program in programs]
            prefixes = _training_prefixes(programs, self.seed)
            task_samples = _balanced_transitions(
                prefixes, programs, max_per_task, rng)
            program_depth = max((len(program) for program in programs),
                                default=1)
            task_transitions = 0
            for prefix, action in task_samples:
                if (time.process_time() - started >= max_seconds or
                        len(self.traces) >= max_transitions or
                        task_transitions >= max_per_task):
                    timed_out = (
                        time.process_time() - started >= max_seconds)
                    break
                child = (*prefix, action)
                outcomes_seen, invalid, censored = _walk_rollouts(
                    prefix, action, pairs, rng, rollout_count,
                    rollout_steps)
                # Engine-generated successful suffixes are observed
                # rollouts, while the scorer sees only the child state.
                teacher_success = any(
                    program[:len(child)] == child and
                    _is_success(program, pairs)
                    for program in programs)
                if teacher_success:
                    outcomes_seen.append(1)
                if outcomes_seen:
                    label = sum(outcomes_seen) / len(outcomes_seen)
                    vectors.append(_state_features(prefix, action, pairs))
                    labels.append(label)
                    status = "observed_success_or_failure"
                else:
                    label = None
                    status = "censored"
                depth = len(prefix)
                depth_bin = ("early" if depth * 3 <= program_depth else
                             "middle" if depth * 3 <= 2 * program_depth
                             else "late")
                self.traces.append({
                    "signature": str(row["signature"]),
                    "prefix": prefix, "action": action,
                    "depth": depth, "depth_bin": depth_bin,
                    "teacher_suffix_success": teacher_success,
                    "outcomes": tuple(outcomes_seen),
                    "invalid_rollouts": invalid,
                    "censored_rollouts": censored,
                    "label": label, "status": status,
                })
                task_transitions += 1
                if (len(self.traces) >= max_transitions or
                        task_transitions >= max_per_task):
                    break
            if (len(self.traces) >= max_transitions or timed_out or
                    time.process_time() - started >= max_seconds):
                timed_out = time.process_time() - started >= max_seconds
                break
        if not vectors or len(set(labels)) < 2:
            raise ValueError(
                "rollouts need both successful and failed outcomes")
        features = torch.tensor(vectors, dtype=torch.float32)
        targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.008)
        for _ in range(max(1, epochs)):
            prediction = self.network(features)
            loss = nn.functional.binary_cross_entropy(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.fit_seconds = time.process_time() - started
        self.fitted_signatures = tuple(sorted({str(trace["signature"])
                                               for trace in self.traces}))
        payload = repr(self.traces).encode()
        depth_histogram: dict[str, int] = {}
        depth_bin_histogram = {"early": 0, "middle": 0, "late": 0}
        for trace in self.traces:
            depth = str(trace["depth"])
            depth_histogram[depth] = depth_histogram.get(depth, 0) + 1
            depth_bin_histogram[str(trace["depth_bin"])] += 1
        return {"seed": self.seed, "transitions": len(self.traces),
                "labeled_transitions": len(labels),
                "positive_rollouts": sum(
                    sum(trace["outcomes"]) for trace in self.traces),
                "negative_rollouts": sum(
                    trace["outcomes"].count(0) for trace in self.traces),
                "censored_transitions": sum(t["label"] is None
                                             for t in self.traces),
                "censored_rollouts": sum(
                    int(t["censored_rollouts"]) for t in self.traces),
                "eligible_task_count": eligible_task_count,
                "selected_task_count": len(rows),
                "selected_signatures": tuple(sorted(
                    str(row["signature"]) for row in rows)),
                "unique_tasks_used": len(self.fitted_signatures),
                "depth_histogram": depth_histogram,
                "depth_bin_histogram": depth_bin_histogram,
                "fitted_signatures": self.fitted_signatures,
                "excluded_signatures": tuple(sorted(excluded)),
                "fit_cpu_seconds": self.fit_seconds,
                "fit_time_budget_seconds": max_seconds,
                "fit_time_budget_reached": timed_out,
                "max_transitions": max_transitions,
                "max_per_task": max_per_task,
                "max_tasks": max_tasks,
                "trace_sha256": hashlib.sha256(payload).hexdigest()}

    def score(self, prefix: tuple[str, ...], action: str,
              pairs: Pairs) -> float:
        vector = torch.tensor(_state_features(prefix, action, pairs),
                              dtype=torch.float32)
        with torch.no_grad():
            return float(self.network(vector).item())

    def _synthesize(self, pairs: Pairs, budget: int, beam_width: int,
                    learned: bool, max_seconds: float
                    ) -> dict[str, object]:
        if not pairs:
            raise ValueError("at least one input/output pair is required")
        started = time.process_time()
        beam = [tuple()]
        visited: set[tuple[str, ...]] = set()
        traces: list[dict[str, object]] = []
        best: tuple[str, ...] | None = None
        work = 0
        timed_out = False
        while beam and work < budget:
            ranked: list[tuple[float, tuple[str, ...]]] = []
            for prefix in beam:
                try:
                    _, allowed = analyze(prefix)
                except ValueError:
                    continue
                for action in allowed:
                    if time.process_time() - started >= max_seconds:
                        timed_out = True
                        break
                    child = (*prefix, action)
                    if child in visited:
                        continue
                    visited.add(child)
                    if learned:
                        score = self.score(prefix, action, pairs)
                    else:
                        score = _classical_score(child, pairs)
                    exact = _is_success(complete(child) or child, pairs)
                    traces.append({"prefix": prefix, "action": action,
                                   "score": score, "exact": exact})
                    work += 1
                    if exact:
                        best = complete(child)
                        break
                    ranked.append((score, child))
                    if work >= budget or timed_out:
                        break
                if best is not None or work >= budget or timed_out:
                    break
            if best is not None:
                break
            ranked.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
            beam = [prefix for _, prefix in ranked[:beam_width]]
        return {"program": best, "solved_examples": best is not None,
                "proof": "example_fit_only", "traces": traces,
                "actions_scored": work,
                "cpu_seconds": time.process_time() - started,
                "time_budget_seconds": max_seconds,
                "time_budget_reached": timed_out,
                "seed": self.seed if learned else None}

    def synthesize(self, examplepairs: Pairs, *, budget: int = 2000,
                   beam_width: int = 12, max_seconds: float = 3.0,
                   seconds: float | None = None
                   ) -> dict[str, object]:
        """Run learned best-first beam search with an action budget."""
        if seconds is not None:
            max_seconds = seconds
        return self._synthesize(examplepairs, budget, beam_width, True,
                                max_seconds)

    def classical_synthesize(self, examplepairs: Pairs, *, budget: int = 2000,
                             beam_width: int = 12, max_seconds: float = 3.0,
                             seconds: float | None = None
                             ) -> dict[str, object]:
        """Run identical search with the current execution heuristic."""
        if seconds is not None:
            max_seconds = seconds
        return self._synthesize(examplepairs, budget, beam_width, False,
                                max_seconds)

    def untrained_synthesize(self, examplepairs: Pairs, *, budget: int = 2000,
                             beam_width: int = 12, max_seconds: float = 3.0,
                             seconds: float | None = None
                             ) -> dict[str, object]:
        """Run identical search with a fresh seeded, unfitted MLP."""
        if seconds is not None:
            max_seconds = seconds
        baseline = EditValueRanker(seed=self.seed)
        return baseline._synthesize(examplepairs, budget, beam_width, True,
                                    max_seconds)


def _classical_score(prefix: tuple[str, ...], pairs: Pairs) -> float:
    actual, valid = outputs(prefix, [pair[0] for pair in pairs])
    distances = []
    for got, (_, want), ok in zip(actual, pairs, valid):
        if not ok:
            distances.append(1.0)
            continue
        span = max(1, len(got), len(want))
        difference = abs(len(got) - len(want))
        difference += sum(a != b for a, b in zip(got, want))
        distances.append(min(1.0, difference / span))
    return 1.0 - sum(distances) / max(1, len(distances))
