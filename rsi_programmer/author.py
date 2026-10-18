#!/usr/bin/env python3
"""SNN edits: greedy author (#37), beam (#38), operands (#39)."""
from __future__ import annotations
import math
import random
import statistics
from typing import Any
import torch
from rsi_programmer.actions import ActionSpace
from rsi_programmer.search import guided_search
from rsi_programmer.policy import DIM, OP_ORDER, Group, Rows
from rsi_programmer.policy import SearchPolicy, Vals
from rsi_programmer.policy import Triple, encode, layer_triples
from rsi_programmer.rich import listwise_loss
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import Expr, const_node, const_terms
from rsi_programmer.probe import enumerate_programs, minimize_graph
from rsi_programmer.probe import signature, terminals
# Classical author bonus terms appended after SNN scores; encode_author keeps
# them outside the DIM neural input, including exact_flag.
EXTRA = ("nodes_per_bound", "open_gap", "exact_flag", "self_apply")
OPDIMS = tuple(range(len(OP_ORDER) + 3, len(OP_ORDER) + 7))
TaskRow = tuple[str, Rows, Vals, int]
Scored = tuple[list[Triple], list[Vals], list[int], list[float]]
Stash = dict[int, list[tuple[float, Triple, Vals]]]
SOLUTION_CANDIDATE_CAP = 50_000
def open_gap(sigs: list[Vals], target: Vals) -> float:
    return 1.0 - max([sum(1 for a, b in zip(s, target) if a == b)
                       / (len(target) or 1) for s in sigs] or [0.0])
def encode_author(triple: Triple, sig: Vals, rows: Rows, target: Vals,
                  size: int, bound: int, tried: int, total: int,
                  nodes: list[Any], gap: float) -> list[float]:
    op, left, right = triple
    feats = encode(op, signature(left, rows), signature(right, rows),
                   target, left.size, right.size, size, bound, tried, total)
    exact, same = float(sig == target), float(left is right)
    return feats + [len(nodes) / (float(bound) or 1.0), gap, exact, same]
def author_scores(policy: SearchPolicy, feats: list[list[float]],
                  weights: list[float]) -> list[float]:
    """SNN scores over DIM fields plus an explicit classical bonus."""
    base = policy.scores([f[:DIM] for f in feats])
    return [b + sum(w * e for w, e in zip(weights, f[DIM:]))
            for b, f in zip(base, feats)]
def solution_sigs_by_size_report(
        rows: Rows, target: Vals, bound: int,
        candidate_cap: int = SOLUTION_CANDIDATE_CAP
        ) -> tuple[dict[int, set[Vals]], dict[str, Any]]:
    """Find completing signatures by size in the train-only graph.

    The retained graph deduplicates nodes by layer and training signature.
    If a candidate cap interrupts a layer, no labels are returned: treating
    unexamined candidates as negative would reintroduce false negatives.
    """
    if type(candidate_cap) is not int or candidate_cap < 1:
        raise ValueError("solution evidence cap must be a positive integer")
    if type(bound) is not int or bound < 1:
        raise ValueError("solution bound must be a positive integer")
    edges: dict[tuple[int, Vals], set[tuple[int, Vals]]] = {}
    roots: set[tuple[int, Vals]] = set()
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    examined = 0
    truncated = False
    found: dict[int, set[Vals]] = {}
    solved = False
    for size in range(1, bound + 1):
        remaining = candidate_cap - examined
        triples = layer_triples(levels, size, limit=remaining + 1)
        if len(triples) > remaining:
            truncated = True
            break
        current: dict[Any, Expr] = {}
        for op, left, right in triples:
            examined += 1
            expr = Expr(op, left, right, size)
            sig = signature(expr, rows)
            if sig == target:
                roots.add((size, sig))
                solved = True
            children: set[tuple[int, Vals]] = set()
            if left.size:
                children.add((left.size, signature(left, rows)))
            if right.size:
                children.add((right.size, signature(right, rows)))
            edges.setdefault((size, sig), set()).update(children)
            current.setdefault((op, sig), expr)
        levels[size] = list(current.values())
    if not truncated and not solved:
        raise ValueError("training target has no solution within bound")
    if truncated:
        return {}, {
            "label_semantics": "retained_train_signature_graph_reachability",
            "candidate_signature_evaluations": examined,
            "candidate_cap": candidate_cap,
            "truncated": True,
            "retained_graph_complete": False,
            "bound": bound,
        }
    pending = list(roots)
    seen: set[tuple[int, Vals]] = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        found.setdefault(node[0], set()).add(node[1])
        pending.extend(edges.get(node, ()))
    return found, {
        "label_semantics": "retained_train_signature_graph_reachability",
        "candidate_signature_evaluations": examined,
        "candidate_cap": candidate_cap,
        "truncated": False,
        "retained_graph_complete": True,
        "scope": "all_retained_layers_through_bound",
        "bound": bound,
    }


def solution_sigs_report(
        rows: Rows, target: Vals, bound: int,
        candidate_cap: int = SOLUTION_CANDIDATE_CAP
        ) -> tuple[set[Vals], dict[str, Any]]:
    """Return the union of completing signatures and its bounded scope."""
    by_size, report = solution_sigs_by_size_report(
        rows, target, bound, candidate_cap)
    return set().union(*by_size.values()), report


def solution_sigs(rows: Rows, target: Vals, bound: int) -> set[Vals]:
    """Completable subtree signatures from the retained train-only graph."""
    found, report = solution_sigs_report(rows, target, bound)
    if report["truncated"]:
        raise ValueError("train solution graph exceeded evidence cap")
    return found


def canonical_solution_sigs(rows: Rows, target: Vals,
                            bound: int) -> set[Vals]:
    """Subtree labels from the first canonical train solution (ablation)."""
    expr, _, _ = enumerate_programs(rows, target, bound)
    if expr is None:
        raise ValueError("training target has no solution within bound")
    found: set[Vals] = set()
    stack = [expr]
    while stack:
        term = stack.pop()
        if term.op == "input":
            continue
        found.add(signature(term, rows))
        if isinstance(term.left, Expr):
            stack.append(term.left)
        if isinstance(term.right, Expr):
            stack.append(term.right)
    return found
def _collect_layers(rows: Rows, target: Vals,
                    bound: int) -> list[tuple[int, list[Triple]]]:
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    layers: list[tuple[int, list[Triple]]] = []
    for size in range(1, bound + 1):
        triples = layer_triples(levels, size)
        layers.append((size, triples))
        exprs: dict[Any, Expr] = {}
        for op, left, right in triples:
            expr = Expr(op, left, right, size)
            exprs.setdefault((op, signature(expr, rows)), expr)
        levels[size] = list(exprs.values())
        if any(signature(e, rows) == target for e in levels[size]): break
    return layers
def _label_layer(size: int, triples: list[Triple], rows: Rows,
                 target: Vals, bound: int, offset: int,
                 budget: int, want: set[Vals]) -> Group:
    group: Group = []
    for i, (op, left, right) in enumerate(triples):
        sig = signature(Expr(op, left, right, size), rows)
        feats = encode(op, signature(left, rows), signature(right, rows),
                       target, left.size, right.size, size, bound,
                       i + 1, len(triples))
        group.append((feats, 1 if sig in want else 0))
    return group
def _base_groups(fixture: str, budget: int = 10 ** 9,
                 supervision: str = "reachability",
                 supervision_report: list[dict[str, Any]] | None = None
                 ) -> list[Group]:
    if supervision not in ("reachability", "canonical"):
        raise ValueError("unknown author supervision mode")
    rec = TraceRecorder(fixture)
    frozen = {r["task"]: r for r in rec.record_all()["records"]}
    groups: list[Group] = []
    for task in rec.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        layers = _collect_layers(rows, target, bound)
        total = sum(len(t) for _, t in layers)
        assert total == len(frozen[task["id"]]["choice_pairs"])
        if supervision == "canonical":
            want = canonical_solution_sigs(rows, target, bound)
            want_by_size = {size: want for size, _triples in layers}
            task_report = {
                "task": task["id"],
                "bound": bound,
                "mode": "canonical",
                "included": True,
                "label_semantics": "canonical_train_solution_subtrees",
            }
        else:
            want_by_size, report = solution_sigs_by_size_report(
                rows, target, bound)
            task_report = {"task": task["id"], "bound": bound,
                           "mode": "reachability",
                           "included": not report["truncated"], **report}
            if report["truncated"]:
                # Unknown candidates are not safe negatives. Omit this task's
                # groups rather than training on incomplete reachability.
                if supervision_report is not None:
                    supervision_report.append(task_report)
                continue
        if supervision_report is not None:
            supervision_report.append(task_report)
        offset = 0
        for size, triples in layers:
            groups.append(_label_layer(size, triples, rows, target,
                                       bound, offset, budget,
                                       want_by_size.get(size, set())))
            offset += len(triples)
    return groups
def _kept(pairs: Group) -> bool:
    var = [statistics.pvariance([f[d] for f, _ in pairs]) for d in OPDIMS]
    return 0 < sum(y for _, y in pairs) < len(pairs) and max(var) > 0.0
def _contrastive_from(groups: list[Group]) -> list[Group]:
    out: list[Group] = []
    for group in groups:
        parts: dict[int, Group] = {}
        for feats, label in group:
            op = max(range(len(OP_ORDER)), key=lambda i: feats[i])
            parts.setdefault(op, []).append((feats, label))
        out += [p for p in parts.values() if _kept(p)]
    return out


def contrastive_groups(fixture: str, budget: int = 10 ** 9,
                       supervision: str = "reachability",
                       supervision_report: list[dict[str, Any]] | None = None
                       ) -> list[Group]:
    return _contrastive_from(_base_groups(
        fixture, budget, supervision, supervision_report))


def clone_groups(fixture: str, budget: int = 10 ** 9,
                 supervision: str = "reachability",
                 supervision_report: list[dict[str, Any]] | None = None
                 ) -> list[Group]:
    groups = _base_groups(fixture, budget, supervision,
                          supervision_report)
    return groups + _contrastive_from(groups)
JOINT_LAMBDA = 0.2
JOINT_EPOCHS = 5


def _tensed_pairs(fixture: str,
                  supervision: str = "reachability"
                  ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Operand pairs as score/label tensors."""
    tensed: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group in contrastive_groups(fixture, supervision=supervision):
        feats = torch.tensor([f for f, _ in group])
        labels = torch.tensor([y for _, y in group])
        tensed.append((feats, labels))
    return tensed


def _tensed_margin(groups: list[Group]) -> list[tuple[torch.Tensor,
                                                     torch.Tensor]]:
    """Positive/negative tensors for the margin pass."""
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group in groups:
        pos = torch.tensor([f for f, y in group if y == 1])
        neg = torch.tensor([f for f, y in group if y == 0])
        out.append((pos, neg))
    return out


def refine_operands(policy: SearchPolicy, groups: list[Group],
                    fixture: str, lam: float = JOINT_LAMBDA,
                    epochs: int = JOINT_EPOCHS,
                    supervision: str = "reachability") -> int:
    """Joint margin/listwise pass lifting operand top-1."""
    pair = _tensed_margin(groups)
    ranked = _tensed_pairs(fixture, supervision)
    for _ in range(epochs):
        policy._opt.zero_grad()
        loss = torch.zeros(())
        for pos, neg in pair:
            if not len(pos) or not len(neg):
                continue
            gap = policy._forward(pos).unsqueeze(1)
            gap = gap - policy._forward(neg).unsqueeze(0)
            loss = loss + torch.nn.functional.softplus(
                1.0 - gap).mean()
        for feats, labels in ranked:
            scores = policy._forward(feats)
            loss = loss + lam * listwise_loss(scores, labels)
        loss.backward()
        policy._opt.step()
    return sum(int(f.shape[0]) for f, _ in ranked)


def train_author(policy: SearchPolicy, fixture: str, epochs: int = 100,
                 supervision: str = "reachability",
                 supervision_report: list[dict[str, Any]] | None = None
                 ) -> int:
    """Fit author labels with reachability or a canonical-label control."""
    groups = clone_groups(fixture, supervision=supervision,
                          supervision_report=supervision_report)
    if not groups:
        raise ValueError(
            "author supervision produced no complete training groups")
    policy.fit(groups, epochs=epochs)
    refine_operands(policy, groups, fixture, supervision=supervision)
    return sum(len(g) for g in groups)
def emit_edit(space: ActionSpace, graph: dict[str, Any],
              action: dict[str, Any], task_id: str, sig: Vals
              ) -> tuple[dict[str, Any], dict[str, Any], bool]:
    rec = {"task": task_id, "op": action.get("op"),
           "verdict": "signature-mismatch", "signature": list(sig)}
    try: nxt = space.apply(graph, action)
    except ValueError as exc:
        rec["verdict"] = "validator: " + str(exc)
        return graph, rec, False
    return nxt, rec, True
def _new_ctx(consts: tuple[int, ...] = ()) -> dict[str, Any]:
    """Author context, plus const nodes when values given."""
    terms = list(terminals()) + list(const_terms(consts))
    graph = {"input_types": {"x": "Int", "y": "Int"},
             "nodes": [], "output": "x"}
    refs: dict[int, Any] = {}
    for term in terms:
        if term.op == "const":
            refs[id(term)] = len(graph["nodes"])
            graph["nodes"].append(const_node(term.left))
        else:
            refs[id(term)] = term.left
    levels: dict[int, list[Expr]] = {0: terms}
    return {"space": ActionSpace(), "graph": graph, "levels": levels,
            "refs": refs, "sigs": [], "tried": 0, "log": [],
            "signature_evaluations": 0,
            "operand_signature_evaluations": 0, "examined": 0}
def _commit(ctx: dict[str, Any], triple: Triple, sig: Vals,
            task_id: str, size: int, target: Vals) -> bool:
    op, left, right = triple
    action = {"kind": "apply", "op": op, "left": ctx["refs"][id(left)],
              "right": ctx["refs"][id(right)]}
    ctx["graph"], rec, ok = emit_edit(ctx["space"], ctx["graph"],
                                      action, task_id, sig)
    if ok and sig == target: rec["verdict"] = ""
    ctx["log"].append(rec)
    if not ok: return False
    ctx["sigs"].append(sig)
    expr = Expr(op, left, right, size)
    ctx["refs"][id(expr)] = len(ctx["graph"]["nodes"]) - 1
    ctx["levels"][size].append(expr)
    return sig == target
def _score_order(policy: SearchPolicy, ctx: dict[str, Any],
                 rows: Rows, target: Vals, size: int, bound: int,
                 budget: int, w: list[float],
                 candidate_limit: int | None = None) -> Scored:
    if ctx["tried"] >= budget:
        return [], [], [], []
    levels = ctx["levels"]
    width = len(OP_ORDER) * sum(
        len(levels[left_size]) *
        len(levels[size - left_size - 1])
        for left_size in range(size))
    triples = layer_triples(levels, size, limit=candidate_limit)
    if not triples:
        return [], [], [], []
    gap = open_gap(ctx["sigs"], target)
    sigs = [signature(Expr(o, l, r, size), rows) for o, l, r in triples]
    ctx["signature_evaluations"] += len(triples)
    ctx["operand_signature_evaluations"] += 2 * len(triples)
    nodes = ctx["graph"]["nodes"]
    feats = [encode_author(t, s, rows, target, size, bound, i + 1,
                           width, nodes, gap)
             for i, (t, s) in enumerate(zip(triples, sigs))]
    scored = author_scores(policy, feats, w)
    order = sorted(range(len(feats)), key=scored.__getitem__, reverse=True)
    return triples, sigs, order, scored
def _author_step(policy: SearchPolicy, ctx: dict[str, Any],
                 rows: Rows, target: Vals, size: int, bound: int,
                 budget: int, task_id: str, topk: int,
                 w: list[float]) -> bool:
    triples, sigs, order, _ = _score_order(policy, ctx, rows,
                                            target, size, bound,
                                            budget, w,
                                            max(0, budget - ctx["tried"]))
    ctx["tried"] += len(triples)
    ctx["levels"][size] = []
    for idx in order[:topk]:
        ctx["examined"] += 1
        triple, sig = triples[idx], sigs[idx]
        if _commit(ctx, triple, sig, task_id, size, target): return True
    return False
def _pack(ctx: dict[str, Any], solved: bool, graph: dict[str, Any] | None,
    verdict: str, path: str = "author") -> dict[str, Any]:
    unit = "generated_triples" if path == "author" else "committed_nodes"
    output = minimize_graph(graph) if graph is not None else None
    return {"solved": solved, "tried": ctx["tried"], "graph": output,
            "output_nodes": len(output["nodes"]) if output else 0,
            "rejected": ctx["log"], "verdict": verdict, "path": path,
            "signature_evaluations": ctx["signature_evaluations"],
            "operand_signature_evaluations":
            ctx["operand_signature_evaluations"],
            "examined": ctx["examined"],
            "committed_nodes": len(ctx["sigs"]),
            "tried_unit": unit, "budget_unit": unit,
            "label_semantics": "policy-specific; see training report"}


def _add_work(out: dict[str, Any], previous: dict[str, Any]) -> None:
    """Include work in discarded retries/fallbacks in the final totals."""
    for key in ("signature_evaluations", "operand_signature_evaluations",
                "examined", "committed_nodes"):
        out[key] = out.get(key, 0) + previous.get(key, 0)


def _use_fallback(policy: SearchPolicy, ctx: dict[str, Any], rows: Rows,
    target: Vals, bound: int, budget: int, task_id: str,
    consts: tuple[int, ...] = (),
    previous: dict[str, Any] | None = None) -> dict[str, Any]:
    ranked = guided_search(policy, rows, target, bound, budget,
                           task_id, consts=consts)
    ranked["path"] = "fallback"
    ranked["author_log"] = (previous["rejected"] if previous is not None
                            else ctx["log"])
    ranked["author_tried"] = (previous["tried"] if previous is not None
                              else ctx["tried"])
    _add_work(ranked, previous if previous is not None else
              _pack(ctx, False, None, ""))
    return ranked
def author_search(policy: SearchPolicy, rows: Rows, target: Vals,
                  bound: int, budget: int, task_id: str = "",
                  topk: int = 3, weights: list[float] | None = None,
                  fallback: bool = True,
                  consts: tuple[int, ...] = ()
                  ) -> dict[str, Any]:
    ctx, w = _new_ctx(consts), list(weights) if weights \
        else [0.0] * len(EXTRA)
    for size in range(1, bound + 1):
        if _author_step(policy, ctx, rows, target, size, bound,
                        budget, task_id, topk, w):
            return _pack(ctx, True, ctx["graph"], "")
    if fallback:
        return _use_fallback(policy, ctx, rows, target, bound, budget,
                             task_id, consts)
    verdict = "budget-exhausted" if ctx["tried"] >= budget else \
        "no-authored-solution"
    return _pack(ctx, False, None, verdict)
def _fitness(policy: SearchPolicy, tasks: list[TaskRow], weights: list[float],
    topk: int, budget: int) -> tuple[int, int]:
    solved, tried = 0, 0
    for tid, rows, target, bound in tasks:
        out = author_search(policy, rows, target, bound, budget, tid,
                            topk, weights, fallback=False)
        solved, tried = solved + out["solved"], tried + out["tried"]
    return (solved, -tried)
Refined = tuple[list[float], dict[str, int]]
def refine_author(policy: SearchPolicy, tasks: list[TaskRow],
                  weights: list[float] | None = None, rounds: int = 12,
                  sigma: float = 0.1, seed: int = 0, topk: int = 3,
                  budget: int = 10 ** 9) -> Refined:
    best = list(weights) if weights else [0.0] * len(EXTRA)
    best_fit = _fitness(policy, tasks, best, topk, budget)
    for step in range(rounds):
        rng = random.Random(seed + step)
        cand = [w + rng.gauss(0.0, sigma) for w in best]
        fit = _fitness(policy, tasks, cand, topk, budget)
        if fit > best_fit: best, best_fit = cand, fit
    return best, {"solved": best_fit[0], "tried": -best_fit[1]}
def _fresh(sig: Vals, ctx: dict[str, Any]) -> bool:
    """A signature worth committing has no committed twin."""
    return sig not in ctx["sigs"]

def _beam_scan(policy: SearchPolicy, ctx: dict[str, Any], rows: Rows,
    target: Vals, size: int, bound: int, budget: int, task_id: str,
    width: int, w: list[float], stash: Stash) -> bool:
    triples, sigs, order, scored = _score_order(
        policy, ctx, rows, target, size, bound, budget, w)
    if size not in ctx["levels"]:
        ctx["levels"][size] = []
    base = len(ctx["levels"][size])
    for idx in order:
        if ctx["tried"] >= budget: return False
        ctx["examined"] += 1
        if not _fresh(sigs[idx], ctx): continue
        if len(ctx["levels"][size]) - base >= width:
            stash[size].append((scored[idx], triples[idx],
                                sigs[idx]))
            continue
        ctx["tried"] += 1
        triple, sig = triples[idx], sigs[idx]
        if _commit(ctx, triple, sig, task_id, size, target): return True
    return False
def _backtrack(ctx: dict[str, Any], task_id: str, bound: int,
               budget: int, stash: Stash, target: Vals,
               order: str = "fifo") -> bool:
    for size in range(1, bound + 1):
        bucket = stash[size]
        if not bucket or ctx["tried"] >= budget:
            continue
        if order == "score" and not any(
                math.isnan(score) for score, _, _ in bucket):
            # Stable reverse sorting matches repeated max(): ties retain
            # their earliest original entry. Defer compaction so an early
            # solve/budget stop leaves the unconsumed stash in original order.
            ranked = sorted(range(len(bucket)),
                            key=lambda i: bucket[i][0], reverse=True)
            consumed: set[int] = set()
            try:
                for idx in ranked:
                    if ctx["tried"] >= budget:
                        break
                    consumed.add(idx)
                    _, triple, sig = bucket[idx]
                    ctx["examined"] += 1
                    if not _fresh(sig, ctx):
                        continue
                    ctx["tried"] += 1
                    if _commit(ctx, triple, sig, task_id, size,
                               target):
                        return True
            finally:
                if consumed:
                    bucket[:] = [entry for i, entry in enumerate(bucket)
                                 if i not in consumed]
            continue
        while stash[size] and ctx["tried"] < budget:
            if order == "score":
                idx = max(range(len(stash[size])),
                          key=lambda i: stash[size][i][0])
                _, triple, sig = stash[size].pop(idx)
            else:
                triple, sig = stash[size].pop(0)[1:]
            ctx["examined"] += 1
            if not _fresh(sig, ctx): continue
            ctx["tried"] += 1
            if _commit(ctx, triple, sig, task_id, size, target): return True
    return False
MAX_PASSES = 3


def _beam_sweep(policy: SearchPolicy, rows: Rows, target: Vals,
                bound: int, budget: int, task_id: str,
                width: int, w: list[float],
                order: str,
                consts: tuple[int, ...] = ()
                ) -> tuple[dict[str, Any],
                           dict[str, Any]]:
    """Width-capped scans plus ordered backtrack, to fixpoint.

    One bottom-up pass strands late backtrack commits: upper
    sizes already scanned without them. Repeat scan plus
    backtrack until a full pass commits nothing new or the
    budget is spent, so late intermediates feed upper sizes.
    Signature dedup keeps every triple committed at most
    once; passes are capped so never-solve tasks cost at
    most a small multiple of one sweep.
    """
    ctx = _new_ctx(consts)
    for _ in range(MAX_PASSES):
        stash: Stash = {s: [] for s in range(1, bound + 1)}
        for size in range(1, bound + 1):
            if ctx["tried"] >= budget: break
            if _beam_scan(policy, ctx, rows, target, size, bound,
                          budget, task_id, width, w, stash):
                return _pack(ctx, True, ctx["graph"], "", "beam"), ctx
        if ctx["tried"] >= budget: break
        known = len(ctx["sigs"])
        if _backtrack(ctx, task_id, bound, budget, stash, target,
                      order):
            return _pack(ctx, True, ctx["graph"], "", "beam"), ctx
        if len(ctx["sigs"]) == known: break
    verdict = "budget-exhausted" if ctx["tried"] >= budget else \
        "no-authored-solution"
    return _pack(ctx, False, None, verdict, "beam"), ctx


def beam_search(policy: SearchPolicy, rows: Rows, target: Vals,
                bound: int, budget: int, task_id: str = "",
                width: int = 3, weights: list[float] | None = None,
                fallback: bool = True, order: str = "fifo",
                widen: int = 0,
                consts: tuple[int, ...] = ()
                ) -> dict[str, Any]:
    """Beam sweep, optional wider retry, optional fallback."""
    w = list(weights) if weights else [0.0] * len(EXTRA)
    out, ctx = _beam_sweep(policy, rows, target, bound, budget,
                           task_id, width, w, order, consts)
    if not out["solved"] and widen > 0 and \
            out["tried"] < budget:
        again, ctx = _beam_sweep(
            policy, rows, target, bound,
            budget - out["tried"], task_id, widen, w, order,
            consts)
        again["tried"] = out["tried"] + again["tried"]
        again["rejected"] = out["rejected"] + again["rejected"]
        _add_work(again, out)
        out = again
    if out["solved"] or not fallback:
        return out
    return _use_fallback(policy, ctx, rows, target, bound, budget,
                         task_id, consts, out)
