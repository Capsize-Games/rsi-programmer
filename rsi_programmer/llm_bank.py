"""Frozen LLM-judge bank: 180 tasks, pinned rows, guarded sealed held.

Bank home is ``fixtures/llm_bank_v1/`` with ``protocol``, ``tasks``,
``visible``, ``query`` and ``held`` documents. Ninety development
tasks and ninety sealed tasks share six shape families: ``A6``,
``A7`` and ``A8`` arithmetic graphs at witness tree-operation sizes
6, 7 and 8, plus ``S1`` (loop plus accumulate), ``S2`` (loop plus
branch) and ``S3`` (loop plus branch plus update) statement shapes
on the scalar ``curlee_source`` interface. Every row carries both
backends under a stated i64-range assumption; the learner never
sees sealed held rows through the public loader.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

from rsi_programmer import statement_syntax
from rsi_programmer.benchmark_v2_corpus import (
    Node,
    Polynomial,
    apply,
    leaf,
    polynomial,
)

SEED = 20261125
POOLS = ("dev", "sealed")
FAMILIES = ("A6", "A7", "A8", "S1", "S2", "S3")
PER_FAMILY = 15
VISIBLE_N = 3
QUERY_N = 24
HELD_N = 40
HELD_BOUNDARY_N = 6
INT64_LIMIT = 2 ** 62
GRID_LO = -6
GRID_HI = 6
ITEM_LO = -9
ITEM_HI = 9
A_OPS = {"A6": 6, "A7": 7, "A8": 8}
CURLEE_BINARY = Path(
    "/home/joe/Projects/curlee/build/linux-debug/curlee")
GENERATOR_SOURCES = (
    "rsi_programmer/llm_bank.py",
    "rsi_programmer/statement_source.py",
    "rsi_programmer/target_source.py",
    "rsi_programmer/benchmark_v2_corpus.py",
    "rsi_programmer/construction_hard_corpus.py",
)
POLY_FIXTURES = (
    "fixtures/construction_corpus_v1.json",
    "fixtures/construction_hard_corpus_v1.json",
    "fixtures/construction_scaling_corpus_v1.json",
    "fixtures/benchmark_pre_eval_corpus_v2.json",
)
TOKEN_FIXTURES = (
    "fixtures/frozen_branch_transfer_v1/tasks.json",
    "fixtures/frozen_branch_transfer_v2/tasks.json",
)
DEV_NAMES = (
    "positive_total", "negative_count", "even_total",
    "positive_square_total", "odd_count", "above_two_count",
    "maximum_floor_zero", "minimum_ceiling_zero",
    "largest_magnitude",
    "positive_values_capped_at_two_total",
    "positive_count_capped_at_two",
)
BANK_DIRNAME = "fixtures/llm_bank_v1"


class LlmBankGenerator:
    """Build the frozen 180-task bank with pinned, guarded rows."""

    def __init__(self, root: Path | None = None,
                 per_family: int = PER_FAMILY,
                 seed: int = SEED) -> None:
        """Bind the repo root, family quota and master seed."""
        base = Path(__file__).resolve().parents[1]
        self.root = Path(root) if root is not None else base
        self.per_family = per_family
        self.seed = seed
        self.bank_dir = self.root / BANK_DIRNAME

    @staticmethod
    def _sha256(data: bytes) -> str:
        """Hash raw bytes with sha256 and return hex."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _canonical(value: Any) -> bytes:
        """Encode a value as canonical sorted-key JSON bytes."""
        return (json.dumps(value, indent=2, sort_keys=True)
                + "\n").encode("utf-8")

    @staticmethod
    def _tree_ops(node: Node) -> int:
        """Count operation nodes in an arithmetic witness."""
        if node["op"] == "input":
            return 0
        left = LlmBankGenerator._tree_ops(node["left"])
        right = LlmBankGenerator._tree_ops(node["right"])
        return 1 + left + right

    @staticmethod
    def _poly_value(identity: Polynomial, x: int,
                    y: int) -> int:
        """Evaluate one exact integer polynomial identity."""
        total = 0
        for xp, yp, coeff in identity:
            total += coeff * (x ** xp) * (y ** yp)
        return total

    @staticmethod
    def _remainder(left: int, right: int) -> int:
        """Mirror the statement emitter signed remainder."""
        if right == 0:
            raise ValueError("zero remainder divisor")
        value = abs(left) % abs(right)
        return -value if left < 0 else value

    @classmethod
    def _expr_value(cls, expr: tuple[Any, ...], acc: int,
                    value: int, length: int) -> int:
        """Evaluate one parsed statement expression."""
        head = expr[0]
        if head == "acc":
            return acc
        if head == "value":
            return value
        if head == "zero":
            return 0
        if head == "one":
            return 1
        if head == "two":
            return 2
        if head == "minus_one":
            return -1
        if head == "length":
            return length
        left = cls._expr_value(expr[1], acc, value, length)
        right = cls._expr_value(expr[2], acc, value, length)
        if head == "add":
            return left + right
        if head == "sub":
            return left - right
        if head == "mul":
            return left * right
        if head == "rem":
            return cls._remainder(left, right)
        raise ValueError(f"unknown expr head: {head!r}")

    @classmethod
    def _cond_value(cls, cond: tuple[Any, ...], acc: int,
                    value: int, length: int) -> bool:
        """Evaluate one parsed statement condition."""
        name, lexpr, rexpr = cond
        left = cls._expr_value(lexpr, acc, value, length)
        right = cls._expr_value(rexpr, acc, value, length)
        if name == "gt":
            return left > right
        if name == "ge":
            return left >= right
        if name == "lt":
            return left < right
        if name == "le":
            return left <= right
        if name == "eq":
            return left == right
        if name == "ne":
            return left != right
        raise ValueError(f"unknown condition: {name!r}")

    @classmethod
    def _run_block(cls, block: tuple[Any, ...], acc: int,
                   data: Any, length: int) -> int:
        """Execute parsed statements; the top frame holds items."""
        for statement in block:
            kind = statement[0]
            if kind == "set":
                acc = cls._expr_value(statement[1], acc,
                                      data, length)
            elif kind == "if":
                if cls._cond_value(statement[1], acc,
                                    data, length):
                    acc = cls._run_block(statement[2], acc,
                                         data, length)
            elif kind == "for":
                if not isinstance(data, list):
                    raise ValueError("loop needs the item list")
                for item in data:
                    acc = cls._run_block(statement[1], acc,
                                         item, length)
            elif kind == "return":
                return cls._expr_value(statement[1], acc,
                                       data, length)
            else:  # pragma: no cover - grammar closed
                raise ValueError(f"unknown form: {kind!r}")
        return acc

    @classmethod
    def run_tokens(cls, tokens: tuple[str, ...],
                   items: list[int]) -> int:
        """Execute validated statement tokens on one input list."""
        block = statement_syntax.parse(tokens)
        return cls._run_block(block, 0, items, len(items))

    def _poly_sources(self) -> tuple[set[Polynomial],
                                     dict[str, Any]]:
        """Collect excluded polynomial identities from fixtures."""
        excluded: set[Polynomial] = set()
        record: dict[str, Any] = {}
        for name in POLY_FIXTURES:
            path = self.root / name
            if not path.exists():
                record[name] = {"status": "missing",
                                "identities": 0}
                continue
            corpus = json.loads(path.read_text())
            found: set[Polynomial] = set()
            for tasks in corpus["splits"].values():
                for task in tasks:
                    gen = task["generator"]
                    key = ("canonical_polynomial"
                           if "canonical_polynomial" in gen
                           else "polynomial")
                    poly = tuple(tuple(t) for t in gen[key])
                    found.add(poly)
            excluded |= found
            record[name] = {"status": "loaded",
                            "identities": len(found)}
        return excluded, record

    def _token_sources(self) -> tuple[set[tuple[str, ...]],
                                      dict[str, Any]]:
        """Collect excluded statement token programs from banks."""
        excluded: set[tuple[str, ...]] = set()
        record: dict[str, Any] = {}
        for name in TOKEN_FIXTURES:
            path = self.root / name
            if not path.exists():
                record[name] = {"status": "missing",
                                "programs": 0}
                continue
            bank = json.loads(path.read_text())
            refs = {tuple(t["reference"])
                    for t in bank["tasks"]}
            excluded |= refs
            record[name] = {"status": "loaded",
                            "programs": len(refs)}
        return excluded, record

    def _declared_sources(self) -> dict[str, Any]:
        """Record exclusion scope with no identity list on disk."""
        v1 = self.root / TOKEN_FIXTURES[0]
        teachers = 0
        if v1.exists():
            bank = json.loads(v1.read_text())
            teachers = int(bank.get("training_functions", 0))
        pools: dict[str, Any] = {}
        for label, rel in (
                ("d1", "artifacts/construction-d1-diagnostics"
                 "-20261106/results.json"),
                ("d2", "artifacts/construction-d2-diagnostics"
                 "-20261106/results.json")):
            path = self.root / rel
            if not path.exists():
                pools[label] = {"status": "missing"}
                continue
            try:
                data = json.loads(path.read_text())
                pool = data.get("pools", {})
                sizes = pool.get("sizes", pool)
                pools[label] = {"status": "count-declared",
                                "sizes": sizes}
            except (ValueError, AttributeError):
                pools[label] = {"status": "unreadable"}
        return {
            "teacher_functions": {
                "status": "count-declared",
                "count": teachers,
                "note": ("2647-function curriculum list is not "
                         "identity-listed on disk; bank tokens "
                         "are disjoint from all loaded banks.")},
            "array_teachers": {
                "status": "scope-disjoint",
                "note": ("512 array teachers emit list outputs; "
                         "this bank is scalar-return only.")},
            "array_dev_tasks": {
                "status": "scope-disjoint", "count": 3,
                "note": "Array-output dev tasks; scope disjoint."},
            "statement_dev_names": {
                "status": "named", "names": list(DEV_NAMES),
                "note": ("11 development functions; exact token "
                         "matches excluded via loaded banks.")},
            "diagnostic_pools": pools,
        }

    def load_exclusions(self) -> tuple[set[str],
                                       dict[str, Any]]:
        """Merge every exclusion source into identity keys."""
        polys, poly_record = self._poly_sources()
        tokens, token_record = self._token_sources()
        keys = {f"poly:{p!r}" for p in polys}
        keys |= {f"tok:{t!r}" for t in tokens}
        record = {"polynomial_fixtures": poly_record,
                  "token_fixtures": token_record,
                  "declared": self._declared_sources(),
                  "excluded_identities": len(keys),
                  "excluded_polynomials": len(polys),
                  "excluded_token_programs": len(tokens)}
        return keys, record

    @staticmethod
    def _sample_node(rng: random.Random,
                     ops: int) -> Node:
        """Sample one arithmetic tree with an exact op count."""
        if ops == 0:
            choice = rng.choice(["x", "y"])
            return leaf(choice)
        op = rng.choice(["add", "sub", "mul"])
        left_ops = rng.randrange(ops)
        right_ops = ops - 1 - left_ops
        return apply(op,
                     LlmBankGenerator._sample_node(rng, left_ops),
                     LlmBankGenerator._sample_node(rng, right_ops))

    def _sample_arithmetic(self, ops: int, need: int,
                           excluded: set[str]) -> list[Node]:
        """Sample fresh polynomial witnesses at one op count."""
        rng = random.Random(self.seed * 31 + ops)
        found: list[Node] = []
        taken: set[Polynomial] = set()
        attempts = 0
        while len(found) < need and attempts < 200000:
            attempts += 1
            node = self._sample_node(rng, ops)
            identity = polynomial(node)
            if not identity:
                continue
            bare = identity in (((1, 0, 1),), ((0, 1, 1),))
            const = all(x == 0 and y == 0
                        for x, y, _ in identity)
            if bare or const:
                continue
            if identity in taken:
                continue
            if f"poly:{identity!r}" in excluded:
                continue
            grid_ok = True
            for gx in range(GRID_LO, GRID_HI + 1):
                for gy in range(GRID_LO, GRID_HI + 1):
                    if abs(self._poly_value(identity, gx, gy)) \
                            > INT64_LIMIT:
                        grid_ok = False
                        break
                if not grid_ok:
                    break
            if not grid_ok:
                continue
            taken.add(identity)
            found.append(node)
        if len(found) < need:
            raise ValueError(f"only {len(found)} fresh A{ops} "
                             f"witnesses for need {need}")
        return found

    @staticmethod
    def _s1_deck() -> list[tuple[str, ...]]:
        """Enumerate loop-plus-accumulate token candidates."""
        deck = []
        leaves = ("acc", "value", "one", "two")
        for op, left, right, ret in itertools.product(
                ("add", "sub", "mul"), leaves,
                ("value", "acc", "one", "two"),
                ("acc", "one")):
            if ret == "one":
                tails: tuple[str, ...] = ("add", "acc", "one")
            else:
                tails = ("acc",)
            deck.append(("for", "set", op, left, right, "end",
                         "return", *tails, "stop"))
        for right in ("one", "two"):
            deck.append(("for", "set", "rem", "value", right,
                         "end", "return", "acc", "stop"))
        return deck

    @staticmethod
    def _s2_deck() -> list[tuple[str, ...]]:
        """Enumerate loop-plus-branch token candidates."""
        deck = []
        for cmp, lft, rgt, op in itertools.product(
                ("gt", "lt", "ge", "le", "eq", "ne"),
                ("value", "acc"), ("zero", "one", "two"),
                ("add", "sub", "mul")):
            deck.append(("for", "if", cmp, lft, rgt, "set",
                         op, "acc", "value", "end", "end",
                         "return", "acc", "stop"))
        return deck

    @staticmethod
    def _s3_deck() -> list[tuple[str, ...]]:
        """Enumerate branch-plus-update token candidates."""
        deck = []
        for op1, a1, cmp, cl, cr, op2, a2 in itertools.product(
                ("add", "sub"), ("value", "one"),
                ("gt", "lt", "ge", "le"), ("value", "acc"),
                ("zero", "one", "two"), ("add", "sub"),
                ("value", "one")):
            deck.append(("for", "set", op1, "acc", a1,
                         "if", cmp, cl, cr, "set",
                         op2, "acc", a2, "end", "end",
                         "return", "acc", "stop"))
        return deck

    @staticmethod
    def _probes() -> list[list[int]]:
        """Fixed behavior probes for signature dedup."""
        return [[], [0], [1, -2, 3], [5, 5, -5],
                [2, 0, -1, 4], [9, -9, 9, -9],
                [1, 2, 3, 4, 5], [0, 0, 0, 0]]

    def _sample_statements(self, family: str, need: int,
                           excluded: set[str]
                           ) -> list[tuple[str, ...]]:
        """Pick fresh, behavior-distinct programs for one shape."""
        deck = {"S1": self._s1_deck, "S2": self._s2_deck,
                "S3": self._s3_deck}[family]()
        rng = random.Random(self.seed * 131 + len(family) * 7
                            + FAMILIES.index(family))
        order = list(deck)
        rng.shuffle(order)
        found: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for tokens in order:
            if len(found) >= need:
                break
            try:
                statement_syntax.parse(tokens)
            except ValueError:
                continue
            if f"tok:{tokens!r}" in excluded:
                continue
            rem = "rem" in tokens
            probes = [[abs(v) for v in row] if rem else row
                      for row in self._probes()]
            try:
                outs = [self.run_tokens(tokens, row)
                        for row in probes]
            except ValueError:
                continue
            if any(abs(v) > INT64_LIMIT for v in outs):
                continue
            sig = ("rem:" if rem else "gen:") + "|".join(
                str(v) for v in outs)
            if sig in seen:
                continue
            seen.add(sig)
            found.append(tokens)
        if len(found) < need:
            raise ValueError(f"only {len(found)} fresh {family} "
                             f"programs for need {need}")
        return found

    @staticmethod
    def _a_inputs(task_seed: int) -> tuple[list[list[int]],
                                           list[list[int]],
                                           list[list[int]]]:
        """Split the pair grid into visible, query and held rows."""
        visible = [[0, 1], [2, -1], [-2, -3]]
        boundary = [[0, 0], [GRID_HI, GRID_HI],
                    [GRID_LO, GRID_LO], [GRID_HI, GRID_LO],
                    [GRID_LO, GRID_HI], [0, GRID_HI]]
        used = {tuple(p) for p in visible + boundary}
        rest = [[x, y] for x in range(GRID_LO, GRID_HI + 1)
                for y in range(GRID_LO, GRID_HI + 1)
                if (x, y) not in used]
        rng = random.Random(task_seed)
        rng.shuffle(rest)
        return visible, rest[:QUERY_N], boundary + rest[
            QUERY_N:QUERY_N + HELD_N - HELD_BOUNDARY_N]

    @staticmethod
    def _s_boundary() -> list[list[int]]:
        """Six structural boundary lists, all within capacity."""
        return [[], [0], [0] * 256, [ITEM_HI] * 256,
                [ITEM_LO, ITEM_HI] * 128,
                [5, -3, 0, 7, -9, 2]]

    def _s_inputs(self, task_seed: int,
                  rem: bool) -> tuple[list[list[int]],
                                      list[list[int]],
                                      list[list[int]]]:
        """Split list inputs into visible, query and held rows."""
        if rem:
            visible = [[1], [2, 3], [4, 0, 1, 2]]
            boundary = [[], [0], [0] * 256, [ITEM_HI] * 256,
                        [0, ITEM_HI] * 128, [5, 3, 0, 7, 9, 2]]
        else:
            visible = [[1], [-2, 3], [4, 0, -1, 2]]
            boundary = self._s_boundary()
        used = {tuple(v) for v in visible + boundary}
        rng = random.Random(task_seed)
        lo = 0 if rem else ITEM_LO
        taken: list[list[int]] = []
        guard = 0
        while len(taken) < QUERY_N + HELD_N - \
                HELD_BOUNDARY_N and guard < 50000:
            guard += 1
            size = rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 12,
                               32, 64, 200])
            items = [rng.randint(lo, ITEM_HI)
                     for _ in range(size)]
            key = tuple(items)
            if key in used:
                continue
            used.add(key)
            taken.append(items)
        if len(taken) < QUERY_N + HELD_N - HELD_BOUNDARY_N:
            raise ValueError("list input pool exhausted")
        return visible, taken[:QUERY_N], boundary + taken[
            QUERY_N:]

    def _build_tasks(self, excluded: set[str]
                     ) -> list[dict[str, Any]]:
        """Sample every family and split dev/sealed task specs."""
        specs: list[dict[str, Any]] = []
        counters = {"dev": 0, "sealed": 0}
        for family in FAMILIES:
            if family.startswith("A"):
                nodes = self._sample_arithmetic(
                    A_OPS[family], 2 * self.per_family,
                    excluded)
                witnesses: list[Any] = [
                    {"node": n,
                     "polynomial": [list(t) for t in
                                    polynomial(n)]}
                    for n in nodes]
            else:
                tokens = self._sample_statements(
                    family, 2 * self.per_family, excluded)
                witnesses = [{"tokens": list(t)} for t in
                             tokens]
            for pos, witness in enumerate(witnesses):
                pool = "dev" if pos < self.per_family else \
                    "sealed"
                num = counters[pool]
                counters[pool] += 1
                if family.startswith("A"):
                    ident = tuple(tuple(t) for t in
                                  witness["polynomial"])
                    key = f"poly:{ident!r}"
                else:
                    key = f"tok:{tuple(witness['tokens'])!r}"
                excluded.add(key)
                task_seed = int(self._sha256(
                    f"{self.seed}:{family}:{pos}".encode()),
                    16)
                specs.append({
                    "id": f"llm_{pool}_{num:03d}",
                    "pool": pool, "family": family,
                    "interface": ("pair" if family.startswith(
                        "A") else "list"),
                    "witness": witness,
                    "identity_key": key,
                    "task_seed": task_seed,
                })
        return specs

    def _rows_for(self, spec: dict[str, Any]
                  ) -> tuple[list[Any], list[Any], list[Any]]:
        """Freeze visible, query and held rows for one task."""
        if spec["interface"] == "pair":
            ident = tuple(tuple(t) for t in
                          spec["witness"]["polynomial"])
            ins = self._a_inputs(spec["task_seed"])
            return tuple(
                [[x, y, self._poly_value(ident, x, y)]
                 for x, y in part] for part in ins)
        tokens = tuple(spec["witness"]["tokens"])
        rem = "rem" in tokens
        ins = self._s_inputs(spec["task_seed"], rem)
        out: list[list[Any]] = []
        for part in ins:
            rows = []
            for items in part:
                value = self.run_tokens(tokens, items)
                if abs(value) > INT64_LIMIT:
                    raise ValueError("held row exceeds range")
                rows.append([items, value])
            out.append(rows)
        return out[0], out[1], out[2]

    def _curlee_hash(self) -> dict[str, Any]:
        """Pin the local-only CURLEE compiler binary hash."""
        if not CURLEE_BINARY.exists():
            return {"path": str(CURLEE_BINARY),
                    "sha256": "unavailable"}
        return {"path": str(CURLEE_BINARY),
                "sha256": self._sha256(
                    CURLEE_BINARY.read_bytes())}

    def _source_hashes(self) -> dict[str, str]:
        """Hash every generator source pinned by the protocol."""
        hashes = {}
        for name in GENERATOR_SOURCES:
            path = self.root / name
            hashes[name] = self._sha256(path.read_bytes())
        return hashes

    def generate(self) -> dict[str, Path]:
        """Freeze the bank, pin hashes and write five documents."""
        excluded, ex_record = self.load_exclusions()
        specs = self._build_tasks(set(excluded))
        visible: dict[str, Any] = {}
        query: dict[str, Any] = {}
        held: dict[str, Any] = {}
        tasks: list[dict[str, Any]] = []
        for spec in specs:
            vis, qry, hld = self._rows_for(spec)
            visible[spec["id"]] = vis
            query[spec["id"]] = qry
            held[spec["id"]] = hld
            tasks.append({
                "id": spec["id"], "pool": spec["pool"],
                "family": spec["family"],
                "interface": spec["interface"],
                "witness": spec["witness"],
                "identity_key": spec["identity_key"],
                "task_seed": spec["task_seed"],
                "sha256": self._sha256(self._canonical({
                    "id": spec["id"], "pool": spec["pool"],
                    "family": spec["family"],
                    "interface": spec["interface"],
                    "witness": spec["witness"]}))})
        by_pool: dict[str, Any] = {}
        for pool in POOLS:
            members = [t["id"] for t in tasks
                       if t["pool"] == pool]
            by_pool[pool] = {
                "count": len(members),
                "by_family": {
                    fam: sum(1 for t in tasks
                             if t["pool"] == pool and
                             t["family"] == fam)
                    for fam in FAMILIES}}
        protocol = {
            "bank_id": "llm_bank_v1", "seed": self.seed,
            "generator": "rsi_programmer.llm_bank",
            "design": ("tracks/programmer/inversion-design.md"
                       " Bank v1"),
            "counts": {"tasks": len(tasks),
                       "per_pool": self.per_family * len(
                           FAMILIES),
                       "per_family_per_pool": self.per_family,
                       "pools": by_pool},
            "row_discipline": {
                "visible": VISIBLE_N, "query": QUERY_N,
                "held": HELD_N,
                "held_boundary": HELD_BOUNDARY_N,
                "held_seeded": HELD_N - HELD_BOUNDARY_N,
                "inputs_disjoint_within_task": True,
                "backends": ["python", "curlee"],
                "int64_range": [-2 ** 63, 2 ** 63 - 1],
                "acceptance_bound": INT64_LIMIT,
                "note": ("Python ints are unbounded; CURLEE "
                         "Ints are i64. Acceptance states the "
                         "i64 range; rows stay within "
                         "acceptance_bound.")},
            "shapes": {
                "A6": "arithmetic graph, 6 tree ops",
                "A7": "arithmetic graph, 7 tree ops",
                "A8": "arithmetic graph, 8 tree ops",
                "S1": "statement loop plus accumulate",
                "S2": "statement loop plus branch",
                "S3": ("statement loop plus branch plus "
                       "update")},
            "minimality_note": (
                "A strata are witness-size bands, not "
                "catalog-certified minimal sizes; contamination "
                "control rests on polynomial-identity "
                "exclusion."),
            "interfaces": {
                "pair": ("integer pair (x, y); emitters "
                         "target_source.emit_function"),
                "list": ("bounded integer list; emitters "
                         "statement_source python/curlee")},
            "exclusions": ex_record,
            "generator_sources": self._source_hashes(),
            "curlee_binary": self._curlee_hash(),
            "seal": {
                "sealed": True,
                "held_access": ("solvers load protocol, tasks, "
                                "visible and query only; held "
                                "opens via load_held(unseal=True) "
                                "after choices end."),
                "file_guard": "load_held refuses without unseal"},
            "files": {},
        }
        docs = {"protocol": protocol, "tasks": tasks,
                "visible": visible, "query": query,
                "held": held}
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name in ("tasks", "visible", "query", "held"):
            path = self.bank_dir / f"{name}.json"
            path.write_bytes(self._canonical(docs[name]))
            paths[name] = path
        files = {n: self._sha256(p.read_bytes())
                 for n, p in paths.items()}
        protocol["files"] = files
        files["protocol"] = self._sha256(self._canonical(
            protocol))
        protocol["files"] = files
        path = self.bank_dir / "protocol.json"
        path.write_bytes(self._canonical(protocol))
        paths["protocol"] = path
        return paths

    def load_public(self) -> tuple[dict[str, Any],
                                   list[dict[str, Any]],
                                   dict[str, Any],
                                   dict[str, Any]]:
        """Load protocol, tasks, visible and query; verify pins."""
        protocol = json.loads(
            (self.bank_dir / "protocol.json").read_text())
        tasks = json.loads(
            (self.bank_dir / "tasks.json").read_text())
        visible = json.loads(
            (self.bank_dir / "visible.json").read_text())
        query = json.loads(
            (self.bank_dir / "query.json").read_text())
        for name in ("tasks", "visible", "query"):
            digest = self._sha256((self.bank_dir /
                                   f"{name}.json").read_bytes())
            if protocol["files"][name] != digest:
                raise ValueError(f"{name}.json hash mismatch")
        for source, digest in protocol[
                "generator_sources"].items():
            actual = self._sha256(
                (self.root / source).read_bytes())
            if actual != digest:
                raise ValueError(f"source drift: {source}")
        return protocol, tasks, visible, query

    def load_held(self, unseal: bool = False) -> dict[str, Any]:
        """Load sealed held rows only on an explicit unseal."""
        protocol = json.loads(
            (self.bank_dir / "protocol.json").read_text())
        if protocol["seal"]["sealed"] and not unseal:
            raise ValueError("sealed held pool: pass unseal=True "
                             "after solver choices end")
        held = json.loads(
            (self.bank_dir / "held.json").read_text())
        digest = self._sha256(
            (self.bank_dir / "held.json").read_bytes())
        if protocol["files"]["held"] != digest:
            raise ValueError("held.json hash mismatch")
        return held
