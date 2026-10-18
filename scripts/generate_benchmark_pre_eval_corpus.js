#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function leaf(name) {
  return { op: "input", name };
}

function apply(op, left, right) {
  return { op, left, right };
}

function key(node) {
  if (node.op === "input") return node.name;
  return `${node.op}(${key(node.left)},${key(node.right)})`;
}

function evaluate(node, x, y) {
  if (node.op === "input") return node.name === "x" ? x : y;
  const left = evaluate(node.left, x, y);
  const right = evaluate(node.right, x, y);
  if (node.op === "add") return left + right;
  if (node.op === "sub") return left - right;
  return left * right;
}

function choices() {
  const operators = ["add", "sub", "mul"];
  const result = {
    balanced: [],
    repeated_subtree: [],
    skewed: [],
  };

  for (const root of operators) {
    for (const middle of operators) {
      for (const a of ["x", "y"]) {
        for (const b of ["x", "y"]) {
          for (const c of ["x", "y"]) {
            for (const d of ["x", "y"]) {
              const right = operators[(operators.indexOf(middle) + 1)
                % operators.length];
              result.balanced.push(apply(
                root,
                apply(middle, leaf(a), leaf(b)),
                apply(right, leaf(c), leaf(d)),
              ));
            }
          }
        }
      }
    }
  }

  for (const root of operators) {
    for (const middle of operators) {
      for (const a of ["x", "y"]) {
        for (const b of ["x", "y"]) {
          const subtree = apply(middle, leaf(a), leaf(b));
          result.repeated_subtree.push(apply(root, subtree, subtree));
        }
      }
    }
  }

  for (const root of operators) {
    for (const middle1 of operators) {
      for (const middle2 of operators) {
        for (const a of ["x", "y"]) {
          for (const b of ["x", "y"]) {
            for (const c of ["x", "y"]) {
              for (const d of ["x", "y"]) {
                result.skewed.push(apply(
                  root,
                  leaf(a),
                  apply(middle1, leaf(b),
                    apply(middle2, leaf(c), leaf(d))),
                ));
              }
            }
          }
        }
      }
    }
  }
  return result;
}

const familySeeds = {
  balanced: 61031,
  repeated_subtree: 61032,
  skewed: 61033,
};

function main() {
  const families = choices();
  const random = {
    balanced: rng(familySeeds.balanced),
    repeated_subtree: rng(familySeeds.repeated_subtree),
    skewed: rng(familySeeds.skewed),
  };
  const tasks = { train: [], eval: [] };
  let taskIndex = 0;

  for (const family of Object.keys(families)) {
    const seen = new Set();
    const programs = [];
    while (programs.length < 12) {
      const options = families[family];
      const program = options[Math.floor(random[family]() * options.length)];
      const expressionKey = key(program);
      if (!seen.has(expressionKey)) {
        seen.add(expressionKey);
        programs.push(program);
      }
    }

    for (let index = 0; index < programs.length; index += 1) {
      const split = index < 4 ? "train" : "eval";
      const program = programs[index];
      const base = taskIndex * 10 - 180;
      const train = [];
      const heldOut = [];
      taskIndex += 1;

      for (let row = 0; row < 6; row += 1) {
        const x = base + row;
        const y = ((taskIndex - 1) * 17 + row * 11) % 29 - 14;
        train.push([x, y, evaluate(program, x, y)]);
      }
      for (let row = 0; row < 4; row += 1) {
        const x = 1000 + (taskIndex - 1) * 10 + row;
        const y = ((taskIndex - 1) * 19 + row * 7) % 31 - 15;
        heldOut.push([x, y, evaluate(program, x, y)]);
      }

      const ordinal = String(tasks[split].length + 1).padStart(2, "0");
      tasks[split].push({
        id: `frozen_${split}_${ordinal}`,
        family,
        train,
        held_out: heldOut,
        max_tree_ops: 3,
        generator: {
          seed: familySeeds[family],
          program,
          program_key: key(program),
        },
      });
    }
  }

  const corpus = {
    corpus_id: "benchmark-pre-eval-v1",
    generator: "independent-expression-enumerator-v1",
    generation_acceptance:
      "all generated programs retained; no search-outcome filtering",
    dsl: {
      operators: ["add", "sub", "mul"],
      terminals: ["x", "y"],
      max_tree_ops: 3,
    },
    splits: tasks,
  };
  const destination = path.join(
    __dirname, "..", "fixtures", "benchmark_pre_eval_corpus_v1.json",
  );
  fs.writeFileSync(destination, `${JSON.stringify(corpus, null, 2)}\n`);
}

main();
