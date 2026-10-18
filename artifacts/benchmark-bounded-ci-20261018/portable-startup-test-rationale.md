# CI startup-test portability candidate

The failed CI run reached the real startup integration test and failed inside
`benchmark_total_work._make_models_with_cpu` at its pinned post-fit hash check.
The checked-in v2 reference records seed-0 trained-SNN weights as
`f4893eae9c00c213e3184e599c06ac627863233f69b767d4d4a5de673a3ad791`.
CI's traceback reports the mismatch but does not print the produced hash.

CI and the active local SpikeForge source have matching clean commit
`2981b28105a0d9a8ddcb76ae4e21a4358a1d96f6`; the CI install log and local v2
manifest both report `torch==2.8.0+cpu`. The local v2 manifest reports Python
3.13.5 on Debian 13, while CI reports Python 3.13.16 on Ubuntu 24.04.5. Both
are x86_64. Local CPU is an AMD Ryzen 7 5800X; the CI log does not identify its
CPU model, so hardware is not established as the cause. These facts establish
environment differences but do not isolate which changed the trained weight
bytes.

This candidate changes only tests. The startup test spies on the actual
`build_training_groups` invocation, compares its full task list with the
frozen training split, checks all frozen sampling-cap keywords, and verifies
24 generated groups. Its fit spy checks that the runner passed the original
pinned hash map, then sends a copy with expected hashes set to `None` for
neural models only within this portable test. The actual model construction
and 40-epoch fitting still run; the test checks all three trainable controls
changed weights and performed 40 optimizer updates. The untrained SNN is
checked for unchanged weights without comparing its hash to a golden value.
The test still stops at the first memory-measurement boundary, before any
evaluation.

A separate small regression invokes the real model builder with zero epochs
and a deliberately incorrect trained-SNN expected hash, asserting that its
strict production guard still rejects the mismatch. This verifies that the
portable startup test's local override does not remove the guard contract. No
production benchmark code or pinned CLI model checks are changed.

PyTorch 2.8 documentation states that reproducible results are not guaranteed
across platforms, and that seeds alone do not ensure cross-platform equality:
https://docs.pytorch.org/docs/2.8/notes/randomness.html

This is an untested candidate patch. It has not been applied.
