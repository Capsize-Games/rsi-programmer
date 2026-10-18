# Complete-shortlist cost plots

`plot_costs.py` reads the audited bundle at
`artifacts/benchmark-complete-shortlist-v1-20261018/` and writes
standalone PNG and SVG cost curves in this directory. It requires the final
protocol fixture hash, all runner infrastructure checks, 11,808 timed rows,
960 memory rows, and matching result-file hashes before plotting.

The plot covers ten engine/control configurations and all 41 declared cap
points. It uses mean attempt CPU for the first panel and the runner's N=48
amortized CPU values for the second. The latter are checked against the
protocol-pinned selection charge for shortlist configurations and zero
selection charge for streaming and full-cached baselines. It does not
recompute amortization from raw rows.

Colors identify controls. Streaming is black; full-cached curves are dashed;
complete-shortlist and cheap-only curves are solid. Marker shape identifies
the cap. Lines connect recorded points only and do not estimate values between
caps. Accepted percentage is shown on the horizontal axis; both cost axes are
logarithmic.

Artifact-only schema corrections: the default bundle path now matches the
actual output directory, `benchmark-complete-shortlist-v1-20261018`, and the
required runner check now uses `shortlist_metrics_valid` as published in
code-v82. These corrections update the visualization helper and this note
only.

The parent benchmark completed and its independent audit passed before the
plot ran under `/tmp/rsi-programmer-builder.lock`. The command exited 0 in
1.269 s. The separate AST and 79-column check exited 0 in 0.024 s. The PNG
was opened and inspected: both panels, log axes, all ten configuration labels,
the five cap markers, and the footer are visible without clipping. The script
imports Matplotlib only inside rendering and does not inspect or search
evaluation tasks itself.

See [execution.json](execution.json) for the exact commands, checks and hashes.
The PNG SHA256 is `321d6b37d78c15cda751b8a599b7c644ecca45feb92ef613bd04620e6d6ab2a7`;
the SVG SHA256 is
`2cc55e07bf5aa32f945504e0fab1c2cf0f5e5aa89327a9b9fe383b4883f4e943`.
