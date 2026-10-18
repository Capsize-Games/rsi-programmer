# Complete-shortlist cost plots

`plot_costs.py` reads the audited bundle at
`artifacts/benchmark-complete-shortlist-v1-verified-20261018/` and writes
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

The script is prepared but has not been executed. Run it only after the parent
benchmark has completed and its independent audit has passed. It imports
Matplotlib only inside the rendering function and does not inspect or search
evaluation tasks itself.
