# Status: finished

Every step ran to completion on 2026-11-11. Nothing is left to resume.

- Results and the decision: [REPORT.md](REPORT.md), `results.csv`.
- `run_all.py --status` lists every inference step as done; running `run_all.py`
  again reads nothing new (all readings are cached in `predictions\cache\`).
- To add an invoice: put its photo under `bills\`, its truth file under
  `evaluation\ground_truth\`, then run `prepare.py --surya`, `make_crops.py --force`
  and the commands under "Reproducing" in the report. Cached readings of the
  existing cells are reused; only the new crops are read.
