# Internal Scripts

These scripts belong to the current formal workflow, but they are not intended
to be the primary user-facing entrypoints.

Use the top-level `scripts/*.py` drivers for normal runs:
- `eval_method_compare.py`
- `eval_method_compare_matrix.py`
- `eval_dynamic_compare.py`
- `export_dynamic_visuals.py`
- `export_report_figures.py`

Current internal items:
- `summarize_method_compare_dataset.py`
- `summarize_method_compare_overall.py`

Use these directly only when you need to rebuild summary tables without
rerunning the full suites.
