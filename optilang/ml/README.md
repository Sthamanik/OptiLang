# OptiLang ML Extension

This package contains the optional machine-learning workflow for clustering,
classifying, and estimating payoff for OptiLang optimization suggestions.

## Phase 1: Raw Program Corpus

Raw programs live directly under:

```text
optilang/ml/data/raw/
```

The current raw corpus contains 1000 real Python algorithm/problem programs
copied from TheAlgorithms/Python, preserving the original category folders and
filenames. There is no manifest, strategy label, or family label in the raw
dataset path.

## Phase 2/3: Runner, Extractor, and Storage

Generate the execution CSVs with:

```bash
python3 -m optilang.ml.src.runner
```

The runner replaces `executions.csv` by default so repeated runs do not
duplicate the dataset. Use `--append` only when intentionally collecting
multiple independent runs in one CSV.

Useful development options:

```bash
python3 -m optilang.ml.src.runner --limit 20
python3 -m optilang.ml.src.runner --skip-pathological
python3 -m optilang.ml.src.runner --timeout 3
python3 -m optilang.ml.src.runner --append
```

The runner writes:

```text
optilang/ml/data/
+-- executions.csv       # one row per suggestion per execution
+-- executions_meta.csv  # suggestion metadata exported by 01_eda.ipynb
+-- executions_clustered.csv          # metadata plus learned strategy cluster
+-- program_cluster_improvement.csv   # one row per (program_id, cluster)
```

`executions.csv` includes stable identity fields so downstream notebooks can
aggregate suggestions back to the program level:

- `program_id`: corpus-relative source path.
- `execution_id`: `program_id` plus a short source hash.
- `suggestion_id`: stable id for a suggestion within one execution.
- `source_path` and `source_hash`: traceability/debugging fields.

Prediction is program-cluster level. `04_prediction.ipynb` no longer predicts
the heuristic `impact_score`; it predicts `expected_score_improvement_pct`,
the expected percentage score recovery if one cluster is fixed for a program.
This is a score proxy built from current scorer rules, not measured runtime
speedup. True runtime improvement requires a future auto-fix and re-run stage.

Current full generated dataset:

- 690 execution meta rows
- 684 successful executions
- 6 intentionally pathological/error executions
- 1410 suggestion rows
