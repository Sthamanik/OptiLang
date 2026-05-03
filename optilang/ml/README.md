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
+-- executions_meta.csv  # one row per generated program execution
```

Current full generated dataset:

- 690 execution meta rows
- 684 successful executions
- 6 intentionally pathological/error executions
- 1410 suggestion rows
