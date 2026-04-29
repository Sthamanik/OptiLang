# OptiLang ML Extension

This package contains the optional machine-learning workflow for clustering,
classifying, and estimating payoff for OptiLang optimization suggestions.

## Phase 1: Synthetic Program Generator

Generate the raw synthetic fixtures with:

```bash
python3 -m optilang.ml.src.generator
```

Output is written to:

```text
optilang/ml/data/raw/
+-- manifest.csv
+-- base/<family>/*.py
+-- variants/<variant>/<family>/*.py
```

Current generator shape:

- 120 base programs
- 6 intentionally pathological base programs
- 5 variants applied to every non-pathological base program
- 690 total `.py` fixtures

The manifest records each program id, family, expected strategy, intended
pattern tags, pathological flag, variant name, and source path relative to the
raw data directory. The Phase 2 runner should read `manifest.csv`, execute each
listed source file through the existing OptiLang pipeline, and preserve the
manifest metadata beside the extracted execution results.

## Phase 2/3: Runner, Extractor, and Storage

Generate the execution CSVs with:

```bash
python3 -m optilang.ml.src.runner
```

Useful development options:

```bash
python3 -m optilang.ml.src.runner --limit 20
python3 -m optilang.ml.src.runner --skip-pathological
python3 -m optilang.ml.src.runner --timeout 3
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
