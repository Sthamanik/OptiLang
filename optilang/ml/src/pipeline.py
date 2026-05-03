import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

_PATHS = {
    "knn_proxy": os.path.join(MODEL_DIR, "clusterer_knn_proxy.pkl"),
    "classifier": os.path.join(MODEL_DIR, "classifier.pkl"),
    "predictor": os.path.join(MODEL_DIR, "predictor.pkl"),
}

KNOWN_PATTERNS = [
    "constant_folding",
    "dead_code",
    "early_return",
    "expensive_calls",
    "hot_loop",
    "loop_invariant",
    "nested_loops",
    "repeated_computation",
    "string_concat_loop",
    "unused_vars",
]

KNOWN_COMPLEXITIES = ["O(1)", "O(n)", "O(n²)"]

SEVERITY_MAP = {"low": 1, "medium": 2, "high": 3}

REQUIRED_FIELDS = {
    "pattern",
    "severity",
    "impact_score",
    "line_number",
    "co_occurring_patterns",
    "complexity_class",
    "score",
    "total_suggestions",
    "source_lines",
}


class MLPipeline:
    def __init__(self, model_dir: str = MODEL_DIR):
        paths = {
            "knn_proxy": os.path.join(model_dir, "clusterer_knn_proxy.pkl"),
            "classifier": os.path.join(model_dir, "classifier.pkl"),
            "predictor": os.path.join(model_dir, "predictor.pkl"),
        }

        # validate all files exist before loading any
        missing = [k for k, p in paths.items() if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}\n"
                f"Run clustering.py, classification.py, and prediction.py first."
            )

        # phase 1: KNN proxy (cluster assignment)
        with open(paths["knn_proxy"], "rb") as f:
            knn_bundle = pickle.load(f)
        self._knn = knn_bundle["knn"]
        self._knn_scaler = knn_bundle["scaler"]
        self._knn_features = knn_bundle["features"]
        self._cluster_label_map = {
            int(k): v for k, v in knn_bundle["label_map"].items()
        }

        # phase 2: classifier (decision tree)
        with open(paths["classifier"], "rb") as f:
            clf_bundle = pickle.load(f)
        self._clf = clf_bundle["model"]
        self._clf_encoder = clf_bundle["label_encoder"]
        self._clf_features = clf_bundle["features"]

        # phase 3: predictor (linear regression)
        with open(paths["predictor"], "rb") as f:
            pred_bundle = pickle.load(f)
        self._predictor = pred_bundle["model"]
        self._pred_features = pred_bundle["features"]
        self._payoff_min = pred_bundle["target_min"]
        self._payoff_max = pred_bundle["target_max"]

        print(f"Loaded all models from {model_dir}")
        print(
            f"KNN proxy   : {len(self._knn_features)} features, "
            f"{len(self._cluster_label_map)} clusters"
        )
        print(
            f"Classifier  : {len(self._clf_features)} features, "
            f"{len(self._clf_encoder.classes_)} classes"
        )
        print(
            f"Predictor   : {len(self._pred_features)} features, "
            f"payoff range [{self._payoff_min}, {self._payoff_max}]"
        )

    # public API

    def run(self, suggestions: list[dict]) -> list[dict]:
        if not suggestions:
            raise ValueError("Suggestions list is empty — nothing to process")

        # validate input fields early
        self._validate(suggestions)

        # engineer all features from raw fields
        df = self._engineer_features(suggestions)

        # phase 1 — cluster assignment
        cluster_ids, cluster_labels = self._assign_clusters(df)

        # phase 2 — classification
        strategy_labels = self._classify(df)

        # phase 3 — payoff prediction
        payoffs = self._predict_payoff(df)

        # assemble output — one dict per input suggestion
        results = []
        for i, sugg in enumerate(suggestions):
            results.append(
                {
                    # passthrough fields (useful for caller context)
                    "pattern": sugg["pattern"],
                    "severity": sugg["severity"],
                    "line_number": sugg["line_number"],
                    # phase 1 — clustering
                    "cluster_id": int(cluster_ids[i]),
                    "cluster_label": cluster_labels[i],
                    # phase 2 — classification (authoritative strategy label)
                    "strategy_label": strategy_labels[i],
                    # phase 3 — regression
                    "predicted_payoff": round(float(payoffs[i]), 2),
                }
            )

        return results

    def run_single(self, suggestion: dict) -> dict:
        return self.run([suggestion])[0]

    def summary(self, results: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        summary = (
            df.groupby("strategy_label")
            .agg(
                count=("pattern", "count"),
                mean_payoff=("predicted_payoff", "mean"),
                max_payoff=("predicted_payoff", "max"),
                patterns=("pattern", lambda x: ", ".join(sorted(set(x)))),
            )
            .sort_values("mean_payoff", ascending=False)
            .reset_index()
        )
        summary["mean_payoff"] = summary["mean_payoff"].round(2)
        summary["max_payoff"] = summary["max_payoff"].round(2)
        return summary

    # private helpers

    def _validate(self, suggestions: list[dict]) -> None:
        errors = []
        for i, sugg in enumerate(suggestions):
            missing = REQUIRED_FIELDS - set(sugg.keys())
            if missing:
                errors.append(f"Suggestion[{i}] Missing: {sorted(missing)}")
        if errors:
            raise ValueError(
                f"Input validation failed ({len(errors)} suggestion(s)):\n"
                + "\n".join(errors)
            )

    def _engineer_features(self, suggestions: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(suggestions)

        # ordinal encoding of severity
        # fillna(2) = 'medium' default for any unrecognised severity string
        df["severity_encoded"] = df["severity"].map(SEVERITY_MAP).fillna(2).astype(int)

        # pattern_frequency
        # How many suggestions of the same pattern exist in this batch?
        df["pattern_frequency"] = df.groupby("pattern")["pattern"].transform("count")

        # line_proximity
        # How far is each suggestion's line from the batch mean line?
        mean_line = df["line_number"].mean()
        df["line_proximity"] = (df["line_number"] - mean_line).abs().round(2)

        # one-hot encoding — pattern
        for pattern in KNOWN_PATTERNS:
            df[f"pat_{pattern}"] = (df["pattern"] == pattern).astype(float)

        # one-hot encoding — complexity
        for cplx in KNOWN_COMPLEXITIES:
            df[f"cplx_{cplx}"] = (df["complexity_class"] == cplx).astype(float)

        return df

    def _assign_clusters(self, df: pd.DataFrame):
        X_knn = df[self._knn_features].astype(float).values
        X_knn_sc = self._knn_scaler.transform(X_knn)
        ids = self._knn.predict(X_knn_sc)
        labels = [self._cluster_label_map[int(c)] for c in ids]
        return ids, labels

    def _classify(self, df: pd.DataFrame) -> list[str]:
        X_clf = df[self._clf_features].astype(float)
        y_enc = self._clf.predict(X_clf)
        return self._clf_encoder.inverse_transform(y_enc).tolist()

    def _predict_payoff(self, df: pd.DataFrame) -> np.ndarray:
        X_pred = df[self._pred_features].astype(float)
        payoff = self._predictor.predict(X_pred)
        return np.clip(payoff, self._payoff_min, self._payoff_max)


if __name__ == "__main__":

    # load pipeline
    pipeline = MLPipeline()
    print()

    # test scenarios
    SCENARIOS = {
        "A — Single high-payoff pattern (memoize)": [
            {
                "pattern": "expensive_calls",
                "severity": "high",
                "impact_score": 18.0,
                "line_number": 12,
                "co_occurring_patterns": "expensive_calls",
                "complexity_class": "O(n)",
                "score": 92.0,
                "total_suggestions": 1,
                "source_lines": 35,
            },
        ],
        "B — Nested loops (reduce nesting)": [
            {
                "pattern": "nested_loops",
                "severity": "high",
                "impact_score": 12.0,
                "line_number": 8,
                "co_occurring_patterns": "nested_loops",
                "complexity_class": "O(n²)",
                "score": 87.0,
                "total_suggestions": 1,
                "source_lines": 20,
            },
        ],
        "C — Multi-pattern cleanup (dead code + unused vars + early return)": [
            {
                "pattern": "dead_code",
                "severity": "medium",
                "impact_score": 7.0,
                "line_number": 5,
                "co_occurring_patterns": "dead_code|early_return|unused_vars",
                "complexity_class": "O(1)",
                "score": 95.67,
                "total_suggestions": 3,
                "source_lines": 12,
            },
            {
                "pattern": "unused_vars",
                "severity": "medium",
                "impact_score": 7.0,
                "line_number": 10,
                "co_occurring_patterns": "dead_code|early_return|unused_vars",
                "complexity_class": "O(1)",
                "score": 95.67,
                "total_suggestions": 3,
                "source_lines": 12,
            },
            {
                "pattern": "early_return",
                "severity": "medium",
                "impact_score": 7.0,
                "line_number": 20,
                "co_occurring_patterns": "dead_code|early_return|unused_vars",
                "complexity_class": "O(1)",
                "score": 95.67,
                "total_suggestions": 3,
                "source_lines": 12,
            },
        ],
        "D — Cache + hoist (repeated computation + loop invariant)": [
            {
                "pattern": "repeated_computation",
                "severity": "high",
                "impact_score": 8.0,
                "line_number": 10,
                "co_occurring_patterns": "loop_invariant|repeated_computation",
                "complexity_class": "O(n)",
                "score": 98.5,
                "total_suggestions": 4,
                "source_lines": 50,
            },
            {
                "pattern": "loop_invariant",
                "severity": "medium",
                "impact_score": 3.0,
                "line_number": 15,
                "co_occurring_patterns": "loop_invariant|repeated_computation",
                "complexity_class": "O(n)",
                "score": 98.5,
                "total_suggestions": 4,
                "source_lines": 50,
            },
            {
                "pattern": "repeated_computation",
                "severity": "high",
                "impact_score": 8.0,
                "line_number": 25,
                "co_occurring_patterns": "loop_invariant|repeated_computation",
                "complexity_class": "O(n)",
                "score": 98.5,
                "total_suggestions": 4,
                "source_lines": 50,
            },
            {
                "pattern": "hot_loop",
                "severity": "medium",
                "impact_score": 8.0,
                "line_number": 30,
                "co_occurring_patterns": "loop_invariant|repeated_computation",
                "complexity_class": "O(n)",
                "score": 98.5,
                "total_suggestions": 4,
                "source_lines": 50,
            },
        ],
    }

    all_passed = True

    for scenario_name, suggestions in SCENARIOS.items():
        print(f"{'─'*70}")
        print(f"Scenario {scenario_name}")
        print(f" {len(suggestions)} suggestion(s)")

        results = pipeline.run(suggestions)

        for r in results:
            print(f"\npattern={r['pattern']!r:28s} severity={r['severity']!r}")
            print(f"cluster_id: {r['cluster_id']}")
            print(f"cluster_label: {r['cluster_label']}")
            print(f"strategy_label: {r['strategy_label']}")
            print(f"predicted_payoff: {r['predicted_payoff']}")

        # summary table
        summary = pipeline.summary(results)
        print("\n Strategy summary:")
        for _, row in summary.iterrows():
            print(
                f" {row['strategy_label']:<45} "
                f"n={row['count']}  payoff={row['mean_payoff']:.1f}  "
                f"patterns=[{row['patterns']}]"
            )

        # sanity checks
        for r in results:
            assert (
                r["strategy_label"] in pipeline._clf_encoder.classes_
            ), f"Unknown strategy: {r['strategy_label']}"
            assert (
                pipeline._payoff_min <= r["predicted_payoff"] <= pipeline._payoff_max
            ), f"Payoff out of range: {r['predicted_payoff']}"
            assert isinstance(
                r["cluster_id"], int
            ), f"cluster_id should be int, got {type(r['cluster_id'])}"

    # error handling test
    print("\nError handling — missing field")
    try:
        pipeline.run([{"pattern": "dead_code"}])  # missing most required fields
        print(" ERROR: should have raised ValueError")
        all_passed = False
    except ValueError as e:
        print(f" OK — caught ValueError: {str(e)[:80]}...")

    # empty input test
    print("\n Error handling — empty list")
    try:
        pipeline.run([])
        print(" ERROR: should have raised ValueError")
        all_passed = False
    except ValueError as e:
        print(f" OK — caught ValueError: {e}")

    # run_single test
    print("\n Run_single convenience method")
    single_result = pipeline.run_single(
        {
            "pattern": "nested_loops",
            "severity": "high",
            "impact_score": 12.0,
            "line_number": 5,
            "co_occurring_patterns": "nested_loops",
            "complexity_class": "O(n²)",
            "score": 86.0,
            "total_suggestions": 1,
            "source_lines": 18,
        }
    )
    assert (
        single_result["strategy_label"] == "reduce_nesting"
    ), f"Expected reduce_nesting, got {single_result['strategy_label']}"
    print(
        f" OK — strategy={single_result['strategy_label']}  "
        f"payoff={single_result['predicted_payoff']}"
    )

    status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"Smoke test: {status}")
