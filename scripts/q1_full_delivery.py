"""Produce the complete 250-beer Q1 profile with frozen tuned parameters."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from q1_evaluation import load_inputs, apply_cutpoints  # noqa: E402
from q1_models import semantic_group  # noqa: E402

OUT = ROOT / "reports/modeling/q1_full_delivery_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, styles, _ = load_inputs(ROOT)
    params = pd.read_csv(ROOT / "reports/modeling/q1_mainline_retrain_v1/grouped_model_best_params.csv").set_index("model_group")
    cut = pd.read_csv(ROOT / "reports/modeling/q1_mainline_retrain_v1/proxy_grade_cutpoints.csv", index_col=0)
    groups = pd.Series({label: semantic_group(label) for label in y.columns})
    fitted = pd.DataFrame(index=x.index, columns=y.columns, dtype=float)
    model_dir = OUT / "fitted_models"
    model_dir.mkdir(exist_ok=True)
    for group, members in groups.groupby(groups):
        labels = list(members.index)
        row = params.loc[group]
        model = ExtraTreesRegressor(n_estimators=500, max_depth=None if pd.isna(row.max_depth) else int(row.max_depth),
                                    max_features=float(row.max_features), min_samples_leaf=int(row.min_samples_leaf),
                                    random_state=20261026, n_jobs=-1)
        model.fit(x, y[labels].iloc[:, 0] if len(labels) == 1 else y[labels])
        joblib.dump({"estimator": model, "features": list(x.columns), "targets": labels},
                    model_dir / f"{group}.joblib", compress=3)
        fitted[labels] = np.asarray(model.predict(x)).reshape(len(x), len(labels))
    grades = pd.DataFrame(apply_cutpoints(fitted.to_numpy(), cut.loc[y.columns].to_numpy()),
                          index=x.index, columns=y.columns)
    prefix = pd.DataFrame({"beer": x.index, "style": styles.loc[x.index].to_numpy(),
                           "prediction_source": "full_data_fit"}).set_index("beer")
    prefix.join(fitted).to_csv(OUT / "continuous_250x50.csv")
    prefix.join(grades).to_csv(OUT / "proxy_grade_250x50.csv")
    (OUT / "metadata.json").write_text(json.dumps({"rows": 250, "targets": 50,
        "model": "eight_group_extra_trees", "parameters_from": "q1_mainline_retrain_v1/grouped_model_best_params.csv",
        "training": "all 250 beers", "assessment": "do not use these fitted values for validation; use q1_mainline_retrain_v1/test_predictions.csv.gz",
        "grades": "training-split empirical septiles; proxy scale, not recovered original 0-6 panel scores"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {fitted.shape} continuous and grade tables")


if __name__ == "__main__":
    main()
