"""Test whether beer chemistry predicts RateBeer vocabulary beyond expert panels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from q1_evaluation import load_inputs  # noqa: E402

OUT = ROOT / "reports/modeling/consumer_vocabulary_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, _, _, _ = load_inputs(ROOT)
    forum = pd.read_csv(ROOT / "data/raw/7.csv").set_index("beer").reindex(x.index)
    y = forum.select_dtypes(include="number").drop(columns="beer_id", errors="ignore")
    y = y.loc[:, y.notna().any()]
    assert y.shape == (250, 183) and not y.isna().any().any()
    plan = pd.read_csv(ROOT / "data/splits/seed_0/split_manifest.csv")
    tr = plan.loc[plan.split.eq("train"), "beer"].astype(int)
    te = plan.loc[plan.split.eq("test"), "beer"].astype(int)
    heldout_style = forum.loc[te, "tasting_category_fine"].to_numpy()
    Xtr, Xte = x.loc[tr].to_numpy(), x.loc[te].to_numpy()
    Ytr, Yte = y.loc[tr].to_numpy(), y.loc[te].to_numpy()
    mu, sd = Ytr.mean(axis=0), Ytr.std(axis=0)
    sd[sd < 1e-12] = 1
    Ztr = (Ytr - mu) / sd
    Zte = (Yte - mu) / sd
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13), cv=5)),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.5,
                                           random_state=20261029, n_jobs=-1),
    }
    rows, predictions = [], []
    for name, model in models.items():
        model.fit(Xtr, Ztr)
        joblib.dump({"estimator": model, "features": list(x.columns),
                     "descriptors": list(y.columns), "target_mean": mu, "target_std": sd},
                    OUT / f"{name}_trained.joblib", compress=3)
        pred = model.predict(Xte)
        for j, label in enumerate(y.columns):
            rmse = float(mean_squared_error(Zte[:, j], pred[:, j]) ** .5)
            base = float(mean_squared_error(Zte[:, j], np.zeros(len(te))) ** .5)
            rho = (float(spearmanr(Zte[:, j], pred[:, j]).statistic)
                   if np.unique(Zte[:, j]).size > 1 and np.std(pred[:, j]) > 1e-12 else np.nan)
            obs = pd.Series(Zte[:, j])
            fitted = pd.Series(pred[:, j])
            groups = pd.Series(heldout_style)
            obs_resid = obs - obs.groupby(groups).transform("mean")
            pred_resid = fitted - fitted.groupby(groups).transform("mean")
            within_rho = (float(spearmanr(obs_resid, pred_resid).statistic)
                          if np.unique(obs_resid).size > 1 and np.std(pred_resid) > 1e-12 else np.nan)
            rows.append({"model": name, "descriptor": label, "context": label.split("_")[0],
                         "rmse_reduction_pct": 100 * (1 - rmse / base), "spearman": rho,
                         "within_style_spearman": within_rho,
                         "test_std": float(np.std(Zte[:, j])), "train_std_raw": float(sd[j])})
            predictions.extend({"model": name, "beer": int(beer), "descriptor": label,
                                "observed": float(Zte[i, j]), "predicted": float(pred[i, j])}
                               for i, beer in enumerate(te))
        if name == "ridge":
            print("ridge_alpha", model[-1].alpha_)
    pd.DataFrame(rows).to_csv(OUT / "descriptor_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "test_predictions.csv.gz", index=False, compression="gzip")
    d = pd.DataFrame(rows)
    print(d.groupby("model").agg(median_gain=("rmse_reduction_pct", "median"),
                                  n_gain_10=("rmse_reduction_pct", lambda s: int((s > 10).sum())),
                                  n_gain_20=("rmse_reduction_pct", lambda s: int((s > 20).sum())),
                                  median_rho=("spearman", "median")).to_string())
    print(d.loc[d.model.eq("extra_trees")].nlargest(20, "rmse_reduction_pct")
          [["descriptor", "rmse_reduction_pct", "spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
