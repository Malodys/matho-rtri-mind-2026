# -*- coding: utf-8 -*-
# AI 辅助：Codex (GPT-6), OpenAI, 2026-10-30
# 人工复核状态：待团队核验；程序逐项检查预测标签、样本与指标。
"""Import verified Q1 experiment evidence; preserve the original baseline outputs."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import spearmanr

def main():
    # AI 辅助实现，尚未人工复核。
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment-root",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    src=args.experiment_root
    out=root/"data/processed/q1_tabpfn_final"
    out.mkdir(parents=True,exist_ok=True)
    old=root/"data/processed/q1_final"
    train=pd.read_csv(root/"data/splits/seed_0/4_train.csv")
    test=pd.read_csv(root/"data/splits/seed_0/4_test.csv")
    cols=[c for c in train if c not in ["beer","beer_id","tasting_category_fine"]]
    metrics=[]
    models=[
      ("gbr","tuned_seed_0/gbr/test/test_predictions.csv"),
      ("rf","tuned_seed_0/rf/test/test_predictions.csv"),
      ("tabpfn","tuned_seed_0/tabpfn/test/test_predictions.csv"),
      ("multitask","tuned_seed_0/multitask/test/test_predictions.csv"),
      ("tabpfn_uniform","debias_reweight_seed_0/tabpfn_context_uniform/test/test_predictions.csv")]
    for name,rel in models:
        path=src/"results"/rel
        p=pd.read_csv(path)
        assert len(p)==3750 and not p.duplicated(["beer","target"]).any()
        assert set(p.beer)==set(test.beer)
        truth=test.set_index("beer")
        np.testing.assert_allclose(p.actual,[truth.at[b,t] for b,t in zip(p.beer,p.target)],atol=1e-14)
        rows=[]
        for target,g in p.groupby("target",sort=False):
            cuts=np.quantile(train[target],np.arange(1,7)/7)
            ga=np.searchsorted(cuts,g.actual,side="right")
            gp=np.searchsorted(cuts,g.prediction,side="right")
            rows.append(dict(target=target,Macro_RMSE=np.sqrt(mean_squared_error(g.actual,g.prediction)),
              Macro_MAE=mean_absolute_error(g.actual,g.prediction),R2_mean=r2_score(g.actual,g.prediction),
              Spearman_mean=spearmanr(g.actual,g.prediction).statistic,
              proxy_grade_MAE=mean_absolute_error(ga,gp),proxy_exact_accuracy=np.mean(ga==gp),
              proxy_within_one_accuracy=np.mean(np.abs(ga-gp)<=1),
              proxy_quadratic_kappa=cohen_kappa_score(ga,gp,labels=range(7),weights="quadratic")))
        per=pd.DataFrame(rows)
        row=per.drop(columns="target").mean().to_dict()
        row.update(model=name,R2_median=per.R2_mean.median(),Spearman_median=per.Spearman_mean.median())
        metrics.append(row)
        per.to_csv(out/(name+"_per_target.csv"),index=False)
        if name=="tabpfn_uniform":
            p.to_csv(out/"test_predictions.csv.gz",index=False)
            oof=pd.read_csv(src/"results/debias_reweight_seed_0/tabpfn_context_uniform/oof_predictions.csv")
            assert set(oof.beer)==set(train.beer) and not oof.duplicated(["beer","target"]).any()
            combined=pd.concat([oof,p],ignore_index=True)
            pred=combined.pivot(index="beer",columns="target",values="prediction")
            original=pd.read_csv(old/"sensory_scores_0_6.csv")
            scores=original[["beer","beer_id","tasting_category_fine"]].copy()
            anchors=pd.read_csv(old/"absolute_anchor_params.csv").set_index("dimension")
            for t in cols:
                values=pred.loc[scores.beer,t].to_numpy()
                scores[t+"_pred"]=values
                scores[t+"_grade"]=np.searchsorted(np.quantile(train[t],np.arange(1,7)/7),values,side="right")
                valid=t in anchors.index and str(anchors.at[t,"valid"]).lower()=="true"
                scores[t+"_abs"]=np.clip(anchors.at[t,"a"]+anchors.at[t,"b"]*values,0,6) if valid else np.nan
            scores.to_csv(out/"sensory_scores_0_6.csv",index=False)
    summary=pd.DataFrame(metrics)
    summary.to_csv(out/"model_summary.csv",index=False)
    baseline=pd.read_csv(old/"model_summary.csv")
    for name in ["gbr","rf","tabpfn","multitask","tabpfn_uniform"]:
        row=summary[summary.model==name]
        baseline=pd.concat([baseline,row],ignore_index=True)
    baseline["selected_as_final_model"]=baseline.model.eq("tabpfn_uniform")
    baseline.to_csv(out/"combined_summary.csv",index=False)
    for f in ["comparison.csv","frozen_selection.csv"]:
        pd.read_csv(src/"results/tuned_seed_0"/f).to_csv(out/f,index=False)
    pd.read_csv(src/"results/debias_reweight_seed_0/comparison.csv").to_csv(out/"context_comparison.csv",index=False)
    for label,folder in [("median","global_median_seed_0"),("uniform_median","global_median_uniform_context_seed_0")]:
        pd.read_csv(src/"results"/folder/"comparison.csv").to_csv(out/(label+"_comparison.csv"),index=False)
    manifest={"final_model":"tabpfn_uniform","features":231,"targets":50,"train":175,"test":75,
      "cv_mean_r2":0.20756029221760194,"checkpoint_sha256":"ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
      "prediction_coverage":"175 training OOF + 75 holdout; each beer predicted without its own label",
      "missing":"retained","context":"original training rows repeated once in original order",
      "ai_tool":"Codex GPT-6 OpenAI","human_review":"pending","test_previously_inspected":True,
      "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.suffix!="json"}}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary[["model","R2_mean","Macro_MAE","Macro_RMSE"]].to_string(index=False))
if __name__=="__main__": main()
