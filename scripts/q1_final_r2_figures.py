# -*- coding: utf-8 -*-
# AI 辅助：Codex (GPT-6), OpenAI, 2026-10-30
# 人工复核状态：待团队核验；图中数值来自已核验模型产物。
"""Plot Q1 model and target R2 using the shared paper figure style."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import apply_style, tidy, simple_title, save, dim_cn, ACCENT, BASELINE

def main():
    # AI 辅助实现，尚未人工复核。
    root=Path(__file__).resolve().parents[1]
    out=root/"data/processed/q1_final_r2_figures"
    out.mkdir(parents=True,exist_ok=True)
    apply_style()
    df=pd.read_csv(root/"data/processed/q1_tabpfn_final/combined_summary.csv").sort_values("R2_mean")
    names={"independent_gbr":"原逐维GBR","grouped_extra_trees":"分组ET","global_extra_trees":"全局ET","pls2":"PLS","multitask_elastic_net":"多任务弹性网","ridge":"岭回归","global_mean":"均值基线","gbr":"原网格GBR","rf":"原网格RF","tabpfn":"TabPFN","multitask":"多任务网络"}
    fig,ax=plt.subplots(figsize=(5,3.8))
    ax.barh([names.get(x,x) for x in df.model],df.R2_mean,color=[ACCENT if x=="tabpfn" else BASELINE for x in df.model])
    ax.set_xlabel("留出集平均R²"); ax.set_xlim(-.03,.28)
    simple_title(ax,"感官预测比较"); tidy(ax,grid_axis="x")
    fig.tight_layout(); save(fig,out/"model_r2.png"); plt.close(fig)
    target=pd.read_csv(root/"data/processed/q1_tabpfn_final/tabpfn_per_target.csv").sort_values("R2_mean")
    fig,axes=plt.subplots(1,2,figsize=(5,5.7),sharex=True)
    for ax,part in zip(axes,[target.iloc[:25],target.iloc[25:]]):
        ax.barh([dim_cn(x) for x in part.target],part.R2_mean,color=ACCENT)
        ax.set_xlabel("留出集R²"); tidy(ax,grid_axis="x")
    fig.suptitle("逐维预测效果")
    fig.tight_layout(); save(fig,out/"target_r2.png"); plt.close(fig)
    records=[{"file":"model_r2.png","title":"感官预测比较","caption":"同一75款留出酒的50维平均R²；AI辅助：Codex，GPT-6，OpenAI，2026-10-30。"},
             {"file":"target_r2.png","title":"逐维预测效果","caption":"最终TabPFN在75款留出酒上的逐维R²；AI辅助：Codex，GPT-6，OpenAI，2026-10-30。"}]
    (out/"figures_manifest.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()
