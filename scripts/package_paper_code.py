# -*- coding: utf-8 -*-
# AI 辅助：Codex (GPT-6), OpenAI, 2026-10-30
# 人工复核状态：待团队核验；按显式清单收录论文代码。
"""Package source files used by the paper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

MAIN_SCRIPTS = """
data_contract build_data_contract audit_scale resolve_corr_type build_label_dictionary split_dataset
q1_pipeline q1_models q1_evaluation q1_predict q1_absolute_anchor q1_bootstrap q1_figures
q1_final_r2_figures q1_final_tabpfn import_q1_tabpfn_final
q2_common q2_graph q2_evidence q2_stability q2_chem_kb q2_figures q2_verify
q3_copula q3_generate q3_transfer q3_sensory_profile q3_figures q3_report q3_verify
q4_consumer q4_figure_manifest plot_style build_paper paper_prose check_paper_pdf build_ai_usage_doc
""".split()
EXPERIMENT_SCRIPTS = """
split_dataset tuning_protocol paper_baselines tune_rf tune_tabpfn
tune_multitask multitask_cv tabpfn_cv finalize_tuning
""".split()
MAIN_TESTS = """
test_data_contract test_scale_audit test_corr_type test_consumer_layer
test_plot_style test_literature test_final_tabpfn
""".split()
EXPERIMENT_TESTS = """
test_finalize_tuning test_multitask_cv test_paper_baselines test_tabpfn_cv
test_tune_multitask test_tune_rf test_tune_tabpfn test_tuning_protocol
""".split()

README = """# 论文代码支撑材料

## 目录
- scripts/：数据处理、三道题、消费者分析、绘图与论文生成。
- prediction-models/scripts/：问题一的五折调参与基线评估。
- data/processed/nightly/overnight-q123-20261030/：问题二、三补充验证。
- tests/、tests_q1/、prediction-models/tests/：对应检查代码。
- requirements.lock.txt：论文分析依赖。
- prediction-models/requirements-tabpfn.txt：TabPFN实验依赖。
- docs/：字段字典与文献清单。
- MANIFEST.json：文件清单和SHA256。

## 运行顺序
在解压后的项目根目录运行，各脚本的参数见 --help。
将赛题附件放在 data/raw/，训练/测试划分放在 data/splits/seed_0/。
prediction-models/ 是独立实验项目，准备相同的 data/raw/ 与 data/splits/seed_0/。

1. 数据：
   python scripts/build_data_contract.py
   python scripts/audit_scale.py
2. 问题一基础模型：
   python scripts/q1_pipeline.py
   实验项目中的 paper_baselines.py、tune_rf.py、tune_tabpfn.py、
   tune_multitask.py 负责调参，finalize_tuning.py 负责完整训练和留出评估。
3. 最终TabPFN：
   python scripts/q1_final_tabpfn.py --experiment-root prediction-models --model-path ../pretrained/Prior-Labstabpfn_3_5/tabpfn-v3.5-20260909.safetensors
   输出在 prediction-models/results/tabpfn_final_seed_0/。
   参数为4个估计器、温度1、先平均后softmax、seed 0；训练上下文按原顺序重复两次。
   采用原实验的五折划分，输出175款啤酒的OOF预测及75款留出预测。
   import_q1_tabpfn_final.py为既有实验归档的论文结果导入脚本；
   独立入口产生的 comparison.csv、oof_predictions.csv 和 test/test_predictions.csv 可直接核对最终模型。
4. 问题二：
   python scripts/q2_graph.py
   python scripts/q2_evidence.py
   python scripts/q2_stability.py --bootstrap-b 200
   补充验证：nightly目录下q2/q2_robustness_audit.py，参数 --repo-root . --output-dir <输出目录>。
5. 问题三：
   python scripts/q3_generate.py
   python scripts/q3_transfer.py
   python scripts/q3_sensory_profile.py
   补充验证：nightly目录下q3/q3_bounded_validation.py。
6. 消费者、价格与剂量响应：
   python scripts/q4_consumer.py
7. 图表与论文：
   各题 *_figures.py、q1_final_r2_figures.py 生成图表；
   build_paper.py 生成修订稿，docx_to_pdf_word.ps1 负责Word转PDF。

原始附件、模型权重、训练产物和论文模板需在本地准备。
"""

def main():
    # AI辅助打包逻辑；文件名白名单与归档内容逐项核对。
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("code.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment = args.experiment_root.resolve()
    files = {}
    def add(source, target):
        files[target] = source.read_bytes()
    for name in MAIN_SCRIPTS:
        add(root / "scripts" / (name + ".py"), "scripts/" + name + ".py")
    add(root / "scripts/docx_to_pdf_word.ps1", "scripts/docx_to_pdf_word.ps1")
    for name in EXPERIMENT_SCRIPTS:
        add(experiment / "scripts" / (name + ".py"), "prediction-models/scripts/" + name + ".py")
    for name in MAIN_TESTS:
        add(root / "tests" / (name + ".py"), "tests/" + name + ".py")
    for name in EXPERIMENT_TESTS:
        add(experiment / "tests" / (name + ".py"), "prediction-models/tests/" + name + ".py")
    for path in sorted((root / "tests_q1").glob("*.py")):
        add(path, path.relative_to(root).as_posix())
    base = "data/processed/nightly/overnight-q123-20261030/"
    for relative in ["q2/q2_robustness_audit.py", "q3/q3_bounded_validation.py", "q3/q3_holdout_figure.py"]:
        add(root / (base + relative), base + relative)
    for relative in ["requirements.lock.txt", "docs/data_dictionary.md", "docs/refs_verified.json"]:
        add(root / relative, relative)
    add(experiment / "requirements-tabpfn.txt", "prediction-models/requirements-tabpfn.txt")
    files["README.md"] = README.encode("utf-8")
    for name, data in files.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
        if any(word in name for word in ["causal", "debias", "probe", "smoke", "peek"]):
            raise ValueError("Unrelated source: " + name)
    files["MANIFEST.json"] = json.dumps({
        "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(files.items())]
    }, ensure_ascii=False, indent=2).encode("utf-8")
    output = args.output.resolve()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(files)
    print(json.dumps({"files": len(files), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))

if __name__ == "__main__":
    main()
