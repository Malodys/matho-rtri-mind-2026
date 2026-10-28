# 感官画像：pending_q1

问题 3 的「生成化合物谱 → 50 维感官画像」依赖**问题 1 的感官预测接口**，
本次运行时该接口尚不可用，因此本步骤**未产出**感官画像（状态 `pending_q1`）。

## 依赖缺失的具体原因

Q1 接口存在但调用失败：Blond/constrained: NameError("name 'np' is not defined")；Blond/unconstrained: NameError("name 'np' is not defined")；Dubbel/constrained: NameError("name 'np' is not defined")；Dubbel/unconstrained: NameError("name 'np' is not defined")；Hoppy/constrained: NameError("name 'np' is not defined")；Hoppy/unconstrained: NameError("name 'np' is not defined")；Kriek/constrained: NameError("name 'np' is not defined")；Kriek/unconstrained: NameError("name 'np' is not defined")；Lambic/constrained: NameError("name 'np' is not defined")；Lambic/unconstrained: NameError("name 'np' is not defined")；Saison/constrained: NameError("name 'np' is not defined")；Saison/unconstrained: NameError("name 'np' is not defined")；Strong ale/constrained: NameError("name 'np' is not defined")；Strong ale/unconstrained: NameError("name 'np' is not defined")；Tripel/constrained: NameError("name 'np' is not defined")；Tripel/unconstrained: NameError("name 'np' is not defined")

## 接口约定（问题 1 侧就绪后可直接跑通，无需改动问题 3 脚本）

调用方：`scripts/q3_sensory_profile.py`

```python
from q1_predict import predict_sensory
scores = predict_sensory(compound_matrix, model=None)   # model=None 用最终选定模型
```

* **输入** `compound_matrix`：统一 log10(mg/L) 标度的化合物矩阵，**231 列**
  （即 `data_contract.to_unified_scale()` 的全部列，**含** `esters_sum`、
  `aroma_hops_sum`、`acids_sum`、`sulfur_sum` 四个 `*_sum` 派生列；
  Q1 接口内部按 `*_sum` 后缀自动剔除，调用方无需自行处理）。
* **索引**：样本 id；合成谱用 `SYN_<style>_NNNN` / `SYNU_<style>_NNNN` 字符串 id，
  原样回传为输出索引。
* **输出** `scores`：DataFrame，索引与输入**逐行一致**，列为 **50 个感官维度名**，
  值为标准化后的连续分（4.csv 口径）。
* **模型工件**：需存在 `data/processed/q1_final/model_<name>.joblib`，
  并由 `data/processed/q1_final/model_summary.csv` 的 `selected_as_final_model` 标记最终模型。

## 恢复方式

```powershell
.\.venv\Scripts\python.exe scripts\q3_sensory_profile.py
```

接口与工件就绪后，本脚本会自动为每个风格产出
`sensory_profile_<style>.csv`（有约束版）与
`sensory_profile_<style>_unconstrained.csv`（无约束版），
并把 `generation_summary.csv` 的 `sensory_profile_status` 更新为 `ok`。

## 口径声明

附件 3.xlsx 为空白品评卡模板，附件中**没有任何原始 0–6 评分**；
4.csv 的感官分是标准化值。因此问题 3 的「感官画像」只能建立在**标准化尺度**上，
论文与图表须显式声明这一点。
