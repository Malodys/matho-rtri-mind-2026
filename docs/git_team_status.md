# Git 与协作状态快照

快照日期：2026-10-29。此文件只记录 Git 可观测事实，不把提交作者或 detached worktree 推断为队友当前在线或仍在开发。

| 对象 | 可观测状态 |
| --- | --- |
| 当前分支 | `release/toolchain-hardening-20261029`，从 `d01eb42` 创建；待首次推送到 `origin` |
| 相对 `origin/main` | 基线时超前 20 个提交；不改写历史 |
| 预测模型远端分支 | `origin/experiments/prediction-models` → `15d0c99`；对应 `worktrees/prediction-models` 为 detached worktree |
| 初始分析远端分支 | `origin/feat/initial-data-analysis` → `2d72cae` |
| 主分支 | `origin/main` → `10cfca6` |
| 提交作者身份 | 当前线索为 `2579qwe@users.noreply.github.com`；预测模型分支提交作者为 `1138709375@qq.com`；初始分析分支提交作者为 `99719526+Malodys@users.noreply.github.com` |
| PR / CI | 仓库中未发现 `.github` CI 配置或 PR 元数据；不能据此声称已有 PR 或持续集成 |

合并前应分别执行：

```powershell
git merge-base HEAD origin/experiments/prediction-models
git log --oneline --left-right HEAD...origin/experiments/prediction-models
git diff --name-status HEAD...origin/experiments/prediction-models
git merge-base HEAD origin/feat/initial-data-analysis
git log --oneline --left-right HEAD...origin/feat/initial-data-analysis
git diff --name-status HEAD...origin/feat/initial-data-analysis
```

不要使用 `git reset --hard` 或重写历史。合并前以分支提交清单、merge-base 和未合并文件为准，并由队友本人确认工作状态。发布前还需确认当前工作树无未提交 tracked 修改和意外未跟踪文件；本文件不替代该最终检查。
