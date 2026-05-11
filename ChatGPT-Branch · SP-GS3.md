# Branch · SP-GS

**User:** Anonymous  
**Created:** 5/10/2026 13:58:36  
**Updated:** 5/11/2026 16:35:17  
**Exported:** 5/11/2026 17:28:27  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370)  

## Prompt:
2026/5/11 16:00:24

修改后的实验验证结果反而更差了：
已继续把 `ChatGPT-Branch · SP-GS2.md` 这一轮建议完整落到了代码和实验上，并做了全 Cambridge 验证。结论比较明确：专家指出的实现风险是存在的，代码里也确实有关键 bug/缺口；但修完并正式训练后，最新 SP-GS2 full 配置没有带来论文级稳定提升，不能作为当前主结果默认配置。

**已实现内容**
- 修复 differentiable PnP 的核心问题：matching logits 和 PnP target logits 分离，PnP top-k 不再从错误的 valid-pixel logits 里取。
- 新增 landmark matchability 校准脚本：`loc_gs/scripts/calibrate_landmark_matchability.py`，支持 TP/FP 统计、visibility/depth/alpha 过滤、sidecar cache。
- 新增 calibrated matchability prior、coverage-aware filtering、多假设 PnP+dense verification、gated residual descriptor 等 SP-GS2 机制。
- 新增/更新测试，覆盖 calibration、PnP loss、CLI、matchability prior 等路径。
- 全部测试通过：`114 passed, 16 warnings`。

**全 Cambridge 最新 SP-GS2 P0 full 结果**
结果表已写到：

`output/paper_figures/spgs2_20260511/cambridge_spgs2_p0_full_table.csv`

核心指标如下：

| Scene | STDLoc te / R5 | 历史 selected te / R5 | 最新 SP-GS2 P0 te / R5 |
|---|---:|---:|---:|
| GreatCourt | 10.890cm / 23.60% | 15.654cm / 13.29% | 15.755cm / 11.84% |
| KingsCollege | 17.690cm / 1.70% | 19.880cm / 2.04% | 23.685cm / 0.58% |
| OldHospital | 10.720cm / 15.90% | 8.990cm / 32.97% | 13.451cm / 17.58% |
| ShopFacade | 2.647cm / 80.58% | 2.401cm / 81.55% | 3.093cm / 77.67% |
| StMarysChurch | 3.690cm / 63.80% | 3.570cm / 67.17% | 6.246cm / 36.23% |

Macro：
- STDLoc：`9.127cm / 37.12% R@5cm`
- 历史 selected：`10.099cm / 39.40% R@5cm`
- 最新 SP-GS2 P0：`12.446cm / 28.78% R@5cm`

所以最新完整训练版相比 STDLoc 和历史 selected 都是退化的，尤其 GreatCourt、StMarysChurch、KingsCollege 不适合作为主 claim。

**原因判断**
我不认为专家理论本身完全错误。更准确地说：这轮建议里指出的工程问题是真问题，也已经修复；但当前项目里的实现和训练目标还没有把“matchability / landmark score / differentiable PnP”转化成 query-time 真实定位收益。

主要瓶颈是：
- per-landmark 的静态 matchability 不够，它无法处理 GreatCourt 这类重复结构里的 query-specific 误匹配。
- hard filtering / coverage filtering 太容易删掉有用匹配，ShopFacade 和 OldHospital q80 消融都显示退化。
- calibrated prior 加进 sparse matcher 后权重大一点会破坏初始位姿，小权重又几乎无收益。
- gated residual descriptor 容易扰动 STDLoc/PLY descriptor 的稳定结构。
- 当前最好结果主要来自“protected STDLoc sampled bank + PLY/STDLoc descriptor + 稳健 localizer 配置”，不是来自 P0 full retraining。

**论文当前建议**
主实验不要默认使用最新 SP-GS2 P0 full checkpoint。更稳妥的是：
- 主表使用历史 selected/protected STDLoc+PLY 配置。
- SP-GS2 P0/P1/P3/P4/P5 作为 ablation 和负结果分析：说明哪些定位驱动设计有效、哪些简单形式无效。
- claim 收窄为：继承 STDLoc/SplatLoc 的定位先验和 landmark 筛选思想后，部分场景能提升；但需要 query-conditioned matchability 才能稳定跨场景泛化。

论文图表已生成在：

`output/paper_figures/spgs2_20260511/`

包括 median error、R@5cm、GreatCourt calibrated prior 消融图，以及 LaTeX 表格。当前没有残留训练/评测进程，显卡已释放。

## Response:
2026/5/11 16:00:26

我会把这轮结果当作一次反证实验来分析：重点不再加新模块，而是判断 SP-GS2 哪些机制应回滚、哪些保留为诊断/ablation，并收敛论文主线。

## Response:
2026/5/11 16:00:38

我已确认新机制并非未落地，而是全链路强干预破坏了 STDLoc/PLY 的稳定匹配分布。下一步应降级 P0 full，回到保守融合与可回退主线。

## Response:
2026/5/11 16:10:02

这轮结果已经把问题定位得很清楚了：**SP-GS2 P0 full 不应该再作为主线推进。**  

不是因为你实现错了；相反，`match_logits / pnp_logits` 分离、`valid_target_pixels`、TP/FP calibration、coverage-aware filtering、多假设 PnP、gated residual 这些都已经落到代码里了。当前 repo 里 `DifferentiablePnPMatchLoss` 确实已经把 matching logits 和 PnP target logits 分开，PnP top-k 从 `pnp_logits` 取，并用 `geometry_target` 形成 PnP 权重；calibration 脚本也已经按 TP/FP、depth、alpha、reprojection threshold 统计 landmark matchability。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/differentiable_pnp.py))

但**修完以后更差**，这说明一个更深的问题：  
**这些机制虽然理论上合理，但当前形式没有与最终 test-time localization 分布对齐。它们不是 bug fix 后自然会提升的“缺失项”，而是引入了新的训练/推理偏置。**

---

## 1. 现在应该停止把 SP-GS2 P0 full 当主结果

最新结果：

| 配置 | Mean te | Macro R5 |
|---|---:|---:|
| STDLoc | 9.127 cm | 37.12% |
| 历史 selected | 10.099 cm | 39.40% |
| 最新 SP-GS2 P0 | 12.446 cm | 28.78% |

这个差距已经不是调一点权重能解释的。尤其 StMarysChurch 从 `67.17%` 掉到 `36.23%`，OldHospital 从 `32.97%` 掉到 `17.58%`，ShopFacade 从 `81.55%` 掉到 `77.67%`，说明 P0 full 不是只在 GreatCourt 失败，而是在多个原本正向的场景上破坏了稳定性。

所以结论应该改成：

> **SP-GS2 P0 full 是一组有价值的负结果与诊断模块，不是当前论文主方法。主方法应回退到 protected STDLoc/PLY backbone，并只做保守的 branch selection / pose-level reliability selection。**

---

## 2. 为什么“修了正确问题”反而更差？

### 2.1 PnP top-k 修复正确，但变成了 oracle-style training path

现在 PnP top-k 从 `valid_target_pixels` mask 后的 `pnp_logits` 取候选，这解决了“PnP 从 GT 不可见/无效候选取点”的工程问题。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/differentiable_pnp.py))

但它也带来一个副作用：**训练时的 PnP path 被限制在 GT-visible target candidates 中，而测试时 sparse matcher 仍然必须在所有 candidate landmarks 中自己找正确匹配。**

也就是说：

```text
train-time PnP: 在 oracle-valid candidate set 内优化
test-time PnP: 在全量重复结构 candidate set 内匹配 + RANSAC
```

这会造成 train-test mismatch。它能让 PnP loss 更“干净”，但未必让 descriptor / locability 学会如何排除 GreatCourt 那些高相似度错误窗口、立面、柱廊。

更严重的是，PnP loss 被 target mask 限制后，很多 hard negatives 不再直接进入 pose path。对 GreatCourt 来说，最需要学习的不是“有效 target 内怎么拟合 pose”，而是：

> **在大量视觉相似但几何错误的 landmark 中，如何不选错。**

所以这个 fix 是工程正确的，但不能作为 full training 的强主线。

---

### 2.2 TP/FP landmark calibration 是静态的，解决不了 query-specific ambiguity

`calibrate_landmark_matchability.py` 现在确实会统计 TP/FP：满足 visibility/depth/alpha/reprojection 条件的 match 记 TP，否则记 FP。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/calibrate_landmark_matchability.py))

但它仍然是 **per-landmark static prior**。这对 GreatCourt 不够，因为 GreatCourt 的错误不是“某个 landmark 永远坏”，而是：

> 某些 landmark 在某些 query 下是正确候选，在另一些 query 下是高危 false positive。

重复结构里的窗户、柱子、墙面往往具有相似 descriptor。一个 landmark 的全局 TP/(TP+FP) 不能表达 query-specific matching context。

所以 calibrated prior 大权重会破坏初始位姿，小权重又没收益，这完全符合预期。它不是没有信息，而是信息粒度不对。

---

### 2.3 Coverage-aware filtering 删掉了 RANSAC 需要的冗余

你观察到 hard filtering / coverage filtering 容易让 ShopFacade 和 OldHospital 退化。这个也很合理。

在传统 sparse matching + PnP/RANSAC 里，冗余不是坏事。RANSAC 本来就是靠大量候选中的内点子集来求解。你在 PnP 前用 coverage 或 calibrated score 过滤，等于提前替 RANSAC 做决定；一旦 score 没有和真实 inlier 完全对齐，就会删掉弱但正确的 matches。

所以当前应把 filtering 改成：

```text
hard filtering  →  soft ordering / PROSAC-style priority
```

也就是说，不要删匹配，只改变采样优先级或候选排序。

---

### 2.4 Gated residual 仍然扰动了 STDLoc/PLY descriptor 的稳定结构

你已经验证 residual descriptor 多数退化。这个信号很硬：当前 learned descriptor / residual descriptor 不该进入主链路。

STDLoc 的核心是 Feature Gaussian + matching-oriented Gaussian sampling + scene-specific detector + sparse-to-dense localization；它本身已经是一个完整强 baseline。([arXiv](https://arxiv.org/html/2503.19358v1?utm_source=chatgpt.com)) SplatLoc 也强调 unbiased 3D descriptor 与 salient 3D landmark selection，而不是简单把 learned blended descriptor 直接塞进 PnP。([arXiv](https://arxiv.org/html/2409.14067v1?utm_source=chatgpt.com)) 你当前结果说明：在 Cambridge 上，尤其 GreatCourt / StMarysChurch，PLY/STDLoc descriptor 的稳定性比 learned residual 更可靠。

所以 residual descriptor 应该退到 ablation，不要再作为默认模块。

---

## 3. 现在最重要的策略：不要再试图“统一改进所有场景”

当前数据已经说明，不同场景的最佳策略不同：

| Scene | 最合理选择 |
|---|---|
| GreatCourt | 回退 STDLoc baseline |
| KingsCollege | 基本回退 STDLoc baseline；R5 微升不值得换 te 大退 |
| OldHospital | 用历史 selected |
| ShopFacade | 用历史 selected |
| StMarysChurch | 用历史 selected |

如果用这个**保守 scene-gated 选择**，按你已有表格估算：

| 配置 | te / R5 |
|---|---:|
| GreatCourt | STDLoc：10.890 / 23.60% |
| KingsCollege | STDLoc：17.690 / 1.70% |
| OldHospital | selected：8.990 / 32.97% |
| ShopFacade | selected：2.401 / 81.55% |
| StMarysChurch | selected：3.570 / 67.17% |

宏平均大约是：

```text
Mean te ≈ 8.71 cm
Macro R5 ≈ 41.40%
```

这比 STDLoc 的 `9.127 cm / 37.12%` 更强，而且比历史 selected 的 `10.099 cm / 39.40%` 更稳。

但注意：**不能用 test set 结果来决定每个 scene 用哪个 branch。**  
正确做法是：

> 从 train images 中切一个 calibration-val split，用这个 split 选择每个 scene 的 branch；然后 test set 只跑一次最终配置。

这条路线比继续训练 SP-GS2 P0 full 更有希望达到投稿级。

---

## 4. 下一步应该做的是 branch selection，而不是继续训练 P0 full

我建议把主方法从“改 matcher / 改 descriptor / 改 bank”改成：

> **Baseline-preserving reliability selection for STDLoc-style Gaussian localization**

也就是：默认保护 STDLoc，只有当 learned branch 在 calibration-val 上证明可靠时才接管。

### 4.1 Scene-level branch selection

先做最简单版本。

每个 scene 准备候选 branch：

```text
A: STDLoc / PLY baseline
B: historical selected / protected STDLoc+PLY
C: KC strict
D: SP-GS2 P0
E: residual / locability prior branch
```

在 train split 中划出 calibration-val，例如：

```text
每 5 或 8 张 train image 取 1 张作为 calibration-val
其余仍用于训练 / cache / calibration
```

在 calibration-val 上选：

$$
Score = R5 - \lambda \cdot \max(0, te - te_{baseline})
$$

或者更直接：

```text
优先最大化 R5；
如果 R5 差距 < 1%，选 te 更低的；
如果 learned branch 比 baseline R5 低，强制回退 baseline。
```

然后 test set 只用选出的 branch。

这会把当前实验里最有价值的东西保住：OldHospital、ShopFacade、StMarysChurch 的收益；同时避免 GreatCourt / Kings 被错误 branch 拖垮。

---

### 4.2 Query-level pose candidate selector

如果 scene-level selection 还不够，再做 query-level selector。

对每个 query 同时跑两个或三个候选 localizer：

```text
pose_A = STDLoc baseline
pose_B = historical selected
pose_C = SP-GS2 / calibrated branch，可选
```

然后不要融合 matches，只在 pose-level 选择结果。

特征用这些：

```text
sparse_inlier_count
sparse_inlier_ratio
median_reprojection_error
PnP success flag
2D coverage
3D coverage
dense_refine_delta
dense verification score
top1/top2 pose score gap
final dense residual
number of valid dense matches
```

训练一个非常小的 selector：

```text
输入: 每个 candidate pose 的 confidence features
输出: 选择哪个 pose
label: calibration-val 上 pose error 最小或 R5 成功的 branch
```

这比继续改 match filtering 更稳，因为它不破坏 RANSAC 的输入，只在最终 pose 层做保守选择。

**这也是目前最像“投稿级工程贡献”的方向。**

---

## 5. SP-GS2 P0 full 应该怎么处理？

不要删。它很有论文价值，但位置要改。

建议论文里这样写：

> We further explored stronger localization-guided training with differentiable PnP, calibrated landmark matchability, coverage-aware filtering, multi-hypothesis PnP, and gated residual descriptors. Surprisingly, the full combination degraded performance, revealing a mismatch between static matchability proxies and query-time PnP success. This motivates our final baseline-preserving reliability selection design.

也就是说，SP-GS2 P0 作为**负结果分析 / ablation**，而不是主方法。

主表不要放 SP-GS2 P0 full 默认结果。主表应该放：

```text
STDLoc baseline
历史 selected
scene-val gated selected
query-level gated selected，若完成
SP-GS2 P0 full ablation
```

这样读者会看到你不是盲目堆模块，而是通过负结果收敛到了一个更稳的定位系统。

---

## 6. 具体代码修改建议

### 6.1 默认配置回滚

把默认 eval / paper config 改成：

```text
descriptor_source = ply_loc
landmark_source = sampled 或 protected sampled
landmark_score_mode = legacy / protected
disable calibrated prior in sparse matcher by default
disable coverage filtering by default
disable gated residual descriptor by default
disable KC strict replacement by default
disable SP-GS2 P0 checkpoint as default
```

当前 repo 的 eval 里已经有 landmark score weights、pnp hypotheses、match filter、hybrid residual alpha 等配置入口。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/eval_cambridge_hybrid.py)) 接下来不是继续增加入口，而是把 default 收敛到最稳组合。

---

### 6.2 新增 `select_cambridge_branch.py`

功能：

```text
输入:
  scene
  candidate eval result dirs
  calibration-val image ids
  target metric: R5 / te / combined score

输出:
  selected_branch.json
```

`selected_branch.json` 类似：

```json
{
  "GreatCourt": "stdloc_baseline",
  "KingsCollege": "stdloc_baseline",
  "OldHospital": "historical_selected",
  "ShopFacade": "historical_selected",
  "StMarysChurch": "historical_selected"
}
```

注意：这个 JSON 必须由 calibration-val 生成，不能由 test set 手选。

---

### 6.3 新增 `eval_cambridge_branch_selected.py`

功能：

```text
读取 selected_branch.json
每个 scene 用对应 branch 跑 test
汇总 mean te / macro R5
```

这样最终主表是干净的，不会变成“每个 scene 按 test 最优挑结果”。

---

### 6.4 新增 query-level selector，但先不要上复杂模型

第一版不需要 MLP，先做规则：

```text
if selected_branch_pose has:
    inlier_ratio >= baseline_inlier_ratio + δ1
    dense_residual <= baseline_dense_residual - δ2
    coverage >= baseline_coverage * 0.8
then use selected_branch_pose
else use baseline_pose
```

如果规则版有效，再训练 logistic selector。

这条线比 learned descriptor 更稳。

---

## 7. 关于 PnP loss：不要继续强训 full P0

我建议接下来把 PnP loss 降级：

```text
pnp_pose_weight = 0 或极小
pnp_reproj_weight = 0 或极小
matchability / locability 只作为 auxiliary diagnostic
```

原因是：现在的 PnP training path 修得更“正确”，但更像 oracle target path；它不能直接改善 test-time sparse matching。继续强训只会让模型对 train-time valid target distribution 过拟合。

如果还要保留训练贡献，建议改成 hard-negative contrastive，而不是 PnP pose loss：

```text
positive: GT reprojection < 2-5 px
hard negative: descriptor similarity high but GT reprojection > 20 px
loss: supervised contrastive / binary ranking
```

核心目标变成：

> 让重复结构中的错误高相似 landmark 降分。

这比 differentiable PnP 更直接针对 GreatCourt。

但这条是中长期，不要作为当前投稿主结果前置条件。

---

## 8. 当前最实用的投稿级路线

### 主方法命名

可以考虑：

> **Reliability-Gated Loc-GS for STDLoc-style Gaussian Relocalization**

或者更保守：

> **Loc-GS: A Reliability Study and Baseline-Preserving Extension of Gaussian Feature Localization**

### 主 claim

不要写：

> We improve STDLoc on all Cambridge scenes.

写：

> We show that localization-guided Gaussian feature learning is scene-dependent. By preserving the STDLoc/PLY backbone and selecting reliability-calibrated branches, Loc-GS improves macro recall while avoiding degradation on repetitive scenes.

### 主表结构

| Method | GreatCourt | Kings | Old | Shop | StMary | Mean te | Macro R5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| STDLoc | baseline | baseline | baseline | baseline | baseline | 9.127 | 37.12 |
| Historical selected | worse GC | slight Kings | better Old | better Shop | better StMary | 10.099 | 39.40 |
| SP-GS2 P0 full | worse | worse | worse | worse | worse | 12.446 | 28.78 |
| Val-gated branch selection | expected protected | expected protected | selected | selected | selected | target ≈ 8.7 | target ≈ 41.4 |

最后一行必须用 calibration-val 选择后重新跑 test，不能直接用目前 test 结果拼表。

---

## 9. 论文里如何解释 SP-GS2 P0 负结果

这段可以写得很有价值：

1. **Static landmark matchability is insufficient.**  
   Per-landmark TP/FP priors cannot model query-dependent ambiguity in repetitive architecture.

2. **Hard pre-PnP filtering reduces useful redundancy.**  
   RANSAC benefits from many candidate matches; filtering with imperfect scores deletes weak but correct matches.

3. **Differentiable PnP supervision is not automatically test-aligned.**  
   Valid-target masking cleans training but creates an oracle-like candidate set that does not match test-time all-candidate matching.

4. **Learned descriptor residuals disturb strong PLY descriptors.**  
   The STDLoc/PLY descriptor structure is already robust; small learned residuals can still damage nearest-neighbor geometry.

5. **Pose-level reliability gating is safer than match-level replacement.**  
   Late selection preserves baseline behavior and only accepts learned branches when confidence is high.

这比“我们调不出来”强得多。它是一个明确的科学结论。

---

## 10. 下一步实验优先级

现在只建议跑三件事。

### 实验 A：calibration-val scene branch selection

目的：验证是否能合法得到 `≈8.7cm / ≈41.4% R5` 级别结果。

流程：

```text
1. 从 train set 划 calibration-val
2. 在 calibration-val 上比较 STDLoc / historical selected / SP-GS2 P0
3. 每个 scene 选 branch
4. test set 只跑选定 branch
```

如果成功，直接作为主结果。

---

### 实验 B：query-level fallback selector

目的：比 scene-level selection 更细。

流程：

```text
1. 每个 query 跑 STDLoc + selected
2. 用 confidence rule 选择 pose
3. 不改 match，不改 descriptor，不改 bank
```

如果它能在 GreatCourt 自动回退 baseline，同时在 Old/Shop/StMary 接受 selected，就够投稿。

---

### 实验 C：hard-negative ambiguity diagnostic

目的：解释 GreatCourt。

输出：

```text
GreatCourt top false-positive landmarks
FP/TP distribution
descriptor similarity vs reprojection error scatter
baseline vs selected false-positive heatmap
```

这会让论文中的 failure analysis 非常有说服力。

---

## 11. 最终判断

这轮更差不是坏事，它说明你已经把一条看似合理但不稳的路线排除了。

**该回滚的：**

```text
SP-GS2 P0 full checkpoint as default
calibrated prior in sparse matcher
coverage hard filtering
gated residual descriptor
KC strict replacement
strong differentiable PnP training
```

**该保留的：**

```text
PnP loss修复作为代码正确性
calibration脚本作为诊断工具
KC / matchability / locability作为分析指标
SP-GS2 P0作为negative ablation
historical selected/protected STDLoc+PLY作为当前最强 learned branch
```

**该新做的：**

```text
scene-level calibration-val branch selection
query-level pose fallback selector
hard-negative ambiguity diagnostic
```

一句话：  
**不要再试图让 learned SP-GS2 branch 全场景接管 STDLoc。当前最有投稿希望的方案是 baseline-preserving：默认信任 STDLoc，只在验证集和 query confidence 都支持时，让 Loc-GS learned branch 接管。**

---

**Sources:**

- [Loc-GS/loc_gs/losses/differentiable_pnp.py at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/differentiable_pnp.py)
- [From Sparse to Dense: Camera Relocalization with Scene ...](https://arxiv.org/html/2503.19358v1?utm_source=chatgpt.com)

## Prompt:
2026/5/11 16:30:24

现在项目中针对不同数据有单独类似于调参的不同配置吗？我是想实现针对不同数据调整的超参数较少的这种强泛化性？目前这条定位引导重建的主要方向上。精度还有优化空间吗？

## Response:
2026/5/11 16:30:27

我会先核对最新配置入口和评测脚本，判断是否存在 per-scene 调参或分支选择，并区分默认配置、实验开关与过拟合残留。然后聚焦少超参数泛化方案，以及定位引导重建还剩哪些可稳健提升的空间。

## Response:
2026/5/11 16:30:48

> **Searching for "selected_branch" or branch selector**
>
> [Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/eval_cambridge_hybrid.py)
>
> **Examining configuration options for data and parameters**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/ChatGPT-Branch%20%C2%B7%20SP-GS2.md)
>
> **Considering adaptive data and hyperparameter approaches**
>
> I’m thinking about how data-adaptive methods affect scene size and calibration, especially with threshold distributions, but I’m not sure if per-scene selection makes sense. It could be too specific and may hurt generalization. A more universal approach might be better, using validation-free quality rules to keep things simpler.
>
> Thought for 4m 46s

## Response:
2026/5/11 16:35:14

是的，**现在项目里已经存在“事实上的按数据/场景调参”倾向**，但它还没有被整理成一个干净、少超参数、强泛化的正式方法。

我建议你现在把目标改成：

> **不是继续找每个场景最优配置，而是收敛出一个 baseline-preserving、少开关、可跨场景固定使用的 Loc-GS recipe。**

否则即使某些场景刷上去了，论文里也很难说服评审这是方法泛化，而不是 per-scene tuning。

---

## 1. 当前项目是否已经有针对不同数据的独立配置？

**公开配置层面：有 Replica 的 per-scene YAML，但 Cambridge 线主要是 CLI 实验配置。**

`configs/` 目录目前主要是 `superpoint_hybrid_room_0/room_2`、`superpoint_localization_room_0/room_2` 这些 Replica 配置，没有看到正式的 Cambridge 每场景 YAML 配置文件。也就是说，Cambridge 的配置还没有沉淀成“统一 recipe + 少量数据路径参数”的形式。([GitHub](https://github.com/Arthurshen926/Loc-GS/tree/main/configs))

README 里 Cambridge hybrid localization 也是通过 `train_cambridge_hybrid` / `eval_cambridge_hybrid` 的命令行参数运行，而不是固定配置文件；仓库主线仍然写的是 SuperPoint teacher → Gaussian feature field → descriptor/detector rendering → matching/PnP relocalization。([GitHub](https://github.com/Arthurshen926/Loc-GS))

**但实验实践层面：已经有 per-scene 调参。**

因为现在不同场景事实上用了不同的 historical selected / protected / SP-GS2 / KC / residual / localizer 配置，而且 GreatCourt、OldHospital、ShopFacade、StMarysChurch 的最佳配置明显不同。这种做法对研究探索是必要的，但不能直接作为“强泛化方法”主结果。

代码也说明当前超参数面非常大。训练端有 PnP loss、locability prior、same-view、cross-view、hard-negative、SplatLoc saliency、PLY residual 等大量权重和开关；评测端还有 landmark score、calibrated matchability、match filtering、multi-hypothesis PnP、hybrid residual alpha 等参数。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

所以当前状态可以概括为：

> **项目具备很多可调组件，但还没有形成一个少超参数、跨场景稳定的默认配置。**

---

## 2. 如果你想强调强泛化，必须减少“可自由调”的维度

现在最危险的是论文里出现这种情况：

| Scene | 用法 |
|---|---|
| GreatCourt | 回退 STDLoc |
| OldHospital | historical selected |
| ShopFacade | selected/protected |
| StMarysChurch | selected |
| KingsCollege | 另一个 localizer 设置 |

如果这是根据 test set 结果挑出来的，评审会直接认为是 per-scene overfitting。  
但如果你用 **train 内部 validation split 自动选择**，就可以合理化。

不过你现在想要“针对不同数据调整的超参数较少”，那我建议更进一步：**不要做复杂 scene-specific branch selection，而是固定一个 conservative default。**

---

## 3. 建议收敛成三层配置

### 第一层：允许随数据变化的参数

这些不是方法超参数，而是数据输入，允许变化：

| 参数 | 是否允许变化 | 原因 |
|---|---:|---|
| scene name / data root | 是 | 数据路径 |
| image width / height | 是，但最好固定 640×360 | Cambridge 已基本固定 |
| camera intrinsics | 是 | 数据属性 |
| PLY / STDLoc map path | 是 | 数据资产 |
| checkpoint path | 是 | 模型资产 |
| number of train views | 是 | 数据规模 |
| landmark budget | 可以自动化 | 应按 map size / train views 自动设定 |

### 第二层：应该全场景固定的方法参数

这些最好不要再 per-scene 调：

| 模块 | 建议默认 |
|---|---|
| descriptor source | **PLY / STDLoc descriptor** |
| landmark bank | **STDLoc sampled / protected bank** |
| learned descriptor residual | 默认关闭，只做 ablation |
| KC strict selection | 默认关闭 |
| calibrated matchability prior | 默认不直接加进 sparse matcher |
| coverage hard filtering | 默认关闭 |
| MNN / second-best margin | 默认关闭 |
| multi-hypothesis PnP | 默认关闭或只作为 ablation |
| strong differentiable PnP training | 默认降权或关闭 |
| locability | 只作为 confidence / diagnostic / soft score，不硬替换 backbone |

这听起来保守，但这是目前实验结果支持的最稳路线。

### 第三层：自动校准，不人工调参

如果还想利用 learned 部分，建议只允许这种自动校准：

1. 从 train set 切一个 calibration-val；
2. 用固定规则或固定小模型学习 confidence；
3. test set 不再调任何参数；
4. 所有场景共用同一套 selection/fallback 规则。

这样可以写成方法，而不是调参。

---

## 4. 当前定位引导重建方向还有没有精度优化空间？

**有，但不在“继续堆 PnP loss / residual / KC / filtering”这条路上。**

最新 SP-GS2 P0 full 的结果已经说明：这些机制组合起来反而从 `39.40% macro R5` 掉到 `28.78% macro R5`。这不是小波动，而是强信号：**当前定位引导重建的训练目标没有和 test-time localization distribution 对齐。**

更具体地说，优化空间在下面四个方向。

---

# 方向 1：从“定位引导 descriptor 重建”改成“定位引导 reliability 学习”

目前最不稳的是 learned descriptor / residual descriptor。你已经反复验证：一旦 learned descriptor 进入主匹配链路，就容易破坏 STDLoc/PLY descriptor 的稳定结构。

所以主线应该改成：

> **不让 Loc-GS 替代 STDLoc descriptor，而是让 Loc-GS 预测哪里可靠、哪里危险。**

也就是：

- PLY/STDLoc descriptor 负责匹配；
- Loc-GS 学 locability / ambiguity / confidence；
- learned signal 只影响 pose confidence 或 soft rerank；
- 不做 hard replacement。

这条线更容易泛化，因为它不破坏已有强 backbone。

---

# 方向 2：用 hard-negative ranking 替代 differentiable PnP pose loss

你已经实现了 TP/FP matchability calibration。脚本会根据 top-k query-to-landmark candidates，用 GT pose 计算 reprojection error，并结合 depth/alpha visibility 过滤，统计 TP/FP，再用 beta-smoothed 方式估计 landmark reliability。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/calibrate_landmark_matchability.py))

这说明项目已经有了训练 hard negative 的基础。

我建议下一步不要再强化 PnP pose loss，而是训练一个更直接的 loss：

$$
s(q, l^+) > s(q, l^-)
$$

其中：

- $l^+$：GT reprojection error 小于阈值的 positive landmark；
- $l^-$：descriptor similarity 很高但 reprojection error 很大的 hard negative landmark。

loss 可以很简单：

$$
\mathcal{L}_{rank}
=
\max(0, m - s(q,l^+) + s(q,l^-))
$$

或者 binary CE：

$$
P(y=1|q,l)
$$

输入不要只用 landmark 静态 score，而要用 query-conditioned features：

- descriptor similarity；
- top1-top2 margin；
- landmark TP/FP reliability；
- locability；
- visibility；
- geometry score；
- detector score；
- 2D/3D coverage indicator。

这个方向比 differentiable PnP 更直接，因为 GreatCourt 的核心问题是：

> 错误 landmark 和正确 landmark 都能形成高 descriptor similarity。

PnP loss太后置；hard-negative ranking正好打这个问题。

---

# 方向 3：从 hard filtering 改成 confidence fallback / pose-level selection

你已经看到 coverage filtering、margin filtering、MNN 都不稳。原因很简单：RANSAC 需要冗余，hard filtering 会删掉弱但正确的 matches。

更稳的是：

> 不在 match 级别硬删，而是在 pose 级别做保守 fallback。

也就是说同时跑：

1. STDLoc / PLY baseline pose；
2. Loc-GS selected / learned branch pose；

然后用固定 confidence 选择哪个 pose。

confidence 可以用：

- sparse inlier ratio；
- inlier count；
- median reprojection error；
- 2D coverage；
- 3D coverage；
- dense verification score；
- dense refinement 前后 pose 是否稳定；
- PnP success flag。

这比 per-scene 调参更容易泛化。  
因为你不需要为每个场景决定“该不该用 Loc-GS”，而是每个 query 根据自身 confidence 决定是否回退 baseline。

如果这个规则固定，论文里可以说：

> Loc-GS is baseline-preserving: it never replaces the STDLoc pose unless the learned branch passes a reliability check.

这比“每个场景选一个配置”更强。

---

# 方向 4：多分辨率，而不是 original teacher 强行压到 45×80

你之前验证过：original teacher 自身没问题，但 original SuperPoint 下采样到 45×80 训练很难。这个判断很重要。

所以不要再做：

> original SuperPoint descriptor → 45×80 field → 替代 descriptor。

更稳的是：

- query side 保留 original STDLoc/SuperPoint feature；
- map side 保留 PLY/STDLoc descriptor；
- Loc-GS field 只学低分辨 locability / reliability；
- 如果要学 descriptor，只学小 residual，而且默认不进入主表。

如果还有资源，可以做一个全局固定的多分辨率实验：

| 配置 | 目的 |
|---|---|
| 45×80 locability | 当前默认 |
| 90×160 locability | 看 GreatCourt 重复结构是否改善 |
| 45×80 descriptor | 只做 ablation |
| 90×160 descriptor residual | 只做 ablation |

注意：这不能变成每场景调分辨率。只能选一个全局分辨率。

---

## 5. 我建议的“少超参数默认方法”

现在可以收敛成下面这个方法版本。

### Loc-GS-General 默认训练

| 模块 | 默认设置 |
|---|---|
| teacher feature source | resized |
| descriptor reconstruction | 保留，但作为 auxiliary |
| detector reconstruction | 保留，低权重 |
| geometry update | 默认 freeze，不 train xyz / scaling / opacity |
| strong differentiable PnP pose loss | 默认关闭或极低权重 |
| locability | 训练 |
| hard-negative ranking | 新增，作为主定位引导目标 |
| PLY residual descriptor | 默认关闭 |
| KC / SplatLoc score | 作为 input feature，不作为 hard selector |
| calibration | train split 自动生成，不人工调参 |

### Loc-GS-General 默认评测

| 模块 | 默认设置 |
|---|---|
| descriptor | PLY / STDLoc |
| landmark bank | STDLoc sampled / protected |
| hard match filtering | 关闭 |
| calibrated prior in matcher | 关闭或极小固定权重 |
| branch selection | query-level confidence fallback |
| learned descriptor | 不进主结果 |
| KC strict | 不进主结果 |
| SP-GS2 P0 full | 只做 ablation / negative result |

这个配置的精神是：

> **Loc-GS 不接管 STDLoc，而是给 STDLoc 增加可靠性判断。**

这才更像强泛化方法。

---

## 6. 精度优化空间到底有多大？

按你目前结果，**短期内最现实的提升不是把 SP-GS2 P0 full 训好，而是把历史 selected 的收益保住，同时避免 GreatCourt/Kings 退化。**

当前三个参考点是：

| 方法 | Mean te | Macro R5 |
|---|---:|---:|
| STDLoc | 9.127 cm | 37.12% |
| historical selected | 10.099 cm | 39.40% |
| SP-GS2 P0 full | 12.446 cm | 28.78% |

如果能做到：

- GreatCourt 自动回退 STDLoc；
- Kings 自动回退 STDLoc；
- OldHospital / ShopFacade / StMarysChurch 保留 selected 收益；

那么理论上会接近：

```text
Mean te ≈ 8.7 cm
Macro R5 ≈ 41.4%
```

这个结果就比较像投稿级主表了。但前提是：**这个选择必须来自 train/calibration-val，不能来自 test set。**

所以我会把短期目标设成：

| 指标 | 目标 |
|---|---:|
| Macro R5 | ≥ 41% |
| Mean te | ≤ 9.1 cm |
| GreatCourt R5 | 不低于 STDLoc 超过 2% |
| OldHospital R5 | 保持 ≥ 28% |
| ShopFacade R5 | 不低于 baseline |
| StMarysChurch R5 | 保持正收益 |
| KingsCollege | R5 持平即可，te 不明显退化 |

这比追求全场景全面提升更现实，也更能支撑论文。

---

## 7. 下一步最该做的三件事

### 第一，建立一个固定 default recipe

不要再让主表依赖一堆命令行开关。建议新增：

```text
configs/cambridge_locgs_general.yaml
configs/cambridge_locgs_ablation_spgs2.yaml
configs/cambridge_locgs_negative.yaml
```

其中 `cambridge_locgs_general.yaml` 只允许改：

```text
scene
data_root
checkpoint
ply_path
output_dir
```

其他参数全固定。

### 第二，做 leave-one-scene-out 验证

用 Cambridge 五个场景做：

```text
用 4 个 scene 决定默认规则 / confidence selector
在剩下 1 个 scene 测
轮换 5 次
```

如果这个仍能提升 macro R5，你就可以比较有底气地说“少超参数泛化”。

### 第三，训练 hard-negative matchability，而不是继续 full PnP training

现在 calibration 已经能产生 TP/FP。下一步把它变成训练目标：

```text
positive: reprojection error < 4/8 px
hard negative: high descriptor similarity but reprojection error > 32 px
model: small query-conditioned matchability head
output: reliability score
usage: pose-level confidence / soft rerank
```

这条线和“定位引导重建”仍然一致，但比 differentiable PnP 更贴近当前失败原因。

---

## 8. 论文叙事也要同步收敛

现在不要再写：

> Loc-GS learns a better Gaussian descriptor field.

更稳的是：

> Loc-GS investigates localization-guided reliability learning for Gaussian feature localization. Instead of replacing the STDLoc/PLY descriptor backbone, Loc-GS learns matchability and locability cues to preserve baseline robustness while improving recall on scenes where learned reliability is confident.

中文理解就是：

> **本文不是证明 learned descriptor 全面更强，而是证明定位引导的可靠性学习可以作为 STDLoc/SplatLoc 风格定位系统的保守增强。**

这样你既能解释 OldHospital / Shop / StMary 的收益，也能解释 GreatCourt 的风险，还能把 SP-GS2 P0 full 的负结果写成有价值的 ablation。

---

## 9. 最终回答

**现在项目中确实已经出现了类似 per-data/per-scene 调参的现象，但还没有正式整理成配置体系。**  
如果目标是强泛化，就必须把主方法收敛到一个固定 recipe，并把场景差异交给自动 calibration / confidence fallback，而不是手动调 KC、residual、MNN、match filter、Pnp hypotheses 等开关。

**定位引导重建还有精度优化空间，但不是继续加强 descriptor field 或 full differentiable PnP。**  
最有希望的是：

1. 保留 STDLoc/PLY descriptor 主干；
2. Loc-GS 只学习 locability / matchability / confidence；
3. 用 TP/FP hard negatives 训练 query-conditioned reliability；
4. 不做 hard filtering，改做 pose-level fallback；
5. 用固定配置 + calibration-val / leave-one-scene-out 证明泛化。

这条路线的目标不是“每个模块都比 STDLoc 强”，而是：

> **默认不伤害 STDLoc，在模型有把握时带来额外收益。**

这才是目前最可能达到论文投稿级的方向。

---

**Sources:**

- [Loc-GS/configs at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/tree/main/configs)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)