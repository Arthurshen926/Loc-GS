# Branch · SP-GS

**User:** Anonymous  
**Created:** 5/10/2026 13:58:36  
**Updated:** 5/11/2026 0:44:07  
**Exported:** 5/11/2026 0:54:01  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00-shen-qi-yuan/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00-shen-qi-yuan/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370)  

## Prompt:
2026/5/11 00:38:49

目前又按照我们讨论的内容推进了一下，具体情况请看https://github.com/Arthurshen926/Loc-GS 最新推送。阶段性推进的内容总结：
第一轮：
已继续完成这一轮实现和验证。 

实现侧新增了：
- `keypoint_consensus_score` 支持跨视角 descriptor consistency，可作为 SplatLoc/ULF-Loc 风格 landmark score 的补充项。
- eval CLI 新增 `--landmark_score_keypoint_consensus_descriptor_weight`。
- 相关单测已补齐。

验证结果比较明确：

- `KC strict selection` 不是全局默认好配置。
- ShopFacade full 有小幅正向：`3.005cm / R5 76.70%` → `2.978cm / R5 78.64%`。
- GreatCourt full 只小幅改善 median：`15.765cm` → `15.613cm`，但 R5 仍 `12.11%`，没有超过 MNN 历史最好 `15.479cm / R5 12.76%`。
- OldHospital、StMarysChurch 相比当前最强配置明显退化：
  - OldHospital best current `8.990cm / R5 32.97%`，KC strict 只有 `13.560cm / R5 18.13%`。
  - StMarysChurch best current `3.570cm / R5 67.17%`，KC strict 只有 `6.255cm / R5 36.60%`。
- KingsCollege q40 strict selection 也不接受：sampled q40 `11.563cm / R5 2.5%`，KC strict `12.080cm / R5 0%`。
- descriptor-consistency 版本 q40 没改善，GreatCourt R5 还从 `7.5%` 掉到 `5%`，所以保留接口但不设为默认。

判断：专家建议里的“landmark saliency / keypoint consensus”方向不是理论错误，但当前直接替换 landmark bank 的实现不够稳。问题主要是 score 和最终 PnP 可用匹配质量没有充分对齐；45x80 grid 上的 teacher descriptor consistency 也不能可靠地区分重复结构。

验证：
`/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests`  
结果：`95 passed, 16 warnings`

当前没有遗留训练/评测进程。GPU 0/1/3/4 已空，2/5 仍有其他上下文占用。下一步应避免继续扩大 KC replacement，转向训练阶段的 locability calibration / residual descriptor 约束，而不是 eval-time 替换 bank。
第二轮：
本轮继续推进后，结论比较硬：继续堆 residual / locability-prior 训练没有带来精度提升，反而多数场景退化；当前最稳的收益仍来自保留 STDLoc/PLY 主干，只把 learned 部分作为辅助。

**已完成**
- 增加并验证了 locability prior alignment、PLY residual regularization、PnP visibility/top-k 相关接口与测试。
- 跑了 Cambridge 多场景 full eval 和 GreatCourt/ShopFacade/Old/StMary/Kings 的多组消融。
- 生成了新的论文图和汇总表：
  - `/root/Loc-GS/output/paper_figures/cambridge_summary_20260510/`
- 测试通过：`99 passed`，只有第三方 warning。
- 当前没有残留 train/eval 进程。

**当前 selected 指标**
```text
Scene          STDLoc te/R5      Selected te/R5
GreatCourt    10.890 / 23.60%    15.654 / 13.29%
KingsCollege  17.690 /  1.70%    19.880 /  2.04%
OldHospital   10.720 / 15.90%     8.990 / 32.97%
ShopFacade     2.647 / 80.58%     2.401 / 81.55%
StMarysChurch  3.690 / 63.80%     3.570 / 67.17%
```

宏平均 R@5cm 从 `37.12%` 提到 `39.40%`，但 mean median translation 从 `9.13cm` 退化到 `10.10cm`。所以目前可以支撑“部分场景和 macro recall 有收益”，还不能支撑“全 Cambridge 定位精度全面优于 STDLoc”的强 claim。

**关键判断**
- `residualprior_b8`、显式 locability-prior-only、matchability-prior、`image_xyz_grid` prefilter、减少 landmark 到 8k/12k，这些在 q80/q200 上偶尔有信号，但 full eval 多数不稳，不能作为默认。
- GreatCourt 是核心失败项：不是简单 batch size、RANSAC threshold、MNN、landmark 数能解决，问题更像重复结构下 landmark ambiguity 和 descriptor bias。
- 下一步应该聚焦 GreatCourt/Kings：做真正的 landmark-side ambiguity calibration 和 per-query coverage-aware selection，而不是继续加强 same-view reconstruction loss。当前实验已经说明“训练 loss 更好”不等于定位更好。

综上所述，仍然不符合预期啊，请具体帮我分析问题在哪里？需要继续优化精度以达到论文投稿级，该如何修改？

## Response:
2026/5/11 00:38:52

我看了最新实现后，关键判断是：新增模块大多仍是 proxy score，尚未按 PnP-inlier 目标校准。另一个具体缺口是 PnP top-k 仍可能从非 `valid_target_pixels` 候选取点。

## Response:
2026/5/11 00:43:59

我看完最新代码和你给的两轮结果后，判断比上一轮更清楚了：

**当前主要问题不是“还有一个显眼 bug 没修”，而是方法仍在用一组静态 proxy score 去替代真正的 PnP-useful matching reliability。**  
这些 proxy 在 OldHospital、ShopFacade、StMarysChurch 上能带来收益，但在 GreatCourt 这种大尺度、重复结构场景里会系统性误判，所以 strict KC、residual、locability-prior、score replacement 都不稳。

现在要达到投稿级，下一步不该继续堆 saliency / residual / eval heuristic，而应该把方法从：

> **static landmark saliency scoring**

升级成：

> **query-conditioned calibrated matchability + coverage-aware PnP matching**

这才是当前最可能把 GreatCourt 拉回来，同时保住 OldHospital / Shop / StMary 增益的方向。

---

## 1. 先明确：理论方向没有错，但当前实现的“优化对象”偏了

仓库 README 现在仍把主线写成：SuperPoint teacher features → low-dimensional per-Gaussian latent feature field → decoded descriptor/detector maps → geometry-aware matching and PnP relocalization，并且包括 differentiable matching、reprojection proxy、observability、per-Gaussian locability 等 localization-guided losses。这个总体方向是合理的。([GitHub](https://github.com/Arthurshen926/Loc-GS))

但你的结果说明，真正能支撑投稿的 claim 不能是：

> learned Gaussian descriptor / learned landmark bank 全面替代 STDLoc。

而应该收敛成：

> 在 STDLoc/PLY 主干上，学习 query-conditioned matchability、locability 和 landmark weighting，从而提升部分场景与宏平均 recall；但直接替换 landmark bank 或 descriptor 在重复结构场景中不稳定。

STDLoc 本身已经提出 sparse-to-dense localization、matching-oriented Gaussian sampling 和 scene-specific detector。([ZJU3DV](https://zju3dv.github.io/STDLoc/?utm_source=chatgpt.com)) SplatLoc 也已经有 salient 3D landmark selection 和 unbiased 3D descriptor learning。([arXiv](https://arxiv.org/html/2409.14067v1?utm_source=chatgpt.com)) ULF-Loc 进一步明确提出 keypoint-consensus sampling 和 local geometric consistency verification 来处理 3DGS feature bias / mismatch。([arXiv](https://arxiv.org/html/2605.04730v1?utm_source=chatgpt.com))

所以你现在的创新点不能只是“我也加了 landmark score”，而应是：

> **把 downstream PnP 中的真实 TP/FP 匹配行为校准成 matchability，并用于 query-time match selection。**

---

## 2. 目前结果为什么不符合预期？

你给的 selected 指标是：

| Scene | STDLoc te/R5 | Selected te/R5 | 判断 |
|---|---:|---:|---|
| GreatCourt | 10.890 / 23.60% | 15.654 / 13.29% | 核心失败 |
| KingsCollege | 17.690 / 1.70% | 19.880 / 2.04% | R5 微升，te 退 |
| OldHospital | 10.720 / 15.90% | 8.990 / 32.97% | 明显成功 |
| ShopFacade | 2.647 / 80.58% | 2.401 / 81.55% | 小幅成功 |
| StMarysChurch | 3.690 / 63.80% | 3.570 / 67.17% | 稳定成功 |

这说明方法不是无效，而是**场景选择性有效**。真正拖垮投稿 claim 的是 GreatCourt。GreatCourt 一退，mean median translation 从 9.13cm 变成 10.10cm；即使 macro R5 从 37.12% 提到 39.40%，论文也不能写成“全面优于 STDLoc”。

当前失败不是随机波动，原因主要有五个。

---

## 3. 问题一：KC strict selection 把“稳定显著点”误当成“PnP 有用点”

你新增的 `keypoint_consensus_score` 是合理的：它会统计 landmark 在多视角中投影到 keypoint map 的 hit rate，并可选用跨视角 descriptor consistency 加权。代码里确实是 keypoint hits + optional descriptor consistency；descriptor consistency 通过 hit 视角中采样 descriptor 的平均向量范数来估计。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/landmark_selection.py))

但这个 score 在 GreatCourt 失效很正常。因为 GreatCourt 的重复窗户、立面、柱廊上，很多错误 landmark 也会：

- 多视角可见；
- 经常落在 keypoint 附近；
- descriptor consistency 很高；
- 但 query matching 时容易和其他重复结构混淆。

也就是说：

> **KC 能找到“稳定的局部结构”，但不能保证这个结构具有全局可区分性。**

这正好解释了你的实验：KC strict 对 ShopFacade 有小幅帮助，但对 OldHospital、StMarysChurch、KingsCollege 和 GreatCourt 都不稳。它不是理论错，而是不能作为 hard replacement。

**修改：KC 只能作为弱特征，不要作为 strict selection criterion。**

建议默认：

```text
KC weight <= 0.1 或只进入 calibrated matchability feature
不直接替换 STDLoc sampled bank
不单独决定 landmark ranking
```

---

## 4. 问题二：当前 composite score 是手工 proxy 的几何平均，和最终 PnP 目标没有校准

`geometric_mean_score()` 会对各分量做 min-max normalization，然后用加权几何平均组合。常数分量已修成中性 1.0，这是对的；但几何平均仍然有一个结构性风险：任何一个 proxy 被误判为低分，好 landmark 就会被压掉。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/landmark_selection.py))

eval 里当前参与 score 的分量包括 locability、visibility、geometry、ambiguity、keypoint_consensus；rendered 情况下还会加入 detector、alpha、depth、observability、distinctiveness。随后 composite score 与 detector prior blend，再通过 `legacy_keep_ratio` 保留部分原始 sampled bank，并用 composite score fill 剩余 landmark。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/eval_cambridge_hybrid.py))

这个设计比直接 top-k 好，但本质还是：

> 用 detector / visibility / geometry / KC / ambiguity 等 proxy 近似 PnP inlier probability。

GreatCourt 的失败说明这个近似不够。重复结构里，proxy 分数高的 landmark 不一定是 PnP 内点；甚至可能是最危险的 false-positive landmark。

**修改：不要再手调几何平均权重作为主线。把这些 score 当作 feature，学习一个 calibrated matchability predictor。**

---

## 5. 问题三：当前 PnP top-k 还有一个训练路径缺口

你已经加了 GT depth/alpha visibility，非常正确。当前 `DifferentiablePnPMatchLoss` 会接收 `gt_depth_map` 和 `gt_alpha_map`，用 GT pose 下的 depth/alpha 检查 target visibility，并构造 `valid_target_pixels`。训练脚本也已经把 `gt_render["depth"]` 和 `gt_render["alpha"]` 传给 PnP loss。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

但当前代码里还有一个关键细节：matching logits 只用 `valid_pixels` 做 mask，然后 top-k PnP 直接从这个 logits 里取 top-k；并没有单独构造只允许 `valid_target_pixels` 的 `pnp_logits`。代码流程是：`logits = logits.masked_fill(~valid_pixels...)`，然后 `top_ids = logits.detach().topk(...)`，再用这些 top ids 加权 world points 进入 PnP。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/differentiable_pnp.py))

这意味着：  
**match loss 的 target 是 GT-visible 的，但 PnP pose path 的 top-k 仍可能选到 GT 不可见或几何无效的候选。**

在 GreatCourt 里，这个问题会被重复结构放大：descriptor 很相似，top-k 容易选到错误立面或错误窗口，即使这些点不属于 GT 可监督目标。

**直接修改：把 match path 和 PnP path 分开。**

现在：

```python
logits = logits.masked_fill(~valid_pixels[:, None, :], -1e4)
top_ids = logits.detach().topk(k=topk_pnp, dim=-1).indices
```

建议改成：

```python
match_logits = logits.masked_fill(~valid_pixels[:, None, :], -1e4)

pnp_logits = logits.masked_fill(~valid_target_pixels[:, None, :], -1e4)

top_ids = pnp_logits.detach().topk(k=topk_pnp, dim=-1).indices
top_logits = match_logits.gather(-1, top_ids)
top_probs = F.softmax(top_logits / tau_pnp, dim=-1)
```

然后 PnP 权重也不要继续只用 descriptor confidence：

```python
pnp_weights = valid_queries.float() * confidence.detach()
```

建议改成 geometry-aware confidence：

```python
top_geom = geometry_target.gather(-1, top_ids)
geom_conf = (top_probs.detach() * top_geom).sum(dim=-1)
pnp_weights = valid_queries.float() * geom_conf.clamp_min(1e-3)
```

这样 PnP pose loss 就不会被“高 descriptor 置信但几何错误”的候选主导。

这是我认为当前代码层面最值得马上改的一处。

---

## 6. 问题四：locability prior target 仍然太像 detector/geometry proxy，不是真正 matchability

现在 `locability_prior_alignment_loss()` 的目标是把 rendered locability 对齐到外部 detector/geometry prior；训练脚本会用 detector、alpha、depth consistency 构造 `locability_target_prior_from_render()`，再监督 locability map。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

这个 target 有用，但它不是最终目标。PnP 需要的是：

> 这个 landmark 在真实 query matching 中是否经常成为正确 2D-3D 内点。

而 detector/alpha/depth consistency 只能说明：

> 这个位置看起来显著、可见、几何平滑。

GreatCourt 的重复结构恰好会让这两者分离：显著且稳定的点，可能非常容易被错误匹配。

**修改：locability target 应该从 train-query TP/FP 统计中来，而不是只从 rendered proxy 来。**

---

## 7. 问题五：residual descriptor 继续退化，说明它不能作为主模块

`descriptor_residual_alignment_loss()` 当前只是把 learned descriptor 拉近 PLY/STDLoc descriptor basis，本质上是“不要偏离太远”。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py)) 但实验已经说明：即使加了 residual regularization，full eval 仍多数不稳。

这说明问题不是“residual 约束强度不够”，而是：

> learned descriptor 一旦进入主匹配链路，就可能破坏 STDLoc/PLY descriptor 里已经形成的全局区分结构。

所以 residual descriptor 现在不应作为主贡献。最多保留成：

```text
descriptor = normalize(PLY + small gated residual)
```

并且 residual gate 默认很小。

建议：

```python
alpha_j = alpha_max * sigmoid(gate_j)
desc_j = normalize(ply_desc_j + alpha_j * residual_j)
```

其中：

```text
GreatCourt: alpha_max = 0.03 或 0.05
Shop/Old/StMary: alpha_max = 0.05 或 0.10
L_gate = mean(alpha_j)
```

如果 gated residual 仍然退化，就把 descriptor residual 放到 ablation，不要放主方法。

---

## 8. 最关键的下一步：做 landmark-side TP/FP calibration

现在你需要停止扩大 KC replacement，转向**真正的 matchability calibration**。

具体做法：

### 8.1 用 train views 给每个 candidate match 打标签

对每个训练 query view：

1. 用原始 STDLoc/PLY landmark bank；
2. 提取 query keypoints/descriptors；
3. 对每个 query descriptor 取 top-K landmark candidates；
4. 用 GT pose 计算 reprojection error；
5. 如果满足：

$$
e_{ij} < \epsilon
$$

且 depth/visibility consistent，则：

$$
y_{ij}=1
$$

否则：

$$
y_{ij}=0
$$

这里的 $y_{ij}$ 才是真正的 “PnP-useful matching label”。

### 8.2 统计 landmark false-positive rate

对每个 landmark $j$：

$$
TP_j = \sum_i 1[y_{ij}=1]
$$

$$
FP_j = \sum_i 1[y_{ij}=0 \land j \text{ appears in top-K}]
$$

得到：

$$
R_j =
\frac{TP_j+\alpha}{TP_j+FP_j+2\alpha}
$$

这个 $R_j$ 是比当前 descriptor distinctiveness 更有用的 landmark ambiguity score。当前 `descriptor_landmark_distinctiveness()` 只是看非局部 descriptor 最近邻相似度；它能发现 descriptor 重复，但不能发现“这个 landmark 是否经常在真实 query 中被误匹配”。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/landmark_selection.py))

GreatCourt 最需要的是后者。

### 8.3 训练一个很小的 calibrated matchability model

输入 feature 可以包括：

```text
descriptor similarity
top1-top2 margin
STDLoc sampled score
detector prior
KC hit rate
KC descriptor consistency
visibility
geometry score
observability
locability
landmark TP/(TP+FP)
3D voxel / image cell statistics
```

输出：

$$
P(y_{ij}=1)
$$

模型不用复杂，先用 logistic regression 或 2-layer MLP。每个 Cambridge scene 单独训练一个 calibrator 完全合理，因为 STDLoc/SplatLoc 本身就是 scene-specific localization framework。

推理时不要再用：

```text
score = geometric_mean(proxy scores)
```

而用：

```text
score_ij = descriptor_similarity_ij + λ * logit(P_inlier_ij)
```

这一步是整个项目从“proxy saliency”走向“PnP-calibrated matchability”的关键。

---

## 9. 第二个关键改法：static bank selection 改成 query-time coverage-aware selection

现在 eval 里是静态选择 landmark bank：保留部分 legacy sampled ids，然后用 composite score 填充剩余名额。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/eval_cambridge_hybrid.py)) 这会有一个问题：同一个 landmark 对不同 query 的可靠性不同，尤其在 GreatCourt 这种重复结构场景里，静态 bank 无法处理 query-specific ambiguity。

建议改成：

```text
large candidate bank
→ query top-M descriptor matches
→ calibrated matchability scoring
→ coverage-aware match subset selection
→ PnP-RANSAC
→ dense verification/refinement
```

不要先把 bank 缩小到 8k/12k。保留 16k/32k candidate bank，然后在 query-time 筛 match。

### 9.1 Coverage-aware match selection

对 top-M matches 选择一个子集 $\mathcal{S}$：

$$
\max_{\mathcal{S}}
\sum_{(i,j)\in\mathcal{S}} s_{ij}
+
\lambda_{2D} C_{2D}
+
\lambda_{3D} C_{3D}
-
\lambda_{dup} R
$$

实际实现不用复杂优化，可以做 greedy：

1. 按 calibrated score 排序；
2. 每个 2D image cell 最多保留 $m$ 个；
3. 每个 3D voxel 最多保留 $n$ 个；
4. 每个局部重复 cluster 最多保留 $r$ 个；
5. 保留最低 match 数，避免过滤过猛。

这比 MNN / margin 稳，因为 MNN/margin 是全局硬过滤；coverage-aware selection 是在保留候选的基础上控制空间分布。

---

## 10. GreatCourt 需要 multi-hypothesis PnP

GreatCourt 的问题很可能不是单个 match 错，而是**整片重复结构形成错误但自洽的 pose hypothesis**。单次 RANSAC 很容易被错误 cluster 吸走。

建议增加 multi-hypothesis PnP：

1. top-M matches 按 3D voxel、reference view、visibility cluster 分组；
2. 每组跑 PnP-RANSAC；
3. 保留 top-K pose hypotheses；
4. 对每个 pose 做 lightweight dense verification；
5. 选 dense verification score 最好的 pose。

Verification score 可以先简单：

$$
S(T)
=
N_{inlier}(T)
-
\lambda \cdot median\_reproj(T)
+
\mu \cdot coverage(T)
+
\nu \cdot dense\_descriptor\_agreement(T)
$$

这比继续调 RANSAC threshold 更有意义。因为 GreatCourt 的错误通常不是“threshold 不合适”，而是“错误假设也有足够内点”。

---

## 11. 具体代码修改优先级

### P0：马上修 PnP top-k 的 mask

在 `DifferentiablePnPMatchLoss.forward()` 中新增：

```python
match_logits = logits.masked_fill(~valid_pixels[:, None, :], -1e4)
pnp_logits = logits.masked_fill(~valid_target_pixels[:, None, :], -1e4)
```

`log_probs/probs/match_loss/quality_loss` 用 `match_logits`。  
`topk_pnp/expected_world/pose_loss` 用 `pnp_logits` 的 top ids。  
`pnp_weights` 用 geometry-aware confidence。

这是当前最像“仍会削弱结果”的实现缺口。

### P1：新增 `calibrate_landmark_matchability.py`

输出文件：

```text
landmark_tp_count.pt
landmark_fp_count.pt
landmark_fp_rate.pt
matchability_calibrator.pt
```

先不用 MLP，只做统计版：

$$
score_j = \frac{TP_j+\alpha}{TP_j+FP_j+2\alpha}
$$

然后 eval 里新增：

```text
--landmark_score_calibrated_matchability_weight
--match_filter_calibrated_score_weight
```

### P2：改 eval pipeline，不再 strict replacement

默认配置改成：

```text
candidate bank: STDLoc sampled 16k/32k
legacy_keep_ratio: 0.9 或 1.0
KC: disabled by default 或 small feature only
composite score: 不用于 hard replacement
calibrated matchability: 用于 query-time match filtering
```

也就是说，先恢复 STDLoc bank 的保护性，再在 query-time 过滤错误 match。

### P3：实现 query-time coverage-aware match filtering

新增参数：

```text
--match_filter_mode calibrated_coverage
--match_filter_top_m 2000
--match_filter_image_grid 8
--match_filter_max_per_image_cell 8
--match_filter_xyz_grid 8
--match_filter_max_per_xyz_cell 8
--match_filter_min_matches 128
```

### P4：GreatCourt 加 multi-hypothesis PnP

新增：

```text
--pnp_hypotheses 8
--pnp_cluster_mode xyz_voxel
--pnp_dense_verify_topk 4
```

先只在 GreatCourt/Kings 上用，不要默认全场景。

### P5：residual descriptor 改成 gated residual

不要再 full residual 训练。改成：

```text
alpha_max <= 0.05
L1 gate regularization
pure hybrid descriptor only as ablation
```

---

## 12. 下一轮实验不要再大范围扫参数，做 5 个关键实验

### 实验 1：PnP top-k mask fix

只改 P0，重训 GreatCourt / ShopFacade。

看：

```text
GreatCourt R5 是否从 13.29% 回到 15%+
ShopFacade 是否不退
PnP loss 是否稳定
valid query 数是否变化
```

### 实验 2：STDLoc protected bank

```text
legacy_keep_ratio = 0.9 / 1.0
candidate_source = sampled
no KC strict
no residual descriptor
```

目标是确认 GreatCourt 是否能回到接近 baseline。如果能，说明之前主要是 replacement 过猛。

### 实验 3：landmark TP/FP calibration

用 train views 统计 false-positive-heavy landmarks。

看：

```text
GreatCourt RANSAC inlier ratio
GreatCourt sparse initial R5
Old/StMary 是否保留收益
```

### 实验 4：query-time coverage-aware filtering

在不改 bank 的情况下过滤 matches。

看：

```text
#matches
inlier ratio
2D coverage
3D coverage
sparse PnP success
dense refine 前后的 te
```

### 实验 5：multi-hypothesis PnP

只跑 GreatCourt/Kings。

看：

```text
错误 cluster 是否减少
dense verification 是否能选对 hypothesis
GreatCourt R5 是否恢复到 20%+
```

---

## 13. 投稿级目标怎么定

当前状态可以支撑：

> macro recall 有收益，部分场景明显提升，但 GreatCourt 失败说明直接 learned-bank replacement 不稳。

但要达到更强投稿，建议最低目标：

| 指标 | 目标 |
|---|---:|
| GreatCourt R5 | 至少恢复到 20% 左右，最好接近 STDLoc 23.6% |
| macro R5 | ≥ 41% |
| mean median te | ≤ 9.3cm，至少不能明显高于 STDLoc |
| OldHospital | 保持 R5 ≥ 28% |
| ShopFacade | R5 不低于 baseline |
| StMarysChurch | 保持正收益 |
| KingsCollege | R5 微升即可，但 te 不要明显退化 |

如果 GreatCourt 恢复不了，论文 claim 必须收敛成 failure-analysis / reliability-learning paper，而不是 Cambridge full improvement paper。

---

## 14. 最终判断

这轮结果不符合预期，但不是坏消息。它把问题定位得更清楚了：

**不是理论完全错。**  
matchability、locability、observability、SplatLoc/ULF-Loc-style landmark saliency 是合理方向。

**也不是只剩一个简单 bug。**  
你已经修掉了很多主路径 bug，测试也在增长。

**真正问题是：当前 score 和训练 proxy 没有直接对齐 PnP inlier / pose success。**  
KC、visibility、geometry、locability prior、residual descriptor 都是 proxy；proxy 在简单场景有用，在 GreatCourt 这种重复结构里会系统性误导。

下一步最该做的是：

```text
1. PnP top-k 只从 valid_target_pixels 取候选
2. 用 train-query TP/FP 训练 calibrated matchability
3. eval 改成 query-time coverage-aware match selection
4. GreatCourt 加 multi-hypothesis PnP + dense verification
5. 保留 STDLoc/PLY descriptor 和 sampled bank 为主干
6. KC / locability / geometry / residual 只作为辅助 feature，不再 strict replacement
```

把这个做完，项目主线就会从“我们学习了一个更好的 Gaussian descriptor field”变成更稳的：

> **我们学习并校准了 Gaussian localization 中的 matchability / locability，使 STDLoc-style sparse-to-dense localization 在部分场景显著提升，并在重复结构场景中通过 query-conditioned filtering 缓解 false-positive landmarks。**

这才是现在最有投稿希望的版本。

---

**Sources:**

- [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
- [From Sparse to Dense: Camera Relocalization with Scene ...](https://zju3dv.github.io/STDLoc/?utm_source=chatgpt.com)
- [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://arxiv.org/html/2409.14067v1?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)