# Branch · SP-GS

**User:** Anonymous  
**Created:** 5/10/2026 13:58:36  
**Updated:** 5/12/2026 20:41:00  
**Exported:** 5/12/2026 20:53:34  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370)  

## Prompt:
2026/5/12 19:22:38

又继续推进了几轮（git仓库已更新），但感觉进展缓慢且精度好像还不如baseline，是技术路线有问题还是我的实现有bug？主线claim似乎得不到实验数据支撑，如何进行调整？帮我整个梳理一下，架构该如何调整，下一步计划是什么？之前推进的几轮总结如下：
------
继续推进完了，这轮有实质进展，但我先把结论说直：**还不能诚实宣称已经超过当前 SOTA**，尤其 KingsCollege 仍是短板；不过现在终于有一条比“多候选 pose trick”更干净、也比静态删点更有效的主线。

**这次改了什么**
- 在 [hybrid_localizer.py](/root/Loc-GS/loc_gs/localization/hybrid_localizer.py) 增加 `opencv_prosac` PnP：按当前 query 的 match/reliability score 排序，用 OpenCV PROSAC 采样，而不是跑 N 条 pose 分支。
- 在 [eval_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py) 把 sparse/dense match scores 传入 PnP。
- 在 [launch_cambridge_reliability_eval.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_reliability_eval.py) 增加三个固定 recipe：
  - `covisibility_select`：硬选择 ablation
  - `covisibility_soft_select`：保守静态选择
  - `covisibility_prosac`：当前最推荐主线，**不删 landmark，只做单路径 PROSAC priority**
- 新增调研/路线文档：[loc_gs_related_work_and_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_related_work_and_mainline_20260512.md)

**full test split 结果**
对比 `protected`，`covisibility_prosac` 在 5 个 Cambridge 场景 full split 上：

| Scene | Protected Dense | PROSAC Dense |
|---|---:|---:|
| GreatCourt | 25.286cm / 0.097deg, R@5 8.95% | **16.698cm / 0.056deg, R@5 12.50%** |
| KingsCollege | 23.640cm / 0.321deg, R@5 0.58% | **23.338cm / 0.306deg**, R@5 0.29% |
| OldHospital | 22.529cm / 0.332deg, R@5 11.54% | **13.633cm / 0.177deg, R@5 16.48%** |
| ShopFacade | 3.690cm / 0.137deg, R@5 66.99% | **3.245cm / 0.128deg, R@5 72.82%** |
| StMarysChurch | 8.018cm / 0.180deg, R@5 26.23% | **6.169cm / 0.125deg, R@5 36.42%** |

Aggregate:
- Dense mean median: **16.633cm → 12.617cm**
- Dense mean R@5: **22.86% → 27.70%**
- Sparse mean median: **96.140cm → 15.147cm**

这说明之前的 matchability / visibility / observability 不是没用，问题是**硬删 landmark 会伤覆盖率**；把它们改成 query-conditioned PnP priority 后，故事顺了很多。

**投稿角度判断**
当前最稳的 claim 应改成：**localization-feedback reliability priority for Gaussian feature-field localization**。它支持“定位反馈影响特征场使用方式/可靠性选择”，但还不够完整支持“训练期定位反馈直接学出更好的 Gaussian descriptor”。下一步应把 PROSAC 这套 query-conditioned hard-negative 信号前移到训练期 fine-tuning，而不是继续堆静态 selector。

调研参考主要看了 STDLoc、GLACE、SplatLoc、GSplatLoc/GSFeatLoc、MultiLoc 这些方向：  
STDLoc: https://arxiv.org/abs/2503.19358  
GLACE: https://arxiv.org/abs/2406.04340  
SplatLoc: https://arxiv.org/abs/2409.14067  
GSplatLoc: https://arxiv.org/abs/2409.16502  
GSFeatLoc: https://arxiv.org/abs/2504.20379  
MultiLoc: https://arxiv.org/abs/2603.27170

验证：
- `pytest tests -q`：**137 passed, 16 warnings**
- 全量 `pytest -q` 仍会被 `third_party/Mask2Former` 的 `detectron2` 缺失卡住，这不是本轮改动引入的。
----------
已继续推进并完成一轮 DIM/LoFTR 接入和实验验证。结论比较明确：`deep-image-matching` 这条线适合做外部 matcher ablation / 辅助 verifier，目前不适合作为主线替代 SuperPoint/STDLoc feature-field dense matcher。DIM repo 本身支持 LoFTR、RoMa、LightGlue、ALIKED、DISK 等前端/匹配器：<https://github.com/3DOM-FBK/deep-image-matching>。

本轮改动：
- 新增 [dim_image_matcher.py](/root/Loc-GS/loc_gs/localization/dim_image_matcher.py)，封装 Kornia LoFTR 的 query RGB ↔ rendered RGB 匹配。
- 在 [eval_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py) 加入 `dense_matcher=loftr_rendered`：渲染 RGB+depth，LoFTR 匹配，rendered 像素反投影到 3D，再走同一套 PnP/PROSAC。
- 在 [train_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/train_cambridge_hybrid.py) 给 `render_hybrid_superpoint()` 增加按需 RGB 渲染，避免非 LoFTR 路径变慢。
- 在 [launch_cambridge_reliability_eval.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_reliability_eval.py) 加入 `covisibility_prosac_loftr` recipe。
- 在 [loc_gs_related_work_and_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_related_work_and_mainline_20260512.md) 记录 DIM/LoFTR 结论和实验表。

实验结果，q20 Cambridge 三场景：
```text
ShopFacade:  PROSAC 2.265cm / 0.084deg R@5 95.0%
             LoFTR  3.670cm / 0.128deg R@5 70.0%

OldHospital: PROSAC 4.575cm / 0.077deg R@5 55.0%
             LoFTR  6.432cm / 0.092deg R@5 45.0%

GreatCourt:  PROSAC 23.188cm / 0.066deg R@5 0.0%
             LoFTR  21.252cm / 0.108deg R@5 5.0%
```

也试了把当前主线 dense refinement 从 2 轮加到 3 轮：Shop/GreatCourt 小幅改善，但 OldHospital 回退，平均不优，所以没有改主线默认值。

验证：
```text
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests -q
142 passed, 16 warnings
```

当前建议：主线继续保持 `covisibility_prosac`，LoFTR 保留为 DIM ablation。下一步真正可能涨 SOTA 的方向不是替换 matcher，而是把 LoFTR/RoMa 这类外部匹配作为训练期 hard-negative / matchability 标签来源，反馈进特征选择和 reliability 场。
--------
已完成下一步推进，并且跑了 full-test 验证。结论比较关键：q80 的 query-like calibration 正信号没有泛化到 full test，所以这条线不能直接作为主方法，只能作为负结果/消融；投稿主线仍应保持 baseline-preserving。

本轮新增/修改：

- 加入 query-like self-localization calibration：rendered RGB → frozen SuperPoint/STDLoc query extractor → GT rendered pose 标注 TP/FP。
- 新增 3 GPU calibration launcher：[launch_cambridge_matchability_calibration.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_matchability_calibration.py:15)
- eval launcher 支持 `--calibrated_matchability_template`，可批量挂载每场景 reliability cache。
- 新增三分支 manifest：`docs/cambridge_branch_manifest_querylike_20260512.json`
- 结果与判断已写入：[loc_gs_related_work_and_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_related_work_and_mainline_20260512.md:221)

full-test 三分支结果：

| Branch | Macro Median | R@10 | R@5 | R@2 |
|---|---:|---:|---:|---:|
| STDLoc baseline | 9.13cm / 0.156° | 0.576 | 0.371 | 0.133 |
| historical learned | 10.10cm / 0.158° | 0.568 | 0.394 | 0.144 |
| query-like calibrated prior | 12.56cm / 0.159° | 0.487 | 0.282 | 0.090 |

clean guarded selector 加入 query-like 分支后仍选择：

- GreatCourt: baseline
- KingsCollege: baseline
- OldHospital: baseline
- ShopFacade: historical selected
- StMarysChurch: baseline

clean guarded full-test 保持：`9.08cm / 0.153°`，macro `R@5=0.373`，`R@2=0.133`。

这说明目前已经达到“投稿逻辑更自洽”的状态：可以讲清楚 localization feedback、query-like calibration、static reliability 的失败边界，以及为什么需要 guarded baseline-preserving。但还没达到“稳定 SOTA”的水平。下一步真正该做的是 query-conditioned reliability，或把这些反馈学习进 feature field/locability head，而不是继续加强静态 landmark prior。

验证：`pytest tests -q` 通过，`161 passed, 16 warnings`。3 张 3090 当前都空闲。

## Response:
2026/5/12 19:22:41

我会先核对最新主线代码和文档，再围绕“结果为何支撑不了原 claim、是路线还是实现问题、架构如何收敛、下一步最小实验闭环”给出可执行判断。

## Response:
2026/5/12 19:22:45

我已经看到新分歧：PROSAC 是有效信号，但主要相对较弱 protected 分支成立，必须重新对齐最强 STDLoc baseline 后再判断主方法价值。

## Response:
2026/5/12 19:29:21

结论先说直：**现在主要不是一个实现 bug，而是原先“定位引导重建会自然学出更强 descriptor / 更强 landmark bank”的强路线没有被实验支持。** 代码里确实修过很多关键 bug，但现在 `161 passed`、PROSAC 路径、query-like calibration、LoFTR ablation 都跑过后，继续不如 baseline 的主要原因已经不是“小修一下就会翻盘”的工程问题，而是**训练信号、使用方式和最终定位目标没有完全对齐**。

现在应该把主线从：

> 学一个更强的 Gaussian descriptor / learned landmark bank 去替代 STDLoc

调整为：

> **保留 STDLoc/PLY 特征主干，用定位反馈学习 query-conditioned reliability，并只通过 PROSAC priority / confidence gating 影响 2D-3D PnP。**

这不是退缩，是把项目从“结果支撑不了的强 claim”收敛到“实验支撑得住、还能继续提升”的方向。

---

## 1. 现在不是单点 bug，而是 claim 过强

仓库 README 仍然把 Loc-GS 主线写成 SuperPoint teacher features → per-Gaussian latent feature field → novel-view descriptor/detector maps → geometry-aware matching and PnP localization，并包含 differentiable matching、reprojection proxy、observability、per-Gaussian locability 等定位引导 loss。这个设计方向没有问题。([GitHub](https://github.com/Arthurshen926/Loc-GS))

但你最新文档已经把 paper-facing position 改得很准确：**defensible mainline 不是 learned descriptor replacement，而是 matchability-guided fine-tuning and selection**；核心 claim 应该是构建稳定的 SuperPoint/STDLoc-compatible feature field，然后通过 GT/perturbed/interpolated/rendered poses 做 localization rehearsal，把 self-matching / self-localization feedback 转成 detector saliency、visibility、observability、geometry quality、distinctiveness、calibrated matchability 等 reliability signals，并保守使用这些信号。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

所以判断是：

**技术路线不是完全错；强 claim 错了。**

更具体地说：

- **错的 claim**：Loc-GS 可以直接学出比 STDLoc/PLY 更强的 descriptor 或 landmark bank。
- **对的 claim**：定位反馈信号能帮助判断哪些匹配/landmark 更可靠，并能通过 PROSAC priority 或 guarded selection 改善部分场景。
- **还没证明的 claim**：这些信号已经能稳定、全场景、SOTA 级提升 Cambridge localization。

---

## 2. 你现在的结果其实说明了一个很重要的事实

你最新的 `covisibility_prosac` 结果相对 `protected` 是显著进步：

| Scene | Protected Dense | PROSAC Dense |
|---|---:|---:|
| GreatCourt | 25.286cm / R5 8.95% | 16.698cm / R5 12.50% |
| KingsCollege | 23.640cm / R5 0.58% | 23.338cm / R5 0.29% |
| OldHospital | 22.529cm / R5 11.54% | 13.633cm / R5 16.48% |
| ShopFacade | 3.690cm / R5 66.99% | 3.245cm / R5 72.82% |
| StMarysChurch | 8.018cm / R5 26.23% | 6.169cm / R5 36.42% |

这说明：**matchability / visibility / observability 不是没用。**  
之前硬删 landmark 伤覆盖率；改成“不删 landmark，只用 score 排序，让 OpenCV PROSAC 优先采样高可靠匹配”之后，信号终于变得合理。仓库代码里也能看到：`opencv_prosac` 会按 `match_scores` 降序重排 2D-3D correspondences，并使用 OpenCV `SAMPLING_PROSAC`。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/localization/hybrid_localizer.py))

但要注意一个关键点：**这个表是相对 `protected` 进步，不是相对 local STDLoc baseline 进步。** 你的文档里 full-test 后也明确写了：PROSAC 相对 protected 把 dense mean median 从 16.633cm 提到 12.617cm，R@5 从 22.86% 到 27.70%，但这还不能作为最终 SOTA claim。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

而当前 paper-clean guarded selector 是 `9.078cm / 0.153°`，只比 local STDLoc reproduction `9.127cm / 0.156°` 略好，而且文档里也承认还没有超过 ULF-Loc 报告的 `8.3cm / 0.13°`。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

所以目前状态是：

> **PROSAC priority 是正确方向，但还没有把主线结果推到足够强。**

---

## 3. 为什么 LoFTR / DIM 没救回来？

LoFTR-rendered 这条线的结论也很清楚：它适合做 ablation 或 verifier，不适合作为主 matcher。你的文档里 q20 结果显示 ShopFacade 和 OldHospital 都明显不如当前 PROSAC，GreatCourt 只在 median 上略好但角度更差。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

这不是意外。`deep-image-matching` 本身确实支持 LoFTR、RoMa、LightGlue、SuperGlue、DISK、ALIKED 等很多 matcher。([GitHub](https://github.com/3DOM-FBK/deep-image-matching)) 但这些 matcher 解决的是 **image-to-image matching**，而你的主系统需要的是：

> query image feature ↔ 3D Gaussian / PLY landmark descriptor ↔ PnP。

LoFTR-rendered 需要先有一个 rendered RGB/depth，再从 rendered pixel 反投影到 3D。它更像 pose verification 或 refinement，不是稳定的 no-pose-prior initial localization 主干。它能作为 auxiliary verifier，但直接替换 SuperPoint/STDLoc feature-field matcher，实验已经不支持。

---

## 4. 为什么 query-like calibration q80 有信号，full test 反而崩？

这点非常关键。

你做了 query-like self-localization：rendered RGB → frozen SuperPoint/STDLoc query extractor → 用已知 rendered pose 标注 TP/FP。这个思想是对的，而且 sidecar 里 pose delta / overlap 也说明 rendered views 不再只是小扰动。文档里给出的平均 pose delta 很大，比如 GreatCourt 17.61m / 29.27°，KingsCollege 33.33m / 40.32°。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

但是 full-test 表明，query-like calibrated prior 没有泛化：

| Full-test branch | Macro median | R@10 | R@5 | R@2 |
|---|---:|---:|---:|---:|
| STDLoc baseline | 9.13cm / 0.156° | 0.576 | 0.371 | 0.133 |
| Historical learned | 10.10cm / 0.158° | 0.568 | 0.394 | 0.144 |
| Query-like calibrated prior | 12.56cm / 0.159° | 0.487 | 0.282 | 0.090 |

文档也已经写得很准确：这说明 static per-landmark reliability prior，即使用更广的 rendered views 生成，仍然不足以处理 full-scene query-conditioned ambiguity。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

根因是：

> **GreatCourt / Kings 这种重复结构里的错误不是“某个 landmark 永远坏”，而是“某个 query 下某个 landmark 是高危 false positive”。**

所以 per-landmark static prior 会失效。你需要的是 **query-conditioned pair reliability**，不是 landmark-level reliability。

---

## 5. 当前架构该怎么调整？

建议你把架构彻底收敛成下面四层。

---

### Layer 1：冻结强主干，不再替代 descriptor

主干保留：

```text
query extractor: STDLoc/SuperPoint original feature
3D descriptor: STDLoc/PLY loc_feature
landmark bank: STDLoc sampled bank
localization: sparse-to-dense PnP + dense refinement
```

这是底座，不要再让 learned descriptor、teacher fusion、KC strict、LoFTR-rendered 默认接管。

原因很简单：STDLoc 已经是强 baseline，它的论文核心就是 Feature Gaussian + matching-oriented sampling + scene-specific detector + sparse-to-dense relocalization。([arXiv](https://arxiv.org/abs/2503.19358)) 你的实验也反复证明：**一旦 learned descriptor 或 hard selector 大幅干预这个底座，多数场景会退化。**

---

### Layer 2：Loc-GS 只学习 reliability，不学习 replacement

Loc-GS feature field 继续保留，但角色改掉：

```text
不输出“替代 PLY 的 descriptor”
而输出：
  locability
  visibility prior
  observability prior
  geometry consistency
  distinctiveness / ambiguity
  query-conditioned matchability score
```

其中最重要的是最后一个：**query-conditioned matchability score**。

形式上，不要再只学习：

$$
s_j = P(\text{landmark } j \text{ reliable})
$$

而要学习：

$$
s_{ij} = P((q_i, X_j) \text{ is a PnP-useful inlier})
$$

输入是 pair-level feature：

```text
descriptor similarity
top1-top2 margin
query detector score
landmark detector / locability
visibility
observability
geometry quality
landmark static TP/FP prior
2D cell / 3D voxel coverage
local descriptor ambiguity
co-visibility group
```

输出是 `p_inlier(q_i, X_j)`。

---

### Layer 3：推理只做 PROSAC priority，不做 hard deletion

这是你最近最有价值的发现。

主推理流程应该固定为：

```text
1. 取 STDLoc/PLY landmark bank
2. query feature 与 3D landmark descriptor 做标准匹配
3. 对每个 candidate match 计算 reliability score
4. score = descriptor_similarity + β * logit(p_inlier)
5. 按 score 排序
6. 用 OpenCV PROSAC / PROSAC-MAGSAC 采样
7. dense refinement 保持原流程
```

不要删 landmark，不要 coverage hard filtering，不要 MNN 默认，不要 margin 默认。

因为你已经验证：硬过滤会伤覆盖率；PROSAC 排序保留冗余，才真正符合 RANSAC 的机制。

---

### Layer 4：guarded fallback 作为安全网，不作为核心贡献

当前 train-calibrated guarded selector 是 paper-safe，但它的提升太小：`9.078cm / 0.153°` vs local STDLoc `9.127cm / 0.156°`。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

所以它可以保留为：

```text
safety mechanism / conservative deployment recipe
```

但不要把它写成主贡献。真正的主贡献应是：

> query-conditioned localization-feedback priority for 2D-3D Gaussian feature localization.

---

## 6. 下一步不要再做什么？

下面这些方向我建议停止作为主线：

### 不要继续做 hard landmark selection

`covisibility_select`、KC strict、static score top-k 这类已经充分证明不稳。文档里也写了 hard recipe 只能做 ablation，不适合默认。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

### 不要继续做 hard descriptor fusion

teacher fusion q80 看起来有信号，但 GreatCourt full 没泛化。文档里也已经总结：hard descriptor fusion 不是 mainline，只能作为“feature fusion destabilizes STDLoc descriptor distribution”的负结果。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

### 不要继续把 LoFTR/RoMa 当替代 matcher

LoFTR-rendered 可以作为 verifier 或 hard-negative source，但不是默认 matcher。你自己的 q20 结果已经支持这个结论。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

### 不要再把 q20/q80 结果当方向性胜利

q80 多次给出假阳性，full test 崩。以后 q20/q80 只能用于 smoke test，不能指导主 claim。

---

## 7. 接下来真正该做的：Query-conditioned Reliability Head

这是现在最重要的一步。

### 7.1 数据生成

用 train split，不碰 test。

对每个 scene：

1. 取真实 train images；
2. 取 rendered-perturbed images 作为 augmentation；
3. 用 frozen STDLoc/SuperPoint query extractor 提取 query keypoints/features；
4. 用 STDLoc/PLY landmark bank 做 top-K candidate matching；
5. 用 GT pose 标注每个 pair：

$$
y_{ij}=1[
\|\pi(T_{gt}X_j)-u_i\| < \epsilon
\land depth/alpha/visibility consistent
]
$$

6. 保存 `(query_id, keypoint_id, landmark_id, feature vector, label)`。

关键：**label 必须是 pair-level，不是 landmark-level。**

---

### 7.2 训练目标

先不要训练 descriptor field，只训练一个很小的 reliability head：

$$
p_{ij} = f_\theta(\phi(q_i, X_j))
$$

loss 用二分类 + hard-negative ranking：

$$
L_{bce} = BCE(p_{ij}, y_{ij})
$$

$$
L_{rank} = \max(0, m - s(q_i, X^+) + s(q_i, X^-))
$$

Hard negative 定义：

```text
descriptor similarity high
但 GT reprojection error large
```

这正是 GreatCourt 的病根。

---

### 7.3 推理使用

不要过滤，只排序：

```text
final_score = descriptor_similarity + beta * logit(p_inlier)
```

然后：

```text
OpenCV PROSAC PnP
```

`beta` 用 calibration split 固定，不能 per-scene 手调。可以试：

```text
beta ∈ {0.0, 0.1, 0.25, 0.5}
```

最终选一个全局 beta，或者用 calibration-val 自动选择但必须在论文里说明。

---

## 8. 必须加一个 Oracle 实验，否则你不知道还有没有上限

现在最需要的不是再跑一个 recipe，而是判断“reliability priority 这条路还有多少理论空间”。

加四个排序版本：

| Ordering | 含义 |
|---|---|
| random | sanity check |
| descriptor similarity | 当前 baseline |
| current hand-crafted reliability | 当前 PROSAC |
| oracle reprojection error | 按 GT reprojection error 排序，只用于分析 |

如果 **oracle PROSAC** 都不能明显超过 baseline，说明问题不是 reliability score，而是 landmark bank / descriptor / dense refinement 上限不够。  
如果 oracle PROSAC 明显超过 baseline，而 learned reliability 不行，说明还有学习空间。  
如果 current hand-crafted reliability 比 descriptor similarity 稍好但远低于 oracle，说明下一步 query-conditioned head 是正确方向。

这个实验非常重要，建议马上做。

---

## 9. 必须统一 baseline，否则现在容易误判

你现在同时有：

```text
STDLoc baseline
protected
historical learned
covisibility_prosac
query-like prior
guarded selector
LoFTR-rendered
```

但这些不是同一层级。下一版报告要重排成：

| Method | 是否改变 descriptor | 是否删 landmark | 是否使用 learned reliability | 是否 branch selection | 作用 |
|---|---:|---:|---:|---:|---|
| STDLoc reproduced | 否 | 否 | 否 | 否 | baseline |
| protected | 否 | 否/弱 | 否 | 否 | internal conservative baseline |
| historical learned | 轻微 | 否/弱 | 是 | 否 | learned branch |
| covisibility_prosac | 否 | 否 | 是，排序 | 否 | current main candidate |
| query-like static prior | 否 | 否 | static prior | 否 | negative ablation |
| guarded selector | 不定 | 不定 | 间接 | 是 | safety mechanism |
| LoFTR-rendered | 换 matcher | 否 | 否 | 否 | external matcher ablation |

主表必须只比较：

```text
STDLoc reproduced
covisibility_prosac
query-conditioned reliability PROSAC
guarded selector
```

其余放 ablation。

---

## 10. 现在的论文 claim 应该如何调整？

不要再写：

> Loc-GS outperforms SOTA Cambridge localization.

也不要写：

> Localization-guided reconstruction learns a better Gaussian descriptor.

建议改成：

> **Loc-GS studies localization-feedback reliability for Gaussian feature-field localization. Instead of replacing the STDLoc/PLY descriptor backbone, it converts self-localization feedback into match reliability signals and uses them to prioritize single-path 2D-3D PnP solving.**

中文可以写：

> **本文研究如何将定位反馈转化为高斯特征场中的可匹配性可靠度，并在不破坏 STDLoc/PLY 描述子主干的前提下，用该可靠度引导 2D-3D PnP 采样。**

这个 claim 能被你当前最强结果支持：

- static hard selection 失败；
- learned descriptor replacement 失败；
- PROSAC priority 有实质改善；
- guarded selector 能保证不按 test 手选；
- query-like static prior 作为负结果说明“query-conditioned”必要性。

---

## 11. 投稿状态判断

### 如果按强会 full paper 标准

**暂时不够。**

原因：

- 没有稳定超过 STDLoc / 新 SOTA；
- clean guarded selector 提升太小；
- PROSAC 相对 protected 强，但相对 STDLoc baseline 还不够；
- query-like calibration full test 退化；
- KingsCollege 仍是短板。

### 如果按 workshop / short paper / negative-ablation-rich paper

**可以形成一个自洽故事。**

题目可以偏研究问题：

> **When Does Localization Feedback Help Gaussian Feature Localization?**

或者：

> **Loc-GS: Reliability-Guided Gaussian Feature Fields for 2D-3D Camera Relocalization**

主贡献不是 SOTA，而是：

1. 系统研究 learned descriptor replacement、static landmark saliency、query-like calibration、external matcher、PROSAC priority；
2. 证明 hard selection / descriptor fusion / static prior 会不稳；
3. 发现并验证 query-conditioned PROSAC priority 是目前最合理的使用方式；
4. 提供一套 baseline-preserving evaluation recipe。

---

## 12. 下一步计划：按优先级执行

### P0：做 Oracle PROSAC 上限实验

每个 scene 跑：

```text
descriptor-order PROSAC
handcrafted-reliability PROSAC
oracle-reprojection PROSAC
random-order PROSAC
```

输出：

```text
sparse median / R5
dense median / R5
inlier ratio
2D/3D coverage
```

目的：判断是否还有 headroom。

---

### P1：实现 query-conditioned reliability head

不要再做 static prior。

输入 top-K candidate pair feature，输出 `p_inlier(q, landmark)`。

训练数据来自：

```text
real train images
held-out train calibration images
rendered RGB auxiliary views
```

但 label 是 pair-level reprojection/depth/visibility label。

---

### P2：把 learned reliability 接入 PROSAC 排序

推理时：

```text
score = descriptor_similarity + beta * logit(p_inlier)
```

只排序，不过滤。

固定全局 beta，或者用 train-calibration split 选择。

---

### P3：只在 reliability head 成功后，再考虑 feature-field fine-tuning

如果 P1/P2 没有效，不要微调 feature field。  
如果有效，再做：

```text
freeze descriptor base
train locability/matchability adapter
trust-region regularization
no full descriptor replacement
```

目标不是改变 descriptor，而是让 feature field 输出更好的 reliability signals。

---

### P4：重写论文结构

论文结构建议：

1. Motivation：feature reconstruction ≠ localization reliability。
2. Baseline：STDLoc/PLY backbone。
3. Self-localization rehearsal：从 GT/perturbed/rendered poses 生成 matchability feedback。
4. Reliability signals：visibility、observability、geometry、distinctiveness、calibrated matchability。
5. Inference：single-path PROSAC priority，不删点、不多分支。
6. Negative ablations：hard selection、descriptor fusion、static query-like prior、LoFTR-rendered。
7. Main result：query-conditioned reliability PROSAC 或 guarded selector。

---

## 13. 最后判断

你现在遇到的不是“实现有 bug 所以主 claim 不成立”。更准确是：

> **前几轮实现 bug 修掉后，实验暴露出原主 claim 本身太强：定位反馈不能简单转化为更好的 descriptor 或静态 landmark prior。它真正有效的方式是 query-conditioned reliability，用来排序 PnP 采样，而不是替换特征、删 landmark 或覆盖 descriptor bank。**

架构应该调整为：

```text
STDLoc/PLY descriptor backbone
        ↓
Loc-GS feature/reliability field
        ↓
self-localization episodes generate pair-level labels
        ↓
query-conditioned reliability head
        ↓
PROSAC priority PnP
        ↓
dense refinement
```

下一步最关键不是再跑一个新 recipe，而是做 **oracle headroom + query-conditioned reliability head**。  
如果 oracle 排序有大幅上限，这条路还有明显精度空间；如果 oracle 排序也不强，那就说明 Cambridge 上的瓶颈已经不在 matchability priority，而在 descriptor/landmark representation 或 dense refinement，需要换更强 3D descriptor backbone。

---

**Sources:**

- [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
- [[2503.19358] From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/abs/2503.19358)

## Prompt:
2026/5/12 20:10:44

hybrid feature field 和scene-specific detector架构是否需要调整？这个好像有一直没验证过。以及不要有branch selection这种肯定被审稿人喷的操作，肯定是需要一套方法适应整个数据集的。我觉得项目目的就是让匹配环节、pnp算位姿环节反过来影响重建过程，不管是微调地图特征还是detector都行，然后还要影响一些指标来做直接用于query定位时的特征选择，并且这套方法最后的指标要么位姿精度超越原始STDLoc论文精度，要么recall超越原始STDLoc论文精度

## Response:
2026/5/12 20:10:46

我会按“统一方法、无分支选择、定位反馈反传到重建/检测器”的投稿约束重新审视这条线。重点核对 hybrid feature field 与 scene-specific detector 是否需改，并制定对齐 STDLoc 原论文指标的实验闭环。

## Response:
2026/5/12 20:11:08

我已核对最新主线：目前明确正向的是 PROSAC 排序，而 detector 更像 SuperPoint 重建头，尚未被 PnP 内点/成功率监督成真正 scene-specific detector。

## Response:
2026/5/12 20:17:47

你这个判断是对的：**branch selection 不能作为主方法**。它最多是诊断工具，论文主线必须是一套固定方法、固定超参数、跑完整 Cambridge 五个场景。否则评审会认为是按场景挑结果。

现在我建议把项目主线明确收敛成：

> **用 matching / PnP 的反馈训练 scene-specific detector、locability / matchability reliability，并通过 PROSAC priority 影响 query 定位；不要替换 STDLoc/PLY descriptor，不做 per-scene branch selection，不做 hard landmark deletion。**

更直接地说：**现在最该验证和调整的是 detector 与 reliability architecture，而不是继续调 hybrid descriptor。**

---

## 1. hybrid feature field 和 scene-specific detector 确实需要重新验证

我认为这里有一个关键缺口：**你目前一直在验证 descriptor / landmark score / PROSAC / calibration，但没有系统验证“scene-specific detector 是否真的被 Loc-GS 改好”。**

STDLoc 里 scene-specific detector 不是可有可无的模块。它明确认为 SuperPoint 这种 scene-agnostic detector 不一定适合直接匹配 Feature Gaussian landmarks，因此训练一个场景特定 detector，从 dense feature map 里预测哪些 2D features 更可能对应 sampled landmarks；推理时再 NMS 取 keypoints。STDLoc 论文也把 matching-oriented sampling + scene-specific detector 作为 sparse 初始定位的核心。([arXiv](https://arxiv.org/html/2503.19358v1))

更重要的是，STDLoc 的 ablation 表显示 scene-specific detector 对 recall 有实质贡献：在 Cambridge ablation 中，`RS+M.O.+S.S.D.` 相比 `RS+M.O.+SuperPoint` 的 5m/10° 和 2m/5° recall 都进一步提升。([arXiv](https://arxiv.org/html/2503.19358v1))

而你当前 Loc-GS 的 `SuperPointOutputHead` 是一个 **descriptor + detector reconstruction head**，它从 fused feature map 输出 256-d descriptor 和 65-channel SuperPoint detector logits；这更像“重建 SuperPoint detector”，不是 STDLoc 意义上的“scene-specific detector”。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/models/hybrid_gaussian.py))

这点非常关键：

> **重建 SuperPoint detector ≠ 学到适合当前 3D landmark / PnP 的 scene-specific detector。**

所以，如果项目目标是让匹配和 PnP 反过来影响重建过程，那么 detector 是最应该被改造的地方。

---

## 2. 当前 hybrid feature field 的问题不是容量不够，而是职责混乱

当前 `HybridFeatureGaussian` 是 per-Gaussian latent + spatial hash grid 的双路结构：fine path 是 alpha-blended Gaussian latent，coarse path 是 3D position hash grid，两路融合后输出 feature。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/models/hybrid_gaussian.py)) 代码里也有 per-Gaussian `locability_logit`，可以输出 locability。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/models/hybrid_gaussian.py))

这个结构本身可以继续保留，但现在它承担了太多目标：

1. 重建 SuperPoint / STDLoc descriptor；
2. 重建 detector；
3. 学 locability；
4. 参与 descriptor residual；
5. 参与 landmark scoring；
6. 有时还影响 PnP / PROSAC。

实验已经反复说明：**一旦它强行替代 PLY / STDLoc descriptor，结果不稳。**

所以我建议把 hybrid field 的职责改成：

| 分支 | 是否保留 | 作用 |
|---|---:|---|
| Descriptor reconstruction | 保留，但只做 auxiliary / trust region | 保证 feature field 不崩 |
| Descriptor replacement | 不作为主方法 | 只做 ablation |
| Small descriptor residual | 可选，默认极小或关闭 | 只有证明有效后再上 |
| Locability / reliability | 作为主训练目标之一 | 影响 PROSAC priority |
| Scene-specific detector | 必须新增或重构 | 直接决定 query sparse keypoints |
| Pairwise matchability | 新增 | 直接预测 query-landmark match 是否 PnP-useful |

一句话：**hybrid field 不再主打“学更强 descriptor”，而是主打“学哪里可匹配、哪个 query-landmark pair 可靠”。**

---

## 3. 你应该新增一个真正的 Loc-GS scene-specific detector

当前最值得改的架构是这个：

### Loc-GS Scene-Specific Detector，LSD

输入：

```text
query dense feature map
```

输出：

```text
query keypoint heatmap
```

训练目标不是 SuperPoint detector logits，而是：

```text
哪些 query pixels 更可能产生 2D-3D PnP inlier
```

它应该像 STDLoc detector 那样是 scene-specific 的，但监督比 STDLoc 更强。STDLoc 的 detector target 是 sampled landmark center 投影到图像后得到 heatmap；你的 target 应该进一步使用 PnP / matching 反馈。

建议 target 分三层：

### 3.1 基础正样本：projected sampled landmarks

和 STDLoc 一样，把 sampled landmarks 投影到 train image，visible 且 depth consistent 的位置作为 detector 正样本。

### 3.2 强正样本：PnP inlier landmarks

跑一次 self-localization rehearsal：

```text
query train image
→ dense feature extraction
→ 2D-3D matching
→ PnP / PROSAC
→ GT pose 判断 inlier
```

如果某个 keypoint-landmark match 是 PnP inlier，就把对应 query pixel 作为强正样本。

### 3.3 强负样本：high-similarity false positives

对 GreatCourt 这种重复结构，最有价值的是强负样本：

```text
descriptor similarity 很高
但 GT reprojection error 很大
或导致 PnP outlier
```

这些位置应该降低 detector 或 matchability score。

最终 detector target 可以是：

$$
y(u)=
\lambda_1 y_\text{proj}(u)
+
\lambda_2 y_\text{pnp-inlier}(u)
-
\lambda_3 y_\text{hard-neg}(u)
$$

loss 用 focal BCE 更合适：

$$
\mathcal{L}_{det}
=
\mathrm{FocalBCE}(D_\theta(F_q), y)
$$

这比重建 SuperPoint detector 更贴近你的目标。

---

## 4. 新主线架构：一套固定方法，不 branch selection

我建议把最终方法改成下面这个固定架构。

### 4.1 固定 backbone

不要动：

```text
query descriptor: STDLoc / SuperPoint dense feature
3D descriptor: STDLoc / PLY loc_feature
landmark bank: STDLoc sampled landmarks
dense refinement: STDLoc-style dense feature refinement
```

STDLoc 的主流程就是 sparse 2D-3D matching + PnP 得到初始 pose，再渲染 dense feature/depth 做 dense matching refinement。([arXiv](https://arxiv.org/html/2503.19358v1)) 这个骨架已经很强，不要再用 learned descriptor 强行替代。

### 4.2 Loc-GS 学三个东西

Loc-GS 只学：

1. **scene-specific detector**：选 query 2D keypoints；
2. **per-Gaussian locability**：landmark 是否可靠；
3. **query-conditioned matchability**：当前 query keypoint 和当前 3D landmark 是否可能是 PnP inlier。

### 4.3 推理时只影响 PROSAC priority，不删点

你最近最有价值的结果就是 `covisibility_prosac`。它保留完整 STDLoc landmark bank，只用 matchability prior + 当前 query match scores 对 OpenCV PROSAC 采样排序，不做 hard deletion。文档里也明确写了它是当前 preferred direction，并且 full test 相对 protected 明显提升。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

代码里 `solve_pnp_ransac()` 已经支持：如果 solver 是 `opencv_prosac` 或 `opencv_prosac_magsac` 且传入 `match_scores`，就按分数降序排序 correspondences，再用 OpenCV `SAMPLING_PROSAC`。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/localization/hybrid_localizer.py))

这正是正确接口。

最终固定推理流程应该是：

```text
1. query image → dense STDLoc/SuperPoint feature
2. Loc-GS scene-specific detector → query keypoints
3. query keypoints ↔ STDLoc/PLY landmarks matching
4. Loc-GS matchability head 计算每个 match 的 reliability
5. final_score = descriptor_similarity + β * reliability_logit
6. OpenCV PROSAC / MAGSAC PnP, 不删 matches
7. dense feature refinement
```

注意：**没有 branch selection，没有 per-scene recipe，没有 hard landmark deletion。**

---

## 5. 需要改的不是“是否有定位反馈”，而是反馈进入哪里

你现在的目标是：

> 匹配环节、PnP 算位姿环节反过来影响重建过程。

这个目标没错，但反馈不能主要进入 full descriptor replacement。应该进入这三个位置：

| 反馈来源 | 应该影响 | 不应该主要影响 |
|---|---|---|
| matching TP/FP | detector、matchability head | full descriptor replacement |
| PnP inlier / outlier | detector heatmap、locability | 大幅改 PLY descriptor |
| PnP reprojection error | PROSAC priority score | hard deletion |
| dense refine success/failure | reliability calibration | per-scene branch |
| observability | soft score feature | 单独决定 landmark 选择 |

也就是说，真正的 localization-guided reconstruction 应该是：

```text
base feature reconstruction
→ self-localization rehearsal
→ generate PnP inlier / hard negative labels
→ train detector + locability + matchability
→ query-time PROSAC priority
```

而不是：

```text
base reconstruction
→ PnP loss
→ learned descriptor replacement
→ hope matching gets better
```

后一条已经被你的实验基本否定了。

---

## 6. query-conditioned matchability head 是必须的

静态 landmark prior 失败的原因你已经验证了：q80 有信号，full test 崩。文档也写了 query-like calibrated prior full-test 不泛化，说明 per-landmark static reliability 不足以处理 full-scene query-conditioned ambiguity。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_related_work_and_mainline_20260512.md))

所以必须从：

$$
s_j = P(\text{landmark } j \text{ reliable})
$$

变成：

$$
s_{ij}=P((q_i, X_j)\text{ is PnP inlier})
$$

输入特征可以是：

```text
descriptor similarity
top1-top2 margin
query detector score
landmark locability
visibility score
observability score
depth / alpha consistency
landmark TP/FP prior
2D cell / 3D voxel coverage
co-visibility group id / score
```

输出：

```text
matchability_logit
```

训练 label：

$$
y_{ij}=1
$$

当且仅当：

```text
GT reprojection error < threshold
且 depth consistent
且 alpha visible
且该 match 进入 PnP inlier 或几何 inlier set
```

loss：

$$
\mathcal{L}_{match}
=
\mathrm{BCE}(s_{ij}, y_{ij})
+
\lambda
\max(0, m - s_{i+} + s_{i-})
$$

其中 $i-$ 是 high-similarity false positive。

这一步直接对应 GreatCourt 的问题：重复结构里错误 landmark descriptor 很像，所以必须学会压低 high-similarity false positives。

---

## 7. hybrid feature field 要做的 ablation

你说“hybrid feature field 好像一直没验证过”，这个判断也对。现在必须补一个小而硬的 ablation，不要继续混在大 pipeline 里。

建议固定 query detector、固定 PLY descriptor、固定 PROSAC，然后只换 field / reliability 来源：

| Variant | Descriptor used in matching | Reliability source | 目的 |
|---|---|---|---|
| STDLoc | PLY | none | baseline |
| PLY + hand-crafted reliability | PLY | visibility / observability / locability | 看 score 是否有用 |
| PLY + learned locability | PLY | per-Gaussian locability | 看 hybrid field 学 reliability 是否有用 |
| PLY + pair matchability | PLY | query-conditioned head | 主方法候选 |
| Hybrid descriptor | hybrid | same reliability | 验证 descriptor replacement 是否失败 |
| PLY + small residual | PLY + tiny residual | same reliability | 验证微调是否有用 |

这张表会很关键。它能回答：

> 提升来自 descriptor？来自 detector？来自 PROSAC？还是来自 reliability？

目前我的判断是：**最大提升空间在 detector + pair matchability，而不是 hybrid descriptor。**

---

## 8. scene-specific detector 要单独做实验

这是最紧急的验证。

用同一个 STDLoc/PLY landmark bank，比较四个 query keypoint source：

| Detector | 训练方式 | 预期 |
|---|---|---|
| SuperPoint detector | off-the-shelf | baseline |
| STDLoc S.S.D. reproduction | projected landmarks BCE | 强 baseline |
| Loc-GS detector: projected landmarks | same as STDLoc but native | 应接近 STDLoc |
| Loc-GS detector: PnP-inlier supervised | projected + PnP TP/FP | 应超过 STDLoc |

主指标先看 sparse stage：

```text
#matches
RANSAC inlier ratio
sparse median te/re
sparse R@5
dense final R@5
```

如果 **PnP-inlier detector** 不能提升 sparse stage，那说明你说的“PnP 反馈影响 detector”这条路也暂时不成立，需要回到 STDLoc detector。

如果它能提升 sparse stage，但 dense 后退，那就去改 dense refinement。

---

## 9. 你需要一个固定 dataset-wide recipe

你明确说不想要 branch selection，我同意。建议最终方法只允许这些固定参数：

```text
landmarks = 16384
query keypoints = 2048
NMS radius = 4
PnP solver = opencv_prosac_magsac 或 opencv_prosac
dense iterations = 1
detector = Loc-GS PnP-inlier scene-specific detector
descriptor = PLY / STDLoc loc_feature
match score = cosine_similarity + β * matchability_logit
β = fixed globally
no hard landmark deletion
no branch selection
no per-scene MNN / margin / LoFTR switch
```

Scene-specific 训练是允许的，因为 STDLoc 本身就是 scene-specific detector / scene-specific map。**不允许的是 scene-specific hyperparameter tuning。**

---

## 10. 关于“要么位姿精度超 STDLoc，要么 recall 超 STDLoc”

这个目标可以设，但要先把比较对象说清楚。

STDLoc paper 的 Cambridge median table 是：

```text
Court      15.7cm / 0.06°
King's     15.0cm / 0.17°
Hospital   11.9cm / 0.21°
Shop        3.0cm / 0.13°
St. Mary's  4.7cm / 0.14°
Average    10.1cm / 0.14°
```

这是论文公开表格。([arXiv](https://arxiv.org/html/2503.19358v1))

所以如果你走 **precision claim**，最低要求是固定 recipe 达到：

```text
Average translation < 10.1cm
或者 average rotation < 0.14°
最好二者都不差
```

如果你走 **recall claim**，STDLoc paper 在 Cambridge 主表主要报 median cm/deg；它的 ablation 里报的是 5m/10°、2m/5° 这类较宽阈值 recall，不是你现在常用的 5cm/5°、2cm/2°。([arXiv](https://arxiv.org/html/2503.19358v1)) 因此若要 claim recall 超 STDLoc，最好用你本地复现的 STDLoc under same protocol 作为 recall baseline，比如你之前的：

```text
STDLoc local reproduction: R@5cm/5deg ≈ 37.12%
```

然后固定 recipe 必须明显超过它，例如：

```text
R@5cm/5deg ≥ 41% 或 42%
R@2cm/2deg 不下降
```

不要用 q20/q80。必须 full test。

---

## 11. 下一步计划，按优先级

### P0：先做 detector ablation

这是最重要的。

实现：

```text
loc_gs/models/scene_detector.py
```

包含：

```text
SceneSpecificDetectorCNN
输入: dense descriptor map [C,H,W]
输出: heatmap [1,H,W]
```

训练 target：

```text
projected sampled landmarks
+ PnP inlier positives
- hard false positives
```

固定 STDLoc/PLY descriptor 和 sampled bank，只替换 query keypoint detector。

成功标准：

```text
Sparse R@5 提升
RANSAC inlier ratio 提升
Final dense R@5 不下降
```

如果 detector 不涨，后面别急着做 descriptor 微调。

---

### P1：做 oracle PROSAC 上限实验

四个排序：

```text
descriptor similarity
hand-crafted reliability
learned detector score + hand-crafted reliability
oracle GT reprojection error
```

如果 oracle 排序明显强，说明 reliability 学习还有空间。  
如果 oracle 也不强，说明瓶颈不在排序，而在 descriptor / landmark / dense refinement。

---

### P2：训练 query-conditioned matchability head

输入 top-K candidate pair features，输出 inlier probability。

推理：

```text
final_score = cosine_similarity + β * matchability_logit
PROSAC uses final_score
```

只排序，不过滤。

---

### P3：把 PnP feedback 反向用于 detector / matchability，不急着改 descriptor

训练损失：

$$
\mathcal{L}
=
\mathcal{L}_{desc-rec}
+
\lambda_{det}\mathcal{L}_{pnp-det}
+
\lambda_{match}\mathcal{L}_{pair-matchability}
+
\lambda_{loc}\mathcal{L}_{locability}
+
\lambda_{trust}\mathcal{L}_{descriptor-trust}
$$

其中 descriptor trust 必须强：

```text
descriptor 不允许大幅偏离 PLY / base feature
```

---

### P4：最后才试 descriptor residual

只有当 detector + matchability 已经提升，再试：

```text
descriptor = normalize(PLY + α * residual)
α <= 0.03 or 0.05
```

如果 full test 不涨，就彻底把 residual 放 ablation。

---

## 12. 最终架构建议

我建议把项目架构改成这样：

```text
              Train-time self-localization rehearsal
        query train image / rendered pose / GT pose
                       ↓
             2D-3D matching + PnP
                       ↓
        TP / FP / inlier / hard-negative labels
                       ↓
 ┌────────────────────────────────────────────────┐
 │ Loc-GS localization-feedback learning           │
 │                                                │
 │ 1. base Gaussian feature reconstruction         │
 │ 2. scene-specific detector head                 │
 │ 3. per-Gaussian locability head                 │
 │ 4. query-conditioned matchability head          │
 └────────────────────────────────────────────────┘
                       ↓
                 Test-time query
                       ↓
      scene-specific detector selects keypoints
                       ↓
      STDLoc/PLY descriptor matching to landmarks
                       ↓
      matchability score orders PROSAC sampling
                       ↓
              PnP + dense feature refinement
```

这套架构完全符合你的项目目的：

- matching 反馈影响 detector；
- PnP inlier/outlier 反馈影响 detector 和 reliability；
- feature reconstruction 仍然存在，但不强行替代 STDLoc descriptor；
- reliability 直接用于 query 定位；
- fixed recipe，不做 branch selection；
- 最终目标就是在 full Cambridge 上超过 STDLoc median 或 recall。

---

## 13. 最后判断

**hybrid feature field 不一定要大改，但它的角色要改。**  
不要让它继续承担“替代 STDLoc descriptor”的主任务，而是让它提供 localization-aware reliability。

**scene-specific detector 必须重做和验证。**  
当前 SuperPoint detector reconstruction head 不是 STDLoc-style scene-specific detector，也没有充分证明 PnP feedback 能改善 query keypoint selection。这个是当前最值得补的主线缺口。

**下一阶段不要再做 branch selection，也不要继续 hard filtering。**  
固定一套方法：PnP-inlier scene-specific detector + query-conditioned matchability + PROSAC priority + STDLoc/PLY descriptor backbone。

如果这套方法在 full Cambridge 上达不到：

```text
Avg median < 10.1cm / 0.14°
或 R@5cm/5deg 明显高于本地 STDLoc reproduction
```

那就说明主线 claim 需要降级为 “localization-feedback reliability analysis”，而不是 SOTA 方法。但在现在所有尝试里，**detector + pairwise matchability 是最后一条还没有被充分验证、也最符合你原始目标的路线。**

---

**Sources:**

- [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/html/2503.19358v1)
- [Loc-GS/loc_gs/models/hybrid_gaussian.py at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/models/hybrid_gaussian.py)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)