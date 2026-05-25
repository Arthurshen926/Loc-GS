可以做到更优雅，但前提是你要**停止把本项目解释成“STDLoc 上叠加了很多反馈模块”**，而是像 STDLoc、ULF-Loc 一样，用一个清晰的缺陷、一条核心哲学、三个紧密模块来组织。

2026-05-21 Round 5 状态更新：本文是主线重构建议，不是最终投稿结果清单。仓库入口已经改为 LSF-Loc；real-grouped `feedback_bank_v2` 已生成并通过审计，LSF v2 native proposal 已产生同预算 map edit，并在 ShopFacade train-dev q20/q80/q160 上得到单路径 OpenCV STDLoc dense 正向验证。q160 sparse 中位误差改善但 strict sparse recall 小幅下降；OldHospital q80 dense 回退、StMarysChurch translation/recall 未改善。因此当前 claim 必须限定为“ShopFacade train-dev sparse-to-dense 正向 + hard-scene failure boundary”，不能写成 Cambridge test 或 SOTA claim。旧的 selector、residual、quality-gate、SceneMatchNet、LoFTR 和 oracle 叙事均应按 diagnostic / ablation / rejected 处理。

我建议把本项目重新命名和重构为：

> **LSF-Loc: Solvability-Guided Localization Support Fields for 3D Gaussian Relocalization**
> 中文：**面向可解性的 3DGS 定位支持场**

核心一句话：

> **STDLoc 解决了“如何从 Feature Gaussian 做 pose-prior-free sparse-to-dense localization”；ULF-Loc 解决了“α-blending 学特征会产生 biased landmark feature”；LSF-Loc 要解决的是“3DGS 定位中的 landmark / residual selection 不应由 matchability 或 support count 决定，而应由 PnP 可解性、hard-query 风险和 dense residual 可靠性决定”。**

---

## 1. 先看 STDLoc 和 ULF-Loc 为什么显得“优雅”

STDLoc 的优雅在于它不是零散改模块，而是提出了一个完整范式：**Feature Gaussian scene representation → matching-oriented Gaussian sampling → scene-specific detector → sparse PnP initial pose → dense feature matching refinement**。它的论文明确把 Feature Gaussian、matching-oriented sampling、scene-specific detector 和 sparse-to-dense localization 串成一个 pose-prior-free pipeline；sampling 的动机也很清楚：全量 Gaussians 匹配太慢，且大量 ambiguous Gaussians 会伤害匹配，因此要选出多视角可见、空间均匀、可匹配的 landmarks。([arXiv][1])

ULF-Loc 的优雅在于它更进一步：它先指出一个根本问题——通过 3DGS α-blending 优化高维 feature field 会让单个 Gaussian 的 feature 与邻近 Gaussians 纠缠，从而造成 3D point feature bias；然后它不再修修补补 Feature Gaussian descriptor，而是绕开这个机制，用 **Keypoint-Consensus Landmark Sampling、Geometry-Weighted Feature Fusion、Local Geometric Consistency Verification** 三个模块构成新 pipeline。([arXiv][2])

也就是说，这两篇工作的共同写法是：

```text
不是：我们加了一个模块，让指标涨了一点。

而是：
1. 找到当前范式中的一个结构性错误假设；
2. 给出一个更合理的中间表示；
3. 用三个模块把这个表示贯穿 sparse 和 dense localization；
4. 用 accuracy、recall、ablation、效率和 failure case 证明它。
```

你的项目要变得优雅，也必须按这个范式重写。

---

## 2. 本项目现在最应该抓住的结构性错误假设

你现在不要再主打：

> “我们重建了更好的 feature Gaussian descriptor。”

因为你自己的实验已经显示：**native STDLoc descriptor + solver/sampling support 方向比 residual descriptor 主叙事更稳**。仓库当前应把 clean mainline 写成 native STDLoc descriptor/backend → audited `feedback_bank_v2` self-localization traces → solver-consensus support → solvability-aware sampling / dense support → single-path OpenCV PROSAC/RANSAC PnP → STDLoc-style dense refinement。旧的 selector-native ablation 只能说明 selector/sampling-field 有局部信号，不能支撑强 SOTA claim。([GitHub][3])

真正应该抓住的是这个假设：

> **现有 3DGS localization 方法把 landmark selection 当成 point-wise matchability ranking，但最终 pose 成败是 set-level solver problem。**

更直白地说：

```text
一个 Gaussian 看起来 matchable，不代表它对 PnP 有用；
support count 没掉，不代表 pose 可解性没掉；
dense rendered match 置信度高，不代表它不会把 pose refinement 拉坏。
```

你最新的负结果已经很好地支持了这个方向：supportguardall-ret90 虽然 query support loss 为 0，但 Stage A 仍失败，macro dense median translation 退化约 0.513 cm；文档也总结为“support count can be non-decreasing while pose accuracy degrades”。([GitHub][4])

这就是 LSF-Loc 的核心问题：

> **3DGS 定位中的特征选择，应该从 matchability-driven selection 转为 solvability-driven support selection。**

---

## 3. 像 STDLoc / ULF-Loc 一样，给本项目一个三模块结构

我建议最终方法只保留三个模块。

---

# Module 1：Solver-Consensus Support Estimation

对应 ULF-Loc 的 **Keypoint-Consensus Landmark Sampling**，但你不要做 keypoint consensus，而要做：

> **Solver-Consensus Landmark Scoring**

ULF-Loc 的 K.C. Sampling 判断一个 Gaussian 是否可靠，是看它在多视角下是否稳定投影到 2D keypoints 附近。([arXiv][2])
LSF-Loc 应该判断一个 Gaussian 是否可靠，则看它在 self-localization episodes 中是否稳定参与 **successful PnP / low-error dense refinement / hard-query recovery**。

每个 Gaussian / landmark 不再只有一个 selector score，而是有一个 support vector：

```text
s_i = {
  inlier_consensus,
  hard_query_support,
  pose_information_gain,
  ambiguity_risk,
  hard_negative_risk,
  dense_residual_risk,
  visibility_stability
}
```

这个模块的输入是 self-localization feedback bank：

```text
train / rendered / perturbed / shifted query
→ sparse matches
→ PnP inliers
→ reprojection errors
→ dense improved / dense worsened
→ query cluster / hard-query label
```

输出是：

```text
per-Gaussian localization support field
```

注意，这里不要再叫 “locability score” 就结束。更准确的叫法是：

> **Solver-observed support field**

因为它来自下游几何求解器的实际行为，而不是来自 feature reconstruction loss。

---

# Module 2：Solvability-Aware Landmark Sampling

对应 STDLoc 的 **Matching-Oriented Sampling**，但目标函数要升级。

STDLoc 的 sampling 是从 millions of Gaussians 中选出少量 landmarks，原则是可匹配、多视角可见、空间分布均匀。([arXiv][1])
LSF-Loc 的 sampling 则应该是：

> **选出让 PnP 更容易成功、更稳定、更少 hard-tail failure 的 landmarks。**

也就是说，你不再优化：

[
\sum_i \text{matchability}(i)
]

而是优化：

[
U_q(S)
======

\lambda_1 V_q(S)
+
\lambda_2 \log \det(H_q(S)+\epsilon I)
--------------------------------------

## \lambda_3 A_q(S)

\lambda_4 N_q(S)
]

其中：

```text
S: selected landmark set
V_q(S): viable PnP minimal-set coverage
H_q(S): pose information matrix
A_q(S): ambiguity / redundancy risk
N_q(S): hard-negative risk
```

对整个查询分布，再加 hard-query 风险：

[
S^*
===

\arg\max_{|S|=B}
\mathbb{E}*{q}[U_q(S)]
+
\beta \cdot \mathrm{CVaR}*{\alpha}(U_q(S))
]

这一步就是你项目最需要的理论核心。

为什么它比 support guard 更强？

因为 support guard 只保证：

```text
这个 query 还有多少 positive matches
```

而 solvability-aware sampling 保证：

```text
这些 matches 是否能组成几何条件良好的 PnP minimal sets；
是否覆盖 hard-query tail；
是否减少 descriptor / geometry ambiguity；
是否真的改善 pose solver 的条件数。
```

你仓库里 `solver_centric_lsf_plan_20260518.md` 已经把这个方向写出来了：当前问题被定义为 3DGS localization feature selection 应该优化 PnP pose solvability、hard-query tail risk 和 dense residual reliability，而不是 unary matchability 或 support-count preservation。([GitHub][5])

这个模块的输出是 STDLoc-compatible payload：

```text
sampled_idx.pkl
sampled_scores.pkl
locability
support_audit.json
manifest.json
```

这非常重要：**你不改 PnP，不改 evaluator，不替换 query detector，不做 branch selection。你只重建一个更合理的 landmark support set。**

---

# Module 3：Support-Consistent Sparse-to-Dense Verification

对应 ULF-Loc 的 **LGCV**，但你可以做得更贴近本项目。

ULF-Loc 的 LGCV 用局部几何一致性过滤 rendered-to-query dense matching 中由 blur、artifacts 引起的错误匹配。([arXiv][2])
LSF-Loc 的 dense 部分可以定义为：

> **Support-Consistent Dense Residual Selection**

也就是：dense refinement 时不是所有 rendered features / pixels / matches 都可信。应该只保留那些背后由 reliable Gaussians 支持、局部几何一致、pose leverage 合理、composition ambiguity 低的 dense residuals。

稀疏阶段 selection 的对象是：

```text
Gaussian landmarks
```

稠密阶段 selection 的对象则是：

```text
rendered rays / pixels / dense correspondences / residuals
```

Dense support mask 可以写成：

[
m(u)
====

R_{\text{LSF}}(u)
\cdot
D(u)
\cdot
(1-H(u))
\cdot
G(u)
]

其中：

```text
R_LSF(u): contributing Gaussians 的 support score 聚合
D(u): alpha-composition dominance
H(u): composition entropy / mixture ambiguity
G(u): pose leverage / local geometric consistency
```

这个模块输出：

```text
dense match filter
dense residual weight
dense refinement reliability mask
```

它的目标不是替换 STDLoc dense refinement，而是减少：

```text
sparse pose 已经对了，但 dense refinement 被 rendered artifact / ambiguous feature 拉坏
```

这正好和 ULF-Loc 对 rendering artifact mismatch 的分析形成对话。

---

## 4. 这样整理后，本项目和 STDLoc / ULF-Loc 的关系很清楚

可以放在论文 introduction 里：

```text
STDLoc:
  Learns Feature Gaussian and builds a sparse-to-dense localization pipeline.
  Its selection is matching-oriented.

ULF-Loc:
  Shows alpha-blending feature learning creates biased 3D features.
  It avoids feature-field optimization and constructs unbiased landmark features.

LSF-Loc:
  Keeps native descriptors and geometric backend.
  It argues that reliable localization requires solver-aware support selection.
  It learns which landmarks and dense residuals make PnP and refinement solvable.
```

更凝练地说：

| 方法          | 发现的问题                                                     | 中间表示                       | 核心模块                                                                                          |
| ----------- | --------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| STDLoc      | 3DGS 可以作为 pose-prior-free localization map，但全量匹配不可行       | Feature Gaussian           | matching-oriented sampling + scene detector + sparse-to-dense                                 |
| ULF-Loc     | α-blending feature optimization 让 Gaussian feature biased | Unbiased landmark feature  | K.C. sampling + GWFF + LGCV                                                                   |
| **LSF-Loc** | matchability / support count 不能保证 PnP 可解性                 | Localization Support Field | solver-consensus support + solvability-aware sampling + support-consistent dense verification |

这张表就是你项目应该追求的“优雅”。

---

## 5. 本项目的强 claim 应该改成什么

不要再 claim：

> 我们重建了更好的 Feature Gaussian descriptor。

也不要 claim：

> 我们调出了一个更好的 selector score。

应该 claim：

> **3DGS localization feature selection is a solver-level support selection problem. We show that point-wise matchability and support-count preservation are insufficient, and propose a Localization Support Field that selects sparse landmarks and dense residuals according to PnP solvability, hard-query tail risk, ambiguity, and refinement reliability.**

中文：

> **3DGS 定位中的特征选择本质上不是单点可匹配性排序，而是面向几何求解器的支持集选择问题。我们证明 support count preservation 仍可能导致定位退化，并提出 Localization Support Field，在 sparse PnP 和 dense refinement 两个阶段选择真正可解、稳定、低歧义的 landmarks 和 residuals。**

这个 claim 比“涨点”更强，因为它有理论对象。

---

## 6. 本项目的论文结构可以直接模仿 STDLoc / ULF-Loc

建议论文按下面写。

---

### 1. Observation：Feature selection mismatch

先展示一个反例：

```text
Native STDLoc
Support-preserving candidate
Solver-aware candidate
```

画图说明：

```text
support count 不变或增加
但 pose median / R5 下降
同时 logdet(H) 下降 / ambiguity 上升 / dense worsened count 上升
```

这张图是你的“ULF-Loc alpha-blending bias analysis”对应物。

ULF-Loc 用理论分析说明 α-blending feature learning 有 inherent bias；你要用理论 + 实验反例说明：

> **support count 和 point-wise matchability 对 PnP solvability 是 biased proxy。**

---

### 2. Theory：From matchability to solvability

给出两个命题即可，不要过度数学化。

**命题 1：Point-wise matchability 不保证 set-level pose solvability。**

即使每个 selected landmark 的 matchability 都高，如果它们空间聚集、共面、重复结构强，PnP 的 (H) 也可能病态。

**命题 2：Support count preservation 不保证 PnP success。**

两个 landmark set 可以有相同 support count，但 viable minimal-set coverage 和 (\log\det(H)) 完全不同。

这和你当前负结果高度一致。文档已经记录 support count 不下降但 pose 退化，而且 solver diagnostic 初步显示 supportguardall-ret90 在 hard scenes 增加 cached support / tuple mass 的同时降低 mean logdet(H)。([GitHub][5])

---

### 3. Method：LSF-Loc

三节：

```text
3.1 Solver-Consensus Support Estimation
3.2 Solvability-Aware Landmark Sampling
3.3 Support-Consistent Sparse-to-Dense Verification
```

每节一个核心公式、一个算法框、一个输出。

这就像 STDLoc 的：

```text
Feature Gaussian Training
Matching-Oriented Sampling
Scene-Specific Detector
Sparse-to-Dense Localization
```

也像 ULF-Loc 的：

```text
Feature Bias Analysis
Keypoint-Consensus Sampling
Geometry-Weighted Feature Fusion
Local Geometric Consistency Verification
```

---

## 7. 实验也要像 ULF-Loc 一样完整

ULF-Loc 不只报 median pose，它还报 Cambridge recall、训练时间、显存和 FPS；例如 Cambridge 附录报告 `[50cm/5°]`、`[15cm/5°]`、`[10cm/5°]` recall，且报告了训练时间/显存和速度对比。([arXiv][2])

你的实验应该分成五张表。

---

### Table 1：Full localization accuracy

```text
STDLoc sparse
STDLoc dense
LSF-Loc sparse
LSF-Loc sparse+dense
```

指标：

```text
median t/r
R@50cm/5°
R@15cm/5°
R@10cm/5°
R@5cm/5°
R@2cm/2°
```

这样可以同时对齐 STDLoc 和 ULF-Loc 的 reporting style。

---

### Table 2：Sparse selection ablation

```text
STDLoc matching-oriented sampling
random / FPS / opacity / visibility
unary selector
support-count guard
solver-consensus support
solver-aware sampling
solver-aware + CVaR
solver-aware + ambiguity penalty
```

指标不要只放 pose，还要放：

```text
inlier ratio
viable tuple mass
mean / worst logdet(H)
ambiguity risk
hard-query failure count
```

这张表证明你的理论对象是有效的。

---

### Table 3：Dense support ablation

```text
STDLoc dense no mask
composition dominance mask
local geometric consistency mask
LSF support mask
LSF + geometry consistency mask
```

指标：

```text
dense improved query count
dense worsened query count
dense valid match ratio
final R5/R2
dense latency
```

这张表让你的方法覆盖 dense rendering localization，而不只是 sparse sampling。

---

### Table 4：Efficiency

分 offline 和 online。

Offline：

```text
base map training time
feedback bank generation time
LSF training / estimation time
export time
peak GPU memory
map size
```

Online：

```text
feature extraction
sparse matching
PnP / PROSAC
rendering
dense matching
dense refinement
total latency / FPS
```

不要硬 claim realtime。更稳的 claim 是：

> **accuracy-efficiency Pareto improvement**

---

### Table 5：Failure analysis

选 OldHospital、StMarysChurch 这种 hard cases：

```text
Native succeeds / candidate fails
Supportguard fails
LSF recovers
```

展示：

```text
support count
logdet(H)
minimal-set geometry
ambiguity risk
dense residual mask
pose error
```

这张表会比单纯 R5 +0.3pp 更能打动审稿人。

---

## 8. 代码上要做减法，方法上要做“换核”

为了像 ULF-Loc 一样优雅，你现在必须做减法。

### 继续保留

```text
native STDLoc descriptor
single-path PROSAC PnP
STDLoc dense refinement
feedback bank infrastructure
selector/resampling/export infrastructure
manifest / split audit
solver diagnostic
```

### 降级为 appendix / diagnostic

```text
residual descriptor
SceneMatchNet
quality gate
LoFTR replacement
qcov/churn/retention scalar sweeps
raw dense locability prior
inlier-only early exit
```

### 必须补齐

```text
feedback_bank_v2 with real query_id
solver-consensus support estimation
logdet(H) / viable tuple / ambiguity metrics
solver-aware admissible replacement
dense support mask
ULF-style recall/time/memory/FPS reporting
```

早期 `solver_centric_lsf_plan` 承认旧 audited pair cache 缺真实 per-row `query_id`，所以旧 tuple bank 只能用 synthetic chunks。2026-05-21 之后，real-grouped `feedback_bank_v2` artifact 已经开始补上 `query_id/image_id/keypoint_id` 和 dense outcome；下一步不是再证明 support-count 失败，而是把这些 v2 traces 接入 solver-consensus support、CVaR hard-query aggregation 和 solver-admissible replacement。([GitHub][5])

---

## 9. 一个更“CVPR 风格”的最终方法图

你可以这样画方法图：

```text
Training / Mapping stage

3DGS + native STDLoc descriptors
          │
          ▼
Self-localization episodes
(rendered / perturbed / shifted train views)
          │
          ▼
Solver traces
(matches, inliers, reprojection, logdet(H), dense success/failure)
          │
          ▼
Localization Support Field
(per-Gaussian support, ambiguity, hard-query utility, dense reliability)
          │
          ├── Sparse support sampling
          │       → sampled_idx.pkl / sampled_scores.pkl / PROSAC ordering
          │
          └── Dense support mask
                  → reliable rendered rays / dense correspondences


Inference stage

Query image
  → native STDLoc detector / descriptor
  → LSF-sampled landmarks
  → descriptor matching
  → PROSAC PnP
  → LSF-filtered dense refinement
  → final pose
```

这张图的逻辑非常清楚：
**不是特征重建，而是定位支持重建。**

---

## 10. 最简洁的论文故事版本

如果你要像 STDLoc / ULF-Loc 那样写 introduction，可以这样：

> 3DGS-based localization methods have recently shown strong performance by combining explicit Gaussian maps with feature matching. STDLoc demonstrates that Feature Gaussian can support a sparse-to-dense, pose-prior-free localization pipeline. ULF-Loc further reveals that α-blending-based feature-field optimization introduces biased 3D landmark features and proposes unbiased feature construction.
> However, an orthogonal problem remains underexplored: **which Gaussians or rendered residuals should be trusted by the geometric solver?** Existing sampling strategies rely on point-wise matchability, keypoint consensus, or support count, but PnP localization is a set-level problem whose success depends on minimal-set geometry, pose information, ambiguity, and hard-query tail risk.
> We propose LSF-Loc, a solver-centric localization support field for 3DGS relocalization. LSF-Loc does not replace the native descriptor or the geometric backend. Instead, it uses self-localization episodes to estimate per-landmark solver support and dense residual reliability, then exports a STDLoc-compatible landmark set and support-guided dense refinement mask. Current diagnostics show that support-count-preserving edits can still degrade localization; the next paper-facing result must show that a fixed solvability-aware support field improves hard-query robustness and accuracy-efficiency trade-offs.

这就是你要追求的 “ULF-Loc 式优雅”。

---

## 11. 最后给一个非常明确的方向判断

你现在不要再追求：

```text
更复杂的 residual descriptor
更复杂的 branch selection
更多 qcov / hn / ret 超参
```

而要追求一个更干净的三段式方法：

```text
1. Solver-Consensus Support Estimation
   用 self-localization traces 学哪些 landmarks 真正服务 PnP。

2. Solvability-Aware Landmark Sampling
   用 viable minimal-set coverage、logdet(H)、CVaR hard-query risk 和 ambiguity
   重新生成 STDLoc-compatible sampled_idx / sampled_scores。

3. Support-Consistent Dense Verification
   用 support field 过滤 rendered dense matches / residuals，
   避免 dense refinement 被 artifact 或 ambiguous regions 拉坏。
```

这样，本项目就不再是“STDLoc 的一个复杂增强版”，而是一篇有自己核心命题的工作：

> **From Matchability to Solvability: feature selection for 3DGS localization should be optimized for the geometric solver, not for point-wise feature similarity.**

[1]: https://arxiv.org/html/2503.19358v1 "From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting"
[2]: https://arxiv.org/html/2605.04730v1 "ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting"
[3]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/README.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/negative_results_solver_selection_20260518.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/solver_centric_lsf_plan_20260518.md "raw.githubusercontent.com"
