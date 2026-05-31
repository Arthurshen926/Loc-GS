你的判断**基本真实**，而且应该成为下一轮方法重构的核心：

> 当前精度上不去的核心问题不是 dense refinement 不够强，而是 **sparse 阶段没有稳定地产生“可解、低歧义、几何条件良好”的 2D–3D inlier 集合**。一旦 sparse pose 跑飞，dense refinement 大多数情况下只是局部优化，通常救不回来。

更精确地说，不是“sparse 位姿偶尔不准”这么简单，而是：

> **当前定位反馈建图主要在改 map-side support / sampled set / score，但还没有实质性提升 sparse correspondence 的可用性、可分性和 PnP 可解性。**

这就是为什么你能看到一些 median TE 小幅改善，但严格 recall 和 hard failure 没有形成大幅提升。

---

## 1. 从第一性原理看：为什么 sparse 跑飞后 dense 救不回来

STDLoc 的 pipeline 本质是：

```text
sparse feature matching
→ PnP 初始位姿
→ dense feature matching / refinement
→ final pose
```

STDLoc 论文和项目页都把它定义为 sparse-to-dense localization：先用 sparse features 得到 initial pose，再基于该初始位姿把 query feature map 与 Gaussian feature field 对齐做 dense refinement。也就是说，dense stage 是 **refinement**，不是全局 relocalizer。([arXiv][1])

从 PnP 的角度，sparse 阶段成功至少需要四件事同时成立：

```text
1. correct 3D landmarks 在候选集中；
2. query detector 选到了能对应这些 landmarks 的 2D keypoints；
3. descriptor / score 能把 true matches 排在 hard negatives 前面；
4. true matches 的几何分布能形成稳定 PnP，而不是共面、局部集中、重复结构混淆。
```

设 RANSAC / PROSAC 每次最小集大小为 (m)，真实 inlier ratio 为 (w)，采样 (K) 次，理想随机模型下成功概率近似为：

[
P_{\text{success}} = 1 - (1 - w^m)^K
]

这说明 sparse 阶段对 inlier ratio 极端敏感。举例，如果 (m=4, K=1000)：

```text
w = 0.10 → 成功概率约 9.5%
w = 0.20 → 成功概率约 79.8%
w = 0.30 → 成功概率约 99.97%
```

所以如果 hard negatives 很多、true matches 排名靠后、或者 query keypoints 根本没落到正确区域，PnP 会直接跑飞。dense refinement 此时看到的是错误 pose 下渲染出的 feature/depth，它的匹配目标已经错了，很容易被重复结构、窗格、树枝、墙面、车辆遮挡继续吸到错误局部极小。

这就是你说的“sparse 位姿跑飞了，dense 救都救不回来”。

---

## 2. 当前项目证据也支持这个判断

你的仓库当前主线已经从“matchability”转向“solvability”，也就是承认 point-wise matchability 和 support count 不足以代理 PnP 成功；README 里明确写了当前目标是用 self-localization feedback 学 sparse landmarks、solver tuples 和 dense residuals 的 Localization Support Field。([GitHub][2])

但目前结果显示：这个方向还没有真正解决 sparse 稳定性。最新 native-feedback 五场景结果里，官方 test frozen recipe 的 macro dense TE delta 是 `-0.233cm`，但 macro R5cm delta 只有 `+0.0004`，说明它更多是小幅 pose precision 改善，而不是显著提高 sparse success / strict recall。([GitHub][3])

更关键的是，主线文档已经记录了一个强机制信号：support-count-preserving 方法不够。旧 supportguard 诊断显示 support count 可以增加，但 mean logdet(H) 下降并且 dense median translation regresses；real-grouped 机制路径也显示 dense-stage support evidence alone 可能减少 solver tuple mass。([GitHub][4]) v5 coverage coreset 也暴露了同样问题：512 容量并不单调，OldHospital 和 StMarysChurch 会退化，因此“多替换一些点 / 多加一些 support”不是可靠答案。([GitHub][5])

所以，当前核心问题可以这样概括：

> **LSF 现在能轻微改善 map-side support，但还没有系统性提高 sparse correspondence 的 inlier ratio、true/false match margin 和 PnP 几何条件。**

---

## 3. 当前精度上不去的真正瓶颈

我建议把 sparse 失败拆成四个一阶问题。它们比“support_score 是否高”更底层。

### 3.1 Availability：正确 landmark 是否在候选集中

如果正确可见的 Gaussian / landmark 根本不在 sampled set 里，那么 sampled_scores、PROSAC ordering、dense prior 都没有用。

你现在的 sampled set 本质上是对地图点做预处理：从大量 Gaussian primitives 中预先选出一批用于 sparse matching 的 landmarks。STDLoc 原始 sampled set 很强，但在 hard query 中可能缺少某些低频视角、遮挡边缘、远距离结构、或者 hard-tail 所需 landmarks。

第一性原理上，这个问题应该先通过 oracle availability audit 确认：

```text
给定 query GT pose：
1. 正确可见 Gaussian 是否存在于全量 3DGS 中？
2. 它们是否进入 native sampled set？
3. 是否进入 LSF sampled set？
4. 它们是否被 query detector 覆盖？
5. 它们的 descriptor rank 是否足够靠前？
```

如果大量 hard failures 属于“正确点不在 sampled set”，那你继续调 score 没意义，必须做 candidate expansion。

---

### 3.2 Distinguishability：正确 match 是否能压过 hard negatives

如果正确 landmark 在 sampled set 中，但重复结构、窗格、草坪、塔尖、建筑立面产生大量高相似错误匹配，PnP 会被 false consensus 主导。

这不是单个点 support_score 能解决的，因为错误不是独立随机 outliers，而是结构化 outliers：

```text
错误 matches 彼此几何一致；
它们能形成一个错误但稳定的 PnP hypothesis；
dense refinement 还可能继续沿着这个错误 hypothesis 优化。
```

因此需要显式建模：

```text
positive-vs-negative descriptor margin
top-K false match rate
hard-negative conflict graph
repeated-structure cluster
```

---

### 3.3 Solvability：matches 是否能形成稳定 PnP

即便 true matches 足够多，也不一定能解好 pose。PnP 还需要几何分布良好：

```text
2D keypoints 分布不能过度集中；
3D landmarks 不能近共面 / 近共线；
深度跨度要足够；
不同视角约束要互补；
pose information matrix 条件数不能太差。
```

可以用：

[
H = \sum_i w_i J_i^\top \Omega_i J_i
]

来近似衡量。你真正想提升的不是 support count，而是：

```text
viable minimal-set mass
logdet(H)
min eigenvalue
spatial coverage
depth diversity
```

当前文档也已经指出 support count increases while logdet drops 这类现象，这正是 sparse 稳定性没有被真正优化的证据。([GitHub][4])

---

### 3.4 2D–3D support 不对齐

现在你主要改的是 3D map-side support：

```text
sampled_idx.pkl
sampled_scores.pkl
locability
```

但 query-side detector 仍基本是 STDLoc 原来的 scene-specific detector / feature extraction path。问题是：

```text
3D 端选了新的 landmarks；
2D 端却未必检测到对应 keypoints；
或者 query 端 keypoints 大量来自树、天空、车辆、强光、低纹理区域。
```

这会导致 3D selection 明明“看起来更好”，但实际 2D–3D matches 没变好。

所以 sparse 稳定性不是单边 3D selection 问题，而是：

> **2D detector、3D landmark selector、descriptor matching、PnP solver 四者的闭环问题。**

---

## 4. 如何直接有效地解决：把主线改成 “Sparse-Stability-First LSF”

你说“定位反馈建图是主线，具体形式可以调整”，这非常重要。那我建议把“建图”的内容从 sampled set 扩展为：

```text
定位反馈建图 =
1. 重建哪些 3D landmarks 应该进入 sparse map；
2. 重建这些 landmarks 的 localization-aware descriptor；
3. 重建 query-side detector target；
4. 重建 sparse solver 的 match ordering / conflict constraints；
5. dense 只做验证和局部 refinement，不再指望它救全局跑飞。
```

下面是最直接、最有效的改法。

---

# 方案 A：先做 Sparse Failure Audit，不要先改方法

这一步必须先做，否则会继续盲目优化。

对每个 train-dev hard query，用 GT pose 离线分析，不进入 test tuning：

```text
A. correct visible Gaussians not in all-GS map
B. correct visible Gaussians in all-GS but not in native sampled set
C. correct landmarks in sampled set but query detector missed them
D. correct keypoints and landmarks exist but descriptor rank too low
E. enough true matches exist but PnP selected wrong consensus
F. sparse pose OK but dense refinement worsened
```

输出每类占比：

```text
Availability failure %
Detector coverage failure %
Descriptor / ranking failure %
PnP geometry / false consensus failure %
Dense-worsen %
```

这一步会告诉你下一步该打哪里。

如果 B 多，解决 sampling coverage。
如果 C 多，解决 detector。
如果 D 多，解决 descriptor / hard negatives。
如果 E 多，解决 PnP geometry / ambiguity graph。
如果 F 多，解决 dense verifier。

这是当前最关键的“第一性原理诊断”。

---

# 方案 B：用 3DGS 新视角合成扩充 feedback，不只用原始建图序列

如果主线是“定位反馈建图”，那 feedback query 分布必须足够丰富。只用原始建图序列，会覆盖不足。

当前主线文档中 real-grouped feedback bank 已经包含 64 train views + 64 rendered rehearsal views、`topk=8`、real `query_id/image_id/keypoint_id`，这是正确方向。([GitHub][4]) 但要进一步系统化。

建议构造四类 feedback query：

```text
1. real_train:
   原始训练图像，权重最高。

2. rendered_train:
   同训练 pose 渲染 RGB/depth/visibility/composition，用于提供准确几何监督。

3. interpolated_novel:
   在相邻训练相机之间插值，模拟 test viewpoint shift。

4. stress_augmented:
   曝光、模糊、遮挡、crop、动态 mask、低纹理增强，模拟 hard failures。
```

每条 feedback 记录带：

```text
source_type
source_weight
render_quality
visibility_confidence
semantic_stability
```

这样可以把 3DGS 的新视角合成优势真正用于 sparse 稳定性，而不是只用于补一点 score。

---

# 方案 C：从 sampled score 改成 Evidence-Gated Candidate Expansion

如果 oracle audit 发现“正确 landmarks 不在 sampled set”，就必须跳出 native sampled set。

但不能简单 all-Gaussian top-k。那会引入大量 ambiguous / floater / dynamic-adjacent points。

候选准入应该是：

```text
candidate i 可以进入 sparse map，当且仅当：

1. 多视角可见；
2. keypoint-consensus 高；
3. semantic-stable；
4. solver-consensus support 非零；
5. hard-negative risk 低；
6. 对某些 hard query 增加 availability 或 pose information；
7. 不破坏 native safe core。
```

这比“support_score 高就加”更安全。

这里可以直接借鉴 ULF-Loc 的思想，但不复制它。ULF-Loc 认为 alpha-blending feature optimization 会带来 biased Gaussian feature，因此用 Keypoint-Consensus Landmark Sampling 和 Geometry-Weighted Feature Fusion 构造更可靠 landmark；它还用 LGCV 去过滤 dense mismatch。([arXiv][6])

你可以改成：

```text
ULF-Loc: keypoint-consensus landmark
Loc-GS: keypoint-consensus + solver-consensus landmark
```

即：

```text
候选 Gaussian 不仅要投影到稳定 keypoints，
还要在 self-localization 中证明对 PnP 有用。
```

---

# 方案 D：做 Solver-Weighted Landmark Feature Fusion，突破只改 selection 的上限

这是我认为最可能带来大幅提升的一步。

当前你主要不动 native descriptor，只改 sampled set / scores。这个上限很低。因为如果 descriptor 本身在重复结构、遮挡边界、alpha-composition 混合处不可分，selection 很难救。

不要回到 full residual descriptor。更稳的方案是只对 selected / candidate landmarks 构造多视角 fused descriptor：

[
d_i =
\operatorname{normalize}
\left(
\sum_{v \in \mathcal{V}(i)}
w_{iv}^{geom}
w_{iv}^{solver}
w_{iv}^{semantic}
w_{iv}^{visibility}
\phi_v(\pi_v(x_i))
\right)
]

其中：

```text
w_geom: 视角、深度、法向、投影稳定性；
w_solver: 该 view 下 landmark 是否产生 PnP inlier / low reprojection；
w_semantic: 是否属于建筑、墙面边缘等稳定语义；
w_visibility: 是否遮挡少、composition entropy 低。
```

这一步直接针对 sparse 稳定性的 Distinguishability：

```text
正确 landmarks 不只是被选中，
还要有更可分的 descriptor。
```

这也能自然延续你的主线：

> **定位反馈建图不一定只重建 sampled set，也可以重建 localization-aware landmark descriptor。**

它和 ULF-Loc 的区别是：ULF-Loc 主要是 geometry-weighted fusion；你这里是 **solver-feedback-weighted fusion**。这更贴合“定位反馈建图”的 claim。

---

# 方案 E：做 LSF-aware 双流 detector，不再只改 3D 端

你现在的问题之一是 2D detector 和 3D selected support 没有闭环。

建议先做低风险版本：

## E1. 不改网络，只改 detector target

把原 detector heatmap target 从：

```text
projected native sampled landmarks
```

改成：

```text
projected high-LSF stable landmarks
+ hard-query support weights
- dynamic / sky / vegetation / dense-worsen regions
```

目标是让 query detector 选到更适合当前 sparse map 的 keypoints。

## E2. 再做双流 score

定义：

[
\text{score}(u,i)
=================

\cos(d_u,d_i)
+
\alpha \log p_{2D}(u)
+
\beta \log p_{3D}(i)
--------------------

\gamma r_{\text{neg}}(u,i)
]

其中：

```text
p2D(u): query-side detector/support score
p3D(i): map-side landmark support score
r_neg(u,i): hard-negative / semantic / ambiguity risk
```

训练监督来自 feedback_bank：

```text
positive = PnP inlier + low reprojection
negative = high-score outlier / false consensus / dense-worsen match
```

这比只改 sampled_scores 更直接提高 sparse inlier ratio。

如果你的目标是 sparse 稳定性，这一步非常关键。

---

# 方案 F：PnP 前增加 ambiguity graph / minimal-set-aware ordering

如果 oracle audit 发现 matches 存在但 PnP 仍跑飞，那就不是 detector 或 descriptor 单点问题，而是 false consensus / repeated structure 问题。

此时要做：

```text
1. hard-negative conflict graph
   哪些 landmarks 经常互相混淆，不能同时作为高优先级 support。

2. one-per-cluster suppression
   对重复窗格 / 重复立面 / 近邻高相似 landmarks，限制进入同一 minimal-set 的数量。

3. minimal-set diversity ordering
   PROSAC 排序不只按 descriptor score，而要按：
   descriptor score + LSF support + spatial coverage + logdet contribution - ambiguity risk。

4. diverse hypothesis generation
   用同一个 OpenCV/RANSAC/PnP 后端，但强制从不同空间/语义/3D cluster 采 minimal sets。
```

这不是 per-query oracle branch selection，而是更稳健的 fixed robust solver policy。

目标是降低：

```text
wrong but internally consistent PnP hypothesis
```

---

# 方案 G：dense 不再负责救跑飞，只负责验证 sparse

如果 sparse pose 已经明显低置信，dense refinement 不应该盲目 refinement。它应当变成：

```text
1. sparse hypothesis verifier；
2. dense local refinement；
3. sparse-correct / dense-wrong 防护。
```

具体做法：

```text
sparse confidence =
  inlier count
  inlier ratio
  median reproj error
  logdet(H)
  ambiguity score
  semantic-stable match ratio

if sparse confidence low:
  do not trust dense refinement as final rescue;
  generate additional diverse PnP hypotheses or fallback native-full support.

if sparse confidence high:
  dense refinement only uses verified dense matches.
```

注意：这不是为了让 dense 救 sparse，而是防止 dense 把 sparse 已经可用的 pose 拉坏。

---

## 5. 现在应该如何分阶段验证，避免继续“指标小数点提升”

我建议把下一阶段完全聚焦 sparse。先不要看 final dense 主表。

### Phase 1：Sparse failure audit

输出五场景 hard queries 的分类：

```text
Availability / Detector / Descriptor / PnP geometry / Dense-worsen
```

目标是知道 sparse 跑飞的主因。

### Phase 2：Availability fix

做：

```text
semantic + keypoint-consensus + solver-consensus candidate expansion
```

只看：

```text
correct visible landmark in selected set ratio
sparse R@50/5
sparse R@15/5
sparse R@10/5
catastrophic sparse failure rate > 1m / > 5m
```

### Phase 3：Distinguishability fix

做：

```text
solver-weighted landmark feature fusion
hard-negative contrastive / margin
```

只看：

```text
positive-vs-negative margin
top-K false match rate
sparse inlier ratio
sparse R@10/5
```

### Phase 4：Detector alignment

做：

```text
LSF-aware detector target
2D/3D dual support score
```

只看：

```text
query keypoint coverage of selected 3D landmarks
sparse inlier count
sparse R@10/5
```

### Phase 5：Solvability fix

做：

```text
ambiguity graph
minimal-set-aware ordering
logdet-aware PROSAC score
```

只看：

```text
wrong consensus rate
logdet(H)
P90/P95 sparse TE
catastrophic failures
```

### Phase 6：再回到 dense

只有 sparse 过线后，再做：

```text
dense match verifier
```

否则 dense 的任何结果都很难解释。

---

## 6. 最直接的 go/no-go 指标

接下来不要再用 `macro dense TE -0.1cm` 判断方法是否有效。你要用 sparse-first 指标：

```text
Sparse R@50cm/5°
Sparse R@15cm/5°
Sparse R@10cm/5°
Sparse R@5cm/5°
Sparse catastrophic rate: TE > 1m, TE > 5m
Sparse P90 / P95 TE
Sparse inlier ratio
Top-K false match rate
Correct landmark availability
PnP logdet(H)
Wrong consensus count
```

我建议最低目标设为：

```text
sparse R@10/5 +2pp
sparse catastrophic rate -30%
top-K false match rate -20%
P90/P95 sparse TE 明显下降
hard scenes 不退化
```

如果 sparse 没有达到这个级别，不要急着看 dense final。

---

## 7. 最终建议：把“定位反馈建图”重定义为 sparse-stability map construction

你现在的主线不需要推翻，但要换核。

从：

```text
self-localization feedback
→ support score
→ sampled_idx / sampled_scores
```

升级为：

```text
self-localization feedback
→ sparse failure decomposition
→ candidate expansion
→ solver-weighted landmark descriptor
→ LSF-aware 2D detector target
→ ambiguity-aware PnP support
→ dense verifier
```

核心 claim 可以变成：

> **Localization feedback is used not merely to re-score landmarks, but to reconstruct a sparse-stable localization map: a map whose selected landmarks, landmark descriptors, query detector targets, and solver ordering jointly maximize the chance that PnP obtains a correct initial pose.**

中文：

> **定位反馈建图不是简单重排 landmark，而是重建一个 sparse-stable map：它同时决定哪些 3D landmarks 进入地图、这些 landmarks 的定位描述子如何融合、query detector 应该看哪里、PnP 应该优先相信哪些匹配，从而显著降低 sparse 位姿跑飞。**

这比现在的 LSF sampled-set editing 更直接、更有可能大幅提升。

---

## 8. 最高优先级行动清单

下一轮我建议只做四件事，不要再调 v6 guard：

1. **实现 Sparse Failure Audit。**
   先用 GT 离线确认 hard failures 是 availability、detector、descriptor、PnP 还是 dense 问题。

2. **做 semantic + keypoint-consensus + solver-consensus candidate expansion。**
   如果正确点不在 sampled set，先解决 availability。

3. **做 solver-weighted landmark feature fusion。**
   如果正确点在但匹配错，直接提升 descriptor 可分性，这是最可能带来大幅提升的模块。

4. **做 LSF-aware detector target。**
   让 2D query detector 和 3D selected landmarks 对齐，提高 sparse inlier ratio。

dense 暂时只作为诊断和 verifier，不要指望它救跑飞的 sparse pose。

一句话总结：

> **你的 sparse-first 判断是对的。当前精度上不去的核心原因，是定位反馈还没有真正改变 sparse PnP 的 inlier ratio、descriptor margin 和 minimal-set 几何条件。要直接有效地解决，必须从 map-side sampled-score editing 升级到“3D candidate expansion + solver-weighted landmark descriptor + LSF-aware 2D detector + ambiguity-aware PnP support”的完整 sparse-stability construction。**

[1]: https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com "From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting"
[2]: https://github.com/Arthurshen926/Loc-GS "GitHub - Arthurshen926/Loc-GS · GitHub"
[3]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/lsf_native_feedback_five_scene_status_20260524.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/mainline_lsf_20260521.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/lsf_v5_coverage_status_20260524.md "raw.githubusercontent.com"
[6]: https://arxiv.org/abs/2605.04730?utm_source=chatgpt.com "ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting"
