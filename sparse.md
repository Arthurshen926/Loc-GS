你的判断是**真实的**：如果 sparse pose 已经跑飞，dense refinement 大概率救不回来。原因不是经验性的，而是由几何定位的基本结构决定的。

STDLoc 的完整路径本来就是 sparse-to-dense：先用 sampled landmarks 和 scene-specific detector 做 sparse 2D–3D matching，再用 PnP 求初始 pose，最后基于这个初始 pose 做 dense feature matching/refinement。STDLoc 论文也明确说 sparse 阶段由 matching-oriented Gaussian sampling 和 scene-specific detector 支撑，dense 阶段是在 sparse 初始位姿基础上进一步 refine。([arXiv][1]) 你现在仓库的 LSF-Loc 主线也仍然保持这个单一路径：native STDLoc feature extraction/matching → OpenCV PROSAC/RANSAC PnP → STDLoc-style dense refinement。([GitHub][2])

所以，**如果 sparse 初始位姿落到错误 basin，dense 阶段渲染出的 feature/depth 本身就来自错误视角，后续 matching/refinement 只是局部优化，通常不能跨越大范围错误位姿。** 当前主线如果主要想强调“定位反馈建图”，那么具体形式确实应该向 sparse 稳定性让路：反馈到底写进 sampled set、landmark descriptor、2D detector、PnP ordering，都是可以调整的。

---

## 1. 第一性原理：sparse 位姿为什么会跑飞

Sparse 定位的基本链条是：

```text
query image
→ 2D keypoint detection
→ 2D descriptor extraction
→ 与 3D landmarks 匹配
→ 得到 2D–3D correspondences
→ RANSAC / PROSAC / PnP
→ sparse pose
```

PnP 成功至少需要三件事同时成立：

```text
1. 正确 correspondences 存在；
2. 正确 correspondences 在候选中排名足够高；
3. 正确 correspondences 的几何分布足以稳定求解 pose。
```

RANSAC 的成功概率近似是：

[
P_{\text{succ}} = 1 - (1 - w^m)^N
]

其中 (w) 是 inlier ratio，(m) 是最小采样集大小，(N) 是迭代次数。对 PnP 来说，(m) 通常是 3 或 4 量级。这个公式说明：**inlier ratio 从 0.5 掉到 0.2，不是线性变差，而是指数级变差。**

例如按 4 点 minimal set 粗略看：

```text
w = 0.5  → w^4 = 0.0625
w = 0.2  → w^4 = 0.0016
w = 0.1  → w^4 = 0.0001
```

所以只要 hard negatives 增多、正确 matches 排名靠后、或者 selected landmarks 里正确点不足，PROSAC/PnP 就会非常容易跑飞。

PnP 精度还取决于几何条件。可以近似看成：

[
H = \sum_i J_i^\top \Omega_i J_i
]

最终 pose uncertainty 与 (H^{-1}) 相关。即使 inlier 数量不少，如果这些 inliers 都集中在同一墙面、同一深度层、同一重复立面，(H) 仍然病态。换句话说：

```text
support count 多 ≠ PnP 可解性强
inlier 多 ≠ pose 稳定
descriptor 相似 ≠ 几何正确
```

这也解释了你之前 supportguard 的负结果：support count 修掉了，pose 仍然会退化。你的仓库文档也已经记录了这一点：support-count preservation 不是 safety guarantee，当前方法必须保护 hard-query tuple mass、pose-information、spatial coverage、ambiguity / hard-negative risk，而不能只优化 support count。([GitHub][3])

---

## 2. 当前精度上不去的核心问题

我认为核心问题不是 dense，也不是 LSF 分数本身，而是：

> **当前方法还没有真正提高 sparse 阶段“正确 2D–3D correspondence 形成并进入 PnP”的概率。**

更具体地说，当前 LSF 主要做的是：

```text
native STDLoc map
→ feedback_bank_v2
→ per-Gaussian / per-landmark support
→ sampled_idx / sampled_scores / locability 的 small edit
→ 原 STDLoc matching + PnP
```

这个设计很安全，但它只主要作用在 **3D map-side sampled landmarks / scores**。如果 sparse 跑飞来自以下任何一类问题，仅靠 sampled score 小幅调整都不够：

| 问题                            | 现象                                             | 为什么当前 LSF 难以解决                            |
| ----------------------------- | ---------------------------------------------- | ----------------------------------------- |
| 正确 landmarks 根本不在 sampled set | query 可见区域没有被 native/LSF selected landmarks 覆盖 | 改 `sampled_scores` 没用，必须扩展 candidate pool |
| query 端 detector 选错 keypoints | keypoints 落在天空、树、车、强光、低纹理区域                    | 只改 3D sampled set，2D detector 仍然不对齐       |
| descriptor hard negatives 太强  | 重复窗户/立面中错误 landmark 比正确 landmark 更相似           | selection 无法改变 descriptor 可分性             |
| PnP 几何退化                      | 有 matches，但都集中在同一区域/平面                         | unary support 分数无法保证 minimal-set geometry |
| 错误 matches 形成一致错误 pose        | repeated facade 给出一个“假 inlier cluster”         | RANSAC 会把 coherent outlier mode 当成正确模型    |
| sparse 已接近但 dense 拉坏          | dense residual 来自渲染伪影/重复区域                     | dense 需要 verifier，但这是后续；先解决 sparse        |

你给的 hard-failure 图里，大量 GreatCourt / OldHospital / StMarysChurch / KingsCollege 难例都符合这些模式：重复建筑结构、强光、动态遮挡、树/杆、低可见区域。这些不是“局部 score 权重不够好”的问题，而是 **2D–3D correspondence 生成机制本身不够稳**。

---

## 3. 先建立一个最小理论框架：A–D–G

我建议把 sparse 问题压缩成三个核心变量，不要再被几十个 support/risk 指标淹没。

### A. Availability：正确 landmark 是否存在于候选集合

[
A_q(S)
]

问题是：

```text
query q 需要的正确 3D landmarks，是否在 selected set S 里？
```

如果正确 landmarks 不在 sampled set 里，后面的 descriptor、PnP、dense 都无解。

### D. Distinguishability：正确 match 是否能和 hard negatives 区分开

[
D_q(S)
]

问题是：

```text
正确 2D–3D match 的相似度 / 排名是否高于错误 match？
```

如果正确点在 set 里，但重复结构中的错误点更相似，PnP 仍然会跑飞。

### G. Geometry / Solvability：这些 matches 是否能稳定求解 PnP

[
G_q(S)
]

问题是：

```text
正确 matches 能否组成几何条件良好的 minimal sets？
```

如果点都在同一平面、同一立面、同一小区域，PnP 会不稳定。

这三个量分别对应最直接的改法：

```text
Availability 低 → 扩 candidate pool / keypoint-consensus / semantic-stable sampling
Distinguishability 低 → solver-weighted descriptor fusion / hard-negative training
Geometry 低 → tuple-aware sampling / diversity-aware PnP ordering
```

当前 LSF 的问题是：它试图用一个综合 support score 同时解决 A、D、G，但这三个问题本质上不同。要大幅提升 sparse 稳定性，必须分阶段直接打它们。

---

## 4. 先验证问题是否真实：做 Sparse Failure Decomposition

下一步不要先加模块。先对 hard failures 做一个强诊断。

对每个 sparse 跑飞 query，用 GT pose 或 train/self-map pose 做离线分析，回答五个问题：

### Q1：正确可见 Gaussian 是否存在于全量 3DGS 中？

如果不存在，说明是地图/重建问题，LSF sampling 救不了。

### Q2：正确可见 Gaussian 是否在 native STDLoc sampled set 中？

如果不在，说明是 sampling coverage 问题。

### Q3：正确 Gaussian 在 sampled set 中，但 query detector 是否检测到了对应 2D keypoint？

如果没有，说明是 2D detector / semantic stability 问题。

### Q4：正确 2D–3D match 存在，但 descriptor 排名是否被 hard negatives 压过？

如果是，说明是 descriptor distinguishability 问题。

### Q5：正确 matches 存在且排名不差，但 PnP 是否仍失败？

如果是，说明是 minimal-set geometry / outlier cluster / RANSAC ordering 问题。

这一步可以直接决定下一步改哪里。否则继续做 LSF-v7/v8 仍然是在盲调。

建议新增一个固定报告：

```text
output/sparse_failure_decomposition/<scene>/<query_id>.json
```

每个 query 输出：

```text
visible_positive_in_all_gaussians
visible_positive_in_native_sampled
visible_positive_in_lsf_sampled
detector_hit_rate_on_positive_projection
positive_descriptor_rank_percentile
topk_false_positive_count
pnp_inlier_ratio
pnp_logdet_H
wrong_pose_cluster_score
failure_type = availability / detector / descriptor / geometry / outlier_cluster / dense_only
```

最终你需要一张表：

```text
Failure type                  Ratio among sparse failures
Availability miss             xx%
2D detector miss              xx%
Descriptor hard negative      xx%
Geometry degeneracy           xx%
Coherent wrong-pose cluster   xx%
Other                         xx%
```

这张表会告诉你该把主要精力放哪里。

---

## 5. 最直接有效的解决路线：先重构 sparse 阶段

我建议把主线改成：

> **Localization feedback is used to reconstruct a sparse correspondence system, not only a 3D landmark support score.**

也就是从：

```text
反馈 → 3D landmark score
```

升级为：

```text
反馈 → 3D landmark selection
     → 3D landmark descriptor
     → 2D detector target
     → 2D–3D match ranking
     → PnP minimal-set geometry
```

这仍然是“定位反馈建图”，但建的不是单一 field，而是一个 sparse localization support system。

---

# 6. Phase 1：先解决 Availability

## 核心判断

如果 sparse 跑飞 query 的正确 Gaussian 不在 sampled set 中，那当前 LSF 再怎么改 `sampled_scores` 都没用。

STDLoc 的 sampled set 虽然强，但它的目标是 matching-oriented、均匀、可识别。STDLoc 本身也承认全量 Gaussian 匹配很慢，且 ambiguous Gaussians 会伤害 matching，因此需要采样成 scene landmarks。([arXiv][1]) 但 hard query 的正确支持点可能恰好不在 native sampled set 中，尤其是遮挡、强光、重复结构下，默认 sampled landmarks 可能不是最稳的那一组。

## 直接改法

建立一个新的 candidate pool，不再只从 native sampled set 附近小改：

```text
candidate_pool =
  native sampled landmarks
  ∪ keypoint-consensus landmarks
  ∪ semantic-stable landmarks
  ∪ solver-consensus positive landmarks
  ∪ rendered-novel-view visible hard-query landmarks
```

### keypoint-consensus

ULF-Loc 的 keypoint-consensus sampling 是一个很有用的启发：它通过将 Gaussian 投影到多视角中，检查其是否稳定靠近 2D keypoints，从而选择可靠 landmarks。([arXiv][4])

你可以迁移为：

```text
一个 Gaussian 如果在 train/rendered views 中反复投影到 stable 2D keypoints 附近，
则认为它有较高 sparse availability。
```

### semantic-stable

使用 Cambridge 预处理语义 mask，降低：

```text
sky
vehicle
person
vegetation
glare
thin pole
```

提高：

```text
building facade
window corner
fixed architecture
road/building boundary
```

这直接针对你图中的车、树、强光、天空等 hard failure。

### rendered-novel-view

你前面问得很对：feedback bank 不应该只来自原始建图序列。3DGS 新视角合成的价值在 sparse 阶段就是：

```text
合成训练轨迹附近但不同视角的 query，
看哪些 landmarks 在这些 novel views 下仍然可见、可检测、可匹配。
```

不要只渲染 RGB。要渲染/记录：

```text
depth
visibility
semantic-stable mask
Gaussian contribution / projected support
detector hit
descriptor consistency
```

## Phase 1 成功标准

先不要看最终 dense。看 availability 指标：

```text
hard query visible positive coverage:
native sampled set        -> x%
new candidate pool        -> x + 20% 以上

correct-positive-in-topK:
native sampled set        -> x%
new candidate pool        -> 明显提升
```

如果 availability 没有提升，后面不会有大幅 sparse 改善。

---

# 7. Phase 2：解决 2D detector 与 3D landmark 不对齐

你现在主要改 map-side sampled landmarks，但 query-side detector 基本仍是 STDLoc 逻辑。STDLoc 的 scene-specific detector 是 shallow CNN，通过把 sampled Gaussian centers 投影到训练图像上作为 heatmap target 来训练；推理时用 NMS 取 keypoints。([arXiv][1])

问题是：如果你现在 selected landmarks 变了，query detector 仍然按原 native target 工作，就会出现：

```text
3D 端选了更好的 landmarks，
2D 端却没有在对应位置给出 keypoints。
```

## 直接改法：先做 LSF-aware detector target，不要先上复杂双流

第一步不要改 detector 网络结构，只改 target：

```text
target heatmap =
  projected high-availability / high-solver-support landmarks
  × semantic stable mask
  × visibility confidence
  × hard-query support weight
  - hard-negative / dense-worsen regions
```

这样 detector 会学会：

```text
在 query 图像上优先检测那些能匹配到高质量 3D support 的位置。
```

这一步很可能比继续改 sampled_scores 更有效，因为它直接提高 2D–3D correspondence 的闭环一致性。

## 第二步再做双流 detector

你设想的双流结构是合理的：

```text
3D detector:
  输入 Gaussian / landmark 属性
  输出 p3(i): landmark i 是否适合进入 map-side sparse localization

2D detector:
  输入 query feature map / semantic mask
  输出 p2(u): pixel u 是否适合作为 query keypoint

pair score:
  sim(desc2D(u), desc3D(i))
  + α logit p2(u)
  + β logit p3(i)
```

训练监督来自 feedback bank：

```text
positive:
  2D keypoint u 与 landmark i 匹配后成为 PnP inlier，且 reprojection error 低

negative:
  descriptor 分数高但 PnP outlier，或导致 wrong pose cluster
```

这就把“定位反馈建图”扩展成：

```text
定位反馈同时训练 3D landmark selector 和 2D query detector。
```

这仍然合理，而且比现在只改 3D 端更有潜力。

---

# 8. Phase 3：解决 Descriptor Distinguishability

如果 correct landmarks 在 sampled set 里，detector 也打到了对应 keypoints，但 descriptor ranking 仍然被错误 landmarks 压过，那 sparse 跑飞的根因就是 descriptor hard negatives。

当前你保持 native descriptor 很安全，但上限低。ULF-Loc 之所以提升明显，是因为它指出 α-blending 学 feature 会造成 Gaussian feature bias，并用 geometry-weighted feature fusion 直接从多视角 2D features 构造 landmark feature。([arXiv][4])

你不需要完全复现 ULF-Loc，但应做一个更贴合本项目的版本：

## Solver-weighted landmark feature fusion

对每个 landmark (i)，从训练/渲染视角里收集它投影位置的 2D descriptors：

[
d_i =
\operatorname{normalize}
\left(
\sum_v
w^{geom}*{iv}
w^{solver}*{iv}
w^{semantic}*{iv}
w^{visibility}*{iv}
\phi_v(\pi_v(x_i))
\right)
]

其中：

```text
w_geom:
  视角、法向、深度稳定性

w_solver:
  这个 observation 是否在 self-localization 中成为 PnP inlier
  是否低 reprojection error
  是否 dense 后改善

w_semantic:
  是否落在 building/static 区域，是否避开 sky/vegetation/vehicle

w_visibility:
  是否非遮挡、非高不确定性、可见性稳定
```

它的优点是：

```text
1. 不再依赖 alpha-blending 优化 per-Gaussian descriptor；
2. 不需要全量重训 Feature Gaussian；
3. descriptor 构造直接使用定位反馈；
4. 只对 selected/candidate landmarks 做，成本可控。
```

这个模块可能是当前最有希望带来大幅 sparse 提升的方向。因为只改 sampled set 不改变 descriptor 可分性，通常只能带来小幅 precision gain。

## 成功标准

先只评 sparse：

```text
top-1 / top-5 correct match rate 提升
hard-negative top-K false positives 下降
sparse R@10/5 或 R@5/5 提升 ≥ 1pp
catastrophic sparse failures 减少
```

如果 sparse 这里没有明显提升，不要急着进入 dense。

---

# 9. Phase 4：解决 Geometry / Solvability

如果 correct matches 已经有了，但 PnP 仍然跑飞，就说明：

```text
inliers 数量不够稳定；
或几何分布差；
或错误 matches 形成了 coherent wrong cluster。
```

直接改法有三项。

## 4.1 Minimal-set-aware sampling

不要只选高分 landmarks，要选能组成好 PnP minimal sets 的 landmarks：

[
\max_S
\lambda_A A_q(S)
+
\lambda_D D_q(S)
+
\lambda_G \log\det(H_q(S))
--------------------------

\lambda_N N_q(S)
]

其中：

```text
A: availability
D: distinguishability / descriptor margin
G: geometry / pose information
N: hard-negative conflict
```

## 4.2 Geometry-diverse PROSAC ordering

现在 PROSAC ordering 如果只按 descriptor/support 排序，会把重复结构中一堆相似点排在前面。应该加入 diversity：

```text
ranked list =
  high match score
  + high support
  + low conflict
  + spatial / depth / viewpoint diversity
```

可以做 bucketed PROSAC：

```text
从不同 image regions / 3D regions / depth bins / facade planes 中采 minimal sets。
```

这样能减少“前几百次采样全在同一重复立面”的风险。

## 4.3 Wrong-pose cluster suppression

建立 hard-negative conflict graph：

```text
两个 landmarks 经常在同一 query 中与错误 keypoints 共同支持错误 pose
→ 它们之间加 conflict edge
```

selection 和 matching 时惩罚：

[
\sum_{i,j \in S} \text{conflict}_{ij}
]

这直接针对 GreatCourt / OldHospital 重复结构。

---

# 10. Dense 不作为当前主攻，但要作为 sparse 的安全阀

你现在主要提升 sparse 是对的。Dense 可以暂时只做两件事：

```text
1. 作为 feedback source：
   标记 sparse-correct -> dense-wrong 的 query 和对应 regions。

2. 作为 verifier：
   不让 dense refinement 进一步拉坏 sparse pose。
```

不要再做“dense locability prior”这种全局小权重。你需要的是：

```text
dense match verifier
```

它输入：

```text
local geometric consistency
LSF support aggregation
depth consistency
semantic stability
alpha-composition reliability
pose leverage
```

输出：

```text
keep / drop / weight dense correspondence
```

ULF-Loc 的 LGCV 就是这个方向：在 coarse-to-fine refinement 中用局部几何一致性过滤由 rendering artifacts 导致的 mismatches。([arXiv][4])

---

# 11. 一条最直接、最有效的下一阶段路线

你说“具体形式都可以调整”，那我建议新主线直接改成：

> **Feedback-guided Sparse Correspondence Reconstruction**

核心不再是“一个 support score”，而是：

```text
用定位反馈重建 sparse correspondence system：
1. 哪些 3D landmarks 可用；
2. 哪些 2D keypoints 应该被检测；
3. 3D landmarks 应该用什么 descriptor；
4. 哪些 match 应该优先进入 PnP；
5. PnP minimal sets 应该如何保持几何稳定。
```

## Step 0：Sparse failure decomposition

先做诊断，不加新方法：

```text
Availability miss?
Detector miss?
Descriptor hard negative?
Geometry degeneracy?
Wrong-pose cluster?
```

输出 hard query 类型分布。

## Step 1：Availability upgrade

做：

```text
3DGS novel-view feedback augmentation
keypoint-consensus candidate pool
semantic-stable candidate gate
all-Gaussian oracle coverage check
```

目标：

```text
hard-query correct landmark coverage 明显提升。
```

## Step 2：2D detector target refinement

做：

```text
LSF-aware detector heatmap target
semantic mask downweight
hard-query support upweight
```

目标：

```text
detector_hit_rate_on_positive_projection 提升。
```

## Step 3：Solver-weighted landmark descriptor fusion

做：

```text
用 train/rendered 2D features 融合 selected landmarks descriptor
权重 = geometry × solver inlier × semantic stability × visibility
```

目标：

```text
top-K correct match rate 提升，hard-negative false positives 下降。
```

## Step 4：Geometry-diverse PnP ordering

做：

```text
PROSAC ordering 加 support + descriptor margin + spatial/depth diversity
minimal-set tuple quality
wrong-pose conflict graph
```

目标：

```text
sparse catastrophic failure rate 下降。
```

## Step 5：Dense verifier

只在 sparse 明显变稳后做：

```text
dense match verifier
```

目标：

```text
sparse-correct -> dense-wrong 减少。
```

---

# 12. 当前最该做的三个实验

## 实验 1：Oracle Availability

```text
对 hard failures：
all-Gaussian visible positives 有多少？
native sampled set 中有多少？
LSF sampled set 中有多少？
```

结论会告诉你：是否要跳出 native sampled set。

## 实验 2：Oracle Descriptor / Forced Correspondence

```text
如果用 GT pose 找到正确 2D–3D correspondences，
PnP 是否能成功？
```

如果 forced correspondences 成功，说明地图几何没问题，主要是 detector/descriptor/matching 问题。

## 实验 3：Sparse Transition Audit

对每个 query 分类：

```text
native sparse fail / LSF sparse success
native sparse success / LSF sparse fail
native sparse success / dense fail
native sparse fail / dense also fail
```

你要重点减少：

```text
native sparse success / LSF sparse fail
```

也就是 candidate-induced regression。

---

## 13. 最终判断

你现在的核心问题确实是 sparse。更精确地说：

> **当前方法主要改善了 map-side landmark scoring，但没有系统性提高 sparse 2D–3D correspondence 的 availability、distinguishability 和 geometric solvability。**

因此 sparse pose 一旦跑飞，dense 不可能稳定救回。

最直接有效的解决不是继续调 LSF 权重，而是把“定位反馈建图”改成：

```text
定位反馈重建 sparse correspondence system
```

包括：

```text
3D candidate pool
2D detector target
landmark descriptor
match ranking
PnP minimal-set geometry
```

如果要按优先级排序，我建议：

```text
1. Sparse failure decomposition
2. Keypoint/semantic/novel-view candidate pool
3. LSF-aware 2D detector target
4. Solver-weighted landmark feature fusion
5. Geometry-diverse PROSAC / conflict graph
6. Dense match verifier
```

其中最可能带来大幅提升的是 **2D detector 对齐 + solver-weighted landmark feature fusion**。因为它们直接作用于 sparse 跑飞的根因：正确 correspondence 没有稳定形成，而不是只是在已有 sampled set 上做小幅重排。

[1]: https://arxiv.org/html/2503.19358v1 "From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting"
[2]: https://github.com/Arthurshen926/Loc-GS "GitHub - Arthurshen926/Loc-GS · GitHub"
[3]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/mainline_lsf_20260521.md "raw.githubusercontent.com"
[4]: https://arxiv.org/html/2605.04730v1 "ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting"


