你的判断是对的：**round4.4 之后项目比之前清楚很多，但还没有达到顶刊主结果完成度**。现在的状态更像是：

> **主线已经从“各种 selector / residual / quality gate 试验”收束为 LSF-Loc：solvability-guided localization support field；但当前指标仍是小幅正向，离 ULF-Loc 论文里列出的 SOTA 级幅度还有明显差距。**

我给一个直接估计：

| 维度                |     当前完成度 | 判断                                                                                                                                                                             |
| ----------------- | --------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 问题定义 / 主线收束       |       75% | README 已经明确重置为“from matchability to solvability”，即 point-wise matchability 和 support count 不足以代理 PnP 成功，LSF 应指导 sparse landmarks、solver tuples 和 dense residuals。([GitHub][1]) |
| 代码基础设施            |       70% | round4.4 改了 130 个文件，新增/整理了 feedback-bank-v2、solver tuple、dense support、fixed recipe、ULF alignment、coverage coreset 等大量模块。([GitHub][2])                                         |
| 论文安全 / audit 体系   |       60% | 现在有 split audit、manifest、artifact audit、fixed recipe 等，但历史结果仍有 map contamination、test-camera overlap、modified evaluator 等问题，部分结果只能作为 diagnostic。([GitHub][3])                  |
| 方法完整性             |       50% | 三模块结构已经形成：solver-consensus support、solvability-aware sampling、dense residual verification；但 dense LSF 当前还不能独立带来增益，solver-aware/coverage 仍有非单调和场景不稳定问题。([GitHub][4])            |
| 主实验指标             |    25–35% | 最新 paper-safe/native-feedback 五场景结果是官方 test frozen recipe 宏平均 dense TE `-0.233cm`、4/5 scene translation 改善，但 R5 只有 `+0.0004`，StMarysChurch 仍略负；这不是 SOTA 级幅度。([GitHub][5])      |
| 与 ULF-Loc/SOTA 对齐 |       25% | ULF-Loc 在 Cambridge 平均 median translation 从 STDLoc 的 `10.1cm` 降到 `8.3cm`，且 10cm/5° recall 从 STDLoc 的 `59.9%` 到 `62.2%`；Loc-GS 目前仍是亚厘米级 delta。([arXiv][6])                      |
| 总体投稿完成度           | **约 40%** | 作为“研究平台 + 方向验证”已经不错；作为顶刊主论文，还缺一个方法级大幅提升和完整 paper-safe 表格。                                                                                                                      |

如果目标是顶刊，当前还不能进入写作冲刺；应该进入 **method redesign sprint**，目标不是再提升小数点后几位，而是让方法本身产生一档量级更大的增益。

---

# 1. 和 ULF-Loc / STDLoc / SOTA 的完整指标差距

下面先列 ULF-Loc 论文中 Cambridge Landmarks 的主表。单位是 **cm / deg**。

## 1.1 Cambridge median pose error：ULF-Loc 论文主表

| 方法               |           Court |          College |         Hospital |            Shop |          Church |             Avg |
| ---------------- | --------------: | ---------------: | ---------------: | --------------: | --------------: | --------------: |
| AS/SIFT          |       24 / 0.13 |        13 / 0.22 |        20 / 0.36 |        4 / 0.21 |        8 / 0.25 |       14 / 0.23 |
| HLoc SP+SG       |     17.7 / 0.11 |      11.0 / 0.20 |      15.1 / 0.31 |      4.2 / 0.20 |      7.0 / 0.22 |     11.0 / 0.21 |
| DSAC*            |     33.0 / 0.21 |      17.9 / 0.31 |      21.1 / 0.40 |      5.2 / 0.24 |     15.4 / 0.51 |     18.5 / 0.33 |
| ACE              |        28 / 0.1 |         18 / 0.3 |         25 / 0.5 |         5 / 0.3 |         9 / 0.3 |        17 / 0.3 |
| NeuMap           |        6 / 0.10 |        14 / 0.19 |        19 / 0.36 |        6 / 0.25 |       17 / 0.53 |       12 / 0.29 |
| GLACE            |        19 / 0.1 |         19 / 0.3 |         17 / 0.4 |         4 / 0.2 |         9 / 0.3 |        14 / 0.3 |
| NeRFMatch        |     19.6 / 0.09 |      12.5 / 0.23 |      20.9 / 0.38 |      8.4 / 0.40 |     10.9 / 0.35 |     14.5 / 0.29 |
| GSplatLoc        |               — |        31 / 0.49 |        16 / 0.68 |        4 / 0.34 |       14 / 0.42 |       16 / 0.48 |
| GSFFs-PR Feature |               — |        17 / 0.26 |        18 / 0.36 |        4 / 0.25 |        8 / 0.26 |       12 / 0.30 |
| **STDLoc**       | **15.7 / 0.06** |  **15.0 / 0.17** |  **11.9 / 0.21** |  **3.0 / 0.13** |  **4.7 / 0.14** | **10.1 / 0.14** |
| ACE+GS-CPR       |               — |        20 / 0.29 |        21 / 0.40 |        5 / 0.24 |       13 / 0.40 |       15 / 0.33 |
| **ULF-Loc**      | **7.49 / 0.04** | **17.03 / 0.19** | **10.26 / 0.19** | **2.83 / 0.14** | **3.65 / 0.11** |  **8.3 / 0.13** |

ULF-Loc 的平均 translation 比 STDLoc 论文表中的 `10.1cm` 低到 `8.3cm`，改善约 `1.8cm`，论文称相对 STDLoc 降低约 17%。它不是每个场景都赢：KingsCollege 上 ULF-Loc `17.03cm` 比 STDLoc `15.0cm` 差，但 GreatCourt、OldHospital、ShopFacade、StMarysChurch 和平均值更强。([arXiv][6])

---

## 1.2 Cambridge recall：ULF-Loc 论文附录表

| 方法           | Avg R@50cm/5° | Avg R@15cm/5° | Avg R@10cm/5° |
| ------------ | ------------: | ------------: | ------------: |
| HLoc SP+SG   |          91.4 |          64.8 |          52.0 |
| ACE          |          78.7 |          43.1 |          31.5 |
| GLACE        |          91.0 |          62.8 |          47.6 |
| **STDLoc**   |      **95.4** |      **70.8** |      **59.9** |
| GLACE+GS-CPR |          92.5 |          65.5 |          50.7 |
| ACE+GS-CPR   |          84.6 |          56.8 |          42.6 |
| **ULF-Loc**  |      **93.7** |      **72.0** |      **62.2** |

这里 ULF-Loc 在 R@50/5 略低于 STDLoc，但在更严格的 R@15/5 和 R@10/5 更强，尤其 R@10/5 比 STDLoc 高 `+2.3pp`。这说明 ULF-Loc 的优势不是“更宽松阈值下更多成功”，而是**高精度定位更强**。([arXiv][6])

---

## 1.3 训练时间、显存、速度：ULF-Loc 论文

| 方法          | Training Time |      Memory |
| ----------- | ------------: | ----------: |
| PNeRFLoc    |       58 mins |     6396 MB |
| GSplatLoc   |     1.5 hours |     6986 MB |
| **STDLoc**  |   **50 mins** | **6566 MB** |
| **ULF-Loc** |    **5 mins** | **1086 MB** |

ULF-Loc 论文还报告 Cambridge 平均定位速度：ULF-Loc `4.4 FPS`，STDLoc `3.9 FPS`，GSplatLoc `0.6 FPS`，NeRFMatch `2.2 FPS`。([arXiv][6])

---

## 1.4 Loc-GS / LSF-Loc 当前最新状态对比

注意：下面是**仓库当前报告的 Loc-GS 进度**，不是 ULF-Loc 那种完整 paper table。你目前还没有一个完整、公开可对齐的 ULF-style Cambridge 表格，尤其缺 R@50/5、R@15/5、R@10/5 的最终 paper-safe 版本。

| 项目                                         | 当前 Loc-GS / LSF-Loc 状态                                                                                                                                                            |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 主方法                                        | Native STDLoc map + audited self-localization feedback + LSF for sparse landmarks / solver tuples / dense residuals + single STDLoc-compatible inference path。([GitHub][7])       |
| 最新 paper-safe-ish 五场景方向                    | Native STDLoc feedback chain：train-only / clean valid source maps，feedback_bank_v2，solver-consensus support，same-budget 32-edit map export，固定 native evaluator path。([GitHub][5]) |
| Official test frozen recipe dense TE delta | Macro `-0.2330cm`，4/5 scene translation 改善；GreatCourt `-0.2805`、KingsCollege `-0.0326`、OldHospital `-0.8485`、ShopFacade `-0.0202`、StMarysChurch `+0.0165`。([GitHub][5])           |
| Official test frozen recipe recall         | Macro R5cm delta `+0.0004`，非常小。([GitHub][5])                                                                                                                                      |
| q80 train-dev                              | Macro dense TE delta `-0.2949cm`，3/5 wins，R5 `+0.0025`。([GitHub][5])                                                                                                              |
| q160 train-dev                             | Macro dense TE delta `-0.0857cm`，4/5 wins，R5 `+0.0013`。([GitHub][5])                                                                                                              |
| v5 solver-coverage coreset                 | train-dev q80/q160 有稳定小幅 precision gain，但 512-capacity 非单调，说明方法缺少 robust coverage saturation controller。([GitHub][8])                                                             |
| dense-only LSF                             | q80 `+0.0223cm`、q160 `+0.0179cm`，被拒绝；dense LSF 不能单独作为强模块。([GitHub][9])                                                                                                            |
| 当前结论                                       | 现在强于早期 ShopFacade-only claim，但仍不是 SOTA claim；StMarysChurch 是 neutral/slightly negative boundary。([GitHub][5])                                                                     |

粗略比较：

| 对比对象                                                        |                                                                                                       平均 translation 改善幅度 |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------: |
| ULF-Loc vs STDLoc paper table                               |                                                                                              `10.1cm -> 8.3cm`，约 `-1.8cm` |
| Loc-GS latest native-feedback official test vs local native |                                                                                                                `-0.233cm` |
| 差距                                                          | ULF-Loc 的平均改善幅度约是当前 Loc-GS 的 **7–8 倍**；且 ULF-Loc 还给出 R@10/5 `+2.3pp`，而 Loc-GS 当前 R5cm delta 约 `+0.0004`，几乎可以视为 recall 持平。 |

所以你的判断是准确的：**当前 Loc-GS 指标还没有和 ULF-Loc 这类 SOTA 工作站到同一强度层级。**

---

# 2. 为什么现在还是“小数点后几位提升”

我认为当前架构有四个主要瓶颈。

## 瓶颈 A：当前 LSF 主要仍是 map payload editing，不是 representation-level 改变

你现在最有效的链路是：

```text
native STDLoc descriptor 不变
LSF 修改 sampled_idx / sampled_scores / locability
same-budget map edit
```

这很 paper-safe，但天然增益有限。STDLoc 原始 sampled set 已经很强，尤其在 16k landmarks 下，静态 same-budget 编辑几十到几百个 landmarks 很难产生 ULF-Loc 那种 `1–2cm` 级别平均提升。

ULF-Loc 的核心不是“轻微调采样分数”，而是**换了 landmark feature 构造范式**：它不再通过 alpha-blending feature optimization 学高维 Gaussian feature，而用 geometry-weighted feature fusion 构造 unbiased landmark feature，并用 keypoint-consensus sampling 和 LGCV 支撑整个 sparse-to-dense 流程。([arXiv][10])

你的 LSF 如果只做 support editing，而不改变“2D–3D matching 的候选可分性”，上限就很低。

---

## 瓶颈 B：query detector / selected landmarks / descriptor 三者没有联合闭环

当前主线强调“不替换 query detector”。这是安全的，但也带来一个问题：

```text
STDLoc detector 是为 native sampled landmarks / native score distribution 服务的；
LSF 改了 sampled_idx 或 sampled_scores；
但 query-side keypoints 和 descriptor matching 仍然按原分布工作。
```

如果 2D keypoint 侧不感知 LSF，3D landmark 侧的 support field 很可能只是在已有匹配分布上做小幅扰动。这解释了为什么很多结果表现为 precision 小幅改善，但 recall 不稳定或几乎不动。

---

## 瓶颈 C：dense LSF 仍然是“prior”，不是 dense residual verification

当前仓库明确说 dense-only LSF 被拒绝，q80/q160 都没有带来 macro positive。v4 SPGS-prior 文档也说，dense-only all-selected candidate 被拒绝，正结果来自 sparse support editing + prior-consumed dense support 的组合，不是 dense locability 本身。([GitHub][9])

这说明现在 dense 模块还没有达到 ULF-Loc LGCV 那种“直接过滤渲染 artifact mismatch”的效果。ULF-Loc 的 LGCV 是在 dense matches 上做局部几何一致性验证；你的 dense LSF 目前更像全局 locability prior，还没进入 dense match / residual 层做真正的错误 residual 抑制。

---

## 瓶颈 D：coverage coreset 有思路，但缺 saturation / tail-risk 控制

v5 coverage coreset 的正负结果很有启发：128-edit 版本 q80/q160 macro 是负 delta，但 512 容量非单调，q80 OldHospital 和 q160 StMarysChurch 会被伤害；文档自己也判断“larger edit capacity needs explicit saturation or tail-risk constraint”。([GitHub][8])

这说明你现在的 selection objective 还会“过度覆盖容易 query / 过度编辑某些区域”，导致 hard-tail 被破坏。它还不是一个稳定的 set optimization。

---

# 3. 投稿完成度的实际判断

如果投稿目标是 **顶刊主文**：

```text
当前完成度：约 40%
```

其中：

```text
问题与主线：已经比较清楚
代码与报告基础设施：已经较多
paper-safe 结果：刚开始有可用形态
SOTA 级指标：还远远不够
方法级闭环：还不完备
```

如果投稿目标是 **workshop / 技术报告 / arXiv 方向论文**：

```text
当前完成度：约 60%
```

可以写成“support count 不是正确目标，LSF 有小幅正向”的研究报告。

但如果要顶刊，至少需要达到下面任一条：

```text
1. Cambridge full/test 平均 TE 改善 ≥ 1.0cm，且 R@10/5 或 R@5/5 有 ≥ 1pp 级别提升；
2. 或者平均 TE 改善 0.5–0.8cm，但速度/内存/训练时间显著优于 STDLoc；
3. 或者在一个 ULF-Loc 没覆盖的真实场景维度上有强 failure recovery，比如 dense refinement artifact / hard-query tail 明显恢复；
4. 或者提出一个比 keypoint-consensus / matching-oriented sampling 更强的 solver-level selection 模块，并在 Cambridge + 另一个数据集上稳定成立。
```

当前 `-0.233cm`、R5 `+0.0004` 不够。

---

# 4. 方法级优化脑暴：哪些地方最可能带来“大幅提升”

下面按优先级列。重点不是调参，而是改方法结构。

---

## 方向 1：从 unary LSF 改成 **query-conditioned support banks**

### 当前问题

你现在的 map edit 大体是一个全局静态 sampled set：

```text
one scene -> one sampled_idx / sampled_scores
```

但 Cambridge 的场景差异很大：GreatCourt 大尺度、ShopFacade 小而清晰、StMarysChurch/OldHospital 有 hard-tail。一个全局 sampled set 很难同时兼顾所有 query viewpoint。

### 方法级改法

学习多个 **support banks**：

```text
B0: native safe core
B1: facade / near-range bank
B2: wide-baseline / long-range bank
B3: hard-tail / low-texture bank
B4: dense-risk-safe bank
```

query 进入时，不用 GT，不做 per-query branch oracle，而是用可观测信号路由：

```text
query global descriptor / detector heatmap statistics / top sparse match distribution
-> select one or two support banks
-> run the same matching + PnP + dense path
```

这不是“结果后验 branch selection”，而是 **query-conditioned landmark activation**。它更像 image retrieval 或 HLoc 的 database selection，是合法的前置选择。

### 为什么可能大幅提升

全局 same-budget edit 只有 `-0.2cm` 量级；query-conditioned bank 可以让每个 query 实际匹配的 landmark set 更干净、更局部、更低歧义。ULF-Loc 的 K.C. Sampling 本质也是让 landmark 更贴近 keypoint / query distribution；你可以把它提升为 solver-conditioned query distribution。

### 第一版实现

```text
1. 用 feedback_bank_v2 的 real image_id 对 query 聚类。
2. 每个 cluster 单独构建 solver-coverage coreset。
3. 训练一个轻量 router：
   input = query SuperPoint/global image stats/top sparse match entropy
   output = cluster id or top-2 banks
4. 每个 bank same-budget 或 sub-budget。
5. 固定路由规则，不看 test GT。
```

### 验收

```text
q80/q160 train-dev:
  per-cluster bank > global static LSF
  R@10/5 或 R@5/5 至少 +1pp
  hard scenes 不退化
```

---

## 方向 2：加入 **solver-weighted landmark descriptor fusion**，不要只改 selection

### 当前问题

你现在尽量保持 native descriptor，这保证 fairness，但也限制上限。ULF-Loc 之所以强，是因为它不仅选 landmark，还改变了 landmark feature 的构造范式。它通过 geometry-weighted fusion 避免 alpha-blending feature bias，而不是只调 sampled_scores。([arXiv][10])

### 方法级改法

不是回到 full residual descriptor，而是只对 selected support landmarks 构造一个轻量的 **solver-weighted fused descriptor**：

[
d_i =
\operatorname{normalize}
\left(
\sum_{q \in \mathcal{V}(i)}
w_{iq}^{solver}
w_{iq}^{geom}
w_{iq}^{visibility}
\phi_q(\pi_q(x_i))
\right)
]

其中：

```text
w_solver: 来自 feedback_bank_v2 的 PnP inlier / low reprojection / dense improved
w_geom: normal-view angle / depth stability / projected size
w_visibility: multi-view stability
```

这和 ULF-Loc 的 geometry-weighted feature fusion相似，但你的区别是：

```text
ULF-Loc: geometry-weighted / keypoint-consensus feature fusion
Loc-GS: solver-weighted / self-localization feedback feature fusion
```

这会把你的方法从“只做 selection”推进到“selection + support feature reconstruction”，但仍然不是 full Feature Gaussian residual，不会重新陷入 alpha-blending dense feature field。

### 预期收益

这可能是最有希望把 `-0.2cm` 级别推到 `-0.8cm ~ -1.5cm` 的方向。因为它改变了匹配本身，而不是只改 candidate set。

### 风险

会被质疑接近 ULF-Loc。区别必须写清楚：

```text
ULF-Loc 用 keypoint-consensus + geometry weighting；
你用 solver traces / PnP success / dense transition 作为 feature fusion 权重。
```

---

## 方向 3：把 dense LSF 改成 **dense match verifier**，而不是 locability prior

### 当前问题

dense-only prior 已经失败或近似无效。当前 dense support 如果只是写进 locability prior，很难模仿 ULF-Loc LGCV 的作用。

### 方法级改法

在 dense refinement 前，对 dense matches 做二级验证：

```text
dense match candidate
  -> local geometric consistency
  -> LSF support aggregation
  -> alpha composition reliability
  -> pose leverage
  -> accept / reject / weight
```

也就是把 dense LSF 从：

```text
global prior weight
```

改成：

```text
dense correspondence classifier / verifier
```

输入特征：

```text
query-render descriptor similarity
rendered depth consistency
local 2D neighborhood scale/angle consistency
contributing Gaussian support mean/max
composition entropy / dominance
pose Jacobian leverage
sparse pose reprojection residual
```

输出：

```text
dense_match_weight 或 keep/drop
```

这和 ULF-Loc LGCV 对齐，但你加了 LSF 和 alpha reliability。

### 为什么可能大幅提升

当前很多问题是 dense refinement 会擦掉 sparse stage 的小收益。文档里也反复出现“sparse improvement erased by dense refinement”。如果 dense verifier 能减少 dense-worsened queries，R5/R2 会明显改善。

### 第一版实现

先不要训练深网，用规则 + logistic calibration：

```text
score = a * local_geom_consistency
      + b * LSF_support
      + c * alpha_dominance
      - d * ambiguity
      - e * depth_uncertainty
```

然后在 q80/q160 上只看：

```text
dense worsened_count 是否下降
sparse-correct -> dense-wrong 是否减少
```

目标不是所有 query 变好，而是修掉 dense stage 的负贡献。

---

## 方向 4：query-tail saturation controller，解决 v5 非单调

### 当前问题

v5 coverage coreset 很接近正确方向，但 512 edit 容量非单调：更多 edits 不一定更好。文档已经指出需要 saturation/tail-risk controller。([GitHub][8])

### 方法级改法

每个 query cluster 都有一个 native support target：

[
T_q = \mathrm{percentile}*{p}(U_q(S*{native}))
]

当候选 set 对 query 的 support 达到 target 后，不再给这个 query 的额外 support 加正收益：

[
\tilde{U}_q(S)=\min(U_q(S), T_q)
]

同时对 hard-tail 用 CVaR：

[
\max_S \sum_q \tilde{U}*q(S) + \lambda \mathrm{CVaR}*{10%}(\tilde{U}_q(S))
]

这会防止算法继续向已被覆盖的 easy queries 加点，从而破坏 hard scenes。

### 预期收益

这不是小调参，是 objective 结构修复。它直接针对 v5 的 failure mode。

---

## 方向 5：把 detector 也纳入 LSF，但不是替换 detector 网络

### 当前问题

map-side sampling 改了，但 query-side keypoints 没变。STDLoc detector 仍然按 native target 学 keypoints；LSF selected landmarks 可能没有被 query keypoints 充分覆盖。

### 方法级改法

做一个 **LSF detector target refinement**：

```text
原 STDLoc detector heatmap target:
  projected native sampled Gaussian centers

新的 LSF detector target:
  projected LSF-supported landmarks
  + hard-query solver support weights
  - dense-worsen / hard-negative risk
```

推理时仍然用单个 detector 网络，不做 branch。训练时只是更新 detector target。

### 为什么可能大幅提升

如果 query detector 和 3D selected support 对齐，2D–3D matching 的召回会提升，而不是仅仅改变 3D 端排序。

### 最小实现

```text
1. 冻结 descriptor。
2. 用 LSF support 生成 detector heatmap target。
3. fine-tune detector 1–3 epochs，强正则保持 native heatmap。
4. q80/q160 只比较 sparse initial pose。
```

如果 sparse R@10/R@5 明显提升，再进入 dense。

---

## 方向 6：引入 **negative support memory**，让 hard negatives 真的参与选择

### 当前问题

目前 hard-negative risk 可能只是 scalar penalty。它没有真正改变候选匹配分布。

### 方法级改法

建立 per-scene hard-negative graph：

```text
节点：landmarks / Gaussians
边：在 query top-K 中互相混淆、竞争同一 keypoint、导致错误 PnP
```

selection objective 加入 graph cut：

[
\min_S \sum_{i,j \in S} \text{conflict}_{ij}
]

这比单点 hard_negative_risk 更强，因为重复结构的风险通常是 pairwise/group-wise，不是 unary。

### 预期收益

对 StMarysChurch、OldHospital 这种重复结构 / hard-tail 应该更有帮助。

---

## 方向 7：不要只做 32-edit，小编辑只能验证方向

### 当前问题

32 edits 太保守。它有 paper-safety，但不可能带来 ULF-Loc 级别提升。

### 方法级改法

用两阶段策略：

```text
Stage 1: 32-edit safe candidate，用于验证不崩。
Stage 2: bank-level resampling，允许 10–30% landmark replacement，但必须由 saturation/CVaR/negative graph 约束。
```

关键不是 edits 多，而是：

```text
大规模 edit 必须 query-conditioned + saturation-controlled。
```

否则 v5 512 的非单调会继续出现。

---

# 5. 我建议下一轮“方法级 sprint”这样做

不要再同时做十条线。下一轮只做三件事。

## Sprint A：LSF-v6 Saturated Coverage Coreset

目标：修复 v5 512 非单调。

实现：

```text
loc_gs/stdloc_native/solver_coverage_coreset.py
新增：
  --coverage_saturation native_percentile
  --tail_cvar_alpha 0.1
  --easy_query_cap
  --hard_query_min_gain
```

验证：

```text
q80/q160 train-dev
比较 v5-128、v5-512、v6-saturated-512
目标：v6 在 q80/q160 都不出现 OldHospital/StMarysChurch regress
```

成功标准：

```text
macro TE delta ≤ -0.5cm
R@10 or R@5 ≥ +0.5pp
5/5 scenes median non-worse
```

---

## Sprint B：Support-weighted Landmark Feature Fusion

目标：突破“只改 sampling”上限。

实现：

```text
loc_gs/stdloc_native/solver_weighted_feature_fusion.py
```

输出：

```text
selected_landmark_descriptors.pt
descriptor_mode=solver_fused_landmarks
```

约束：

```text
只对 selected sampled landmarks 构造 fused descriptor；
不训练 full Gaussian descriptor field；
native descriptor 作为 fallback；
rho/trust-region 限制 descriptor shift。
```

验证：

```text
先只跑 sparse initial pose
如果 sparse R@10/R@5 没明显提升，就不进 dense。
```

成功标准：

```text
sparse R@10 +1pp 以上
dense R@10/R@5 不掉
macro TE -0.5cm 以上
```

---

## Sprint C：Dense Match Verifier

目标：减少 dense refinement 抹掉 sparse 改善。

实现：

```text
loc_gs/dense_support/dense_match_verifier.py
```

输入：

```text
dense match similarity
local geometric consistency
LSF support aggregation
alpha reliability
depth consistency
pose leverage
```

输出：

```text
match weight / keep mask
```

验证：

```text
只看 dense transition:
  sparse-correct -> dense-wrong 是否减少
  dense_worsened_count 是否下降
```

成功标准：

```text
dense worsened_count -10% 以上
R@5/R@2 至少 +0.5pp
latency 不增加超过 10%
```

---

# 6. 一个更激进但可能真正有效的版本

如果你愿意接受方法更接近 ULF-Loc，但仍保持差异，我建议最终方法变成：

> **Solver-Guided Unbiased Support Feature Field**

三模块：

```text
1. Solver-consensus support sampling
   不是 keypoint-consensus，而是 PnP/dense-success consensus。

2. Solver-weighted landmark feature fusion
   不是 alpha-blending descriptor optimization，而是从 train views 直接融合 2D descriptors，
   权重来自 geometry + solver success。

3. Support-consistent dense verification
   不是全图 dense prior，而是 dense correspondence verifier。
```

这会更像 ULF-Loc 的“优雅有效”路线，但有自己的核心差异：

| ULF-Loc                          | 你的新版本                                                   |
| -------------------------------- | ------------------------------------------------------- |
| Keypoint-consensus sampling      | Solver-consensus support sampling                       |
| Geometry-weighted feature fusion | Geometry + solver-weighted feature fusion               |
| LGCV dense mismatch removal      | LSF + alpha reliability + local geometry dense verifier |
| 解决 biased Gaussian feature       | 解决 matchability/support-count 不能代理 PnP solvability      |

我认为如果你追求“大幅提升”，迟早要走到这一步。单纯 same-budget sampled_idx editing 的上限已经被你这几轮实验摸到了。

---

# 7. 下一步优先级排序

按“最可能带来方法级提升”排序：

1. **Solver-weighted landmark feature fusion**
   这是最可能把小幅 precision gain 变成大幅 accuracy gain 的模块。

2. **Dense match verifier，而不是 dense prior**
   这是最可能提升 R5/R2 和减少 dense-worsened queries 的模块。

3. **Saturated query-tail coverage coreset**
   这是修复 v5 非单调、让大 edit budget 安全的关键。

4. **LSF detector target refinement**
   这是让 query-side detector 和 selected support 对齐的关键。

5. **Query-conditioned support banks**
   这是大尺度提升和速度-精度 Pareto 的长期方案。

6. **Pairwise hard-negative conflict graph**
   这是解决重复结构和 StMarysChurch hard-tail 的长期方案。

---

# 8. 结论

当前 round4.4 的价值在于：**主线终于像一篇论文了**。README、mainline docs 和新模块已经把项目从“杂乱 selector 实验”变成了 “LSF-Loc：从 matchability 到 solvability”。但指标层面仍然不够：最新五场景 official test frozen recipe 只有 macro dense TE `-0.233cm`，R5 近乎持平；而 ULF-Loc 对 STDLoc 是平均 `-1.8cm`、R@10/5 `+2.3pp` 级别。([GitHub][5])

所以接下来不要继续调小参数。要做方法级升级：

> **从“选择哪些 native landmarks”升级为“solver-guided support feature construction + dense match verification”。**

最推荐下一轮组合是：

```text
LSF-v6 saturated coverage coreset
+ solver-weighted landmark feature fusion
+ dense match verifier
```

这三者分别解决：

```text
selection 非单调
descriptor 不变导致上限低
dense refinement 抹掉 sparse gain
```

如果这三件事仍然只能带来 `0.2–0.3cm`，那就说明这个课题在 STDLoc backbone 上的边际空间确实有限，需要转向更根本的 ULF-style feature construction 或跨数据集真实问题验证。

[1]: https://github.com/Arthurshen926/Loc-GS "GitHub - Arthurshen926/Loc-GS · GitHub"
[2]: https://github.com/Arthurshen926/Loc-GS/commit/8c71d1a6467da22a787a7f90f42b2ef72273f1ba "round4.4 · Arthurshen926/Loc-GS@8c71d1a · GitHub"
[3]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/lsf_canonical_q80_status_20260523.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/mainline_lsf_20260521.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/lsf_native_feedback_five_scene_status_20260524.md "raw.githubusercontent.com"
[6]: https://arxiv.org/html/2605.04730v1 "ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting"
[7]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/README.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/lsf_v5_coverage_status_20260524.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/lsf_v4_spgs_prior_precision_status_20260524.md "raw.githubusercontent.com"
[10]: https://arxiv.org/abs/2605.04730?utm_source=chatgpt.com "ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting"