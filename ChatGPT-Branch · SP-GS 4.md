# Branch · SP-GS

**User:** Anonymous  
**Created:** 5/10/2026 13:58:36  
**Updated:** 5/11/2026 22:55:58  
**Exported:** 5/11/2026 22:56:33  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370)  

## Prompt:
2026/5/11 22:38:12

目前推进后验证的结果无法支撑我们的主要claim，git仓库已更新https://github.com/Arthurshen926/Loc-GS。
如果不是纯 learned descriptor replacement，而是在基础特征重建的基础上做微调呢？现在的实现就是如此过程吗：重建过程中做了一些有真值的辅助定位，有一些信息实时反馈给重建，有一些信息则保留着真实query定位时使用。相当于在地图里先做一次自匹配、自定位，发挥3dgs新视角渲染的能力来提前算出一些用于特征选择的指标？并且服务基于2d3d几何的定位方法，用可匹配性引导特征重建。
对于可投稿状态的目标，该如何改进方法和实验验证呢？

## Response:
2026/5/11 22:38:15

我会先核对最新仓库中训练、PnP loss、calibration、landmark selection 与 eval 默认链路，明确当前实现是否可称为定位反馈式微调。然后把 claim 收敛到可投稿的“可匹配性引导特征重建”版本，并给出最低可行方法与实验协议。

## Response:
2026/5/11 22:38:32

我已确认当前实现更像“重建后的定位可靠性校准/自诊断”，不是纯 replacement。接下来应把反馈信号分清哪些参与微调、哪些只服务真实 query 定位。

## Response:
2026/5/11 22:43:07

可以这样理解，但需要把表述收敛得更准确：

> **不是 learned descriptor replacement，而是 matchability-guided feature-field fine-tuning：先重建 SuperPoint/STDLoc 可用特征场，再利用训练视角中的自匹配、自定位和 GT 几何监督，估计哪些 Gaussian / rendered pixels / matches 对 2D-3D PnP 真正有用，最后用这些可匹配性信号约束特征重建与定位阶段。**

这条路线仍然成立。问题是：**当前实现已经包含了这条路线的很多部件，但还没有被组织成一个干净、稳定、可投稿的“微调方法”。** 现在它更像是把 descriptor reconstruction、locability、PnP loss、landmark score、calibration、branch selection 都尝试了一遍；实验结果说明，强干预版本不稳，但“受约束微调 + 保守使用定位反馈”仍有空间。

---

## 1. 当前实现是不是你说的这个过程？

**部分是，但不是完全。**

当前仓库 README 仍把主线定义为：SuperPoint teacher features → low-dimensional per-Gaussian latent feature field → decoded descriptor/detector maps at novel views → geometry-aware matching and PnP relocalization；并且明确包含 differentiable matching、reprojection proxy、observability、per-Gaussian locability 等 localization-guided losses。这个已经接近你说的“基于基础特征重建，再用定位信息反馈”的框架。([GitHub](https://github.com/Arthurshen926/Loc-GS))

训练代码里，Cambridge hybrid training 的确先渲染 GT pose 下的 SuperPoint-like descriptor/detector，然后用 teacher descriptor/detector 做重建监督；同时会准备 SuperPoint query keypoints/descriptors，并计算 same-view geometric keypoint matching loss。也就是说，基础 feature reconstruction 和“自匹配”已经在训练里发生了。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

之后，训练会对 GT pose 加扰动得到 `loc_pose`，再从扰动视角渲染 `loc_render`，把 query descriptors、query keypoints、rendered descriptors、rendered depth、GT pose、GT depth/alpha 等送进 `DifferentiablePnPMatchLoss`。这就是你说的“在地图里先做一次自定位，用训练视角中的真实几何反馈给重建”。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

`DifferentiablePnPMatchLoss` 现在也确实把 matching logits 和 PnP logits 分开：matching 用 `valid_pixels`，PnP top-k 用 `valid_target_pixels`，再用几何 target 得到 PnP 权重。这说明你前面修的实现风险已经被认真处理过。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/differentiable_pnp.py))

此外，当前仓库已经加入 branch selection 和 query-level pose selection：README 中现在明确写到 paper-facing Cambridge recipe 是 baseline-preserving，默认使用 STDLoc sampled / PLY descriptor backbone，再从 train/calibration-val split 选择 learned branches，而不是按 test 结果手选。([GitHub](https://github.com/Arthurshen926/Loc-GS)) `select_cambridge_branch.py` 也支持用 calibration split、combined/R5/median-te 指标、baseline branch 约束来选择分支。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/select_cambridge_branch.py))

所以，**你这个理解方向是对的**。

但还要补一句：**当前实现还不是一个干净的“基础重建后微调”流程。** 因为训练阶段仍然可能联合更新 latent、hash field、decoder、SuperPoint head、locability、甚至 geometry；eval 阶段又有 descriptor source、landmark source、calibrated prior、coverage filtering、branch selection 等多个分支。因此实验上容易变成“很多机制同时干预 STDLoc”，而不是一个单一可解释的 fine-tuning 方法。

---

## 2. 更准确的方法定位

我建议不要再把论文叫成：

> learned Gaussian descriptor replacement

也不要只叫：

> localization-guided reconstruction

更准确的定位是：

> **Matchability-Guided Fine-Tuning of Gaussian Feature Fields for 2D-3D Camera Relocalization**

中文可以写成：

> **面向 2D-3D 定位的可匹配性引导 Gaussian 特征场微调。**

它的核心不是“我学出了更好的 descriptor”，而是：

1. **先重建一个稳定的基础特征场**，保持 SuperPoint/STDLoc descriptor 的基本可匹配结构；
2. **用 3DGS novel-view rendering 在训练视角中自匹配、自定位**，提前计算 visibility、depth consistency、TP/FP、observability、locability、match ambiguity 等定位相关信号；
3. **用这些信号约束特征场微调**，让特征场不要只追求 teacher reconstruction，而是更偏向 PnP 有用的匹配；
4. **测试时仍服务传统 2D-3D 几何定位**，不推翻 STDLoc/SplatLoc 的 sparse-to-dense localization backbone。

这样和现有工作的关系也更清楚。STDLoc 已经提出 Feature Gaussian 表示、matching-oriented Gaussian sampling、scene-specific detector 和 sparse-to-dense camera relocalization。([arXiv](https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com)) SplatLoc 也已经提出 3DGS visual localization、unbiased 3D descriptor 和 salient 3D landmark selection。([arXiv](https://arxiv.org/html/2409.14067v1?utm_source=chatgpt.com)) 你的贡献不应写成“替代它们”，而应写成：

> 在这些 2D-3D 定位骨架上，进一步研究如何用可匹配性信号反过来微调 Gaussian feature field，使重建出的特征更适合 PnP。

---

## 3. 当前主要问题在哪里？

### 问题一：当前“微调”太强，容易破坏基础 descriptor 分布

你想要的是“在基础特征重建上做微调”，但目前一些配置会让 learned branch 大幅介入 descriptor、landmark bank、matching prior 或 filtering。实验已经证明，这会破坏 STDLoc/PLY descriptor 的稳定结构。

所以现在应明确一个原则：

> **基础 descriptor 分布不可被大幅改变；所有定位引导信号只能做小步、受限、可回退的 fine-tuning。**

也就是说，feature field 的目标不是从零学一个新 descriptor，而是学：

$$
d_\text{final}
=
\mathrm{normalize}(d_\text{base} + \alpha \Delta d)
$$

其中 $\alpha$ 必须很小，甚至默认只训练 locability / confidence，不改 descriptor。

### 问题二：当前有些定位反馈是 oracle-style，不等价于真实 test-time matching

训练里用 GT pose、GT visibility、GT target pixels 来构造几何监督，这是正确的；但测试时 query 要在全量重复结构 candidate 中自己匹配。因此，如果 PnP loss 只教模型“在干净 target set 里优化 pose”，它未必学会“在大量 hard negatives 中不选错”。

这就是为什么 PnP 修得更正确后，full SP-GS2 反而不一定更好。训练目标更干净，不代表和测试分布更一致。

### 问题三：静态 landmark matchability 不足以处理 query-specific ambiguity

你已经验证 per-landmark score、KC strict、calibrated prior 都不稳。这说明很多指标只能回答：

> 这个点总体上是否显著、可见、稳定？

但定位真正需要回答的是：

> 对当前 query 来说，这个 2D feature 匹配到这个 3D landmark 是否会成为 PnP 内点？

GreatCourt 的重复结构正好卡在这里。静态 score 很容易把重复窗户、重复立面这种“稳定但有歧义”的点选出来。

### 问题四：hard filtering 违背 RANSAC 的优势

传统 2D-3D PnP/RANSAC 需要冗余候选。你在 PnP 前硬删 matches，如果 score 没有 100% 对齐真实内点，就会删掉有用弱匹配。实验里 coverage filtering / margin / MNN 不稳，就是这个原因。

所以未来不要再主打 hard filtering，而要主打：

> soft priority / confidence ordering / fallback selection

比如 PROSAC-style 优先采样，而不是删除。

---

## 4. 可投稿版本的方法应该怎么改？

我建议把方法重构成四阶段。这样既符合你说的“自匹配、自定位、提前算指标、服务 2D-3D 定位”，也能避免继续把 learned branch 做成 replacement。

---

### Stage 1：基础特征场重建

目标：只学一个稳定的 feature field，不追求定位收益。

训练：

$$
\mathcal{L}_{base}
=
\mathcal{L}_{desc-recon}
+
\lambda_d \mathcal{L}_{det-recon}
+
\lambda_{same}\mathcal{L}_{same-view-match}
$$

当前代码已经有 descriptor MSE + cosine、detector KL，以及 same-view geometric keypoint matching loss。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

这一阶段要做两件事：

1. 记录 base checkpoint；
2. 不允许 geometry 大幅变化，不允许 descriptor residual 大幅偏离 teacher/PLY。

这阶段的论文指标不是定位，而是：

- descriptor cosine；
- same-view top1@1px / top1@3px；
- rendered detector repeatability；
- valid rendered depth/alpha coverage。

---

### Stage 2：训练视角自匹配 / 自定位，离线生成 matchability labels

用训练图像模拟真实 query：

1. 从 train view 抽 query keypoints/descriptors；
2. 用当前 feature field / PLY bank / rendered bank 做 2D-3D matching；
3. 用 GT pose 判断 reprojection error；
4. 得到 TP/FP、inlier probability、ambiguity、visibility、depth consistency、observability；
5. 存成 sidecar cache。

这一步就是你说的：

> 在地图里先做一次自匹配、自定位，发挥 3DGS 新视角渲染能力，提前算出一些用于特征选择的指标。

但要注意：这些指标不能只做 per-landmark 静态均值，必须至少包含 query-conditioned 信息。

建议 label 定义：

$$
y_{ij}=1[
\|\pi(T_{gt}X_j)-u_i\| < \epsilon
\land \text{depth-consistent}
\land \text{visible}
]
$$

这里 $i$ 是 query keypoint，$j$ 是 3D landmark。  
这比单纯 landmark score 更接近真实定位目标。

---

### Stage 3：受约束 fine-tuning，而不是 full replacement

这是最关键的修改。

新的 fine-tuning loss 不应以 PnP pose loss 为主，而应以 hard-negative matchability 为主：

$$
\mathcal{L}_{ft}
=
\mathcal{L}_{base}
+
\lambda_{rank}\mathcal{L}_{hard-neg}
+
\lambda_{loc}\mathcal{L}_{locability}
+
\lambda_{trust}\mathcal{L}_{trust}
$$

其中：

$$
\mathcal{L}_{trust}
=
\|d_\text{ft} - d_\text{base}\|^2
$$

或者：

$$
d_\text{ft}
=
\mathrm{normalize}(d_\text{base} + \alpha \Delta d),\quad \alpha \le 0.05
$$

Hard-negative loss 要直接针对失败原因：

$$
s(q_i, X^+)
>
s(q_i, X^-)
$$

其中 $X^-$ 是 descriptor similarity 很高但 GT reprojection error 很大的 hard negative。  
这比 differentiable PnP 更直接，因为 GreatCourt 的核心问题不是“PnP 公式不会优化”，而是“错误重复结构也很像”。

Fine-tuning 时建议：

- freeze 大部分 descriptor decoder；
- 只训练 locability / small residual gate / final small adapter；
- PnP pose loss 设为 0 或极小；
- 不训练 geometry，除非只做 isotropy/regularization；
- 不让 learned descriptor 直接替代 PLY descriptor。

---

### Stage 4：测试时服务传统 2D-3D 定位

测试时不要让 Loc-GS 接管整个 pipeline。建议：

1. query descriptor 仍用 STDLoc/SuperPoint original feature；
2. 3D descriptor 主干仍用 PLY/STDLoc descriptor；
3. Loc-GS 输出 locability / matchability / ambiguity / visibility 作为 score；
4. matching 不 hard filter，只做 priority ordering；
5. PnP/RANSAC 使用原始冗余 matches；
6. learned branch 只在 confidence 足够时启用，否则回退 baseline。

这和 README 里目前 baseline-preserving recipe 是一致的：默认 STDLoc sampled/PLY descriptor backbone，通过 train/calibration-val split 选择 learned branches，而不是手选 test 结果。([GitHub](https://github.com/Arthurshen926/Loc-GS))

---

## 5. 现在最值得补的实现

### 5.1 把 fine-tuning 明确拆成两个 checkpoint

当前训练流程容易混在一起。建议在代码和论文里固定为：

```text
base checkpoint:
  descriptor/detector reconstruction + same-view matching

ft checkpoint:
  initialized from base
  freeze most modules
  train locability + tiny adapter + hard-negative ranking
```

不要再让一个 checkpoint 同时承担：

- descriptor reconstruction；
- PnP loss；
- locability prior；
- residual descriptor；
- landmark replacement；
- geometry training。

这会让实验不可解释。

---

### 5.2 新增 query-conditioned matchability head

输入不是 landmark 静态 score，而是 pair-level 特征：

$$
\phi(q_i, X_j)
$$

包括：

- descriptor similarity；
- top1-top2 margin；
- locability；
- visibility；
- depth consistency；
- detector saliency；
- landmark TP/FP prior；
- query keypoint score；
- 3D point observability；
- local ambiguity score。

输出：

$$
P(y_{ij}=1)
$$

这个 head 不需要复杂，MLP 两层就够。  
关键是它是 **query-conditioned**，不是 per-landmark static prior。

---

### 5.3 把 hard filtering 改成 PROSAC-style ordering

不要删除 matches。把 calibrated matchability 用作 RANSAC sampling priority：

```text
matches sorted by:
descriptor_similarity + λ * logit(matchability)
```

然后：

- top matches 更容易被采样；
- 低分 matches 仍保留；
- 避免 hard filtering 误删 weak true positives。

如果 OpenCV PnP-RANSAC 不方便改采样，可以先实现简化版：

```text
run PnP on top N1, top N2, top N3 match prefixes
choose best by inlier count + reprojection + dense verification
```

这比直接删 matches 更稳。

---

### 5.4 把 locability 定义改清楚

现在 locability 容易和 observability / detector / alpha / depth 混在一起。论文里必须定死：

- **observability**：几何上这个点是否能约束 pose；
- **locability**：这个点/区域是否倾向于产生 PnP useful matches；
- **matchability**：query-conditioned 2D-3D pair 是否是正确内点。

三者层级是：

$$
observability \rightarrow locability \rightarrow matchability
$$

但最终优化目标应该是 matchability，不是单独 observability 或 detector saliency。

---

## 6. 可投稿实验应该怎么设计？

现在最大的问题是 claim 没有被结果支撑。所以实验设计必须从“证明全场景超过 STDLoc”改成“证明可匹配性引导微调的具体价值”。

### 主表一：固定配置，不按 test 调参

必须有一个固定 recipe：

| Method | Descriptor | Landmark bank | Learned signal | Test-time usage |
|---|---|---|---|---|
| STDLoc | PLY | sampled | none | baseline |
| Base Loc-GS | rendered/base | sampled | reconstruction only | no fine-tune |
| Loc-GS-FT | PLY + base | sampled | matchability fine-tune | priority/fallback |
| SP-GS2 full | learned | learned/filtered | all modules | ablation negative |

主表必须用同一套参数跑五个 Cambridge scenes。  
不能按 scene 手动切 MNN、KC、residual、filtering。

### 主表二：calibration-val 合法选择

如果你要用 branch selection，必须严格写：

1. train split 内部划 calibration-val；
2. calibration-val 选择 branch / confidence threshold；
3. test split 不再调；
4. 报告所有 scene 的结果。

`select_cambridge_branch.py` 和 `select_cambridge_query_pose.py` 已经具备这条线的基础。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/select_cambridge_query_pose.py))

但这条线更像 system contribution，不是 feature reconstruction contribution。论文里要分清楚。

---

### 中间指标必须补齐

否则评审会说：最终 pose 有波动，不知道为什么。

建议每个 scene 都报：

| Metric | 目的 |
|---|---|
| descriptor cosine | 基础重建质量 |
| same-view top1@1px/3px | 自匹配能力 |
| cross-view inlier@5px | 可匹配性 |
| hard-negative AUC | 区分重复结构能力 |
| RANSAC inlier ratio | PnP 输入质量 |
| 2D/3D coverage | 是否空间分布足够 |
| sparse pose R5 | 初始位姿质量 |
| dense refine delta | refinement 是救回还是拉坏 |
| final R5 / median te | 最终定位 |

论文 claim 应该从这些指标链条推出来：

> feature reconstruction → matchability improves → PnP inliers improve → pose improves.

现在缺的是中间两步的稳定证据。

---

### 消融实验要围绕“受约束微调”

不要再做一堆杂项开关。建议只做：

| Ablation | 要回答的问题 |
|---|---|
| base reconstruction only | 单纯重建是否足够 |
| + locability only | 可靠性地图是否有用 |
| + hard-negative ranking | 是否能处理重复结构 |
| + trust-region residual | 小幅 descriptor 微调是否安全 |
| + PROSAC ordering | soft priority 是否优于 hard filter |
| full hard filtering | 作为负结果，解释为什么不用 |

---

### 失败分析必须保留

GreatCourt 不要藏。它应该成为论文的一部分：

- 展示 GreatCourt 中 repeated structures 的 false positive；
- 可视化 top false-positive landmarks；
- 对比 hard filtering 为什么删掉 true positives；
- 对比 static landmark matchability 与 query-conditioned matchability；
- 解释为什么 full SP-GS2 退化。

这会把你的负结果变成论文价值。

---

## 7. 现阶段最现实的投稿目标

现在不要把目标定成：

> 全 Cambridge 全面超过 STDLoc。

更现实、也更容易被接受的是：

> **提出一种受约束的可匹配性引导微调框架，证明它能提升 matchability / PnP input quality，并在部分 Cambridge 场景与 macro recall 上超过 STDLoc，同时揭示 naive learned descriptor replacement 和 hard landmark filtering 的失败原因。**

如果要冲更强投稿，建议最低实验目标是：

| 目标 | 要求 |
|---|---|
| macro R5 | 高于 STDLoc 至少 2-4 个百分点 |
| mean median te | 不明显高于 STDLoc，最好低于 STDLoc |
| OldHospital / Shop / StMary | 保持正向 |
| GreatCourt | 至少不大幅退化；最好 fallback 到 baseline |
| SP-GS2 full | 放 ablation，不做主方法 |
| 参数 | 固定 recipe 或 calibration-val 合法选择 |
| 中间指标 | 证明 matchability / inlier ratio 提升 |

如果做不到 GreatCourt 提升，至少要做到：

> learned branch 不伤害 GreatCourt。

这就需要 baseline-preserving fallback。

---

## 8. 最终建议：把论文方法重写成这个版本

### 方法名

**Loc-GS-FT: Matchability-Guided Fine-Tuning of Gaussian Feature Fields**

### 核心贡献

1. **Feature reconstruction is not enough.**  
   单纯重建 SuperPoint descriptor 不等于定位好。

2. **Self-localization statistics from 3DGS rendering.**  
   利用训练视角和 3DGS novel-view rendering，提前估计 visibility、depth consistency、observability、TP/FP、hard negatives。

3. **Constrained matchability-guided fine-tuning.**  
   在基础重建 checkpoint 上，只做小幅 adapter / locability / hard-negative ranking 微调，避免破坏 STDLoc/PLY descriptor。

4. **Baseline-preserving 2D-3D localization.**  
   测试时仍使用传统 2D-3D matching + PnP/RANSAC，Loc-GS 只提供 priority / confidence / fallback，不强行 replacement。

### 一句话 claim

> **Loc-GS-FT uses training-time self-localization over a Gaussian feature field to learn matchability-aware reliability cues, improving 2D-3D localization without replacing the strong STDLoc/PLY descriptor backbone.**

这比“learned descriptor replacement”稳得多，也更符合你现在所有实验反馈。

---

## 9. 直接回答你的问题

**如果不是纯 learned descriptor replacement，而是在基础特征重建基础上做微调，可行吗？**  
可行，而且这是现在最合理的方向。但必须是**受约束微调**：freeze 大部分基础 descriptor，训练 locability / matchability / tiny adapter，使用 hard-negative ranking，而不是强 PnP loss 或 full residual replacement。

**当前实现是不是已经如此？**  
部分是。当前已经有 SuperPoint reconstruction、same-view matching、noisy-pose differentiable PnP、locability prior、TP/FP calibration、branch/query selection。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py)) 但它还没有被严格组织成“base checkpoint → constrained FT checkpoint → baseline-preserving test”的清晰流程。

**是不是相当于在地图里先自匹配、自定位，利用 3DGS 新视角渲染提前算特征选择指标？**  
是，这个表述可以作为方法直觉。但论文里要更精确地说：这是在 train/calibration views 上用 GT geometry 生成 matchability supervision，不能写成 test query 的真实先验，也不能暗示使用测试真值。

**如何达到可投稿状态？**  
停止把 SP-GS2 full 当主方法；重构为 Loc-GS-FT：基础重建、离线自定位统计、query-conditioned hard-negative matchability、受约束微调、soft priority/fallback 定位。实验上用固定 recipe 或 calibration-val 选择，补齐中间 matchability / inlier / coverage 指标，并把 SP-GS2 full、KC strict、hard filtering 作为负结果消融。

这条路还值得继续，但要换一个更稳的主 claim：**不是“我们学到更好的 descriptor 替代 STDLoc”，而是“我们用定位可匹配性约束 Gaussian 特征场微调，使其更好地服务传统 2D-3D PnP 定位”。**

---

**Sources:**

- [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
- [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)