# 科研项目定位引导优化

**User:** Anonymous  
**Created:** 5/16/2026 0:57:35  
**Updated:** 5/16/2026 1:54:10  
**Exported:** 5/16/2026 2:12:07  
**Link:** [https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb](https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb)  

## Prompt:
2026/5/16 00:57:34

我现在正在做一个科研项目，从STDLoc这篇工作出发，想要在其基础之上实现一个让定位信息反馈给特征重建的过程，重点是面向视觉定位/定位引导的特征重建，并在其中凸显出特征选择的思想，最终获得定位精度提升（包括位姿精度和召回率）；虽然位姿估计方式仍然是传统的特征匹配+几何pnp，但想要尽可能地体现端到端的概念，主要是定位端到端影响重建。此外，在这个主题之上还是想要尽量解决实际科学问题，特别是baseline在真实场景下遇到的问题，而不仅仅是数据集上的精度涨点，并且用第一性原理去提出解决方案。最终本工作的目标是要投稿顶刊的，所以不光是逻辑合理，方法有创新和精度有提升这三条条件都需要满足。
具体git仓库位于https://github.com/Arthurshen926/Loc-GS，目前已经实现了一部分内容，仓库里也有很多进展记录，但我觉得距离我的目标还有不少距离，精度指标也遇到瓶颈，无法形成投稿级的强实验证据来支撑论文claim。请帮我分析一下下列几项是否合理以及需要如何进一步修改并优化：我的目标和动机、目前的方法架构、代码实现等等

## Response:
2026/5/16 00:57:36

我会先对齐 STDLoc 的核心设定、Loc-GS 仓库现状与你的投稿目标，再从问题定义、方法闭环、代码实现和实验可证性四方面诊断。重点判断哪些 claim 能被真实场景问题支撑，哪些模块需要保留、重构或优先优化。

## Response:
2026/5/16 00:57:48

我现在看到核心矛盾在于：最强提升来自 self-map quality gate/重建选择，而非残差描述子、SceneMatchNet 或 detector 的稳定增益。因此主线应更精确地定位为“定位反馈驱动的地图/特征场选择与轻量重建”，避免被质疑只是模型选择或调参。

## Response:
2026/5/16 01:02:56

> **整理最终答案中的引用**
>
> 我需要检查训练线是否涉及深度问题，虽然不一定必要。我最终的版本需要结合之前的引文，为文献添加清晰的出处，只使用网络引用，而非文件搜索结果。也需要注意根据实际结果整理中文答案。避免重复实验，确保引用可靠，且适应最终审稿提交。
>
> 已思考 8m 31s

## Response:
2026/5/16 01:06:07

我基于公开仓库、README/docs、STDLoc 论文和 2025-2026 年相关 3DGS localization 工作做了审阅；我没有在你的机器环境里复现实验，所以代码层面结论主要是“架构/实现/实验设计诊断”，不是运行后的 debug 报告。

## 1. 总体判断

你的方向是合理的，而且有潜力形成顶刊级问题，但当前主线需要收束。现在最稳的论文 claim 不应是“端到端定位”，而应是：

> **Localization-in-the-loop reconstruction：让自定位反馈反过来决定 Gaussian feature field 中哪些特征值得重建、增强、保留或降权，从而得到更适合 PnP + dense refinement 的定位特征场。**

原因是：STDLoc 本身已经是一个很完整的 sparse-to-dense localization pipeline，包含 Feature Gaussian、matching-oriented Gaussian sampling、scene-specific detector、sparse PnP 和 dense feature refinement。它的论文已经明确把“Feature Gaussian + landmark sampling + scene-specific detector + sparse-to-dense refinement”作为核心贡献。你的工作若只是再加一个 detector、matcher 或 reranker，很容易被认为是 STDLoc 的工程增量。([arXiv](https://arxiv.org/html/2503.19358v1))

真正有新意的切入点是：**STDLoc 的 Feature Gaussian 重建目标和最终定位目标并不一致**。STDLoc 用 feature-field distillation 学一个可渲染特征场，但 PnP 实际需要的是少量“可见、稳定、几何分布好、区分性强、低歧义”的 2D-3D inlier，而不是平均意义上 L1/L2 重建得好的 dense feature map。这个 mismatch 才是你可以从第一性原理展开的科学问题。

## 2. 你的当前目标：合理，但需要换一种表述

你现在的目标里有三件事：定位反馈、特征重建、特征选择。三者都合理，但要避免混成“我做了一个端到端 localization 系统”。因为你的最终位姿估计仍然是 feature matching + PnP + 几何验证，这本身没有问题；问题在于不要把它过度包装成 inference-time end-to-end。

建议改成下面这个逻辑：

**原始表述：**  
“定位信息反馈给特征重建，体现端到端概念。”

**建议表述：**  
“我们提出一种 task-driven / localization-guided Gaussian feature reconstruction 框架。在 mapping / reconstruction 阶段，通过虚拟自定位 episode 评估每个 Gaussian、每个 landmark 和每类特征残差对 PnP 成功率的贡献，并将该反馈用于特征选择、locability 更新和 descriptor trust-region residual 学习。推理阶段仍保持单一路径的传统几何 PnP，因此兼具几何可解释性与定位任务驱动的重建闭环。”

这个表述有两个好处：第一，它不会被审稿人抓住“你并没有端到端反传到 OpenCV PnP”这一点；第二，它把创新点放在 reconstruction-time feedback，而不是 inference-time trick。你的 README 里其实已经在往这个方向走：当前主线写的是 STDLoc/PLY descriptor backbone → virtual self-localization feedback → protected tiny descriptor residual → single PROSAC PnP path → STDLoc-style dense refinement。([GitHub](https://github.com/Arthurshen926/Loc-GS))

## 3. 当前方法架构：保留什么，重构什么

你目前仓库里有几条线：

一条是 **native STDLoc parity / soft prior / LFF descriptor export**。这条最干净，因为它保持 STDLoc native backend，`rho=0` 时能回到源 STDLoc evaluator，审稿风险低。但目前提升很小，例如 soft locability prior 和 LFF descriptor export 的提升幅度接近 conservative ablation，而不是足以支撑主论文 claim 的强结果。你的文档里记录了 native STDLoc parity、soft prior、LFF descriptor export 和 unified gated LFF 的对比，其中 native-backed median improvement 有，但 R5/R2 并不总是同步提升。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

另一条是 **self-map quality gate**。这条数值证据最强：文档记录 native macro 从 `9.127cm / 0.156deg, R5 0.3712, R2 0.1331` 到 self-map quality gate 的 `8.682cm / 0.151deg, R5 0.4144, R2 0.1459`，R5 提升约 4.32 个百分点。这个结果值得保留，但不能简单说成“方法主结果”，否则容易被质疑是 scene-level model selection 或 validation selection。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

我的建议是：

**论文主方法不要直接叫 quality gate。**  
把它重构为 **Self-localization Feedback for Feature-Field Selection and Reconstruction**。也就是说，self-map episode 不只是“选择哪个 branch”，而是产生训练信号：哪些 Gaussian 是稳定 inlier，哪些 descriptor residual 会制造 hard negative，哪些区域虽然重建得好但对 PnP 不贡献，哪些区域因为 floater / alpha blending / occlusion 应该被降权。

这样 quality gate 可以作为“pre-deployment map validation”或“upper-bound diagnostic”，而不是唯一核心方法。主方法应当是单一路径、固定 recipe、无 per-query branch、无测试集选择。

## 4. 第一性原理：把“特征选择”从经验 saliency 改成 PnP 信息选择

视觉定位最终不是为了重建漂亮特征，而是为了让 PnP 有足够多、足够准、几何分布足够好的 inlier。PnP 的稳定性近似由下式控制：

$$
H = \sum_i w_i J_i^\top \Omega_i J_i
$$

其中 $J_i$ 是第 $i$ 个 2D-3D correspondence 对位姿的投影雅可比，$w_i$ 是该 match 成为 inlier 的概率或置信度，$\Omega_i$ 是测量噪声权重。定位失败通常不是因为所有特征都差，而是因为被选中的 matches 在以下维度上失败：

1. inlier 概率低；
2. 2D/3D 几何分布退化；
3. descriptor 在局部或重复结构中歧义高；
4. Gaussian 受 floater、遮挡或 alpha-blending bias 影响；
5. detector 选到了对 PnP 信息量低的点。

STDLoc 已经承认 3DGS floaters 会导致 dense feature map indistinct，从而造成 dense matching 失败；最新 ULF-Loc 也明确指出 3DGS 的 alpha-blending feature optimization 会引入点特征 bias，使直接 2D-3D feature matching 发生 mismatch。([arXiv](https://arxiv.org/html/2503.19358v1)) ([arXiv](https://arxiv.org/abs/2605.04730))

所以你的“特征选择”不应只是 top-k locability、matchability 或 detector heatmap，而应定义为：

$$
s_i = P(\text{Gaussian } i \text{ should be trusted for localization})
$$

并由以下因素共同决定：

$$
s_i = f(
\text{visibility},
\text{multi-view inlier rate},
\text{descriptor ambiguity},
\text{depth/alpha consistency},
\text{PnP information gain},
\text{self-localization residual}
)
$$

这才是顶刊级科学问题：**不是选择看起来 salient 的特征，而是选择能改善几何定位条件数的特征。**

## 5. 建议的新方法主线

我建议把方法压缩成一个清晰闭环：

### 5.1 Stage A：STDLoc-compatible base field

保留 STDLoc/PLY descriptor backbone，不要一开始就完全替换 descriptor。你的 `gated_residual_descriptor_blend` 方向是对的：让新特征作为受限残差进入已有 STDLoc descriptor bank，而不是自由生成全新 descriptor。代码里当前 residual blend 的思想就是 `ply_desc + alpha * gate * residual`，这对稳定性和 reviewer 信任都很重要。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/localization/descriptor_blend.py))

推荐公式：

$$
d_i = \text{normalize}(d_i^{STDLoc} + \alpha \rho s_i \Delta d_i)
$$

其中：

- $d_i^{STDLoc}$：原始 STDLoc/PLY descriptor；
- $\Delta d_i$：Loc-GS 学到的 localization residual；
- $s_i$：per-Gaussian feature selector；
- $\rho$：map-level self-localization reliability；
- $\alpha$：小 trust-region cap，例如 0.03-0.10。

这比“换 backbone”更安全，也更符合你想保留传统几何 PnP 的目标。

### 5.2 Stage B：Virtual self-localization episodes

不要只用 feature reconstruction loss。你需要系统性生成 self-localization episodes：

- train-view perturbation；
- interpolated novel views；
- rendered query-like views；
- illumination / blur / exposure perturbation；
- low-overlap / weak-texture stress views；
- floater-heavy / alpha-uncertain regions。

每个 episode 运行完整 localization：detector → matching → PROSAC/RANSAC PnP → dense refinement。你的代码已经用 OpenCV PROSAC / MAGSAC 方向做 PnP，这是合理的，因为推理阶段可以保持传统几何路径。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/localization/hybrid_localizer.py))

关键是不要只存最终 pose error，而要记录中间监督：

```text
query_id
view_pose
keypoint_id
matched_gaussian_id
descriptor_score
detector_score
is_pnp_inlier
reprojection_error
depth_consistency
visibility
local_ambiguity
pose_jacobian_information
dense_refinement_success
```

这会变成后续 feature selection 和 descriptor residual 的训练数据。

### 5.3 Stage C：Pose-information-aware feature selector

把现在的 locability / matchability / reliability boost 改成一个更 principled selector：

$$
\mathcal{L}_{select}
=
-\log \det \left(
\epsilon I + \sum_i s_i p_i J_i^\top \Omega_i J_i
\right)
+
\lambda_b \left(\frac{1}{N}\sum_i s_i - \tau\right)^2
+
\lambda_r \mathcal{L}_{redundancy}
+
\lambda_c \mathcal{L}_{coverage}
$$

含义是：选择的 Gaussian 不仅要 matchable，还要让 PnP Hessian 条件更好、空间覆盖更均匀、不要集中在重复纹理或同一平面。

你现在的 `locability_budget_loss` 主要约束平均选择比例，`descriptor_ambiguity_loss` 主要约束 descriptor 相似度；这些有用，但不够。它们还没有直接约束 PnP 几何条件、空间覆盖和视角稳定性。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/losses/landmark_selection.py))

### 5.4 Stage D：Hard-negative residual reconstruction

当前瓶颈不太像“PROSAC 排序不够好”，因为你的文档里 oracle ordering headroom 很小。这说明只改排序、阈值、score fusion 不会带来大突破；必须让 feature field 产生新的正确 matches，或者减少原本会进入 PnP 的 hard negatives。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

建议加入 hard-negative descriptor loss：

$$
\mathcal{L}_{HN}
=
\sum_{(q,i^+,i^-)}
\max(0, m + sim(q,d_{i^-}) - sim(q,d_{i^+}))
$$

其中 $i^-$ 不是随机负样本，而是 self-localization episode 中真实导致 PnP 失败、reprojection error 大、但 descriptor 相似度高的 Gaussian。这比普通 contrastive loss 更符合你的科学动机。

### 5.5 Stage E：Self-map reliability 不要只用固定 10cm 阈值

你当前 soft prior 里有类似 `rho = sigmoid((center_cm - median_te_cm) / temperature_cm)` 的设计，默认中心是 10cm。这个想法直观，但顶刊审稿人可能质疑它是 dataset-specific calibration。代码里确实有这种 scene-level reliability weight。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/stdloc_native/soft_prior.py))

建议改为 baseline-relative reliability：

$$
\rho =
\sigma
\left(
\frac{
E_{base}^{self} - E_{candidate}^{self} - \delta
}{
T
}
\right)
\cdot
\sigma
\left(
\frac{
R_{candidate}^{self}@5 - R_{base}^{self}@5
}{
T_R
}
\right)
$$

也就是说，不问“candidate 是否低于 10cm”，而问“candidate 是否比同一场景的 STDLoc base map 更好”。这样不同尺度、不同难度场景之间更公平，也更能解释为什么 GreatCourt/KingsCollege 应该被降权，而 ShopFacade/StMarysChurch 可以信任。

## 6. 对你当前代码实现的具体建议

### 6.1 保留 native STDLoc parity 作为第一锚点

这个非常重要。你的 README 已经强调 source-of-truth STDLoc evaluator、native-backed soft prior、single PROSAC PnP path。继续保留它，不要退回到 hybrid evaluator 作为主 baseline。([GitHub](https://github.com/Arthurshen926/Loc-GS))

每一个新模块都应该有三种开关状态：

```text
rho = 0        -> 完全复现 native STDLoc
rho > 0, Δd=0 -> 只测 locability / selector
rho > 0, Δd>0 -> 完整 Loc-GS
```

这会让 reviewer 很难质疑 baseline fairness。

### 6.2 把 monolithic scripts 拆成“反馈数据-训练-导出-评估”四层

现在 `train_cambridge_hybrid.py` 和 `eval_cambridge_hybrid.py` 承载了太多功能，参数也很多，包括 localization descriptor source、residual alpha、feedback detector、locability target、selfmap reliability、calibration 等。长远看，这会让实验不可审计，也会增加“调参堆结果”的印象。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

建议拆成：

```text
loc_gs/feedback/
  episodes.py          # 生成 self-localization episodes
  labels.py            # inlier / hard-negative / information labels
  selector.py          # pose-information selector
  objectives.py        # feedback reconstruction losses

loc_gs/export/
  native_lff_map.py    # 导出 STDLoc-compatible map
  manifest.py          # 记录所有超参、checkpoint、数据 split

loc_gs/eval/
  native_stdloc.py
  locgs_single_path.py
  aggregate.py
```

每次实验必须生成一个 `manifest.json`：

```json
{
  "baseline_map": "...",
  "candidate_map": "...",
  "train_views": [...],
  "selfmap_views": [...],
  "test_views": [...],
  "feedback_recipe": "...",
  "alpha": 0.03,
  "rho_formula": "...",
  "selector_loss": "...",
  "no_test_pose_used_for_training_or_selection": true
}
```

### 6.3 明确防止 test leakage

你的 Cambridge pipeline 文档提到会加载 train poses 和 `dataset_test.txt` query poses。评估当然需要 test pose，但 self-map selection / calibration / reliability 不能使用任何 test query pose、test image 或由 test query 生成的 synthetic view。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/cambridge_hybrid_localization.md))

建议在代码里硬性 assert：

```python
assert set(selfmap_image_names).isdisjoint(set(test_image_names))
assert set(calibration_pose_ids).isdisjoint(set(test_pose_ids))
```

并在论文 appendix 给出 split 表。顶刊审稿人会非常关注这一点，尤其是你使用 self-map quality gate 的情况下。

### 6.4 SceneMatchNet 暂时不要作为主方法

你文档里已经记录 SceneMatchNet 在 refined full Cambridge 上没有 beat residual default。我的建议是把它保留为 diagnostic 或 future work，不要让主线复杂化。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

若要继续做 SceneMatchNet，必须改成 listwise / query-conditioned ranking，而不是固定 pair score 加权：

```text
输入：一个 query keypoint 的 top-K candidate Gaussians
输出：K 个候选的 calibrated inlier probability
loss：listwise cross entropy / LambdaRank / differentiable AP
约束：score 只能作为 weak prior，不允许替代 descriptor similarity
```

否则它很容易在某些 scene 提升、全局均值下降。

## 7. 实验应该怎么补强

你目前最缺的不是再调一个 eval-time 参数，而是形成“科学问题-失败机制-反馈修复-可解释提升”的证据链。

### 7.1 必做主表

至少要有：

```text
STDLoc official / local native parity
STDLoc + your reproduced map
Loc-GS no feedback reconstruction
Loc-GS + locability selector only
Loc-GS + descriptor residual only
Loc-GS + hard-negative feedback
Loc-GS + pose-information selector
Loc-GS full
Self-map quality gate diagnostic
```

指标：

```text
median translation / rotation
R@10cm,5deg
R@5cm,5deg
R@2cm,2deg
sparse inlier ratio
dense refinement success rate
failed-query recovery rate
runtime / memory / map size
```

只报 median pose error 和 recall 不够。你需要证明 feedback 改善了 match quality 和 failure mode，而不是偶然改善 pose。

### 7.2 必做 failure taxonomy

把 STDLoc baseline 的失败分为：

```text
A. sparse stage insufficient matches
B. sparse stage wrong dominant cluster
C. PnP inlier geometry degenerate
D. dense refinement drift
E. floater / occlusion induced mismatch
F. repetitive structure / descriptor ambiguity
G. illumination / exposure shift
```

然后报告 Loc-GS 对每类失败的 recovery rate。STDLoc 自己提到 weak-texture、illumination、floater 等问题；你的工作要证明解决的是这些真实问题，而不是只在 Cambridge 平均指标上涨点。([arXiv](https://arxiv.org/html/2503.19358v1))

### 7.3 必做跨数据集或真实场景验证

Cambridge 上的提升很有价值，但顶刊会要求泛化。建议至少补一个：

- 7-Scenes：验证室内弱纹理；
- Replica / 12-Scenes：验证 dense/geometry 质量；
- 自采真实场景：光照变化、动态物体、低纹理墙面、玻璃/反光、重复结构；
- Aachen / RobotCar-style day-night：如果你的 pipeline 能承受 outdoor large-scale，则价值更高。

注意：现在 2026 年相关工作已经很强。SplatLoc 已经强调 unbiased scene-specific descriptors 和 landmark selection，ULF-Loc 进一步把 alpha-blending bias、keypoint-consensus sampling 和 local geometric consistency verification 作为核心点；GSVisLoc 也强调不修改/不重训 3DGS 的 generalizable localization。你的 novelty 必须落在“定位反馈驱动重建与选择”上，而不是“3DGS + descriptor + PnP”本身。([DiffWind](https://zju3dv.github.io/splatloc/)) ([arXiv](https://arxiv.org/abs/2605.04730)) ([arXiv](https://arxiv.org/abs/2508.18242))

## 8. 投稿级 claim 应该怎么写

我建议最终论文 claim 控制在三条：

**Claim 1：目标错配。**  
Feature Gaussian 的平均 feature reconstruction loss 与 PnP localization objective 存在错配。可渲染、可重建的特征不一定可定位，尤其在 alpha-blending bias、floaters、重复结构和弱纹理区域。

**Claim 2：定位反馈驱动的特征选择。**  
Loc-GS 通过 virtual self-localization episodes 估计 per-Gaussian / per-landmark 的 inlier probability、pose information gain 和 hard-negative risk，并将这些信号反馈到 Gaussian feature reconstruction，使 feature field 在 mapping 阶段就面向定位优化。

**Claim 3：传统几何推理仍然受益。**  
推理阶段不依赖 per-query branch selection，不替换 PnP，不引入不可解释 pose regression；只使用一个被反馈优化过的 STDLoc-compatible feature field，仍走 feature matching + PROSAC/RANSAC PnP + dense refinement，但 pose accuracy 和 recall 提升。

这个叙事比“端到端”更强，也更不容易被反驳。

## 9. 当前版本的取舍建议

**保留：**

- native STDLoc parity；
- protected descriptor residual；
- self-map reliability；
- single-path PROSAC/PnP；
- quality gate 作为 diagnostic / validation protocol；
- hard-negative branch 作为训练数据来源。

**重构：**

- 把 self-map quality gate 从 scene-level branch selector 改成 training-time feedback label generator；
- 把 locability 从简单 score 变成 pose-information-aware selector；
- 把 descriptor residual 的监督从 feature reconstruction 改成 inlier/hard-negative contrastive；
- 把 fixed 10cm reliability threshold 改成 baseline-relative reliability；
- 把 SceneMatchNet 从主线降级为 ablation，除非 full split 稳定提升。

**暂时放弃作为主线：**

- eval-time 多分支选择；
- LoFTR/RoMa 直接替换 dense matching；
- 单纯调 residual alpha、PnP threshold、score weight；
- 没有 full-split 正结果的 feedback detector / SceneMatchNet。

## 10. 最直接的下一轮实现路线

按优先级做：

1. **生成 self-localization feedback bank**：每个 scene 采样固定 train/interpolated/perturbed views，保存 match、inlier、hard negative、pose Jacobian、dense refinement success。
2. **实现 pose-information selector**：不要只用 locability/matchability，加入 $J^\top J$、coverage、ambiguity、visibility。
3. **训练 residual descriptor with hard negatives**：只对高 $s_i$、高 $\rho$ 的 Gaussian 更新残差，保持 STDLoc trust region。
4. **导出一个 native STDLoc-compatible map**：不走 hybrid evaluator，不做 per-query branch。
5. **跑固定 full Cambridge + 至少一个额外数据集/真实场景**：报告主表、failure taxonomy、ablation、runtime。
6. **把 quality gate 保留为“self-map validation upper bound”**：证明 feedback 信号有价值，但主结果尽量来自单一训练 recipe 的 map，而不是选择多个 branch。

我的最终判断是：**项目方向值得继续，但论文主线必须从“我在 STDLoc 上加了定位反馈模块”升级为“我证明并解决了 feature-field reconstruction 与 geometric localization objective 的错配”。** 当前最有投稿潜力的不是 eval-time reranking 或 branch selection，而是把 self-localization feedback 变成可解释、可导出、STDLoc-compatible 的特征选择与残差重建机制。

---

**Sources:**

- [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/html/2503.19358v1)
- [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/loc_gs_lff_scenematch_mainline_20260512.md)
- [
    SplatLoc
  ](https://zju3dv.github.io/splatloc/)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)