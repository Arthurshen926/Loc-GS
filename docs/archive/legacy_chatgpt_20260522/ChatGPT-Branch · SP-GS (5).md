# Branch · SP-GS

**User:** Anonymous  
**Created:** 5/10/2026 13:58:36  
**Updated:** 5/14/2026 1:07:22  
**Exported:** 5/14/2026 1:21:58  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/6a001e8b-8548-83ea-a2a5-efdfcf08f370)  

## Prompt:
2026/5/13 23:44:53

继续推进了一下，git已更新，但感觉推进的存在很多问题，指标根本没有多少提升，问题出在哪里？多轮进展总结如下：
1.
已按 `ChatGPT-Branch · SP-GS (4).md` 把主线重新收紧到专家认可的方向：**STDLoc/PLY descriptor trust region + 虚拟自定位反馈 + query-conditioned SceneMatchNet + 单路径 PROSAC**。旧的 branch selector、static prior、LoFTR replacement、硬筛点路线没有删除，但已明确归档为诊断/负消融，避免继续误导主线判断。 

主要改动：

- 新主线文档：[docs/loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md)
- 旧路线归档说明：[docs/archive/static_prior_branch_selection_20260512.md](/root/Loc-GS/docs/archive/static_prior_branch_selection_20260512.md)
- 新增 SceneMatchNet：[loc_gs/localization/scene_matcher.py](/root/Loc-GS/loc_gs/localization/scene_matcher.py)
- calibration 现在可导出 pair-level 自定位监督：[loc_gs/scripts/calibrate_landmark_matchability.py](/root/Loc-GS/loc_gs/scripts/calibrate_landmark_matchability.py)
- 新增 matcher 训练脚本：[loc_gs/scripts/train_scene_matcher.py](/root/Loc-GS/loc_gs/scripts/train_scene_matcher.py)
- eval 已支持 `--scene_matcher_path --scene_matcher_topk --scene_matcher_weight`，执行的是 top-K pair rerank 后的**单路径 PROSAC**：[loc_gs/scripts/eval_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py)
- README 已改成新主线命令和 3GPU launcher 流程：[README.md](/root/Loc-GS/README.md)

验证结果：

- 单元测试：`170 passed, 16 warnings`
- `git diff --check`：通过
- 三卡真实 checkpoint smoke：GreatCourt / OldHospital / ShopFacade 都成功生成 pair cache，每场 512 pairs，正样本比例 0.5
- ShopFacade smoke 训练成功生成 checkpoint：`output/scenematch/lff_pair_smoke_20260512/ShopFacade/best.pt`
- ShopFacade q1 eval 成功加载 SceneMatchNet 并跑通单路径 PROSAC：dense `1.26cm / 0.056deg`，这只是链路 smoke，不作为精度 claim

现在项目逻辑链已经从“学一个更好的 descriptor”修正为“把定位反馈转成 detector/matcher 监督，优先采样 PnP-useful correspondences”。下一步应直接跑 full pair generation + per-scene SceneMatchNet training + full Cambridge eval，比较 `covisibility_prosac` 和 `scene_matcher_prosac` 的 median 与 R@10/5/2。
2.
已经完成这一轮全量验证和主线收敛。结论很明确：当前最可投稿、最自洽的路线是 **LFF / SceneMatchNet 作为单路径 PROSAC 排序信号**，不是 top-K 多候选替换，也不是 descriptor replacement。

核心 full Cambridge 结果：

| 方法 | Median | R10 | R5 | R2 |
|---|---:|---:|---:|---:|
| `covisibility_prosac` baseline | 12.572cm | 0.486 | 0.278 | 0.085 |
| `SceneMatch top1, beta=0.35` | **12.399cm** | **0.492** | **0.283** | **0.091** |
| `SceneMatch top1, beta=0.75` | 12.386cm | 0.491 | 0.282 | 0.084 |

所以主线固定为：`scene_matcher_topk=1, scene_matcher_weight=0.35`。`beta=0.75` 只略降 median，但牺牲 R2/R5，不作为主线。

我也系统跑了负/中性消融：`topK=4` 候选替换、logit zscore、balanced/raw SceneMatchNet、global SceneMatchNet、MAGSAC、coverage filtering、query detector score blending，都没有比 `top1 beta=0.35` 给出更干净的 full 证据。相关结论已写入 [docs/loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md:1)，README 的推荐命令也改成 top1 主线了：[README.md](/root/Loc-GS/README.md:145)。

代码层面补了/整理了：
- SceneMatchNet 训练与推理主线：[scene_matcher.py](/root/Loc-GS/loc_gs/localization/scene_matcher.py:1)、[train_scene_matcher.py](/root/Loc-GS/loc_gs/scripts/train_scene_matcher.py:1)
- full/q80 多 GPU launcher 支持 SceneMatch、MAGSAC/coverage 诊断 recipe，并把负向 recipe 归档标记：[launch_cambridge_reliability_eval.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_reliability_eval.py:1)
- eval 增加 logit norm、query-score 等消融开关，但默认不启用：[eval_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py:1)

验证：
- `git diff --check` 通过
- `python -m pytest tests`：172 passed
- 直接跑全仓 `pytest` 会扫进 `third_party/Mask2Former`，因为缺 `detectron2` 在 collection 阶段失败；项目自身 tests 已通过。
- 当前没有遗留训练/评估进程，3 张 3090 已释放。

实话说：现在已经能支撑“定位反馈影响重建/特征场选择，并带来 full Cambridge 小幅稳定收益”的 claim，但离“强 SOTA 级显著领先”还不够。下一步真正该攻的是 **KingsCollege / GreatCourt 的正例覆盖和 pair label 质量**，不是继续调 beta。
3.
继续推进了一轮，结论比较硬：**新尝试没有刷新精度，反而帮助我们把主线收窄了**。

我做了这些改动：

- 在 [calibrate_landmark_matchability.py](/root/Loc-GS/loc_gs/scripts/calibrate_landmark_matchability.py:213) 增加 `--scene_match_pair_train_fraction`，支持把 SceneMatchNet pair cache 分成 train/rendered 两部分。
- 加了预算满后跳过 rendered/train pair 标签计算的短路，减少无效几何标注开销。
- 在 [launch_cambridge_matchability_calibration.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_matchability_calibration.py:27) 接入新参数。
- 更新 [README.md](/root/Loc-GS/README.md:176)，明确默认主线仍保留 v1 train-view pair cache，rendered pair mixing 只作为 ablation。
- 更新 [loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md:97)，把 2026-05-13 的 rendered self-localization 诊断结果写入文档。

关键 q80 结果：

| 方案 | Median cm | R10 | R5 | R2 |
|---|---:|---:|---:|---:|
| v1 主线 train-view SceneMatchNet | **9.074** | **0.6175** | **0.3800** | **0.1200** |
| v3 50/50 train/render | 9.369 | 0.6025 | 0.3800 | 0.1150 |
| v4 75/25 train/render | 9.207 | 0.6100 | 0.3750 | 0.1175 |
| v5 75/25 + balanced positive | 9.281 | 0.6100 | 0.3775 | 0.1075 |
| v4 calibrated matchability, no SceneMatchNet | 9.437 | 0.6100 | 0.3775 | 0.1150 |

所以当前判断是：**rendered/self-localization 视角确实应该保留在故事里，但不能直接混入当前 pair classifier 或 landmark score 当主线**。最强、最干净的投稿主线仍然是 v1：基础 STDLoc/PLY 特征场 + train-view 定位反馈监督的 SceneMatchNet + 单路径 PROSAC rerank。

验证也过了：

- `git diff --check`
- `python -m pytest tests`：172 passed, 16 warnings
- 当前没有残留训练/评估进程，3 张 3090 已空闲。
4.
结论先说清楚：**当前 paper-facing 主线还没有超过 STDLoc 论文指标**。

按现在能正式比较的 full Cambridge single-path 主线结果，也就是 `scene_matcher_prosac, beta=0.35, scene_matcher_topk=1`：

| 方法 | Median cm | Median deg | R@10 | R@5 | R@2 |
|---|---:|---:|---:|---:|---:|
| 当前 Loc-GS 主线 | 12.399 | 0.159 | 0.492 | 0.283 | 0.091 |
| 本地 STDLoc 复现 | 9.127 | 0.156 | 0.576 | 0.371 | 0.133 |
| STDLoc 论文 Cambridge Avg | 10.1 | 0.14 | 论文主表未报这些严格 recall | - | - |

所以现在相对 **STDLoc 论文平均值**还差大约 **+2.30cm / +0.019°**，平移误差约差 **22.8%**。相对我们自己的 **本地 STDLoc 复现**还差 **+3.27cm / +0.003°**，严格 recall 也落后：R@10 差 **8.4 个百分点**，R@5 差 **8.8 个百分点**，R@2 差 **4.2 个百分点**。

注意：之前那个 `9.078cm / 0.153°` 的结果确实略好于本地 STDLoc 复现和 STDLoc 论文平均值，但它属于历史 guarded selector / branch 结果，已经不适合作为当前投稿主线；当前主线文档里正式保留的是 single-path SceneMatchNet + PROSAC。q80 子集上的 `9.074cm / 0.111°` 也只能算诊断信号，不能替代 full Cambridge 结论。对应记录在 [loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md:81) 和历史对比在 [loc_gs_related_work_and_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_related_work_and_mainline_20260512.md:161)。STDLoc 论文 Cambridge 表格给的是 `10.1cm / 0.14°`，并且它的 Cambridge ablation recall 是宽阈值 `5m/10°`、`2m/5°`，不是我们现在常用的 cm 级 recall，见 STDLoc 论文 Table 2 和 Table 5：<https://arxiv.org/html/2503.19358v1>。

关于 `(3).md` 和 `(4).md` 的建议是否全部落实：**没有全部落实**。更准确地说，主线方向已经切过去了，但最能支撑 claim 的几个关键闭环还没完成。

已经落实的部分：

- 放弃“learned descriptor replacement”作为主 claim，改成保留 `STDLoc/PLY descriptor backbone`，用 localization feedback 学可靠性和匹配优先级。
- 当前 pipeline 已整理为 `STDLoc/PLY bank -> self-localization labels -> SceneMatchNet -> single-path PROSAC -> dense refinement`，README 也已对齐，见 [README.md](/root/Loc-GS/README.md:148)。
- `SceneMatchNet` 已实现，见 [scene_matcher.py](/root/Loc-GS/loc_gs/localization/scene_matcher.py:89)。
- pair-level self-localization label cache 已接入，包含 `--scene_match_pair_train_fraction`，见 [calibrate_landmark_matchability.py](/root/Loc-GS/loc_gs/scripts/calibrate_landmark_matchability.py:210)。
- eval 里已经支持 `scene_matcher_topk=1` 的单路径 PROSAC，不再把 top-k 多候选 pose 当主线，见 [eval_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py:3121)。
- rendered / perturbed self-localization 也做了诊断实验，但结果显示直接混入 pair training 会退化，所以目前只作为分析和覆盖性证据。

还没有真正落实或没有验证充分的部分：

- **重建过程中的闭环反馈还不够强**：目前更多是 self-localization 产生标签后用于 pair scorer / rerank，还不是“定位反馈实时影响 Gaussian feature field 重建”的完整闭环。
- **PnP-inlier scene-specific detector** 还没有成为主线核心模块。现在更多复用 STDLoc detector / landmark bank，Loc-GS 自己的 detector 还没有证明能超过 STDLoc detector。
- **小幅 descriptor residual fine-tuning** 还只是保守 ablation 方向，不是已验证有效的最终方案。
- matchability、locality、observability 等思想有部分进入 prior / diagnostics，但还没有形成一个能稳定提升 full Cambridge 的统一模块。
- DIM / LoFTR / LightGlue 路线已经作为诊断和外部 matcher ablation 接入过，但还没有转化成稳定的 hard-negative teacher 或主线训练收益。
- 最关键的一点：**full Cambridge 固定 recipe 还没有达到投稿级 SOTA 指标**。

所以目前状态可以概括为：**路线已经从旧瓶颈转到专家认可的新主线，但完成度大概只是“框架和实验闭环已搭好，核心 claim 还未被 full benchmark 证明”。** 下一步最值得攻的是 PnP-inlier scene-specific detector 和把 self-localization feedback 真正灌回 feature field / locability / residual 训练，而不是继续只调 SceneMatchNet rerank。
5.
你说得对，**feature field 也必须调整**。我上一条回答不够完整，过分聚焦在 SceneMatchNet / PROSAC 当前结果上了。

更准确的判断是：

当前专家最终计划不是“只在 STDLoc/PLY 上加一个 matcher rerank”，而是要把旧的 `HybridFeatureGaussian` 改造成 **Localization Feedback Field, LFF**：

```text
STDLoc/PLY frozen descriptor d0
+ tiny descriptor residual r, alpha <= 0.03~0.05
+ per-Gaussian locability
+ PnP-feedback scene-specific detector
+ query-conditioned pairwise matchability
```

也就是说，feature field 的职责要变：

- 旧职责：重建/替代 SuperPoint 或 STDLoc descriptor field。
- 新职责：在不破坏 STDLoc/PLY descriptor 几何的前提下，学习“哪些 Gaussian / 哪些 query keypoint / 哪些 2D-3D pair 对定位有用”。

目前代码里确实已经有一些零件：`HybridFeatureGaussian` 有 `locability_logit`，eval 支持 `ply_loc`、`hybrid_ply_gated_residual`，训练里也有 locability / residual regularization 相关 loss。但**这还不等于专家文档要求的 feature field 主线已经完成**。现在主线实际仍然更像：

```text
PLY descriptor frozen
+ STDLoc detector/landmark bank
+ SceneMatchNet rerank
+ PROSAC
```

缺的关键是：

1. 还没有真正把旧 hybrid field 简化/收敛成 LFF 主结构。
2. 还没有把 PnP inlier/outlier feedback 系统性反灌到 feature field 训练。
3. 还没有真正实现/验证 Loc-GS 自己的 PnP-feedback scene-specific detector。
4. descriptor residual 目前只是可选接口/消融方向，不是已验证主方法。
5. 当前 `SuperPointOutputHead` detector 仍偏“重建 SuperPoint logits”，不是“定位有用 keypoint detector”。

所以我修正上一条结论：**不是只有 matcher 没到位，feature field 调整本身就是当前最大缺口之一，而且它应该和 detector 一起成为下一步主线。**

下一步应该直接推进这个方向：把当前主线从 “SceneMatchNet post-hoc rerank” 升级为 “LFF 训练闭环”：self-localization 产生 PnP TP/FP → 训练 locability、PnP-feedback detector、pairwise matcher、可选 tiny residual → 再用同一套固定 recipe 跑 full Cambridge。
6.
已继续推进并把专家建议里的 **feature field 主线改造**补上了。现在不是只做 SceneMatchNet rerank，而是形成了代码级 LFF 闭环：

- `PLY/STDLoc descriptor trust region + tiny residual`：训练侧新增 `localization_descriptor_source=hybrid_ply_gated_residual`，默认 `alpha=0.03`。
- `PnP-feedback scene-specific detector`：训练时用 self-localization 的 `query_inlier_score` 监督 query detector，并保存到 checkpoint 的 `feedback_detector_state_dict`。
- eval/calibration 支持 `--query_detector feedback`，可以直接用 Loc-GS 自己训练出来的 detector。
- SceneMatchNet pair cache 新增 `query_score`，训练 matcher 时使用 5 个 scalar channel，eval 时也会把 query detector score 传进去。
- 新增 `lff_feedback_prosac` eval recipe，用 feedback detector + protected residual descriptor + 单路径 PROSAC，不做 branch selection。

主要改动在：
[train_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/train_cambridge_hybrid.py:386)、
[eval_cambridge_hybrid.py](/root/Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py:2183)、
[calibrate_landmark_matchability.py](/root/Loc-GS/loc_gs/scripts/calibrate_landmark_matchability.py:193)、
[train_scene_matcher.py](/root/Loc-GS/loc_gs/scripts/train_scene_matcher.py:1)、
[launch_cambridge_reliability_eval.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_reliability_eval.py:18)，文档也更新到了
[loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md:1)。

验证结果：

- `python -m pytest -q tests`：**178 passed**。
- `git diff --check`：通过。
- 真实 ShopFacade smoke training 跑通，checkpoint 确认包含 `feedback_detector_state_dict`。
- feedback calibration 跑通，pair cache 确认包含 `query_score`。
- SceneMatchNet smoke training 跑通，checkpoint 使用 `scalar_dim=5`。
- feedback detector + protected residual + SceneMatchNet 的 1-query eval smoke 跑通。

有效性结论要分开说：

代码/架构层面：**专家建议的主线闭环已经落实并验证可运行**。

指标层面：**还不能宣布有效或 SOTA**。目前 full Cambridge 的正式指标仍然是旧主线 `scene_matcher_prosac` 的 `12.399cm / 0.159°`，新 LFF feedback detector 只做了 smoke 验证，还没有跑完整 Cambridge 表。因此现在的投稿级结论是：实现已补齐，下一步必须跑 `reliability_recipe` 训练后接 `lff_feedback_prosac` 和 `scene_matcher_prosac` 全量评估，才能判断它是否真正超过 STDLoc。
7.
已继续推进并完成这一轮全量验证。核心结论先说清楚：**目前最稳的主线不是 feedback detector，也不是 SceneMatchNet，而是 `lff_residual_prosac`，即保留 STDLoc detector/PLY descriptor backbone，只使用受限 residual feature field。** 但它的提升还很小，尚未达到“完全超越 SOTA”的投稿级结果。

**本轮完成**
- 跑完 `*_lff_refined_20260513` 五场景全量评估。
- 补跑并验证 SceneMatchNet：
  - 生成 50/50 train/rendered self-localization pair cache。
  - 每场景 400k pairs，balanced batch 训练。
  - 评估 `scene_matcher_weight=0.35` 和 `0.1`。
- 更新文档，避免后续继续误判 SceneMatch/feedback detector 是当前主线：
  - [README.md](/root/Loc-GS/README.md)
  - [docs/loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md)

**关键结果，五场景均值**
```text
covisibility_prosac      12.603cm / 0.159deg   R10 0.485  R5 0.278  R2 0.086
lff_residual_prosac      12.456cm / 0.156deg   R10 0.485  R5 0.284  R2 0.085
lff_feedback_prosac      13.211cm / 0.165deg   R10 0.470  R5 0.281  R2 0.078
oracle_prosac            12.461cm / 0.154deg   R10 0.493  R5 0.283  R2 0.084
SceneMatch beta=0.35     12.766cm / 0.159deg   R10 0.477  R5 0.275  R2 0.086
SceneMatch beta=0.10     12.761cm / 0.159deg   R10 0.487  R5 0.278  R2 0.082
```

**判断**
- `lff_residual_prosac` 是当前唯一有全量正收益的 learned/LFF 组件。
- `feedback detector` 已经修复低分辨率崩溃问题，但 full split 仍不稳，尤其 GreatCourt/Kings 拖后腿。
- `oracle_prosac` 上界很小，说明只做排序/selector 不够，瓶颈更像是候选匹配质量、descriptor bank 或 dense refinement。
- SceneMatchNet 已落实并验证，但当前标签和融合方式不达标；有局部信号，比如 GreatCourt sparse R5、Shop R10，但全量均值不如 residual。

**验证**
- `git diff --check` 通过。
- `pytest -q tests` 通过：`182 passed, 16 warnings`。
- 当前没有训练/评估进程在跑，三张 3090 已空闲。

下一步最合理的方向不是继续调 SceneMatch 权重，而是重做 matcher/候选生成监督：让 pair label 更接近真实 PnP inlier sequence，或者引入更强的外部 matcher/teacher 做 hard-negative mining；同时保留 `lff_residual_prosac` 作为当前默认主线。
8.
继续推进了一轮全量验证，结论比较硬：**目前架构不是靠继续换 matcher 或调 eval 小参数就能达到预期 SOTA 指标**。这轮 3 张 3090 都用上跑完了几组全 Cambridge full-test。

核心结果如下：

| Variant | Mean cm/deg | R10 | R5 | R2 | 结论 |
|---|---:|---:|---:|---:|---|
| `lff_residual_prosac` 默认 | **12.456 / 0.156** | 0.485 | 0.284 | 0.085 | 当前主线 |
| residual `alpha=0.05` | 12.551 / 0.157 | 0.488 | **0.286** | 0.084 | R5 小涨，但 median/R2 退 |
| q-score 0.10 | 12.481 / 0.159 | 0.488 | 0.281 | 0.085 | sparse/R10 有用，dense R5 不行 |
| q-score 0.25 | 12.509 / 0.163 | **0.493** | 0.277 | 0.088 | 更像粗召回 ablation |
| reproj 4px | 13.204 / 0.161 | 0.469 | 0.274 | **0.090** | 高精度少量涨，整体负 |
| PLY+LoFTR | 12.912 / 0.178 | 0.475 | 0.255 | 0.068 | 不能替换 |
| residual+LoFTR | 13.327 / 0.178 | 0.479 | 0.257 | 0.066 | 归档诊断 |

我也补了代码和文档：

- 在 [launch_cambridge_reliability_eval.py](/root/Loc-GS/loc_gs/scripts/launch_cambridge_reliability_eval.py) 加了 `lff_residual_prosac_loftr` 诊断 recipe，并归档为非主线。
- 给 eval launcher 加了全局 `--hybrid_residual_alpha_max`，方便固定全场景测试 residual trust-region，而不是手改代码。
- 在 [docs/loc_gs_lff_scenematch_mainline_20260512.md](/root/Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md) 和 [README.md](/root/Loc-GS/README.md) 记录了这轮全量结论。

判断：**当前最诚实主线仍是 protected residual LFF + single-path PROSAC**。`alpha=0.05` 可以作为 recall ablation，但不足以支撑“已超 SOTA”。LoFTR/DIM 不能做直接替换，应该改成训练期 hard-negative / verifier 标签来源。

验证：

- `git diff --check` 通过。
- `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests`：`185 passed, 16 warnings`。
- 全量 `pytest -q` 仍被 vendored `Mask2Former` 的 `detectron2` 缺失卡住，这是第三方 collection 问题，不是本轮改动引入。

下一步要想真正冲指标，应该停止 eval-time 小调，转向 **LFF-v2 训练**：用 LoFTR/RoMa 只挖 hard negatives 和 verifier labels，扩大 localization batch，训练 query-conditioned reliability / residual，而不是推理时多跑一条匹配路径。

## Response:
2026/5/13 23:44:58

我会先核对最新 LFF/SceneMatch 主线与结果，再判断问题是监督实现、候选质量还是方法上限。重点会放在停止 eval 小调、重构训练闭环和固定可投稿架构。

## Response:
2026/5/13 23:45:04

进一步看，oracle 只重排已有 sparse matches，所以上限小说明候选池或本地 STDLoc parity 本身有问题。优先查 covisibility/protected 为何掉到 12.6cm，而不是继续调 rerank。

## Response:
2026/5/13 23:48:21

> **Browsing STDLoc, SuperGlue, and related resources**
>
> [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/html/2503.19358v1)
>
> **Addressing problem areas and proposing architecture**
>
> I need to address a few key problem areas: identifying whether the issue is a bug or a route problem and explaining why the metrics are small. I'll cite the current code and documentation, emphasizing the lack of branch selection in the current default. 
> 
> For the architecture plan, here are some possible issues:
> - The candidate pool and landmark bank are poor; current methods don't improve much because the oracle ranking is limited. We likely need better candidate generation and descriptor refinement.
> - SceneMatchNet is facing issues with imbalanced labels, poor pair selection, and limited context in training, among other limitations like feature selection and ranking loss.
> 
> I'll propose improving these areas.
>
> **Addressing issues with learning and model performance**
>
> It seems there’s a potential issue where the model might learn redundant cosine/margin behavior, especially if the oracle is too small. The feedback detector might be problematic due to sparse targets from self-localization and a mismatch between keypoint distributions. We should focus on increasing recall while keeping coverage. For residual descriptors, they seem too small because they weren't trained against hard negatives or with a proper pair ranking. Improvements here might be minimal.
>
> **Evaluating models, improvement metrics, and citations**
>
> I’m considering whether to consult literature for GSFeatLoc/ULF—maybe it's unnecessary. I should mention related models like SplatLoc, SuperGlue, and STDLoc and ensure I'm citing both the current code and documentation for the matcher/training. I might compute relative improvements manually, with a few example figures like the residual improvements. User-supplied values don't need external citations, but if referring to the docs tables, I should cite. Lastly, I might review the "hybrid_ply_gated_residual" code for more clarity.
>
> **Citing relevant methods, computing improvements, and inspecting code**
>
> [Loc-GS/loc_gs/scripts/eval_cambridge_hybrid.py at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/eval_cambridge_hybrid.py)
>
> **Investigating detector loss in training**
>
> I’m wondering whether I should review the code around detector loss in the "train_cambridge" sections. It might shed some light on how this part of the training process works, and it's worth checking. Just need to focus on those lines specifically to understand the impact they might have!
>
> **Searching for open lines in train_cambridge detector loss**
>
> [Loc-GS/loc_gs/scripts/train_cambridge_hybrid.py at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py)
>
> **Considering raw data importance**
>
> It seems like the issue is that "find" isn't working because the input might be too low. Hmm, I wonder if I need to use raw data instead. That could address the problem and offer better results, so it's something I should try to ensure everything is processed correctly!
>
> **Troubleshooting raw data issue**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/loc_gs/scripts/train_cambridge_hybrid.py)
>
> **Finalizing Chinese response details**
>
> I’m thinking it’s okay to reference the raw lines, but I should avoid using them unless they need to be cited for clarity. For code examples, I'll cite using webpage references, especially when training arguments appear. Now, my Chinese translation will mention three key issues: firstly, no implementation bug; secondly, focus on bottlenecks like candidate generation and improvement; lastly, identify issues in SceneMatchNet training labels.
>
> **Revising model and feature strategies**
>
> I need to focus on improving label quality, implementing group ranking loss, and using a listwise objective with better diversity among top-K candidates. For the feedback detector, its target might be too noisy and low precision, so I'll adjust how it's trained. Plus, using landmarks and better
>
> **Addressing baseline drift and alignment to STDLoc**
>
> It looks like I need to emphasize that we shouldn’t claim improvements against STDLoc until the native pipeline matches the STDLoc paper exactly. If there's a 3.5cm gap in the baseline (12.6 vs 9.1), any learned components on top will inherit that error. The main point here is: "First, fix the parity gap before chasing SOTA."
>
> **Identifying causes and proposing parity lock for STDLoc**
>
> There are many potential causes here: feature scaling, detector settings, NMS thresholds, descriptor sources, and more. To address this, I propose a "STDLoc parity lock": ensuring exact alignment by running the same settings (detector, features, dense refinement, etc.) and comparing outputs within tolerance. If any gap remains, comparisons are invalid.
> 
> Regarding architecture, I might simplify HybridFeatureGaussian to LFF-Point, freezing certain elements and adjusting layers for better alignment, while adding locability and detector compatibility.
>
> **Updating ranking methodology and refining candidate generation**
>
> For replacing Top1, I’ll implement listwise ranking using softmax cross-entropy, hinge ranking, and calibrated probability. I’ll include more context features like 2D/3D distribution, rank, local density, visibility, and descriptor ambiguity, but if the oracle is small, this may not be sufficient. For candidate generation, expanding keypoints and improving detectors should help, and I’ll track recall accuracy. Dense refinement should be carefully separated if it worsens performance.
>
> **Addressing pipeline issues and priorities**
>
> I should clarify that improving localization-feedback (LFF) could provide slight gains over the current control. However, the user’s goal is to surpass STDLoc, so I need to address the parity and candidate quality issues first. The current oracle size might be limiting, as existing matches are already saturated. If LFF-v2 doesn't surpass STDLoc, I may need to reconsider the dataset or adjust the thesis. It’s important to communicate that the current results still fall short of STDLoc and the baseline.
>
> **Clarifying citations and results**
>
> To ensure everything is properly referenced, I’ll cite the relevant documents for current results, the README for mainline, and the SceneMatchNet code for features and independent BCE. I also need to mention calibration label and STDLoc detector/sampling. It’s important to properly reference citations related to internet facts and avoid raw links. The structured answer will avoid using phrases like “if you want” and stay direct.
>
> Thought for 10m 31s

## Response:
2026/5/13 23:55:29

问题不在“你还没调到好参数”，而在**当前主线优化的对象不够对**。现在最强的全量结果已经说明：

1. `lff_residual_prosac` 相对内部 control 有一点点正收益；
2. `feedback detector` 和 `SceneMatchNet` 已经实现，但目前没有带来稳定 full Cambridge 提升；
3. `oracle_prosac` 的提升也很小，说明**只给现有 sparse matches 排序，已经没有多少上限**；
4. 当前 native Loc-GS control 本身明显弱于你自己的本地 STDLoc 复现，所以还存在一个更根本的问题：**你的当前主线没有先恢复到 STDLoc parity。**

最新文档里也已经写得很清楚：`lff_residual_prosac` 是当前唯一有正向 full-split 均值的 learned 组件；`feedback detector` 和 `SceneMatchNet` 还只能算诊断；`oracle_prosac` 的 small headroom 表明缺口不可能靠排序/selector 单独补上。当前 full Cambridge 表里，`covisibility_prosac` 是 `12.603cm / R5 0.278`，`lff_residual_prosac` 是 `12.456cm / R5 0.284`，`oracle_prosac` 也只有 `12.461cm / R5 0.283`，这三个数非常接近。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

所以直白地说：**目前不是“SceneMatchNet 不够强”这么简单，而是现有候选匹配池、native STDLoc parity、detector 质量、dense refinement 这几个底层环节没有支撑起主 claim。**

---

## 1. 最大异常：你的 current control 已经不是强 STDLoc

现在必须先盯住这个事实：

| 方法 | Mean median | R5 |
|---|---:|---:|
| 本地 STDLoc 复现 | 9.127cm | 0.371 |
| 当前 `covisibility_prosac` control | 12.603cm | 0.278 |
| 当前 `lff_residual_prosac` | 12.456cm | 0.284 |

如果当前方法是建立在 STDLoc/PLY backbone 上，那么 **native Loc-GS control 至少应该接近本地 STDLoc 复现**。但现在 control 比本地 STDLoc 差了约 `3.48cm`，R5 差了约 `9.3` 个百分点。这个 gap 比 LFF residual 带来的 `0.147cm / 0.6% R5` 增益大太多。

这意味着：**现在的主问题不是 LFF 学得不够，而是当前 native evaluation/localizer pipeline 没有复现 STDLoc 的强 baseline。** 在这个基础上做 residual、detector、SceneMatchNet，最多只能在一个弱 control 上做小修小补。

你需要先做 **STDLoc parity audit**。把 native `eval_cambridge_hybrid.py` 的 `ply_loc + sampled bank + STDLoc detector + STDLoc dense refine` 跑到和本地 STDLoc 复现基本一致，再谈 Loc-GS 的提升。STDLoc 原论文的核心就是 Feature Gaussian、matching-oriented Gaussian sampling、scene-specific detector、sparse-to-dense relocalization；如果你的 native pipeline 没有复现这个骨架，后面所有 LFF 组件的结论都会被污染。([arXiv](https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com))

---

## 2. `oracle_prosac` 小上限说明：排序不是瓶颈

你现在做了非常关键的 oracle 实验：用 GT reprojection error 对已有 sparse matches 排序。结果是：

| Variant | Median | R5 | Sparse R5 |
|---|---:|---:|---:|
| `covisibility_prosac` | 12.603 | 0.278 | 0.195 |
| `lff_residual_prosac` | 12.456 | 0.284 | 0.196 |
| `oracle_prosac` | 12.461 | 0.283 | 0.203 |

这说明很硬：**在“现有 sparse candidate matches”这个集合里，即使用 GT 排序，最终也涨不动。** 文档也明确解释：pure sorting / selector improvements cannot explain the missing SOTA gap。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

这直接否定了继续调这些东西作为主线：

- SceneMatchNet 权重；
- PROSAC score fusion；
- MAGSAC / MSAC；
- query score blending；
- coverage filtering；
- top1/topK rerank；
- calibrated prior。

这些都只是**已有候选的排序/筛选**。如果 oracle 排序也只有这点提升，说明真正瓶颈在排序之前或排序之后：

1. **候选匹配池里正确 match 不够多**；
2. **query detector 选的点不对**；
3. **3D landmark descriptor / residual descriptor 仍有 bias**；
4. **dense refinement 没有把 sparse pose 拉回来**；
5. **native pipeline 和 STDLoc 原实现存在 parity 差异。**

---

## 3. SceneMatchNet 当前为什么涨不动？

SceneMatchNet 的当前形式是一个独立 pair MLP：输入是 query descriptor 与 landmark descriptor 的 element-wise product、absolute difference，再加 cosine、margin、landmark prior、calibrated prior、query score 等少量 scalar，输出这个 pair 是否 PnP-useful。代码里默认 feature 就是 `q*d`、`|q-d|` 加几个 scalar；网络是 3 层左右的小 MLP。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/localization/scene_matcher.py))

这有三个问题。

**第一，它是 independent pair scorer，不是 query-level matcher。**  
它看不到同一个 query keypoint 的 top-K candidate 之间的相互竞争，也看不到一组 matches 的空间一致性。GreatCourt/Kings 的核心问题恰恰是重复结构中多个候选都很像，单个 pair 看起来都合理。SuperGlue 这类方法的关键不是简单 pair MLP，而是联合推断 correspondences 并拒绝 non-matchable points；它用 attention / optimal transport 来建模匹配集合之间的关系。([arXiv](https://arxiv.org/abs/1911.11763?utm_source=chatgpt.com))

**第二，它只做 top1 或弱 rerank，本身无法修正“top1 错但 topK 有正解”的情况。**  
你试过 topK candidate replacement 会退化，但这不代表 topK 思路错，而是当前 matcher 没有能力可靠地在 topK 内选对。需要先测 `topK oracle candidate recall`，而不是直接上 learned topK。

**第三，label 可能不贴近真实 test-time PnP sequence。**  
你用 self-localization / rendered labels 生成 pair 监督，但 full test 表明 50/50 train/render pair mixing 反而落后 residual default。文档里也承认当前 pair-label formulation needs work。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_lff_scenematch_mainline_20260512.md)) 这通常意味着训练数据分布、正负样本定义、候选生成方式和真实 evaluation pipeline 不一致。

所以 SceneMatchNet 不应该继续按现在的 “pair MLP + beta” 调权重。它需要升级成 **query-level / listwise matcher**，否则很难解决重复结构。

---

## 4. feedback detector 为什么不稳？

你现在已经实现了 PnP-feedback scene-specific detector：训练时用 self-localization 的 `query_inlier_score` 监督 query detector，eval 也支持 `--query_detector feedback`。文档中说它已经能跑 full-res canvas，但 full Cambridge 结果是 `13.211cm / R5 0.281`，弱于 `lff_residual_prosac`。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

原因大概率不是 detector 模块“没训练够”，而是 target 有问题：

> **PnP inlier score 是结果标签，不是稳定的 detector 标签。**

它依赖当前 matcher、当前 PnP、当前候选池。如果当前 matcher 本身在 GreatCourt/Kings 有系统性 false positives，那么 detector 会学到一套被当前 pipeline 污染的 keypoint preference。

真正的 scene-specific detector 需要三种 target 分开建模：

1. **Projected landmark target**：STDLoc-style，把 sampled landmark 投影到训练图像作为基础正样本；
2. **PnP-inlier target**：只有几何正确且最终进入 inlier set 的点才加权增强；
3. **Hard-negative target**：descriptor 相似度高但 reprojection error 大、或导致 wrong pose 的点要显式压低。

当前 feedback detector 更像只用了第 2 类，而且第 2 类还受到当前 PnP pipeline 噪声影响。所以它不稳是合理的。

STDLoc 明确把 scene-specific detector 作为 robust initial pose 的核心模块，而不是简单重建 SuperPoint detector。([arXiv](https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com)) 你的 detector 应该沿这个方向重做，而不是直接用 PnP inlier score 当唯一监督。

---

## 5. LFF residual 为什么只有小提升？

当前 residual 路径是：

```text
PLY descriptor → render ply descriptor
hybrid descriptor → gated residual
descriptor = normalize(ply_descriptor + alpha * gate * residual)
```

`render_hybrid_superpoint()` 里支持 `hybrid_ply_gated_residual`，用 rendered `locability` 作为 gate，并用 `hybrid_residual_alpha_max` 限制 residual 幅度。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/train_cambridge_hybrid.py))

这条线小幅正向，说明 trust-region residual 是目前最合理的 learned feature 用法。但它提升小，原因也很明显：

1. `alpha=0.03` 很保守，不会大幅改变 descriptor；
2. 当前 residual 主要靠 reconstruction / alignment，不是强 hard-negative descriptor learning；
3. oracle 排序小上限说明很多 query 的正确候选可能不在当前 top1 / candidate pool 里；
4. 如果 native STDLoc control 本身偏弱，residual 很难补回 3cm 级 gap。

所以 `lff_residual_prosac` 可以保留，但它不是能单独冲 SOTA 的模块。它更像一个**安全的 feature-field fine-tuning 证据**。

---

## 6. 技术路线是不是错了？

不完全错，但你现在的路线要再收窄。

**错的路线：**

> 学一个 complex hybrid feature field / SceneMatchNet / feedback detector，然后期待它自然超过 STDLoc。

**正确路线：**

> 先恢复 STDLoc parity；然后证明 localization feedback 在 detector、candidate generation、descriptor residual 中的某一个具体环节带来可解释提升。

现在的结果最强烈地指向这个判断：

> **只在“已有候选 matches 的排序”上做文章，上限很小；要真正涨，需要改善候选生成、detector、descriptor bank 或 dense refinement。**

文档已经得出类似结论：oracle ordering 的 small gap 表明未来收益必须来自 better candidate generation, descriptors, or dense refinement，而不是只靠 PROSAC priority。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_lff_scenematch_mainline_20260512.md))

---

## 7. 下一步不要继续做什么？

我建议立即停止把这些作为主线：

1. **继续调 SceneMatchNet beta。**  
   已经证明 `0.35`、`0.10` 都不如 residual；再调只是局部搜索。

2. **继续 eval-time 小参数。**  
   q-score、alpha=0.05、reproj 4px、MAGSAC、coverage filtering、LoFTR replacement都没有给出 clean full gain。

3. **继续把 rendered pair mixing 直接混进训练。**  
   50/50 train/render refined full run 落后 residual default，说明 label formulation 有问题，不是数据量不够。

4. **继续把 feedback detector 当主方法。**  
   它现在是有价值 ablation，不是主线。

5. **继续 claim “localization feedback already improves full Cambridge”。**  
   目前只能说 residual 有小正收益，但还不能支撑强 claim。

---

## 8. 现在最该做的五件事

### P0：做 native STDLoc parity audit

这是最高优先级。

你需要把 native Loc-GS pipeline 跑到接近本地 STDLoc 复现：

```text
native ply_loc + STDLoc detector + sampled bank + STDLoc dense refine
≈ 9.127cm / R5 0.371
```

如果 native control 仍是 `12.6cm / R5 0.278`，就不要继续训练 LFF。先查：

| 模块 | 要对齐的内容 |
|---|---|
| query detector | keypoint count、NMS radius、score threshold、feature resolution |
| descriptor sampling | yx/xy、align_corners、normalization、original/resized feature |
| landmark bank | sampled ids、loc_feature、visibility、3D coordinates |
| matching | topK、mutual / margin、score threshold、prior 权重 |
| PnP | solver、threshold、iterations、confidence、refine |
| dense refine | iterations、render stride、feature source、acceptance rule |
| image preprocessing | resize、intrinsics scaling、color/gray pipeline |
| map variant | GreatCourt / StMarys map path是否一致 |

做一个 query-by-query parity CSV：

```text
query_id
STDLoc sparse te/re
native sparse te/re
STDLoc dense te/re
native dense te/re
#matches
#inliers
inlier ratio
median reproj
dense delta
```

找出 native pipeline 从哪一步掉下去。

**如果这一步不做，后面所有 LFF 结论都不可靠。**

---

### P1：做 candidate-pool oracle，不只是排序 oracle

你现在的 `oracle_prosac` 只是对已有 top1 sparse matches 排序。下一步要测：

```text
top1 candidate oracle
top4 candidate oracle
top8 candidate oracle
top16 candidate oracle
```

对每个 query keypoint，先取 topK landmark candidates，然后用 GT reprojection error 在 topK 里选最好的 candidate，再送 PnP。

这能回答关键问题：

| 结果 | 结论 |
|---|---|
| top1 oracle 小，topK oracle 大 | 当前 top1 candidate generation 是瓶颈，需要 listwise matcher |
| topK oracle 也小 | landmark bank / descriptor / detector 本身不够 |
| sparse oracle 大，dense final 小 | dense refinement 是瓶颈 |
| detector oracle 大 | query detector 是瓶颈 |

现在只知道“排序已有 matches 没上限”，还不知道“topK candidate pool 有没有上限”。

---

### P2：重做 SceneMatchNet：从 pair MLP 改成 listwise matcher

SceneMatchNet-v2 不要再独立打分每个 pair。应该按 query keypoint 的 topK candidate set 训练：

```text
query keypoint i
topK landmarks {j1...jK}
features for all K candidates
network outputs K logits + dustbin
loss = cross-entropy over correct candidate / dustbin
```

建议结构：

```text
candidate features:
  q*d, |q-d|, cosine, margin, landmark prior, locability,
  visibility, xyz, view-count, covisibility group, query score

model:
  candidate MLP
  + DeepSets / lightweight self-attention over K candidates
  + softmax over K + dustbin
```

为什么必须 listwise？因为 GreatCourt 的重复结构问题是“多个候选之间谁更合理”，不是“单个 pair 是否看起来像正样本”。独立 pair MLP 很难解决这个问题。

训练目标也要改：

- 不要强行 0.5 positive ratio；
- 用真实 topK 候选分布；
- hard negative 必须是 high-similarity wrong landmark；
- label 是 topK 中 GT reprojection 最小且小于阈值的 candidate；
- 如果 topK 中没有正样本，训练 dustbin / no-match。

推理：

```text
for each query keypoint:
  topK candidate logits
  either choose top1 candidate
  or emit all candidates with score
PROSAC uses score
```

这才是 SceneMatchNet 该做的事情。

---

### P3：重做 detector target，而不是继续 feedback detector weight

Detector-v2 的 target 应该是混合目标：

$$
y(u)=
w_1 y_{\text{projected landmark}}
+
w_2 y_{\text{PnP inlier}}
-
w_3 y_{\text{hard negative}}
$$

而不是只用当前 PnP pipeline 产生的 `query_inlier_score`。

实验设计必须固定 PLY descriptor 和 native localizer，只换 detector：

| Detector | 目的 |
|---|---|
| STDLoc detector | baseline |
| SuperPoint detector | weak baseline |
| projected-landmark detector | 复现 STDLoc-style |
| projected + PnP-inlier detector | 看定位反馈是否有用 |
| projected + PnP-inlier - hard-negative detector | 主候选 |

先看 sparse stage：

```text
#keypoints
#matches
#inliers
inlier ratio
sparse median
sparse R5
```

如果 sparse 不涨，dense final 不可能稳定涨。

---

### P4：把 LoFTR/RoMa 只用于 hard-negative mining，不用于 eval 替换

你已经验证 LoFTR replacement 退化。这个结论合理。下一步不要再推理时跑 LoFTR，而是用它做训练标签：

- LoFTR/RoMa 能匹到的 query-rendered correspondences；
- 用 rendered depth lift 到 3D；
- 和 STDLoc candidate 对比，找出 high-similarity but wrong 的 negative；
- 生成 detector hard-negative mask；
- 生成 SceneMatchNet topK labels。

外部 matcher是 teacher/verifier，不是 inference backbone。

---

### P5：只有在 parity + detector/listwise matcher 后，再谈 residual fine-tuning

当前 residual 是唯一正向 learned 组件，但很小。未来 residual 应该由 hard-negative ranking 驱动：

$$
d_j = \mathrm{normalize}(d_j^0 + \alpha r_j), \quad \alpha \le 0.03
$$

loss：

```text
positive descriptor closer
hard-negative descriptor farther
trust-region loss strong
```

不要让 residual 自己成为主 claim。它应该服务于 detector/listwise matcher发现的问题。

---

## 9. 架构应该调整成什么样？

建议最终架构变成：

```text
                STDLoc / PLY Gaussian map
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   Frozen descriptor d0              LFF residual r, gate α
        │                                 │
        └────────── descriptor trust region ──────────┐
                                                       │
Train-time self-localization rehearsal                 │
real train views + rendered/perturbed views             │
matching + PnP + GT pose                                │
→ projected landmark positives                          │
→ PnP inlier positives                                  │
→ high-sim false negatives                              │
                                                       │
        ┌──────────────────────────────────────────────┘
        │
        ├── Detector-v2:
        │     predicts PnP-useful query keypoints
        │
        ├── SceneMatchNet-v2:
        │     listwise topK query-conditioned matcher
        │
        └── Residual descriptor adapter:
              tiny trust-region correction

Test:
query image
→ Detector-v2 keypoints
→ PLY/residual descriptor topK candidate generation
→ SceneMatchNet-v2 listwise candidate scoring
→ single-path PROSAC / MAGSAC PnP
→ STDLoc-style dense refinement
```

这里没有 branch selection，没有 hard landmark deletion，没有 LoFTR replacement，没有 pure learned descriptor replacement。

---

## 10. 当前论文 claim 怎么调整？

现在不能写：

> Loc-GS 已经超过 STDLoc / SOTA。

也不能写：

> SceneMatchNet / feedback detector 带来稳定提升。

现在可以诚实写成：

> 当前实验表明，直接 descriptor replacement、static landmark prior、hard landmark selection、pairwise MLP reranking 都不足以稳定提升 STDLoc-style localization；唯一稳定正向的是受限 descriptor residual，但提升很小。进一步 oracle 结果显示，对已有 sparse matches 的排序上限很小，说明主要瓶颈在 candidate generation、detector、descriptor bank 或 dense refinement，而不是 PROSAC score 本身。

如果要继续冲投稿级主结果，下一版 claim 应该是：

> **Loc-GS-v2 learns PnP-supervised query keypoint detection and listwise 2D-3D candidate matching from virtual self-localization, improving candidate generation rather than merely reranking existing matches.**

这比当前的“SceneMatchNet top1 rerank”更有理论和实验空间。

---

## 11. 最后判断：问题到底在哪里？

按优先级排序：

### 第一问题：native STDLoc parity 没恢复

当前 control 和本地 STDLoc 差距太大。这个不解决，所有改进都是在弱基线上打转。

### 第二问题：oracle 排序上限小

说明你优化了一个 headroom 很小的子问题。排序已有 matches无法补回 SOTA gap。

### 第三问题：SceneMatchNet 是独立 pair MLP，不是 listwise matcher

它不能处理 topK 竞争和重复结构歧义。

### 第四问题：feedback detector target 受当前 PnP pipeline 污染

它没有把 projected landmark positives、PnP inlier positives、hard false negatives 分开建模。

### 第五问题：residual 正向但太弱

它说明 trust-region 微调方向合理，但不是主突破点。

---

## 12. 下一步计划

我建议接下来只做四件事，不要再扩展杂项 recipe。

### Week 1：STDLoc parity audit

目标：native Loc-GS control 接近本地 STDLoc。

交付：

```text
parity_query_table.csv
module-by-module diff
native_stdloc_parity recipe
```

成功标准：

```text
native parity median ≤ 9.5cm
R5 ≥ 0.35
```

如果达不到，先修 evaluator/localizer，不训练新模型。

---

### Week 2：topK candidate oracle

目标：判断 candidate generation 是否有上限。

跑：

```text
top1 oracle
top4 oracle
top8 oracle
top16 oracle
detector oracle
dense-refine oracle
```

成功标准：

```text
topK oracle 明显高于 top1 oracle
```

如果 topK oracle 大，做 SceneMatchNet-v2。  
如果 topK oracle 也小，改 detector / descriptor bank / landmark sampling。

---

### Week 3：Detector-v2

目标：验证 PnP feedback 是否能改善 query keypoint selection。

固定 native parity pipeline，只替换 detector。

不要直接上 full LFF。先做 sparse metrics。

---

### Week 4：SceneMatchNet-v2 listwise

目标：从 independent pair MLP 改为 topK listwise matcher。

固定 detector 和 parity pipeline，验证：

```text
cosine topK oracle
listwise matcher
PROSAC final
```

只有这一步明显超过 native parity，才回到 residual feature fine-tuning。

---

## 13. 一句话总结

你现在不是“还有某个小 bug 导致指标不涨”。更准确地说：

> **当前主线已经把能做的 post-hoc rerank / static prior / pair MLP 都试得差不多了，结果和 oracle 都证明：已有 sparse matches 的排序不是主要瓶颈。真正的问题是 native STDLoc parity、candidate generation、detector 和 topK query-conditioned matching。**

下一步必须先把 native STDLoc baseline 复现回来，然后用 detector-v2 和 listwise SceneMatchNet-v2 改善候选生成。否则继续调 residual、beta、PROSAC、LoFTR replacement，都不会带来论文级结果。

---

**Sources:**

- [Loc-GS/docs/loc_gs_lff_scenematch_mainline_20260512.md at main · Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/loc_gs_lff_scenematch_mainline_20260512.md)
- [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com)

## Prompt:
2026/5/14 01:00:59

如果换特征会有收益吗？（比如https://github.com/3DOM-FBK/deep-image-matching里提到的最新的RIPE特征）或者有没有可能把MASt3R这类方法的特征拿来用？（可能要改架构）

## Response:
2026/5/14 01:01:03

我会先判断 RIPE、DIM、MASt3R 是否能补当前候选匹配池短板，再区分它们适合作为主干替换、teacher/verifier，还是仅做诊断上界实验。

## Response:
2026/5/14 01:01:08

我已确认 DIM 可直接接入 RIPE 等特征与 LightGlue/RoMa/LoFTR 等匹配器；MASt3R 更像完整 3D-aware 定位管线，不是简单 descriptor 插件。

## Response:
2026/5/14 01:04:53

> **Considering the role of RIPE vs MASt3R in 3D feature extraction**
>
> I need to mention that MASt3R uses a non-commercial license (CC BY-NC-SA). I’ll also highlight RIPE, as it's easier to integrate into existing systems than MASt3R. However, to improve results, I'd need to ensure 3D landmarks and features align across queries and maps. Dimensional-specific details will help here.
>
> Thought for 6m 12s

## Response:
2026/5/14 01:07:16

可以换特征，而且**换特征是目前少数仍然有可能带来实质提升的方向**。但要非常克制：不能把 RIPE、MASt3R、LoFTR、RoMa 直接“替换进当前 Loc-GS 主线”就期待涨点。你现在的实验已经说明，**直接替换 matcher / 直接 LoFTR-rendered / 直接 learned descriptor replacement 都不稳**。所以正确问题不是“要不要换特征”，而是：

> **新特征应该替换哪一层？query detector？query descriptor？3D landmark descriptor？dense refinement？还是只作为训练期 teacher / hard-negative miner？**

我的判断是：

1. **RIPE / XFeat / ALIKED / DeDoDe 这类 sparse local feature，值得做 sparse initial pose 的特征替换实验。**
2. **MASt3R / RoMa / LoFTR 这类 dense pair matcher，不适合作为当前 3D Gaussian descriptor bank 的直接替代，但非常适合作为 hard-negative teacher、pose verifier、dense refinement teacher。**
3. **如果当前 oracle PROSAC 上限很小，单纯换 PROSAC 排序没用；真正该测的是“换特征后 top-K correct candidate recall 是否提升”。**

---

## 1. 换特征有没有可能涨？

**有可能，但前提是它提升 candidate generation，而不是只提升 reranking。**

你现在最重要的诊断结论是：`oracle_prosac` 上界很小。也就是说，在当前 SuperPoint/STDLoc/PLY descriptor 产生的 sparse candidate matches 里，即使用 GT 排序，也没有多少可提升空间。这说明瓶颈不在“怎么给已有匹配排序”，而在：

- query keypoint 不够好；
- 3D landmark descriptor 不够好；
- top-1 / top-K candidate 里正确 3D 点不够多；
- dense refinement 没有把 sparse pose 拉回来。

所以换特征值得做，但必须围绕这个指标：

$$
\textbf{Top-K correct candidate recall}
$$

也就是对每个 query keypoint，用新特征找 top-K 3D landmarks，看 GT reprojection 误差小于阈值的正确 landmark 是否出现在 top-K 里。

如果换成 RIPE / MASt3R / RoMa 后：

- top-1 correct candidate recall 明显提升；
- top-4 / top-8 correct candidate recall 明显提升；
- RANSAC inlier ratio 提升；
- sparse pose R@5 提升；

那它值得进主线。  
如果只是 image-to-image matching 看起来好，但 2D-3D candidate recall 没涨，那不值得。

---

## 2. RIPE 值得尝试，但更适合作为 sparse feature backbone，不适合直接做 dense field

RIPE 是 2025 年提出的 keypoint detector + descriptor 方法，核心是只用“图像对是否属于同一场景”的二元标签训练 keypoint extractor，不依赖深度、pose 或人工增强；它用中间层 hyper-column 特征做 descriptor，并用 epipolar / geometric consistency 作为 reward 来学习更鲁棒的 keypoint。RIPE 论文和仓库都强调它是 keypoint detection + description，而不是 dense 3D feature field。([arXiv](https://arxiv.org/html/2507.04839v1))

deep-image-matching 目前已经把 RIPE 列为 supported extractor，同时也支持 XFeat、DeDoDe、ALIKED、DISK、SuperPoint、HardNet、SIFT 等 extractor，以及 LightGlue、RoMa、LoFTR、SuperGlue 等 matcher。([GitHub](https://github.com/3DOM-FBK/deep-image-matching))

但 RIPE 有一个实际限制：RIPE repo 的 Glue Factory 集成里 `dense_outputs=False`，并且 dense outputs 直接写着不支持；它主要输出 sparse keypoints、scores、descriptors。([GitHub](https://github.com/fraunhoferhhi/RIPE)) 所以它不适合直接替代你现在的 dense SuperPoint feature field / dense refinement，但适合做：

### RIPE 路线 A：替换 query detector + query descriptor

流程：

```text
query image
→ RIPE detectAndCompute
→ RIPE keypoints / descriptors
→ 与 3D landmark RIPE descriptors 匹配
→ PnP / PROSAC
```

关键是：**query 和 map 两侧必须是同一种 descriptor**。不能 query 用 RIPE，3D map 仍用 STDLoc/PLY/SuperPoint descriptor，然后直接 cosine matching。descriptor space 不一致，必然不可靠。

所以你需要为每个 3D landmark 建 RIPE descriptor bank：

```text
train image RIPE descriptors
+ landmark projection / depth visibility
→ 每个 Gaussian / landmark 聚合一个 RIPE descriptor
```

聚合方式：

$$
d_j^\text{RIPE}
=
\mathrm{normalize}
\left(
\sum_{v \in \mathcal{V}_j}
w_{vj} d_{v,\pi_v(j)}
\right)
$$

权重可以用：

```text
RIPE keypoint score
depth consistency
view angle
alpha visibility
descriptor consistency
```

然后测试：

```text
RIPE query descriptor ↔ RIPE landmark descriptor
```

### RIPE 路线 B：只用 RIPE detector，descriptor 仍用 STDLoc/PLY

这条更简单，但风险是 RIPE keypoints 与 STDLoc descriptor map 的 sampling 位置不一定最优。可以做：

```text
RIPE keypoints
→ 在 STDLoc/SuperPoint dense descriptor map 上采样 descriptor
→ 匹配 PLY/STDLoc landmarks
```

这可以回答一个关键问题：

> 当前失败是不是因为 SuperPoint/STDLoc detector 选点不好？

如果 RIPE detector + STDLoc descriptor 提升 sparse R@5，说明 detector 是瓶颈。  
如果不提升，说明瓶颈更可能在 descriptor bank / landmark geometry / dense refine。

### RIPE 路线 C：训练期 hard-negative teacher

RIPE 的定位反馈训练思想和你项目很契合：RIPE 本身就是用几何一致性奖励训练 keypoint detector/descriptor。([arXiv](https://arxiv.org/html/2507.04839v1)) 所以可以用 RIPE 生成 alternative matches，和 STDLoc matches 比较，挖：

```text
STDLoc high-sim false positive
RIPE correct / more stable match
```

这些样本再用于训练你的 PnP-feedback detector 或 listwise SceneMatchNet-v2。

---

## 3. MASt3R 可以用，但不应该直接当“可缓存的 Gaussian descriptor”

MASt3R 很强，但它和你的当前架构不是天然对齐。

MASt3R 的核心是把 image matching 看成 3D 任务，基于 DUSt3R 的 pointmap regression，加了 dense local feature head 和 fast reciprocal matching，从而在多个 matching 任务上表现很强。论文摘要里明确说它输出 dense local features，并用 fast reciprocal matching 加速 dense matching。([arXiv](https://arxiv.org/abs/2406.09756)) 官方 repo 也有 visual localization 例子，支持 Aachen、InLoc、Cambridge、7-Scenes 等数据。([GitHub](https://github.com/naver/mast3r))

但注意：MASt3R 是 **pair-conditioned dense matching / 3D reconstruction prior**，不是传统意义上“给每个图像像素一个固定 descriptor，然后缓存到每个 Gaussian”的局部特征。它的优点是 pairwise image matching 强，缺点是：

- 计算重；
- 特征可能依赖 paired image context；
- 不容易直接聚合成一个稳定 per-Gaussian descriptor；
- 推理时如果每个 query 都要和多张 rendered/reference images 跑 MASt3R，成本高；
- 官方代码是 CC BY-NC-SA 4.0 非商业许可，投稿研究一般没问题，但后续商用或开源集成要注意。([GitHub](https://github.com/naver/mast3r))

所以 MASt3R 不建议作为：

```text
MASt3R descriptor → 直接替代 STDLoc/PLY descriptor bank
```

更建议作为下面三种用途。

---

## 4. MASt3R 最合理的三种接入方式

### 4.1 MASt3R as teacher：生成 hard-positive / hard-negative labels

这是最推荐的。

流程：

```text
train/query image
+ rendered RGB/depth at known pose or retrieved reference pose
→ MASt3R dense matches
→ rendered pixel 用 depth 反投影成 3D point
→ GT pose 检查 reprojection / visibility
→ 生成 2D-3D pair labels
```

然后训练：

```text
PnP-feedback detector
SceneMatchNet-v2 listwise matcher
LFF residual descriptor
```

这条路不会在 test-time 增加很多成本，也不会破坏 STDLoc backbone。它把 MASt3R 当“强监督来源”，而不是当“替换 matcher”。

特别是 GreatCourt / KingsCollege，MASt3R 很可能能找到一些 SuperPoint/STDLoc 错过的 cross-view correspondences。你应该用它来挖：

```text
query keypoint / pixel u
STDLoc top-K 都错
但 MASt3R-rendered match 给出了正确 3D point
```

这类样本对 detector 和 listwise matcher 最有价值。

---

### 4.2 MASt3R as pose verifier：不求 pose，只给候选 pose 打分

流程：

```text
STDLoc / Loc-GS 先产生若干 candidate pose
→ 对每个 pose render RGB/depth
→ MASt3R match(query, rendered)
→ 用 match count / reprojection / confidence 打分
→ 选择 pose 或拒绝 pose
```

这适合处理 GreatCourt 的重复结构：错误 pose 可能 sparse PnP 内点也不少，但 rendered image 与 query 的 dense 3D-aware match 会更差。

这条路比 LoFTR-rendered 更可能有用，因为 MASt3R 本身就是 3D-grounded matching。MASt3R 论文也强调它在极端 viewpoint 和 map-free localization 上有明显优势。([arXiv](https://arxiv.org/abs/2406.09756))

但它的风险是算力成本。如果只做 top-2/top-3 pose verification，还能接受；如果每个 query 和很多 render 都跑，成本会很高。

---

### 4.3 MASt3R as dense refinement teacher

不要直接用 MASt3R 替代 dense refinement。可以用它监督你的 dense refinement / residual reliability：

```text
MASt3R dense match confidence
→ 哪些 rendered pixels 和 query pixels 更可靠
→ 训练 dense matchability / locability map
```

这比直接 test-time MASt3R 更适合论文主线，因为它仍然保持：

```text
Loc-GS learns localization feedback
test-time uses efficient 2D-3D PnP
```

---

## 5. 如果真的要换整个 feature backbone，该怎么做？

你需要设计一个“特征替换金字塔”，从低成本到高成本逐步验证。不要一次改全架构。

### Level 0：只做 image-to-image smoke，不进主线

目的：确认 RIPE / MASt3R / RoMa 在 Cambridge train/query 图像上比 SuperPoint/STDLoc 有更高几何 matching quality。

指标：

```text
relative pose AUC
F-matrix inlier count
match count
repeatability
```

这一步只能说明特征有潜力，不能证明 2D-3D localization 会涨。

---

### Level 1：query detector 替换

```text
RIPE keypoints
+ STDLoc/PLY descriptor sampling
+ original PnP
```

如果这能涨，说明 detector 是瓶颈。

同理可以试：

```text
XFeat keypoints
ALIKED keypoints
DeDoDe keypoints
DISK keypoints
```

DIM 支持这些 extractor，所以这一步工程成本可控。([GitHub](https://github.com/3DOM-FBK/deep-image-matching))

---

### Level 2：完整 sparse feature replacement

```text
RIPE query descriptors
+ RIPE landmark descriptors aggregated from train views
+ PnP / PROSAC
```

这是判断 RIPE 是否能替代 SuperPoint/STDLoc sparse feature 的关键实验。

必须输出：

```text
top1 candidate recall
top4 candidate recall
RANSAC inlier ratio
sparse R5
dense final R5
```

如果 topK candidate recall 不涨，就别继续。

---

### Level 3：teacher-only dense matcher

```text
MASt3R / RoMa / LoFTR
→ train-time hard-negative / hard-positive labels
→ train detector / matcher / residual
→ test-time 不跑 MASt3R
```

这是我最推荐的 MASt3R 使用方式。

---

### Level 4：test-time verifier

```text
baseline pose / Loc-GS pose
→ render
→ MASt3R verifier score
→ choose/refine/reject
```

这可以作为 ablation 或 high-accuracy setting，但不一定适合高效主线。

---

### Level 5：MASt3R feature field distillation

这是最大改架构，不建议现在优先做。

要做的话，需要解决：

- MASt3R dense features 是否 pair-independent；
- 如何从多 reference pairs 聚合到 per-Gaussian descriptor；
- 如何处理 pair-conditioned feature inconsistency；
- 如何在 test-time query feature 与 map feature 对齐；
- feature dimension / memory / render cost。

只有当前面 Level 1-4 证明 MASt3R 明显带来 localization headroom，才值得做这个。

---

## 6. 特征替换最该先做哪几个？

我建议优先级如下。

### 第一优先级：RIPE / XFeat / ALIKED sparse replacement

原因：

- 它们是 local feature extractor，和你的 2D-3D PnP 架构最匹配；
- 可以建 3D landmark descriptor bank；
- 不需要大改 dense refinement；
- 可以直接测 topK candidate recall；
- DIM 已经支持很多 extractor，RIPE 也有 `detectAndCompute` 风格接口。([GitHub](https://github.com/3DOM-FBK/deep-image-matching))

实验组合：

```text
SuperPoint/STDLoc baseline
RIPE detector + STDLoc descriptor
RIPE detector + RIPE descriptor bank
XFeat detector + XFeat descriptor bank
ALIKED + LightGlue-style descriptor/matcher if feasible
DISK / DeDoDe optional
```

### 第二优先级：MASt3R teacher for hard negatives

原因：

- 直接 MASt3R 推理替换可能成本高；
- teacher labels 可以改善 detector / matcher；
- 对 GreatCourt 重复结构特别有价值。

实验：

```text
MASt3R-rendered matches
→ hard-negative labels
→ train Detector-v2 / SceneMatchNet-v2
→ same STDLoc/PLY inference
```

### 第三优先级：MASt3R verifier

用于回答：

> 当前 sparse PnP 的错误是否能被 dense 3D-aware verifier 识别？

如果能，后续可以做 pose selection / refinement；如果不能，说明瓶颈更深。

---

## 7. 当前 hybrid / LFF 架构要怎么配合新特征？

不要把每种新特征都塞进同一个旧 hybrid feature field。建议改成更模块化：

```text
FeatureBackbone:
  superpoint_stdloc
  ripe
  xfeat
  aliked
  mast3r_teacher_only

MapDescriptorBank:
  ply_stdloc
  aggregated_ripe
  aggregated_xfeat
  residual_lff

Detector:
  original
  stdloc_scene_specific
  ripe_detector
  pnp_feedback_detector

Matcher:
  cosine
  listwise_scene_matcher
  lightglue_if_2d2d
  mast3r_teacher_labels

PnP:
  RANSAC
  PROSAC
  MAGSAC
```

主方法不要混太多。最终要收敛成一套固定 recipe，例如：

```text
RIPE/XFeat/STDLoc descriptor bank whichever wins
+ PnP-feedback detector
+ listwise matcher
+ PROSAC
+ optional MASt3R teacher labels
```

---

## 8. 需要先做一个上限实验

在真正替换 feature 前，先做这个矩阵：

| Feature | top1 correct | top4 correct | top8 correct | RANSAC inlier | sparse R5 | final R5 |
|---|---:|---:|---:|---:|---:|---:|
| STDLoc/PLY |  |  |  |  |  |  |
| RIPE bank |  |  |  |  |  |  |
| XFeat bank |  |  |  |  |  |  |
| ALIKED bank |  |  |  |  |  |  |
| MASt3R teacher oracle |  |  |  |  |  |  |

这张表比直接跑 final pose 更重要。  
如果 RIPE 的 top8 candidate recall 明显高于 STDLoc，但 final pose 没涨，说明 matcher/PnP/dense refine 没吃到信号。  
如果 RIPE topK 都不如 STDLoc，那不用继续。

---

## 9. 我的判断：哪条最可能涨？

按你当前结果，我会押这三个方向：

### 最可能有中等收益：XFeat / RIPE / ALIKED sparse descriptor bank

因为你的 oracle PROSAC 小，说明当前 candidate pool 弱。换 sparse descriptor bank 是最直接改 candidate pool 的办法。

RIPE 尤其值得试，因为它的训练思想本身就是用 image-pair geometric feedback 学 keypoint detection/description，和你的“定位反馈影响匹配”目标很贴近。([arXiv](https://arxiv.org/html/2507.04839v1))

### 最可能有论文价值：MASt3R teacher hard-negative mining

MASt3R 直接替换可能工程重、速度慢、license复杂，但作为 teacher 非常有价值。它能帮助你生成“STDLoc 错而 MASt3R 对”的 hard examples，从而支撑：

> localization feedback + external dense 3D-aware teacher improves detector/matcher supervision.

### 最不建议优先做：MASt3R feature field distillation

这会大改架构，风险最高。除非你先证明 MASt3R teacher 产生的 2D-3D labels 在 candidate recall 上明显强，否则不要做。

---

## 10. 最后给你一个执行计划

### Step 1：用 DIM/RIPE 做 sparse feature bank baseline

实现：

```text
extract_ripe_features.py
build_ripe_landmark_bank.py
eval_cambridge_feature_bank.py
```

只测：

```text
RIPE sparse PnP
RIPE + PROSAC
RIPE + current dense refine
```

不训练 LFF。

### Step 2：跑 topK candidate oracle

比较：

```text
STDLoc bank
RIPE bank
XFeat bank
ALIKED bank
```

如果新特征 topK 不涨，停止。

### Step 3：MASt3R teacher label generation

实现：

```text
generate_mast3r_teacher_pairs.py
```

输出：

```text
positive 2D-3D pairs
hard negative pairs
pose verifier score
```

用于 detector / matcher 训练，不进 test-time 主链路。

### Step 4：训练 Detector-v2 / SceneMatchNet-v2

用：

```text
STDLoc labels
+ RIPE/XFeat bank labels
+ MASt3R hard negatives
```

训练 listwise matcher，不再用 independent pair MLP。

### Step 5：固定 recipe full Cambridge

最终比较：

```text
STDLoc local reproduction
best new sparse bank
best new sparse bank + Detector-v2
best new sparse bank + Detector-v2 + listwise matcher
```

目标必须是：

```text
mean median < 10.1cm
或 R5 > 本地 STDLoc 0.371，最好 ≥ 0.41
```

---

## 简短结论

**换特征有可能带来收益，而且现在确实应该测。**  
但不要把 RIPE/MASt3R直接塞进旧 hybrid field。

最合理的顺序是：

1. **RIPE/XFeat/ALIKED 先做 2D-3D sparse descriptor bank 替换，测 topK candidate recall。**
2. **MASt3R 不做直接替换，先做 hard-negative teacher 和 pose verifier。**
3. **如果新特征能显著提升 candidate pool，再训练 Detector-v2 / listwise SceneMatchNet-v2。**
4. **只有当这些都有效，再考虑把新特征蒸馏进 LFF。**

你现在真正需要的是改善“候选匹配池”，不是再优化已有候选的排序。新特征最大的价值就在这里。

---

**Sources:**

- [RIPE: Reinforcement Learning on Unlabeled Image Pairs for Robust Keypoint Extraction](https://arxiv.org/html/2507.04839v1)
- [GitHub - 3DOM-FBK/deep-image-matching: Multiview matching with deep-learning and hand-crafted local features for COLMAP and other SfM software. Supports high-resolution formats and images with rotations. Both CLI and GUI are supported. · GitHub](https://github.com/3DOM-FBK/deep-image-matching)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)