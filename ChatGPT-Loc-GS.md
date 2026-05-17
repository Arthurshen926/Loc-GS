# Loc-GS

**User:** Anonymous  
**Created:** 5/16/2026 0:57:35  
**Updated:** 5/16/2026 21:46:59  
**Exported:** 5/16/2026 21:54:08  
**Link:** [https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb](https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb)  

## Prompt:
2026/5/16 20:52:09

又修改项目并推进了一轮，git已更新，但感觉有点跑偏，大致总结是：
已按这条主线继续推进，并拿到了**单一路径 unified selector 的正向 full Cambridge 证据**，但我先不夸大：目前是“方向被验证 + 小幅稳定提升”，还没到“强 SOTA claim”。

**当前最稳主结果候选**

`selector005_native_desc`：  
用 self-localization feedback 蒸馏出的统一 selector / locability，保持 native STDLoc descriptor，不做 branch selection，不做 per-scene recipe，推理仍是：

`STDLoc-compatible map → sampled landmarks → descriptor matching → OpenCV PROSAC PnP → dense refinement`

**q80 Cambridge：**

- Native dense: `9.1045cm / 0.1128deg, R10 0.6000, R5 0.3675, R2 0.1075`
- Selector-only dense: `9.0268cm / 0.1144deg, R10 0.6050, R5 0.3725, R2 0.1175`

也就是 dense median `-0.0777cm`，R10/R5/R2 全部正向，尤其 R2 `+1.0pp`。

**Full Cambridge：**

- Native dense: `12.5725cm / 0.1578deg, R10 0.4860, R5 0.2776, R2 0.0848`
- Selector-only dense: `12.4238cm / 0.1593deg, R10 0.4857, R5 0.2815, R2 0.0879`

full 上 median `-0.1487cm`，R5 `+0.39pp`，R2 `+0.31pp`，R10 基本持平。

报告已保存：

- `output/unified_lff_v2/reports/20260516_rehearsal60k_q80_selector_only_native_desc.json`
- `output/unified_lff_v2/reports/20260516_rehearsal60k_full_summary_selector005_native_desc.json`
- `output/unified_lff_v2/reports/20260516_rehearsal60k_q80_selector010_native_desc.json`

**关键结论**

这轮最重要的发现是：**真正有效的是 self-map feedback 蒸馏出来的 selector / locability 信号；当前 residual descriptor 反而还不够稳。**

对比下来：

- `residual + selector` full dense median 也有提升，但 R10/R2 会掉；
- `selector-only + native descriptor` 更干净，full dense median、R5、R2 都正向；
- `selector 0.10` q80 没有超过 `0.05`，所以当前固定 recipe 先定为 `gate_locability_blend=0.05`。

这对论文叙事反而更清楚：不是“替换 STDLoc descriptor”，而是“定位反馈蒸馏到统一 feature selection / locability field，让原有几何定位后端受益”。

**代码侧推进**

本轮也补了几个关键实现点：

- `UnifiedLFFDescriptor` 中把 residual gate 和 selector gate 解耦；
- export 支持 `descriptor_mode=native`，可做 selector-only 统一表征 ablation；
- 修了 detector score export：源 map 只有 `sampled_idx.pkl` 时也会生成并写入 `sampled_scores.pkl`；
- 修了 differentiable PnP top-k masking 的 logits 使用问题；
- 训练侧补了 detector prior target / false-positive threshold / gate metadata；
- eval launcher 支持 unified detector dir 和 locability prior weight。

验证：

`75 passed in 4.90s`

**投稿判断**

现在可以支撑一个较稳的中间 claim：

> Self-localization feedback can be distilled into a single STDLoc-compatible localization selector, improving downstream localization without branch selection or descriptor replacement.

但还不够支撑“强 SOTA / 大幅超越”claim。下一步应该把 residual descriptor 暂时降级为 ablation，主线聚焦在 selector 的训练信号继续增强：pose-information / coverage / hard-negative false-positive penalty / listwise matchability，这些才可能把现在的 `0.3-1.0pp` recall 增益推到投稿级强幅度。

## Response:
2026/5/16 20:52:10

我会把这轮结果作为单路径 selector 有效性的主线复核，而不是直接推成 SOTA；重点判断哪些证据能进主结果，哪些应降级为诊断，并给出下一轮最小闭环。

## Response:
2026/5/16 20:52:24

我先确认：方向本身没有跑偏，但需要立刻冻结 selector/locability 为主线，并同步收窄 README、claim 与 evidence audit，避免 residual/quality gate 叙事继续误导后续开发。

## Response:
2026/5/16 20:57:36

你的感觉是对的：**这轮不是技术上跑偏，而是论文叙事上开始逼近一个分叉点**。

我建议你不要把这轮看成“又一个小幅涨点实验”，而要把它看成一次非常重要的方向筛选：

> **descriptor residual 暂时没有证明自己；self-localization feedback → selector / locability field 这条线首次在 full Cambridge 单一路径上给出了干净正结果。**

所以接下来应该**主动收束主线**，而不是继续同时推 residual、SceneMatchNet、soft prior、quality gate、selector、hard-negative descriptor 等多个方向。

---

## 1. 先给结论：没有跑偏，但必须改主 claim

当前最稳主线应该从：

> 定位引导的特征重建

收束为：

> **定位反馈蒸馏出的 localization utility / locability field，用于 STDLoc-compatible Gaussian feature map 的特征选择与采样引导。**

也就是说，主贡献不再是“我们改好了 descriptor”，而是：

> 原始 STDLoc/native descriptor 已经足够强；真正缺的是哪些 Gaussian / landmark 应该被信任、被采样、被赋予更高定位优先级。Loc-GS 用 self-localization feedback 学到这个 selector，并把它导出到单一路径 STDLoc-compatible map 中。

这比继续硬讲 residual descriptor 更清楚，也更符合你现在的数据。

仓库当前 README 里本来已经把主线限定为 `STDLoc/PLY descriptor backbone -> virtual self-localization feedback -> ... -> one OpenCV PROSAC PnP path -> STDLoc-style dense refinement`，并明确说当前 paper story 是 reconstruction-time / self-localization feedback，而不是 inference-time 多路径选择。([GitHub](https://github.com/Arthurshen926/Loc-GS)) AGENTS.md 也已经把“不能用 test query 做训练/校准/模型选择、真实 query 必须单路径、quality gate 不能变成 per-query branch selector”写成硬规则。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/AGENTS.md)) 这说明你现在往 selector-only 收束，反而比 residual-heavy 版本更符合项目契约。

---

## 2. 当前结果该怎么解读

你给出的 full Cambridge 结果：

```text
Native dense:
12.5725cm / 0.1578deg
R10 0.4860
R5  0.2776
R2  0.0848

Selector-only dense:
12.4238cm / 0.1593deg
R10 0.4857
R5  0.2815
R2  0.0879
```

可以这样解释：

```text
median translation: -0.1487cm，约 1.18% 相对改善
R10: -0.03pp，基本持平
R5:  +0.39pp，约 1.40% 相对改善
R2:  +0.31pp，约 3.66% 相对改善
rotation median: +0.0015deg，轻微变差
```

q80 上更明显：

```text
median translation: -0.0777cm，约 0.85% 相对改善
R10: +0.50pp
R5:  +0.50pp
R2:  +1.00pp，约 9.30% 相对改善
```

我的判断是：

**这已经足以证明“selector 信号有效”，但还不足以证明“投稿级强方法”。**

比较合理的当前 claim 是你自己写的那条：

> Self-localization feedback can be distilled into a single STDLoc-compatible localization selector, improving downstream localization without branch selection or descriptor replacement.

但不要写成：

> We achieve SOTA localization accuracy by reconstructing better descriptors.

因为当前最强证据恰恰说明：**native descriptor + learned selector 比 residual descriptor 更稳。**

---

## 3. 现在最大的论文风险

### 风险 A：如果 descriptor 是 native，审稿人会问“你到底重建了什么？”

这是当前最大的叙事风险。

解决方式不是硬说 descriptor reconstruction，而是把“feature reconstruction”的定义扩展为：

```text
Gaussian feature map = descriptor field + detector / locability / selection field
```

你的主方法是在重建：

```text
localization utility field
locability field
feature selection field
```

而不是重建 descriptor 本身。

更准确的论文措辞应该是：

> We reconstruct a localization-aware feature utility field on top of a STDLoc-compatible Gaussian feature map. The descriptor field remains anchored to the native STDLoc descriptor bank, while self-localization feedback updates the per-Gaussian locability / selector field that controls landmark trust, sampling, and ranking.

这比“descriptor residual reconstruction”安全很多。

---

### 风险 B：`selector005` 是不是从 q80 / full test 调出来的？

这是必须立即处理的问题。

如果 `gate_locability_blend=0.05` 是因为你看了 q80 / full Cambridge test 结果后选出来的，那它不能直接作为 paper-facing fixed recipe。你可以把它保留为 diagnostic，但主实验需要重新定义选择规则：

```text
gate_locability_blend = 0.05
```

必须来自以下之一：

```text
1. self-map / train-derived calibration；
2. prior fixed hyperparameter，不看 test；
3. 在一个开发场景/开发 split 上确定，然后 frozen；
4. 多个固定值全部报告，不只报告最佳值。
```

最稳做法是：

```text
selector_blend ∈ {0.00, 0.03, 0.05, 0.10}
```

在 self-map calibration 上选出一个全局值，然后只跑一次 full test。或者直接把 `0.05` 声明为 fixed conservative blend，不再根据 q80 / full 改。

你仓库里的 protocol 已经明确写了 Cambridge test 只能用于最终评估，不能用于训练、校准、model selection 或 hard-negative mining。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/experiment_protocol.md)) 所以这点必须严格守住。

---

### 风险 C：当前 selector 可能只是“score nudging”，还没证明是“feature selection”

从代码看，`export_unified_lff_map.py` 已经支持 `descriptor_mode=("checkpoint", "native")`，这让 selector-only ablation 成立。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/loc_gs/scripts/export_unified_lff_map.py)) 进一步看 `unified_lff_export.py`，当前 export 会把 checkpoint 的 gate 混入 PLY locability，并在有 `sampled_idx.pkl` 时更新 `detector/sampled_scores.pkl`，但它主要是在已有 sampled landmarks 上改分数，不一定重新生成 landmark candidate set。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/loc_gs/stdloc_native/unified_lff_export.py))

这意味着当前方法可能仍然是：

```text
fixed sampled_idx
+ learned selector updates sampled_scores / locability
```

而不是真正的：

```text
learned selector changes which landmarks enter localization
```

这也解释了为什么 full Cambridge 提升是稳定但小幅的。

下一步如果想把 `+0.3pp ~ +1pp` 推到投稿级，最有潜力的不是再调 `blend=0.05/0.10`，而是做：

> **selector-guided landmark resampling under the same landmark budget**

也就是保持 descriptor native、保持单一路径 evaluator、保持相同 landmark 数量，但用 learned selector 重新决定 sampled_idx / sampled_scores，并加入 coverage / covisibility / pose-information 约束。

---

## 4. 当前主线应该怎么重命名

我建议暂时不要再叫：

```text
residual + selector
```

也不要叫：

```text
descriptor reconstruction
```

而是把当前主候选命名为：

```text
Loc-GS-Selector
或
Loc-GS-Locability
或
SelfLoc-Selector
```

论文方法可以写成三层：

```text
1. Self-localization feedback generation
   从 train/self-map episode 得到 per-Gaussian localization utility signal。

2. Localization utility distillation
   把 feedback 蒸馏成统一 selector / locability field。

3. STDLoc-compatible deployment
   descriptor 保持 native，selector 导出为 locability / detector score / sampling priority。
   query-time 仍是 descriptor matching + PROSAC PnP + dense refinement。
```

descriptor residual 放到 appendix / ablation：

```text
Residual descriptor reconstruction is explored but not used in the main model,
because native descriptors plus feedback-distilled selection are more stable.
```

这不是退步，而是一个很强的科学结论：**定位失败的瓶颈不一定是 descriptor 表达能力，而是 map 中 landmark 的定位效用选择。**

---

## 5. 下一轮最应该做的不是继续训练，而是做 paper-safety 和 causality audit

你现在已经有三个关键报告：

```text
output/unified_lff_v2/reports/20260516_rehearsal60k_q80_selector_only_native_desc.json
output/unified_lff_v2/reports/20260516_rehearsal60k_full_summary_selector005_native_desc.json
output/unified_lff_v2/reports/20260516_rehearsal60k_q80_selector010_native_desc.json
```

下一步不要马上继续加 loss。先把这三个结果转成 paper-safe evidence bundle。

仓库里的 `paper_evidence_audit.md` 已经指出一个关键问题：当前 checkout 里还缺少 claim-level full Cambridge runs 的 machine-readable `manifest.json`、`metrics_summary.json`、`split_audit.json` 等审计材料。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/paper_evidence_audit.md)) 也就是说，即使你手里有结果，论文证据链还没闭合。

马上让 CodexPotter 做这个任务：

```text
请不要改任何 evaluator、metric、checkpoint 或实验结果。

目标：
把 20260516 selector-only native descriptor 结果整理成 paper-safe audit bundle，并判断它是否可以作为 main_candidate。

需要读取：
- output/unified_lff_v2/reports/20260516_rehearsal60k_q80_selector_only_native_desc.json
- output/unified_lff_v2/reports/20260516_rehearsal60k_full_summary_selector005_native_desc.json
- output/unified_lff_v2/reports/20260516_rehearsal60k_q80_selector010_native_desc.json
- docs/research_contract.md
- docs/experiment_protocol.md
- AGENTS.md
- loc_gs/stdloc_native/unified_lff_export.py
- loc_gs/scripts/export_unified_lff_map.py
- loc_gs/scripts/update_experiment_board.py

请完成：

1. 新建 docs/selector_only_native_desc_20260516.md
   内容包括：
   - q80 和 full Cambridge native vs selector-only 表格；
   - selector005 与 selector010 的比较；
   - descriptor_mode=native 的说明；
   - 推理路径仍为单路径；
   - 当前 claim 和不能 claim 的内容；
   - 是否存在 q80/full tuning 风险；
   - 下一步需要补的 audit。

2. 为 full selector005 结果生成或补齐：
   - manifest.json
   - command.txt
   - metrics_summary.json
   - split_audit.json
   - git_status.txt 或 git_diff.patch

3. 如果原始命令、split 或 checkpoint 路径不可恢复，不要伪造。
   标记 audit_status = unknown，并说明缺失字段。

4. 更新 docs/experiment_board.md 或通过 update_experiment_board.py 生成 board。
   该 run 只有在 audit 完整时才能标 main_candidate；
   否则标 ablation 或 diagnostic。

5. 不允许：
   - 修改 evaluator；
   - 修改 metric；
   - 修改 test split；
   - 删除失败结果；
   - 把 q80 调参结果包装成 fixed recipe。

验收：
- python -m compileall loc_gs
- pytest -q tests/test_update_experiment_board.py tests/test_manifest_split_audit.py
```

这一步很重要。**没有 audit bundle 的结果，不能进入主表。**

---

## 6. 然后做一个“selector 因果性”实验组

你现在只证明了：

```text
learned selector005 > native
```

但还没证明：

```text
是 self-localization feedback 学到的 localization utility 起作用，
而不是某种偶然 score perturbation。
```

下一步至少补这组 ablation：

```text
A. Native STDLoc
B. Selector005 native descriptor
C. Uniform gate, same blend=0.05
D. Random-permuted selector gate, same distribution
E. Inverted selector gate
F. Detector-score-only update
G. PLY locability-only update
H. Detector-score + PLY locability update
```

关键点是 C/D/E。

如果 B 比 C/D/E 都好，才能说明 selector 不是随机扰动。

如果 F 和 G 能拆开，你还能回答：

```text
selector 到底通过 sparse landmark ranking 起作用，
还是通过 dense refinement locability 起作用？
```

给 CodexPotter 的任务：

```text
请设计并实现 selector-only causality ablation 的 export/eval 工具，但不要自动启动 full Cambridge 长实验。

目标：
验证 20260516 selector-only native descriptor 的提升是否来自 self-localization feedback selector 本身，而不是随机 score perturbation。

新增或修改：

1. loc_gs/scripts/export_selector_ablation_maps.py
   输入：
   - --source_map
   - --checkpoint_path
   - --output_root
   - --gate_locability_blend
   - --descriptor_mode native
   - --ablation native,selector,uniform,permuted,inverted,detector_only,locability_only,both
   - --seed

   输出每个 ablation 的 STDLoc-compatible map。

2. loc_gs/stdloc_native/unified_lff_export.py
   在不破坏现有 build_unified_lff_map 的情况下，支持：
   - gate_transform = identity / uniform / permuted / inverted
   - apply_to_detector_scores: bool
   - apply_to_ply_locability: bool
   - descriptor_mode=native

3. tests/test_selector_ablation_export.py
   用合成 map/checkpoint 测试：
   - native descriptor 不变；
   - uniform/permuted/inverted gate 生效；
   - detector_only 不改 PLY locability；
   - locability_only 不写 sampled_scores；
   - manifest 正确记录 ablation type。

4. docs/selector_causality_ablation.md
   写清楚每个 ablation 能排除什么 confound。

禁止：
- 不改 evaluator。
- 不跑 full Cambridge。
- 不使用 test 结果选择 ablation。
- 不改变 descriptor，除非 descriptor_mode 明确不是 native；本任务固定 native。

验收：
- pytest -q tests/test_selector_ablation_export.py tests/test_unified_lff_map_export.py
- python -m compileall loc_gs
```

这组实验比继续直接跑 selector020 更有价值。

---

## 7. 真正可能把增益推大的下一步：selector-guided resampling

当前 selector005 的 full gain 小，原因很可能是它只在已有 sampled landmarks 上改 score。你需要让 selector 真正参与：

```text
哪些 Gaussian / landmarks 被选进 STDLoc map
```

但必须满足：

```text
同样 landmark budget
同样 native descriptor
同样 evaluator
同样 PnP path
无 test tuning
```

方法建议：

```text
score_i =
  source_score_i
  + beta * selector_logit_i
  + gamma * visibility_i
  - eta * hard_negative_risk_i
  + kappa * pose_information_i
```

然后做 coverage-constrained sampling：

```text
1. 按 scene / covisibility / image grid / 3D voxel 分桶；
2. 每桶保留 top-k；
3. 全局补足预算；
4. 写新的 sampled_idx.pkl 和 sampled_scores.pkl；
5. descriptor 仍然 native。
```

这会比只更新 sampled_scores 更接近“feature selection”。

给 CodexPotter 的任务：

```text
请实现 selector-guided landmark resampling，但只实现 map export 和 synthetic tests，不跑 full Cambridge。

目标：
在 descriptor_mode=native 下，用 self-localization feedback selector 重新生成 STDLoc detector/sampled_idx.pkl 和 sampled_scores.pkl，使 selector 不只是重排已有 sampled landmarks，而是真正控制 map-level landmark selection。

新增：

1. loc_gs/stdloc_native/selector_resampling.py
   函数：
   - build_selector_sampling_scores(...)
   - coverage_constrained_topk(...)
   - resample_detector_landmarks(...)
   - write_resampled_detector_payload(...)

2. loc_gs/scripts/export_selector_resampled_map.py
   参数：
   - --source_map
   - --checkpoint_path
   - --output_map
   - --descriptor_mode native
   - --budget same_as_source 或 int
   - --selector_weight
   - --source_score_weight
   - --coverage_grid
   - --seed
   - --dry_run

3. manifest 必须记录：
   - source sampled count
   - output sampled count
   - descriptor_mode
   - selector_weight
   - coverage settings
   - whether sampled_idx changed
   - single_path_deployment=true
   - branch_selection=false

4. tests/test_selector_resampling.py
   合成测试：
   - same budget；
   - high selector landmarks are selected；
   - coverage prevents all selected landmarks collapsing into one region；
   - descriptor payload remains native；
   - dry_run 不写 map。

禁止：
- 不改 evaluator。
- 不改 descriptor。
- 不用 test query。
- 不自动跑 Cambridge。

验收：
- pytest -q tests/test_selector_resampling.py tests/test_unified_lff_map_export.py
- python -m compileall loc_gs
```

如果这一步 full Cambridge 能带来 R5/R2 明显提升，你的主 claim 会强很多。

---

## 8. 训练侧也要收束：做真正的 selector-only training

从当前 `train_unified_lff.py` 看，`UnifiedLFFDescriptor` 同时有 residual、residual gate 和 selector gate；输出 checkpoint 里有 `export_descriptors`、`gate`、`residual_gate`，metadata 里也记录 `selector_gate_decoupled=True`。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/loc_gs/scripts/train_unified_lff.py)) 这很好，但对论文主线来说还不够干净。

如果主结果是：

```text
selector-only + native descriptor
```

那么训练脚本最好也有一个干净模式：

```text
--selector_only
--freeze_descriptor_residual
```

否则审稿人会问：

> 你最终导出 native descriptor，但训练阶段 residual 参与了 listwise descriptor loss，这个 selector 到底是不是独立学到的？

理想的 selector-only 训练应该是：

```text
candidate_desc = frozen native descriptor
selector_bias = logit(selector(candidate_id))
candidate_score = cosine(query_desc, native_desc) / temperature
                + selector_bias_weight * selector_bias

loss =
  listwise CE(candidate_score, positive/dustbin)
  + BCE(selector, gaussian_advantage_target)
  + hard_negative_suppression
  + budget / coverage regularization
```

descriptor residual 完全冻结或不存在。

给 CodexPotter 的任务：

```text
请为 train_unified_lff.py 增加严格 selector-only 训练模式。

目标：
支持一个 paper-facing clean mode：
native descriptor frozen，residual 不参与训练，selector 通过 self-localization feedback 的 listwise / hard-negative / gaussian advantage 信号学习，最终导出 descriptor_mode=native 的 STDLoc-compatible map。

修改：

1. loc_gs/scripts/train_unified_lff.py
   增加参数：
   - --selector_only
   - --freeze_descriptor_residual
   - --selector_bias_weight
   - --lambda_selector_listwise
   - --lambda_selector_gate
   - --lambda_selector_hard_negative
   - --lambda_selector_budget
   - --lambda_selector_coverage

2. selector_only=True 时：
   - 不优化 residual；
   - 不优化 residual gate；
   - candidate descriptor 固定为 base_descriptors[candidate_ids]；
   - candidate logits = native cosine logits + selector_bias_weight * logit(selector(candidate_ids))；
   - checkpoint 仍保存 gate / selector；
   - export_descriptors 应等于 native descriptors，或 manifest 明确写 descriptor_mode=native_required。

3. 训练日志记录：
   - selector_listwise_loss
   - selector_gate_loss
   - selector_hard_negative_loss
   - selector_budget_loss
   - selector_coverage_loss
   - residual_frozen=true

4. tests/test_train_unified_lff.py 增加：
   - selector_only 不改变 export_descriptors；
   - residual 参数没有梯度或未被 optimizer 管理；
   - selector logits 能通过 listwise loss 更新；
   - descriptor_mode=native export 和 checkpoint 一致。

禁止：
- 默认行为不变。
- 不改 evaluator。
- 不自动跑 Cambridge。
- 不用 test 数据。

验收：
- pytest -q tests/test_train_unified_lff.py tests/test_unified_lff_descriptor.py
- python -m compileall loc_gs
```

这一步会让你的论文主线非常干净：

```text
不是 descriptor residual accidentally worked；
而是 localization feedback distilled selector worked。
```

---

## 9. residual descriptor 现在应该怎么处理

我建议：

```text
residual descriptor 暂时降级为 ablation，不再作为主线。
```

原因：

1. full 上 `residual + selector` median 可能好，但 R10/R2 掉；
2. selector-only 更干净，且更符合 single-path / native descriptor anchor；
3. residual 容易引入“你是不是改变了 STDLoc descriptor backbone”的 reviewer 风险；
4. 当前结果说明瓶颈可能是 landmark utility，而不是 descriptor capacity。

但不要删除 residual。保留为：

```text
Ablation: descriptor residual under protected trust region
```

并诚实报告：

```text
Residual descriptor improves some medians but destabilizes strict recall;
therefore the main model uses native descriptors and only distills localization utility.
```

这句话反而会显得你的研究判断很成熟。

---

## 10. 当前论文主表建议

现在主表不要放太多历史路线。建议主表先这样设计：

```text
Method                                      Desc      Selector      Single path   Median t/r       R10     R5      R2
Native STDLoc                               native    no            yes           ...
Loc-GS soft locability prior                native    soft prior    yes           ...
Loc-GS residual descriptor                  residual  no/weak       yes           ...
Loc-GS selector-only, sampled-score update  native    yes           yes           12.4238/...      0.4857  0.2815  0.0879
Loc-GS selector-resampled                   native    yes           yes           TBD
```

消融表：

```text
Ablation                     Median      R10      R5      R2
Native
Uniform gate
Permuted selector
Inverted selector
Selector locability-only
Selector detector-only
Selector both
Selector resampling
Residual + selector
```

诊断表：

```text
SceneMatchNet
LoFTR rendered matching
Oracle PROSAC ordering
Quality gate upper bound
q80 rehearsal
```

quality gate 现在不要再作为主方法，而是作为：

```text
upper-bound / self-map validation diagnostic
```

README 当前仍然把 self-map quality gate 写成 strongest evidence。([GitHub](https://github.com/Arthurshen926/Loc-GS)) 但从你这轮结果看，建议更新 README：把 `selector005_native_desc` 提升为当前 clean main candidate，把 quality gate 降为 diagnostic / upper-bound，否则叙事会冲突。

---

## 11. 你现在应该更新 README / docs 的核心段落

建议把 README 的 mainline 改成：

```text
Current clean main candidate, 2026-05-16:

STDLoc/native descriptor backbone
  -> virtual self-localization feedback
  -> unified localization selector / locability field distillation
  -> STDLoc-compatible map export with native descriptors
  -> one OpenCV PROSAC PnP path
  -> STDLoc-style dense refinement

The current paper-facing result is selector-only native descriptor export:
self-localization feedback improves the feature selection / locability field,
not the descriptor field. Residual descriptor reconstruction is retained as
an ablation because it is less stable on strict recall.
```

然后明确：

```text
Main claim:
Self-localization feedback can be distilled into a single localization utility field
that improves downstream geometric localization under a fixed native descriptor
and single-path PnP pipeline.

Non-claim:
We do not claim descriptor replacement or inference-time branch selection.
```

这个更新能防止 CodexPotter 后续又把项目带回 residual/branch/SceneMatchNet 方向。

---

## 12. 下一轮最小行动清单

按优先级：

```text
1. 把 20260516 selector-only native-desc 结果做成 paper-safe audit bundle。
2. 生成 per-scene 表和 query-level recovered/lost 分析。
3. 做 uniform/permuted/inverted selector causality ablation。
4. 实现 selector-only training mode，冻结 residual。
5. 实现 selector-guided resampling，而不是只更新 sampled_scores。
6. 重新跑 q80 -> full，但 blend / recipe 必须从 self-map 或固定协议决定。
7. residual descriptor 暂时只放 ablation。
```

其中第 1-3 步是证据闭环，第 4-5 步是方法增强，第 6 步是最终验证。

---

## 13. 投稿级目标的现实判断

当前阶段可以说：

```text
方向被验证；
single-path selector-only 有稳定小幅 full Cambridge 正结果；
代码结构和审计规则已经比之前更健康；
main claim 应该从 descriptor reconstruction 转向 localization utility reconstruction。
```

但还不能说：

```text
强 SOTA；
大幅提升；
descriptor reconstruction 成功；
完整顶刊证据闭环完成。
```

我建议你把内部里程碑设为：

```text
最低可投稿线：
full Cambridge selector-only / selector-resampled
R5 +1.5pp 以上，R2 +1.0pp 以上，
median translation 正向，R10 不掉，
至少 3/5 scenes 正向，
有 causality ablation 和 split audit。

较强投稿线：
R5 +2~3pp，R2 +1.5~2pp，
failure taxonomy 能明确恢复 baseline 失败案例，
额外数据集或自采真实场景也有正结果。

顶刊强线：
不仅 Cambridge 提升，
还能证明 selector 改善 PnP inlier geometry / coverage / hard-negative suppression，
并在真实场景问题上解释 baseline 为什么失败、Loc-GS 为什么修复。
```

现在最好的策略不是继续扩张，而是**冻结主线：selector-only native descriptor + localization utility field reconstruction**。这条线最干净、最可解释、最不容易被审稿人质疑，也最符合你“定位反馈影响重建，而不是测试时调参”的初衷。

## Prompt:
2026/5/16 21:42:57

我可不可以这样理解：我们不动feature gaussian本身，而是重建一个selection或者算是sampling field，还有就是不要用codexpotter。现在精度提升太小了，感觉做的所有改进尝试都是小打小闹。此外，除了精度评估，实时性也可以纳入评估标准，如果我们对比STDLoc等方法速度更快也可以作为一个点。

## Response:
2026/5/16 21:42:59

我会先厘清你现在究竟是在重建 feature Gaussian 还是 localization utility / sampling field，再判断速度能否成为主贡献而非附带指标。

## Response:
2026/5/16 21:43:01

> **浏览STDLoc运行时性能与最新仓库信息**
>
> [From Sparse to Dense: Camera Relocalization with Scene ...](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.pdf)
> [From Sparse to Dense: Camera Relocalization with Scene ...](https://zju3dv.github.io/STDLoc/)
> [GitHub · Change is constant. GitHub keeps you ahead. · GitHub](https://github.com/)
> [From Sparse to Dense: Camera Relocalization with Scene ...](https://arxiv.org/html/2503.19358v1)
> [GitHub](https://en.wikipedia.org/wiki/GitHub)
> [StdLoc — SeisComP Release documentation](https://www.seiscomp.de/doc/apps/global_stdloc.html)
> [GitHub - Apps on Google Play](https://play.google.com/store/apps/details?hl=en_US&id=com.github.android)
> [From Sparse to Dense: Camera Relocalization with Scene ...](https://www.researchgate.net/publication/390176191_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_Feature_Gaussian_Splatting)
> [The GitHub Blog: Home](https://github.blog/)
> [CVPR2025 - STDLoc, a new camera relocalization method ...](https://www.youtube.com/watch?v=auodnGC51tw)
> [r/github](https://www.reddit.com/r/github/)
> [GitHub Status](https://www.githubstatus.com/)
> [GSLoc: Visual Localization with 3D Gaussian Splatting](https://www.semanticscholar.org/paper/GSLoc%3A-Visual-Localization-with-3D-Gaussian-Botashev-Pyatov/48d535b0734a7b6d9e92d141af06d41a9f48a23f)
> [Enhancing Visual Relocalization with Dense Scene ...](https://openreview.net/pdf/21bba14db57c3118f7222c5499e180cf675a1fd2.pdf)
>
> **寻找打开STDLoc和可能的仓库**
>
> [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/html/2503.19358v1)
> [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
> [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting
  ](https://zju3dv.github.io/STDLoc/)
>
> **查找GitHub README中的更多内容**
>
> [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
>
> **搜索STDLoc时间与运行时相关信息**
>
> [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/html/2503.19358v1)
>
> **考虑密集姿态与早期退出优化**
>
> 如果允许在稀疏阶段后“早期退出”，或通过减少密集迭代/降低分辨率来提高速度，可能会有所改进。可以避免在高置信度查询中进行密集匹配，虽然准确率可能会降低，但如果稀疏选择器提高了质量，还是能保持召回。需要避免每个查询的分支选择问题。
>
> 已思考 3m 59s

## Response:
2026/5/16 21:46:59

可以这样理解，而且我建议你现在就**明确转向这个理解**：

> **我们不再试图改动 Feature Gaussian 的几何、颜色或 native descriptor 本体，而是在 Feature Gaussian 之上重建一个 localization-aware selection / sampling field。这个 field 决定哪些 Gaussian / landmarks 更值得被采样、匹配、送入 PnP、参与 dense refinement。**

这不是退而求其次。相反，这可能比继续改 descriptor 更像一个清楚的科研问题。

STDLoc 本身已经有 Feature Gaussian、matching-oriented sampling、scene-specific detector、sparse PnP 和 dense refinement。它的 sampling 目标主要是从大量 Gaussians 中选择“匹配友好、分布均匀、多视角可见”的 landmarks；论文也明确说 exhaustive matching 很耗时，并且大量 ambiguous Gaussians 会伤害 feature matching。([arXiv](https://arxiv.org/html/2503.19358v1)) 你的新问题可以定义为：

> **STDLoc 的 sampling score 是 appearance / visibility / distribution driven；Loc-GS 要学习的是 localization-outcome driven utility。也就是说，哪些 Gaussian 不只是“看起来能匹配”，而是在真实 PnP / dense refinement 中真正提高位姿估计质量。**

这比“我重建了更好的 descriptor”更稳，因为你当前实验已经显示 native descriptor 更稳定，而 selector-only 才是有效信号。

---

## 1. 你的新理解应该再精确一点

不是简单地说：

> 不动 Feature Gaussian，只重建 selection field。

而应说：

> **不改变 Feature Gaussian 的底层 descriptor / geometry / radiance field；在其上学习一个附加的 localization utility field，并用它导出 STDLoc-compatible sampling / ranking / locability payload。**

这个 field 可以有三种落点：

```text
per-Gaussian selector score
  -> sampled_idx.pkl        决定哪些 landmarks 进入 sparse matching

per-landmark sampled score
  -> sampled_scores.pkl     决定 detector / PROSAC / ranking 里的优先级

dense locability prior
  -> dense refinement       决定 dense matching / refinement 更信任哪些区域
```

如果你只是改 `sampled_scores.pkl`，那更像 score reweighting，提升小是正常的。要让它成为真正的方法，下一步必须让 selector 参与：

```text
which landmarks are selected
not only how already-selected landmarks are scored
```

也就是从 **selection score field** 进入 **sampling field**。

---

## 2. 这条路线的论文叙事更干净

建议把主线从：

```text
定位引导的特征重建
```

改成：

```text
定位反馈蒸馏的 Gaussian sampling field
```

或者更完整：

```text
Self-localization Distilled Sampling Field for Efficient and Accurate Feature-Gaussian Relocalization
```

中文逻辑可以是：

> 现有 Feature Gaussian localization 方法把 descriptor reconstruction 和 landmark sampling 分开处理。Feature reconstruction 追求像素/特征空间一致性，但 PnP localization 真正需要的是少量高效、稳定、低歧义、几何分布良好的 2D-3D correspondences。我们提出用 virtual self-localization episode 观察每个 Gaussian / landmark 对最终定位的贡献，并将这些反馈蒸馏成一个 localization utility / sampling field。该 field 不替换原始 descriptor，不改变 PnP 后端，而是重新决定哪些 features 被采样、排序和信任。

这个叙事的好处是：

```text
1. 不需要强行解释为什么 residual descriptor 没有稳定提升；
2. 不会被质疑“你只是换了 descriptor backbone”；
3. 和 STDLoc 的 matching-oriented sampling 形成直接对话；
4. 可以自然引入效率指标；
5. 可以把小幅精度提升转化为 accuracy-efficiency Pareto 改善。
```

---

## 3. “所有改进都是小打小闹”的根本原因

你现在的小幅提升很可能不是因为方向错，而是因为你当前修改还没有真正改变 localization 的关键瓶颈。

目前最可能的情况是：

```text
native descriptor 已经很强；
STDLoc sampling 已经很强；
当前 selector005 主要是在已有 sampled landmarks 上做轻微 score nudging；
真实进入 PnP 的候选集合变化有限；
dense refinement 仍然占据主要计算和误差来源；
所以 full Cambridge 上只能得到 0.3-1.0pp 级别 recall 增益。
```

这不是投稿级强结果，但它给了一个清楚判断：

> **不要继续小修小补 residual / alpha / score weight / q80 调参。要做 structural change：让 selector 改变 landmark budget、sampling set、PnP candidate set，以及 dense refinement 触发策略。**

也就是说，下一步应该从“涨点”转向“Pareto 改善”：

```text
同等精度下更快；
同等速度下更准；
同等 landmark budget 下更准；
更小 landmark budget 下不掉点；
高置信 query 可跳过或减少 dense refinement。
```

---

## 4. 实时性完全可以纳入评估，但要小心定位

速度可以成为贡献点，而且很适合你的 selection / sampling field 叙事。

STDLoc 官方论文报告其 Cambridge 上单张 RTX 4090、dense stage 迭代一次时约 7 FPS，总耗时约 152.3 ms；其中 dense pose estimation 72.8 ms、rasterization 23 ms、dense matching 13.2 ms，sparse matching 17.4 ms、sparse pose 15.8 ms。([arXiv](https://arxiv.org/html/2503.19358v1)) 这说明两个事实：

```text
1. 单纯减少 sparse landmarks，最多主要影响 sparse matching + sparse PnP，大约 33ms 量级；
2. 如果 dense refinement 不变，总体速度提升会有限；
3. 想要明显实时性贡献，必须同时影响 dense stage 或引入可靠的 early-exit / budgeted refinement。
```

所以速度主张不能只写：

> 我们 selector 更好，所以更快。

而要写成：

> 我们的 sampling field 支持 budgeted localization：在不同 landmark budget、dense refinement budget 下形成更优 accuracy-speed Pareto frontier。

这比单个 FPS 数字更有说服力。

---

## 5. 建议把方法拆成 full mode 和 fast mode

你可以保留两个部署模式：

### Full mode：主精度模式

```text
native descriptor
+ self-localization distilled sampling field
+ selector-guided landmark resampling
+ normal sparse PnP
+ normal dense refinement
```

目标：

```text
R5 / R2 / median 全部比 native STDLoc 稳定正向。
```

### Fast mode：效率模式

```text
native descriptor
+ selector-guided smaller landmark set
+ fewer sparse candidates / fewer PROSAC iterations
+ confidence-based dense refinement early exit
```

目标：

```text
在接近 native STDLoc 精度的情况下显著降低 latency。
```

这里的 early exit 要非常谨慎。它不能变成“per-query branch selection 选哪个模型”，但可以是一个预先固定的 deterministic inference policy，例如：

```text
if sparse_inlier_count >= T_count
and sparse_inlier_ratio >= T_ratio
and PnP_reprojection_median <= T_px
and pose_information_logdet >= T_info:
    skip dense refinement
else:
    run dense refinement
```

这些阈值必须来自 self-map / validation，不得看 test query 结果。这样它更像 **anytime / budgeted inference**，不是 oracle branch selection。

STDLoc 自己也显示 dense feature refinement能把 Cambridge average median 从 sparse 的 `13.8cm / 0.21deg` 提升到 `10.1cm / 0.14deg`，所以不能粗暴删掉 dense stage；但它也提到 dense stage 会因 floaters 造成 recall 轻微损失。([arXiv](https://arxiv.org/html/2503.19358v1)) 这给了你一个很好的切入点：

> 不是所有 query 都需要 dense refinement；不是所有 dense refinement 都值得做。selection field 可以预测 sparse pose 是否已经足够可靠，以及 dense refinement 是否可能被不可靠区域污染。

---

## 6. 速度评估应该怎么设计

你应该把速度从“附加指标”升级为“正式 Pareto 评估”。

### 6.1 必报指标

```text
Accuracy:
- median translation / rotation
- R@10cm,5deg
- R@5cm,5deg
- R@2cm,2deg
- per-scene results
- recovered / lost query count

Efficiency:
- end-to-end latency, mean / median / p95
- FPS
- sparse matching time
- sparse PnP time
- rasterization time
- dense matching time
- dense pose refinement time
- total GPU memory
- map size
- number of sampled landmarks
- number of query keypoints
- number of 2D-3D candidate matches
- number of PROSAC / RANSAC iterations
- dense refinement invocation rate
```

### 6.2 必须做公平计时

```text
same GPU
same image resolution
same query split
same feature extractor
same batch size, ideally batch=1
same CUDA synchronization policy
same warmup count
same number of timed queries
include feature extraction unless明确说明双方都用 cached features
offline map training / export time 单独报告，不混入 query latency
```

不要直接拿你机器上的数字和 STDLoc paper 的 7 FPS 硬比。STDLoc 的 7 FPS / 152.3 ms 可以作为参考背景，但正式 claim 必须来自你本地同一环境下重跑 native STDLoc 与 Loc-GS。([arXiv](https://arxiv.org/html/2503.19358v1))

---

## 7. 最关键的实验：landmark budget sweep

这是现在最值得做的一组实验。

不要只比较：

```text
STDLoc 16k landmarks
vs
Loc-GS 16k landmarks
```

而要比较：

```text
N = 16384, 8192, 4096, 2048, 1024
```

方法：

```text
A. STDLoc native matching-oriented sampling
B. random / FPS baseline
C. current selector-score update
D. selector-guided resampling
E. selector-guided resampling + coverage
F. selector-guided resampling + coverage + hard-negative penalty
```

你要找的不是单点提升，而是这类结果：

```text
STDLoc native 16k:
12.57cm, R5 27.76%, 152ms

Loc-GS sampling 8k:
12.50cm, R5 28.0%, 130ms

Loc-GS sampling 4k:
12.70cm, R5 27.4%, 115ms

STDLoc native 4k:
13.80cm, R5 24.0%, 115ms
```

如果出现这种趋势，论文就有力很多：

> 在相同 landmark budget 下更准；在相同 accuracy 下更快；在相同 latency 下更高 recall。

这就是 sampling field 的真正价值。

---

## 8. 你的主贡献可以从“精度涨点”变成“有效 landmark 的发现”

现在更合理的科学问题是：

> 在 Feature Gaussian 中，并不是所有可渲染 / 可匹配 Gaussians 都具有定位价值。哪些 Gaussian 应该被保留为 localization landmarks？这个问题不能只由 descriptor reconstruction loss 或 local matching score回答，而应由下游 self-localization outcome回答。

对应贡献可以写成三条：

### Contribution 1：Localization utility field

用 self-localization feedback 学习 per-Gaussian localization utility：

```text
utility = inlier probability
        + pose information
        + multi-view consistency
        - hard-negative risk
        - ambiguity risk
        + coverage contribution
```

### Contribution 2：Selector-guided sampling

不是改 descriptor，而是导出新的 STDLoc-compatible sampling payload：

```text
sampled_idx.pkl
sampled_scores.pkl
locability field
detector target / prior
```

### Contribution 3：Accuracy-efficiency Pareto

证明该 sampling field 在不同 landmark budget / dense refinement budget 下优于 STDLoc native sampling：

```text
same accuracy -> lower latency
same latency -> higher recall
same landmark budget -> better pose accuracy
```

这比“full Cambridge 上 R5 +0.39pp”更像顶刊论文。

---

## 9. 不用 CodexPotter是合理决定

我同意你现在不要用 CodexPotter。

当前阶段不是缺代码产能，而是缺**研究主线收束和假设验证**。CodexPotter 这种自动推进工具很容易把项目带回：

```text
多加一个 loss
多加一个 branch
多加一个 scorer
多加一个 ablation
```

但你现在真正需要的是：

```text
少做东西；
冻结主线；
证明一个结构性假设；
跑更有信息量的实验。
```

建议改成手工推进：

```text
每次只开一个小分支；
每个分支只验证一个 hypothesis；
每个实验必须先写 expected outcome；
如果没达到预期，记录为 negative evidence；
不要让工具自动扩展 scope。
```

现在最好停止“自动化继续开发”，改成“研究假设驱动开发”。

---

## 10. 下一阶段我建议你只做三件事

### 第一件：把主线改名并冻结

更新 README / docs：

```text
Main method:
Self-localization distilled Gaussian sampling field.

We keep native STDLoc descriptors and the standard geometric backend.
We learn an additional localization utility / selection / sampling field.
The field is exported into STDLoc-compatible landmark sampling, ranking, and locability payloads.
```

同时把 residual descriptor 写成：

```text
diagnostic ablation, not main method
```

---

### 第二件：实现真正 selector-guided resampling

不要再只改 score。要生成新的 sampled landmark set：

```text
输入：
- native STDLoc map
- per-Gaussian selector score
- visibility / coverage info
- hard-negative risk optional

输出：
- sampled_idx.pkl
- sampled_scores.pkl
- manifest.json
```

采样目标：

```text
maximize selector utility
subject to landmark budget
subject to spatial coverage
subject to multi-view visibility
subject to descriptor ambiguity penalty
```

核心对比：

```text
same N landmarks:
STDLoc native sampling vs Loc-GS sampling
```

这一步是从“小打小闹”进入“结构性方法”的关键。

---

### 第三件：做 speed-accuracy Pareto

固定测试表：

```text
Budgets:
N_landmarks = 16384 / 8192 / 4096 / 2048 / 1024
dense_mode = full / confidence-gated / sparse-only diagnostic

Methods:
STDLoc native
STDLoc native reduced budget
Loc-GS selector-score update
Loc-GS selector-resampled
Loc-GS selector-resampled + fast mode
```

输出：

```text
Table 1: full accuracy
Table 2: module latency breakdown
Figure 1: R5 vs latency
Figure 2: R2 vs latency
Figure 3: median translation vs number of landmarks
Figure 4: recall vs number of landmarks
```

这套实验比继续做 `selector0.03/0.05/0.10` 有意义得多。

---

## 11. 现在不要这样做

我建议暂停这些方向：

```text
1. 不要继续主推 residual descriptor；
2. 不要继续让 SceneMatchNet 升级成主方法；
3. 不要继续调 selector blend 来找 test 最优；
4. 不要做 per-scene recipe；
5. 不要为了精度引入 query-time branch selection；
6. 不要只看 full Cambridge macro 平均；
7. 不要把 0.3pp recall 提升包装成强 claim。
```

尤其是第 3 点非常重要。`gate_locability_blend=0.05` 现在只能作为当前经验值，不能继续根据 full test 调。如果要 paper-safe，它必须来自 self-map calibration 或固定协议。

---

## 12. 最后的判断

你的理解可以进一步凝练成一句话：

> **Loc-GS 不再试图重建一个更强 descriptor field，而是重建一个 localization-aware sampling field：它学习哪些 Feature Gaussians 对下游几何定位真正有用，并用这个 field 在保持 native descriptor 和 PnP 后端不变的前提下，改善准确率、召回率和速度-精度权衡。**

这是一个比“精度小涨点”更有潜力的方向。

当前最现实的投稿路径不是：

```text
我们比 STDLoc 准很多
```

而是：

```text
我们揭示并解决了 Feature Gaussian localization 中的有效 landmark selection 问题。
在相同 descriptor 和几何后端下，self-localization feedback 学到的 sampling field 能产生更好的 accuracy-efficiency Pareto。
```

如果最终你能证明：

```text
R5/R2 小幅正向
+ landmark 数量减少 2-4 倍
+ latency 降低 20-40%
+ per-scene 不崩
+ failure cases 可解释
```

这会比单纯追求 Cambridge 上 `+0.3pp` 更像一篇完整论文。

---

**Sources:**

- [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/html/2503.19358v1)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)