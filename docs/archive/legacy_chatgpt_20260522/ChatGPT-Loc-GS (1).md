# Loc-GS

**User:** Anonymous  
**Created:** 5/16/2026 0:57:35  
**Updated:** 5/17/2026 23:13:09  
**Exported:** 5/17/2026 23:57:58  
**Link:** [https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb](https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb)  

## Prompt:
2026/5/17 23:07:23

按照我们的讨论内容，又推进了几轮，git仓库也已更新。有效进展是有的，但还没有达到“稳定超越 STDLoc、可作为顶刊主结果”的程度。 

  已经取得的有效进展：

  1. 方向已经收束：主线从 residual descriptor 强 claim 转为更稳的 Loc-GS sampling / localization utility field + native STDLoc descriptor，保持单一路径推理，避免 evaluator drift 和
     test leakage。
  2. 基础设施基本打通：已经有 selector-only 训练、selector-guided resampling/export、audited cache、profile/timing wrapper、causality ablation、manifest/report 体系，且相关单元测试
     之前通过。
  3. 有一个弱但真实的正信号：selector-only native descriptor 在 Cambridge full 旧诊断结果里有小幅提升，例如 median translation 从 12.5725 cm 到 12.4238 cm，R5 从 0.2776 到 0.2815，
     R2 从 0.0848 到 0.0879。但这个旧结果的 split audit 是 unknown，只能算候选/诊断，不是 paper-safe 主表结果。
  4. 排除了很多不稳分支：qcov12 same-budget、native12 子集、fixed prior、inlier-only early exit、qcov f10/r001 8k、pose-pair 小修等都没有形成稳定、可推广、可宣称的结果。这个清理是
     有价值的，避免继续在不可靠方向上消耗实验弹药。
  5. 最新被中断前的实验有一个很弱的新候选：base-qcov12-churn threshold 64 guard 在 train-q80 上宏平均略正：
      - TE -0.00287 cm
      - angle -0.001774 deg
      - R10 +0.0025
      - R5 +0.0025
      - R2 不变
        但 KingsCollege 有小回退，而且还没有 shifted validation、solo timing、paper-safe 全流程验证，所以不能推广。

  还缺的关键东西：

  - 还没有 paper-safe 主结果。
  - 还没有强而稳定的速度收益。
  - 最新候选还没做 shifted train-s3/q75 验证。
  - 不能启动 full/test 结果宣称，除非先把 train/self-map 候选冻结并通过 audit。

  如果继续推进，最合理的下一步是：对 base-qcov12-churn threshold 64 + hq005/protnative995 这条 guard 做 shifted train-s3/q75 accuracy validation；如果过，再做公平 solo timing；如果
  不过，就把它归入负结果，转向更结构性的 sparse-pose-aware geometry sampling objective。
主线有没有需要调整的地方？如何加速项目进度？

## Response:
2026/5/17 23:07:26

我会先把主线判断和下一步验证压缩成“是否调整研究假设、候选门禁、加速策略”三件事；重点是尽快得到 paper-safe yes/no，而不是继续堆模块。

## Response:
2026/5/17 23:07:38

我已核对到新文档和 probe 记录：主线不用大改，但候选过多且晋级标准过宽，部分 shifted 验证已显示只是 accuracy-neutral。接下来应收缩为少量强假设，尤其是受保护的 hard-query coverage / map compression 与硬性停止规则。

## Response:
2026/5/17 23:13:07

你的主线**不需要推翻**，但需要再收束一层：

> **从“Loc-GS sampling / localization utility field”进一步改成“risk-controlled, sparse-pose-aware sampling field”。**  
> 也就是不再只问“哪些 Gaussian 分数更高”，而是问“在不破坏 native STDLoc hard-case support 的前提下，哪些替换或新增 landmarks 能提高 PnP 的 inlier 几何、覆盖和速度-精度 Pareto”。

这一步很重要，因为你现在的实验已经说明：**单纯学一个 selector score 或轻微 resampling 还不足以稳定超越 STDLoc。**

---

## 1. 对当前进展的判断

我看了仓库当前 README 和 2026-05-17 的几个文档，主线已经正确收束到：

```text
native STDLoc descriptor / geometry / radiance field
  -> train/rendered self-localization rehearsal
  -> per-Gaussian localization utility / selector / sampling field
  -> STDLoc-compatible sampling, ranking, locability payload
  -> single-path matching + OpenCV PROSAC PnP + dense refinement
```

README 也已经明确写了：当前 clean candidate 是 `selector005_native_desc`，保持 native descriptor，只把 learned selector 导出到 locability 和 detector score payload；full Cambridge 的提升是正向但小，不能作为强 SOTA claim。([GitHub](https://github.com/Arthurshen926/Loc-GS))

你现在的问题不是“方向错了”，而是：**当前 sampling field 还只是 score-level 或 weak replacement-level 机制，没有真正形成一个能保护 hard cases 的结构性地图选择目标。**

这在最新实验里已经很明显：qcov12 在 train-q80 上确实有速度信号，五个场景的 mean/median latency 多数下降；但 OldHospital 和 StMarysChurch 出现 accuracy / R10 损失，所以它只是 speed/accuracy frontier diagnostic，不是可冻结主结果。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md))

更关键的是，后续 isolation 已经指出：OldHospital / StMarysChurch 的损失不是简单 qcov prefix 导致的，而是 broader source-pool replacement / fill 导致的；qcov+protected-native 能恢复 native metrics，但这样又退回到“几乎不改 sampled set”。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md))

所以结论是：

> **当前 scalar selector / qcov / protected native / churn guard 这一系列已经接近上限。继续调 threshold、fraction、prior weight，不太可能自然长成顶刊主结果。**

---

## 2. 最新 churn64 guard 不建议再推进到 full/test

你提到的 `base-qcov12-churn threshold 64 guard`，公开文档里已经记录了 shifted train-s3/q75 验证：它选择 additive union 用于 ShopFacade、KingsCollege、GreatCourt、StMarysChurch，OldHospital 回退 native12；结果是 macro dense median、R10、R2 都不变，R5 只从 `0.4507` 到 `0.4533`，本质上是一条 GreatCourt query 的变化。文档也明确建议 keep the guard off Cambridge test，除非以后有 same-budget export rule 能保留 qcov12 的速度信号。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md))

因此我建议你直接做这个决策：

```text
base-qcov12-churn64 guard:
  paper role = diagnostic / rejected as main candidate
  reason = shifted validation only accuracy-neutral, no solo timing gain, no strong effect size
  next action = 不跑 full/test，不继续调 hq005/protnative995
```

如果你本地还有一个和文档不同的 `hq005/protnative995` 变体，也应该按同一门槛处理：

```text
只有同时满足：
1. train-q80 明显正向；
2. shifted train-s3/q75 明显正向；
3. solo timing 有 ≥10% 级别收益；
4. 无 scene-level strict recall 明显回退；
5. manifest/split audit 全通过；

才允许进入 full/test。
```

现在这种 `TE -0.00287cm, R5 +0.0025` 的量级不值得消耗一次 full/test 机会。

---

## 3. 主线应该怎么调整

建议把主线从：

```text
Loc-GS sampling / localization utility field
```

升级为：

```text
Risk-controlled sparse-pose-aware Gaussian sampling field
```

核心变化如下。

### 3.1 不再追求“替换更多 landmarks”

现在的实验证据已经说明：替换 native sampled set 很危险。source-pool selector 几乎不改 sampled set 时 accuracy 不变；all-Gaussian 或 qcov replacement 一旦掉 native hard-case support，就会损失 recall。文档里 train-q80 分析显示 source-pool selector 只改 7 个 ID，指标完全等于 native8192；而 all-Gaussian / pose strict replacement 会掉大量 native IDs 并损失 hard cases。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md))

所以新策略不是：

```text
用 learned selector 替换 STDLoc sampling
```

而是：

```text
在 native STDLoc support set 上做风险受控的局部编辑
```

也就是：

```text
native sampled set = safe backbone
learned sampling field = propose additions / replacements
support guard = reject changes that hurt hard-query coverage
```

### 3.2 目标函数要从 average score 改成 hard-query support

现在很多失败来自 scene-specific hard cases，而不是平均分数。建议把 map-level objective 写成：

$$
S^\* =
\arg\max_{S, |S|=B}
\left[
\sum_{i \in S} u_i
+
\lambda_1 \operatorname{CVaR}_{q \in Q}^{\text{worst}}
C_q(S)
+
\lambda_2 \log\det(H_q(S))
-
\lambda_3 R_{\text{hard-neg}}(S)
\right]
$$

其中：

$$
C_q(S) = \sum_{i \in S} p_{qi}^{inlier} \cdot v_{qi} \cdot m_{qi}
$$

含义是：

```text
u_i                  self-localization selector utility
C_q(S)               query q 在当前 sampled set 下的可定位支持量
CVaR worst           关注最差一批 train/self-map hard queries，而不是平均 query
H_q(S)               PnP pose-information / geometry conditioning
R_hard-neg(S)        hard-negative / false positive 风险
```

约束应该包括：

```text
1. same landmark budget；
2. native hard-case support 不得下降超过 δ；
3. 每个 train/self-map pose cluster 至少保留 k 个 reliable landmarks；
4. replacement 必须有 positive self-map evidence；
5. 任何新增 landmark 不得显著提高 hard-negative risk；
6. full inference 仍然 single-path。
```

这才是“sparse-pose-aware geometry sampling objective”，而不是继续调 qcov/hq/protnative 阈值。

### 3.3 speed 目标也要改

你现在的 speed 结果说明：**减少 landmark 数量本身不一定加速**。ShopFacade q20 里 4k / 2k fast-cap extension 反而因为 sparse pose time 增加，没形成 Pareto point；文档也明确写了这是 pure landmark-budget speed claim 的负证据。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/speed_accuracy_probe_20260517.md))

所以 speed 主线不应是：

```text
更少 landmarks -> 更快
```

而应是：

```text
更高 early inlier quality / 更好 proposal ordering / 更稳定 PnP support
  -> 更少 RANSAC/PROSAC 工作量
  -> 固定低 iteration budget 下仍保持 accuracy
```

也就是说，speed claim 应围绕：

```text
same budget, lower PnP workload
same PnP cap, higher recall
same accuracy, lower latency
```

而不是简单扫 `2048/4096/8192`。

---

## 4. 现在应该暂停哪些方向

我建议明确暂停：

```text
1. qcov12 + churn threshold 继续调参；
2. hq005/protnative995 阈值族继续穷举；
3. fixed prior005 / prior weight 调参；
4. inlier-only dense early exit；
5. native12 subset 降预算路线；
6. residual descriptor 主线；
7. additive union 直接进 full/test；
8. 小 q20 test diagnostic 继续指导 recipe。
```

dense early exit 目前也不该成为主线。文档显示，inlier-only threshold 700 只能跳过 3% queries、节省约 4.3 ms，但会掉 R10/R5；唯一不退化的 threshold 800 只跳过 1/400 queries，平均节省约 0.33 ms。后续 reliability signal 的单特征 AUC 也只有 0.581，还不足以实现稳定 fast mode。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md))

---

## 5. 加速项目进度的关键：少跑，但每次跑都带明确淘汰标准

你现在不是缺实现能力，而是实验空间已经膨胀。加速方式不是自动化继续开发，而是把候选推进做成严格 funnel。

### Stage A：只做 train/self-map 候选，不碰 full/test

每个新 candidate 必须先通过：

```text
train-q80 accuracy
shifted train-s3/q75 accuracy
split audit
map-set audit
query-level recovered/lost audit
```

通过条件建议定死：

```text
Accuracy candidate:
  macro R5 ≥ +1.0pp 或 R2 ≥ +0.5pp
  median translation 不得明显变差
  no scene R5/R2 loss > 1.25pp
  no new meter-scale hard failure

Speed candidate:
  same accuracy within tolerance
  solo mean 或 median latency ≥ 10% faster
  no per-scene latency regression larger than noise band
```

达不到就直接归档，不再调第二轮 threshold。

### Stage B：只有 Stage A 通过，才做 solo timing

parallel accuracy profile 只能用于筛选，不能用于 speed claim。仓库当前文档也多次把 parallel timing 标为 ignored。solo timing 只对已经通过 accuracy gate 的 candidate 做，避免你把时间花在“已经不够准”的 candidate 上。

### Stage C：只有 Stage A + B 都通过，才冻结 recipe

冻结内容必须包括：

```text
map export rule
landmark budget
sampling objective weights
PROSAC / dense iteration caps
prior weights
fallback rule
split audit
manifest
```

冻结后才能跑 full/test。否则 full/test 会继续变成调参反馈。

---

## 6. 下一步最值得做的一个结构性方向

我建议你只推进一个新方向：

> **Native-support-preserving, query-risk-aware sampling optimizer.**

具体实现目标：

### 输入

```text
native sampled_idx / sampled_scores
self-map feedback cache
per-query sparse/dense outcome
per-landmark inlier evidence
per-landmark hard-negative risk
visibility / pose cluster / spatial coverage
selector score
```

### 输出

```text
new sampled_idx.pkl
new sampled_scores.pkl
manifest.json
map_set_audit.json
support_audit.json
```

### 核心机制

不是简单 top-k，而是：

```text
1. 先从 train/self-map logs 找 native hard-query support set；
2. 对每个 hard query 建立 support constraint；
3. learned selector 只允许在不破坏 support constraint 的情况下提出 replacement；
4. replacement 必须通过 pose-information / hard-negative / coverage 过滤；
5. 最终 sampled set 必须报告：
   - native kept count
   - native dropped count
   - dropped hard-support count
   - added non-native count
   - per-query support delta
   - worst-10% query support delta
```

### 最小公式

对每个 query $q$，定义 native support：

$$
C_q(S_{native})
=
\sum_{i \in S_{native}} a_{qi}
$$

候选 set 必须满足：

$$
C_q(S_{candidate}) \ge C_q(S_{native}) - \delta_q
$$

特别是 hard queries：

$$
q \in Q_{hard}: \quad C_q(S_{candidate}) \ge C_q(S_{native})
$$

然后在可行集合里最大化 learned utility：

$$
\max_S
\sum_{i \in S}
(
w_s s_i
+
w_p p_i^{pose}
-
w_h h_i^{neg}
+
w_c c_i^{coverage}
)
$$

这比 `churn <= 64` 更有科学含义。`churn <= 64` 只是现象级 guard；support constraint 才是第一性原理解释。

---

## 7. 速度方向如何保留

速度可以保留，但不要再说“实时性”作为当前目标。公开文档里当前最好的诊断也还是约 364 ms/query，并且文档明确写了“不支持 realtime claim”。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md))

速度应该被写成：

```text
budgeted localization / accuracy-latency Pareto
```

而不是：

```text
real-time localization
```

更具体地，速度实验只保留三种：

```text
1. same map budget, lower fixed PROSAC/dense caps；
2. same accuracy, lower mean/median latency；
3. same latency, higher R5/R2。
```

并且需要分解：

```text
sparse matching time
sparse PnP time
dense rendering time
dense matching/refinement time
total latency
```

目前 profile wrapper 已经支持 sparse/dense iteration cap 和 prior weight overrides，并把这些写入 manifest，所以基础设施够用了。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/loc_gs/scripts/profile_stdloc_native.py))

---

## 8. 和 STDLoc / 新相关工作的关系

这个调整也有外部压力。STDLoc 本身已经提出 Feature Gaussian、matching-oriented sampling、scene-specific detector，以及 sparse-to-dense localization pipeline；所以你如果只说“我也做 sampling”，新意会不够。([arXiv](https://arxiv.org/html/2503.19358v1?utm_source=chatgpt.com))

同时，2026 年 ULF-Loc 已经直接攻击“Gaussian feature 有 alpha-blending bias”这个问题，并提出 unbiased landmark feature、keypoint-consensus sampling 和 local geometric consistency verification，还声称 Cambridge 上相对 SOTA mean median translation error 降低 17%、训练时间和显存显著低于 STDLoc。([arXiv](https://arxiv.org/abs/2605.04730))

这意味着你不能再把主 claim 放在：

```text
我们比 STDLoc descriptor 更好
```

而应放在：

```text
我们提出一种 self-localization feedback driven、risk-controlled 的 map-level sampling field；
它不是换 descriptor，而是解决 Feature Gaussian localization 中有效 landmark support 的选择和保守编辑问题。
```

如果最终精度仍然只是小幅，那么至少要有强 Pareto 或强 failure analysis；否则顶刊主结果会很难成立。

---

## 9. 一个更快的执行框架

接下来不要开很多分支。只保留三个轨道：

### Track 1：evidence cleanup

目的不是涨点，而是避免继续误判。

```text
- 把 qcov12、churn64、native12-subset、prior005、early-exit、posepair 全部标注 paper role。
- 每条写清楚：kept / rejected / diagnostic。
- 当前 selector005_native_desc 保留为 old diagnostic，不作为 paper-safe main result。
```

这个轨道完成后，你会少很多心理负担。

### Track 2：support-preserving sampling

只实现一个新 optimizer：

```text
native-hard-support extraction
per-query support matrix
CVaR / worst-query support objective
support-preserving replacement
same-budget export
```

第一轮只在 train-q80 + shifted train-s3/q75 上看：

```text
是否能在不掉 OldHospital / StMarysChurch 的情况下保留 qcov12 的正信号。
```

### Track 3：solo timing only after accuracy passes

不要再对不稳 candidate 做 timing。只有 Track 2 通过 accuracy gate，再做：

```text
native12 vs candidate12
same caps
same GPU
same split
same warmup
solo profile
```

如果 speed 没有 ≥10% 级别优势，就只保留 accuracy claim，不要硬做 realtime。

---

## 10. 我会怎么改你的当前主线文案

建议从：

> Loc-GS sampling / localization utility field improves STDLoc.

改成：

> **Loc-GS learns a risk-controlled localization support field for Feature Gaussian maps. Instead of replacing native descriptors or changing the geometric backend, it uses train/self-localization outcomes to identify which landmark edits preserve hard-query PnP support while improving sampling utility and budgeted localization efficiency.**

中文可以写成：

> Loc-GS 不改 native Feature Gaussian descriptor，也不改 PnP / dense refinement 后端，而是在其上学习一个定位支持场。该场不是简单给 Gaussian 打分，而是通过 train/self-localization episode 估计每个 landmark 对 hard-query PnP 支持、pose-information、hard-negative 风险和覆盖的贡献，并在保持 native hard-case support 的约束下进行保守的 same-budget map editing。

这个叙事能解释你已经观察到的负结果：为什么 all-Gaussian replacement 会坏，为什么 native-protected 能恢复，为什么 churn guard 有用但太弱，为什么下一步必须是 support-preserving objective。

---

## 11. 最直接的决策

我建议你现在做这几个决定：

```text
1. 不再推进 base-qcov12-churn64 / hq005 / protnative995 到 full/test。
2. 把它们归档为：qcov speed signal exists, but guard only accuracy-neutral.
3. 暂停 dense early-exit，除非有多特征 reliability predictor 在 shifted train 上 AUC 明显提升。
4. 放弃“减少 landmark 数量自然加速”的路线。
5. 启动 support-preserving sampling objective。
6. 所有候选必须 train-q80 + shifted train-s3/q75 双通过后，才做 solo timing。
7. solo timing 通过后，才允许一次 paper-safe full/test。
```

当前项目不是没有价值，而是已经进入“弱局部调参基本清完”的阶段。下一步要加速，就要停止小修小补，转向一个能解释当前所有正负结果的结构性目标：**保留 native hard-case support，同时用 self-localization feedback 做风险受控的 map-level sampling 编辑。**

---

**Sources:**

- [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/docs/audited_sampling_fast_probe_20260517.md)
- [From Sparse to Dense: Camera Relocalization with Scene ...](https://arxiv.org/html/2503.19358v1?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)