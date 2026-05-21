# Loc-GS

**User:** Anonymous  
**Created:** 5/16/2026 0:57:35  
**Updated:** 5/18/2026 14:55:31  
**Exported:** 5/18/2026 15:17:46  
**Link:** [https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb](https://chatgpt.com/c/6a07507f-30bc-83ea-80ca-e75a90a5d4fb)  

## Prompt:
2026/5/18 13:47:00

继续推进了一轮，git仓库已更新，这轮有效进展： 

  - 实现了 support-preserving 基础设施：
      - loc_gs/stdloc_native/selector_resampling.py
      - loc_gs/scripts/export_selector_resampled_map.py
      - tests/test_selector_resampling.py
  - 新增能力包括 native support reservation、support preservation audit、--support_guard_* 导出参数、support_audit.json 和 manifest 记录。
  - 验证通过：61 passed；git diff --check 通过。
  - 没有改 Cambridge/test split、STDLoc evaluator 或 metric；没有跑 full/test；所有实验都是 train/self-map Stage A。

  实验结论比较硬：

  - supportguardcount_ret90：hard-query support loss 修掉了，但 Stage A 失败，macro dense median_te +0.519cm，R5 只 +0.005。
  - supportguardall_ret90：所有 query support loss 都为 0，但仍失败，macro dense median_te +0.513cm，R5 +0.010，R2 +0.005，scene gate 不过。
  - OldHospital 早停探针：
      - ret95：R5 回到 native，R2 +0.0125，但 median_te 8.173cm vs native 7.743cm，仍退化。
      - ret975：median_te 8.509-8.580cm，更保守也没恢复。
      - positive-support + hard-negative 加权、hn100 诊断都没解决。
  - StMarysChurch：ret95/hn100 都没有恢复，ret95 median_te 7.932cm vs native 5.407cm，明显失败。

  报告已写入：

  - output/unified_lff_v2/reports/20260518_train_q80_supportguardall_ret90_qcov01_pose025_hq025_hn05_followup.json
  - output/unified_lff_v2/reports/20260518_stagea_earlystop_supportguard_followups.json
  - output/unified_lff_v2/reports/20260518_stagea_hn100_diagnostic_followups.json

  当前判断：这轮不是“没有进展”，而是把专家建议里的 support-preserving 方向做成了可审计实验，并证明“只保 support count / 只调 retention / 只加 hard-negative 权重”不够。下一步应该实
  现更强的 replacement admissibility 或 query-level replacement utility，禁止低证据非 native 替换进入主 candidate，而不是继续调 qcov/churn/weight。GPU 现在已空闲，没有遗留 profile
  进程。
本项目目前距离顶刊投稿的核心痛点问题是什么？请重新深度思考，针对3dgs定位方法中的特征选择这个课题，如何展开前沿、具备理论深度、具备可行性的研究？具体的技术路线如何制定？注意不要过多受之前的会话影响，不要浅尝辄止，而是给我一套完整的解决方案

## Response:
2026/5/18 13:47:02

我先把问题重置成一次顶刊级科学问题诊断：对齐近两年的 3DGS/Feature-Gaussian 定位主线和你仓库现状，重新定义“特征选择/采样/utility”这个课题。然后给出一套尽量不受现有局部 recipe 束缚的完整方案，聚焦核心科学问题、理论目标和可执行技术路线。

## Response:
2026/5/18 13:47:24

我现在会把焦点收紧到一个更像新问题的层面：不是继续给 Gaussian 打分，而是回答“什么样的 3DGS 实体才配当 PnP 地标”。Support-preserving 仍失败这个结果反而很关键，它说明只保 support count 不够，真正缺的是围绕 query-level 可解性、几何条件、歧义风险与 tail-risk 的理论对象和目标函数。

## Response:
2026/5/18 13:50:44

> **安全替代定理与实施**
>
> 提出了一个类似于增强学习中的安全策略改进的“安全替换定理”，即替换必须保持每个单元的支持矩阵和信息不降低。通过对信息矩阵的PSD性质进行测试，替换只有在其信息矩阵差异为PSD或至少支持计数保持不变时才允许。这是个很强的实现方法。
>
> **信息保持的稳健核心集与不确定性拆分**
>
> 现在方法变成了“信息保持支持保护”，而非单纯的支持计数。应该引入许多一对多的结构不匹配问题。可以提议两种模式：标准候选原子和高体积歧义的高斯拆分模式，后者可选，但可能需要以超越现有方法来优化。如果信息选择不足，拆分模式可能成为必要步骤。
>
> **优化控制变量问题**
>
> 核心问题可能是：**你正在优化错误的控制变量。** 瓶颈不仅在于选择哪些高斯中心保留，更在于选择合适的定位支持单元。直接选择未修改的高斯可能提升效果很小，因为候选集合本身存在缺陷。要解决这个问题，需采用更强的目标函数并进行选择性转换。
>
> **缺失的科学规律**
>
> 当前的挑战并非在于代码或评估，而是缺少一个决定哪些高斯应该出现在定位地图中的科学规律。这个规律的缺失使得目前的优化或评估措施难以获得显著的性能提升。我们需要建立一个有力的理论框架来指导高斯选择和定位精度之间的联系。
>
> **风险管理和优化问题**
>
> 在3DGS地图中，我们需要选择本地化原子子集 \(S\) 来最小化最坏情况下的姿态风险和运行时。由于直接优化不可行，我们通过替代方法优化。步骤包括使用安全核心 \(S_{core}\)和自适应外壳，依据局部信息矩阵实现替代可行性。
>
> **优化效果与实验设计**
>
> 我在考虑优化使用的核心方法，包括精简的预算和虚拟原子的组合。这将实现更加稳定的PnP收敛和更高效的查询响应。为验证这些，我想通过常用数据集和自收集的真实世界数据进行测试，特别是面对复杂场景时的表现。
>
> **实验设计与验证**
>
> 我计划使用小型真实场景数据集进行实验，同时结合Cambridge或7-Scenes数据集。评价将包括标准POSE指标、CVaR和RANSAC收敛等。我们还将进行 ablation 实验，验证不同选择策略和理论假设。另外，代码路线会在现有基础上构建，逐步加入各种模块。
>
> **优化3DGS定位方法**
>
> 建议提出的新方法聚焦于提高映射构建和降低训练/运行时间，而不只关注精度。避免重训练和高维描述符的使用可以提升效率。拟定的论文标题可能是 "RISC-GS" 或 "SAFE-Loc-GS"。要确保论文有新理论、证明分析、有效算法，以及强有力的实证分析和诊断证据。
>
> **3DGS定位核心问题**
>
> 文献中已有一些研究提到：STDLoc使用启发式的多视角一致分布，虽然能够饱和回忆度，但不具查询特定性；ULF-Loc指出高维描述符训练代价大且易产生偏差；SplitGS-Loc提到PnP算法在高斯点的近似匹配中会面临不稳定的收敛和运行问题。缺少“鲁棒查询家族感知的定位选择”这一问题的解决。
>
> **硬查询单元和核心点的流程设计**
>
> 我计划添加日志记录，跟踪匹配内点集、许多对一的计数、雅可比矩阵、位姿协方差代理。阶段1是验证替代架构，时间大约1-2周。阶段2，则进行信息保留的可接受性验证，重点是矩阵保护的实施。阶段3则是查询家族稳健的核心选取。
>
> **阶段性优化流程设计**
>
> 在第4阶段，我计划为高各向异性或高许多对一高斯函数创建虚拟原子，并从视图观测中提取无偏原子描述符。在第5阶段，我将比较在相同GPU上的时间，加强对相同预算、相同准确度和延迟的评估。第6阶段则包括论文完整测试及额外数据集。
>
> **特征设置与择优算法描述**
>
> 对于候选j和单元c，特征涉及一些重要因素：例如内点率、与负样本的界限差异、几何不确定性、许多对一风险、可观察性、动态风险等。基于这些特征，我可以推导出一个近似的\(H_{cj}\)模型。至于logdet子模性，我认为无需外部引用，基于已知理论可推导。
>
> **提出3DGS定位问题的核心挑战**
>
> 目前项目缺少明确的答案，关于哪些3DGS原始元素构成了定位地图，而非渲染地图。当前进展已经否定了简单的假设，这为更强的算法奠定了基础
>
> **命名选择与论文结构规划**
>
> 关于命名，我考虑了多个选项。或许“**SAGE-Loc**”是一个直观的名字，能够传达“支持-可接受高斯实验设计定位”的概念。同时，为了确保清晰，我会将回答结构划分为几个部分，包括问题定义、论文方法等，并加入必要的引用确保准确性。
>
> **整理认知总结与技术细节**
>
> 我提到在各个方法中涉及的一些重要技术特性和成果，如STDLoc的采样机制和场景检测器，SplatLoc的显著地标选择和无偏解码器，和ULF-Loc的关键点共识采样等。我也会引用相关数据，确保每个观点都有适当的支持性文献。
>
> **确认引用和进一步探索的方向**
>
> 我认为对于已知来源的内容，需要提供引用，而用户提供的信息可以不引用。直接与用户的对话内容（如“根据你给出的Stage A结果”）可以不需要引用。至于进一步的研究方向，可能考虑增量式的子图/局部地图查询或更强的联合优化定位路径。
>
> **构思新的研究方向**
>
> 单独提出强大的核心集选择可能不足够有新意，除非用深度理论和速度来执行。但也许可以将“信息论查询家族的强大核心集”与“定位原子”结合，这样会更具新颖性。另一个有吸引力的方向是“最小充分定位子图”，这对机器人技术具有重要意义。
>
> **构思新的研究方案**
>
> 当前主要痛点是缺乏不变性。项目可以强调“最小充分定位地图”的研究问题。通过在渲染优化的3D图形结构上提取最小化的核心集，即便与STDLoc相比提升较小，但如果能在困难案例中使用更小的地图达到相同或更好的准确度，这将很有说服力。
>
> **构建最小有效定位地图方法**
>
> 该方法解决了以下问题：
> - 不是所有高斯分布适用于定位
> - 一些高斯分布可能有害（偏差、众对一、视角依赖性）
> - 选择必须考虑最坏情况下的姿态信息和求解器成本。
> 步骤包括：
> 1. 从3DGS构建候选原子池。
> 2. 根据经验统计和不确定性评分。
> 3. 使用硬查询约束进行鲁棒实验设计选择。
> 4. 导出嵌套定位地图。
>
> **解释支持计数失败背后的理论**
>
> 目前的支持计数方法存在缺陷，具体原因如下：
> - 当地标变得共线或远离光学中心时，即使计数未改变，\(H\)变差，导致姿态协方差增加。
> - 如果替代项增加相关的外点，RANSAC迭代次数上升，因为单位3D点的有效内点比例下降。
> - 这些问题无法由计数保护机制识别，理论上可以解释。
>
> **提出实现细节**
>
> 我需要定义“硬查询单元”：即发生基线姿态误差大于指定百分位数或接近阈值的情况下的 episodes。然后，考虑使用查询单元集群训练姿态邻域或召回集群， 每个场景可能有50-100个单元。对每个单元进行统计计算，最终目标是优化计算，并使用贪心法与 matroid 约束进行优化。
>
> **替换可行性分析**
>
> 我可以在阶段A的episodes离线采样并标记i与j替换时性能变化。不过组合爆炸很大。可以通过查询级边际效用来近似。最终方案可包括“替换可接纳性网络”作为优化器的一部分。它会训练小型GNN/MLP用于判断安全替换概率，但从理论深度来看，最好依靠信息矩阵主导。
>
> **实现稳固集合优化**
>
> 这个方案不依赖新建训练网络，而是利用现有的支持保持基础设施，非常好。下一步可能最强的实践方法是实施一个稳健的集合效用函数。这可以通过对每个候选项j在池C中的效用进行估算，使用现有的替换指标对起始集\(S_0\)进行优化。
>
> **优化模型保证准确性和选择性**
>
> 为了减少噪声并提高模型的泛化能力，关键是优化处理单元而非个别查询。我还应该深入探讨理论基础，而不仅仅依赖启发式方法。可以通过Cramér-Rao和RANSAC引理来指导。主要目标是通过最大化某些值来减少不确定性的下限，同时提升有效点比率。
>
> **提出实际的科学问题解决方案**
>
> 为了满足用户的需求，我可能需要自行收集一些真实世界的“压力集”，比如包括反射玻璃、细栏杆、人群动态、光照变化等场景。通过创建这种“高斯定位压力测试”，我可以确保实验结果具有代表性并切合实际需求。同时，提出完整的论文框架。
>
> **提出不推荐的方向**
>
> 我需要明确哪些方向不推荐继续追求，例如增加标量权重、描述符残差、端到端的PnP差分、每个查询的分支选择等。可以在简洁区块中指出这些，引用当前的代码库文档版本以及新支持的基础设施。
>
> **从一元到集合式选择的转变**
>
> 现在的方法依赖于一元特征选择，每个高斯分配一个标量的显著性或可定位性，但实验证明这些不够。推荐采用查询家族集合选择，放眼更大的集合式目标。这可以按三种方式展开。你可以利用这个机会实现集合式选择，特别是在3DGS定位上的缺失。
>
> **关于定位场的集合函数与优化过程**
>
> 可以将“定位场”视为一个定义在原子和查询家族上的集合函数，而非简单的标量场。通过贪心算法利用预计算的对偶修正，能高效地做到近似。虽然GNN可用于预测集合的边际增益，但贪心方法加上logdet和对偶冗余已有足够表现。
>
> **讨论存储成本和硬单元提取**
>
> 存储成本和硬单元提取可以通过加权计算完成，基于任务的重要性。目标函数包含了多个部分，包括加权的对数行列式、M2O、CVaR和成本约束。简化版本通过确保安全核心约束，避免硬单元丢失，在尽可能小的成本下最大化目标。
>
> **讨论α-主导比率和虚拟原子的生成**
>
> 在计算α-主导比率时，如果原子的贡献小于预期，则认为其不可靠。对于具有高各向异性比率或多模态匹配的高风险高原子，逐步收集匹配的点并进行聚类，生成虚拟锚点。这些点通过几何加权池化进行融合，确保原子共享同一渲染Gaussian。
>
> **考虑虚拟原子和兼容性问题**
>
> STDLoc期望对Gaussians进行采样处理，而不是虚拟原子。虚拟原子的处理方式如果与高斯ID不匹配，可能会影响现有的评估器。我可以尝试将每个虚拟原子作为一个不同3D点的高斯副本来编码。这样可能需要自己实现本地器以避免更改评估器，但这增加了工程复杂度。如果方法强大，这或许可以接受。
>
> **制定实际的时间计划**
>
> 我可以考虑提出一个完整的解决方案，开始时介绍一套基本方法，并含有扩展选项。时间规划可以是：1-2周进行仪器化和验证，3-4周处理信息保护和配对可接受性，5-6周进行鲁棒的优化，接下来的时间专注于预算、速度优化，以及最终的实验和报告。也可以提及相关代码模块，尽管不一定是必需的。
>
> **设定成功标准与文档编写计划**
>
> 可以定义“成功”标准为：基于16k预算，R5提升1.52pp，R2提升1pp，或至少不发生场景崩溃；8k预算在较低延迟下匹配或超过16k；R5误差小于0.5pp；另外，hard-cell CVaR在Cambridge和7-Scenes上可改进。完成这些后，文档可以采取段落格式，适当略带些许冗长，保持可读性。
>
> **研究问题与计划概述**
>
> 研究的主要问题包括：从渲染型3DGS中提炼最小充分定位子图，定义理论框架，构建包括候选定位原子、姿态单元库以及稳健的核心集优化等方法。同时，还要考虑实验路线、与现有代码结合、投稿策略，并制定近期任务计划，包括文献引用和实验数据支持。
>
> **科学痛点与改进计划**
>
> 当前的经验痛点是缺乏稳定、完整的结果，但更深层次的科学痛点在于控制变量过于单一。仅通过原始高斯中心的选择器无法解决3DGS定位中的结构模糊，导致性能提升趋于饱和。提出“单变量/上下文/集体”框架进行改进。
>
> **关键点共识抽样与选择优化**
>
> 在多视角支持下，ULF关键点共识抽样部分依赖于选择集。选择集合的价值取决于选中的其他地标，因为姿势信息和冗余是设置的固有属性。顶级解决方案必须是集合式、风险感知的。提出一个核心定理：优化矩阵支持，而非单纯计数。
>
> 已思考 13m 34s

## Response:
2026/5/18 14:00:37

可以，而且我建议你**明确把论文重构成“不动 native feature Gaussian / descriptor，本体上做 solver-centric 的 selection / sampling field”**。这不是退步，反而更像一个能做深、做成体系的课题。

## 核心痛点到底是什么

你现在离顶刊最远的，不是“还差几次调参”，而是**还没有抓住 3DGS 定位里 feature selection 的真正科学对象**。

这两年 3DGS 定位的前沿已经很拥挤了：STDLoc 做了 matching-oriented sampling、scene-specific detector 和 sparse-to-dense 全流程；SplatLoc 做了 unbiased 3D descriptor 和 salient landmark selection；ULF-Loc 进一步从理论上指出 3DGS 里常见的 α-blending feature optimization 会带来 biased feature，并用 keypoint-consensus landmark sampling、geometry-weighted feature fusion 和 local geometric consistency verification 形成了更强更快的系统；SplitGS-Loc/PlugGS-Loc 则把问题指向 many-to-one pixel-Gaussian ambiguity 和冗余 Gaussian，利用 composition weights 与 Gaussian splitting 改善 PnP 收敛和速度；GSVisLoc 走的是 training-free / generalizable 路线；SplatHLoc 又把 query-adaptive retrieval 和 hybrid feature matching 做成了新的主线。也就是说，**“再做一个更好的 score / selector”本身已经不够新**。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.pdf))

你公开仓库现在也已经实质上承认了这一点：主线已经从 descriptor residual 转到 localization utility / sampling field，且当前 clean candidate `selector005_native_desc` 的 full Cambridge 提升是正向但小，下一步被定义为“让 learned field 改变 `sampled_idx.pkl`，而不是只 nudging `sampled_scores.pkl`”。这说明你直觉上已经走到正确方向了，只是理论对象还没有彻底改写。([GitHub](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/README.md))

**真正未解的问题**不是“哪些 Gaussian 更 salient / 更 matchable”，而是：

> **在 3DGS 定位中，怎样选择一组 Gaussian / landmarks，才能在给定查询分布下，让 PnP 更容易成功、更稳定、更快，并且不在 hard queries 上崩掉？**

这和现有方法的差别很大。STDLoc、SplatLoc、ULF-Loc、SplitGS-Loc 都在做 selection，但它们主要还是在优化某种**proxy**：matching score、saliency、keypoint consensus、composition-weight consistency、uniform coverage。它们没有把 selection 直接写成“对 PnP solvability / robust localization success 的优化”。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.pdf))

---

## 你最新实验已经说明了一个很深的结论

你这轮 support-preserving 的负结果，其实给了一个非常强的科学信号：

> **support count preservation 不是 feature selection 的正确不变量。**

这是因为 PnP 成功不取决于“有多少 support”，而取决于：

1. 有没有足够多的**可行最小集**；
2. 这些最小集是不是**几何上非退化**；
3. 对应 correspondence 是不是**低歧义、低相关、低伪重复**；
4. 这些 support 是不是覆盖了**worst-case / hard queries**。

你现在的结果“support loss 修掉了，但 median 仍退化”非常合理，因为**计数不等于可解性**。两组 landmarks 可以有同样的 support count，但一组只提供重复、近共面、局部相关的 correspondences，另一组提供空间分散、深度变化充足、低 ambiguity 的 correspondences；对 PnP 来说，这两者完全不是一回事。

这其实和经典视觉定位 / 地图压缩文献里的发现是一致的：老工作早就发现 landmark selection 不是简单 observation count 问题，而要考虑 co-observability、appearance condition、memory budget，甚至用 ILP 做 landmark selection、用 appearance-dependent selection 做 query-time 选择；更近一些的方法如 AIR-HLoc、ImLoc 也都在强调**query-dependent / budget-dependent selection**比固定全局选择更有效。PoseCompass 虽然做的是 3DGS-based APR synthetic pose selection，但它也把 selection 形式化成了 budget-constrained optimization，值函数由 difficulty、novelty、observability 共同决定--这说明“selection 本身就是问题，而不是附属工程项”。([arXiv](https://arxiv.org/abs/1808.02656))

---

## 我建议你把课题彻底改写成这个命题

### 从 “matchability-aware selection” 改成  
### **“solver-centric, risk-aware Gaussian subset optimization”**

一句话概括：

> **你要研究的不是“怎么给 Gaussian 打分”，而是“怎么在 3DGS 地图上选出一组对 PnP 真正可解、对 hard queries 有保障、对速度-精度 Pareto 更优的 landmarks”。**

这会让你的论文从 STDLoc 的一个变种，变成一个更一般的科学问题：

> **Feature selection in 3DGS localization is a query-conditioned, set-valued, solver-level optimization problem.**

这句话很重要，因为它直接解释了为什么你之前的很多路子都只有小改进：它们本质上还是在做 **unary scoring**，而真正的问题是 **set optimization**。

---

## 这篇论文最该建立的理论框架

我建议你把理论核心写成下面这个形式。

对一个 query $q$ 和选出来的 Gaussian 子集 $S$，定义三个关键量：

### 1. 可行最小集覆盖（minimal-set coverage）

$$
V_q(S)=\sum_{m \in \mathcal{M}_q(S)} P(m\ \text{all-inlier}) \cdot \psi_{\text{cond}}(m)
$$

其中 $\mathcal{M}_q(S)$ 是由 $S$ 提供的、对 PnP 有效的最小 correspondence 子集；$\psi_{\text{cond}}(m)$ 衡量这个 minimal set 的几何条件是否好。

这项直接对应：**RANSAC / PROSAC 有没有足够高概率早早采到一个好 minimal set**。

### 2. 位姿信息量（pose information）

$$
H_q(S)=\sum_{i\in S} p_{qi}\, J_{qi}^{\top}\Omega_{qi}J_{qi}
$$

其中 $p_{qi}$ 是 landmark $i$ 在 query $q$ 下成为 inlier 的概率，$J_{qi}$ 是对位姿的投影雅可比。

这项对应：**即便 PnP 成功了，位姿是不是稳定、精度是不是高**。

### 3. 歧义与冗余风险（ambiguity / redundancy）

$$
A_q(S)=\sum_{i<j \in S} a_{qij}
$$

这里 $a_{qij}$ 应由 descriptor similarity、3DGS composition overlap、co-visibility redundancy、局部重复结构等共同决定。

这项对应：**3DGS 特有的 many-to-one / overlapping Gaussian 问题会不会把 support 数量“虚高”但实际可解性变差**。SplitGS-Loc 明确指出 many-to-one pixel-Gaussian matches 会恶化 PnP convergence，并给出在同样 RANSAC 设定下从 314ms 降到 215ms 的例子；这正说明 selection 必须绑定 solver behavior，而不是只看 saliency。([arXiv](https://arxiv.org/html/2605.07351v1))

于是你真正该优化的不是 support count，而是：

$$
U_q(S)=
\lambda_v \log(V_q(S)+\epsilon)
+
\lambda_h \log\det(H_q(S)+\epsilon I)
-
\lambda_a A_q(S)
-
\lambda_d D_q(S)
$$

其中 $D_q(S)$ 是 dense refinement unreliability，可由 floater risk、render artifact risk、局部 feature blur 风险构成。STDLoc 自己就提到 dense stage 会因 floaters 遮挡导致 feature map / RGB 渲染不完整，进而带来 recall 损失；ULF-Loc 也把 rendering artifacts 作为 coarse-to-fine 阶段 mismatch 的来源。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.pdf))

最后对整个查询分布做 robust optimization：

$$
S^\*=
\arg\max_{|S|\le B}
\ \mathbb{E}_{q}[U_q(S)]
+
\beta\, \mathrm{CVaR}_\alpha(U_q(S))
$$

这里的 $\mathrm{CVaR}$ 很关键。因为视觉定位真正杀 recall 的不是平均 query，而是 hardest 10%-20% queries。你的实验里 OldHospital、StMarysChurch 这种 scene-specific hard failure，本质上就是 tail-risk 问题。

---

## 这套理论带来的一个关键新结论

你现在可以把论文最核心的 insight 定成：

> **3DGS feature selection 不是 unary ranking，而是 hypergraph / set optimization。**

因为 landmark 的价值取决于它与其他 landmarks 共同组成的 viable minimal sets，而不是它自己的单独 score。

这比 “support-preserving” 深一层。  
你真正需要的不是 “support guard”，而是：

> **replacement admissibility**

也就是：一个非 native landmark 能不能替换一个 native landmark，不取决于它自身分数高不高，而取决于**替换之后，hard-query 的 minimal-set coverage、pose information 和 ambiguity risk 是否整体更好**。

这就是你下一步方法的科学核心。

---

## 我建议的完整技术路线

## 第一阶段：构建 solver-level 的监督，而不是继续调 selector 权重

你现在最该做的是一个**Counterfactual Self-localization Hypergraph**。

### 1）建立 query-level episode bank
对 train / self-map / rendered / shifted queries，保存的不只是 inlier count，而是：

- top-M candidate correspondences
- visible landmark pool
- per-candidate inlier probability
- Jacobian / pose leverage
- descriptor ambiguity
- composition-overlap / co-visibility
- dense refinement success/failure
- sampled minimal tuples 的成败与条件数

如果只做一件事，我建议就是这件事。

### 2）把每个 query 变成一个“PnP tuple bank”
对每个 query，不再只统计单点 support，而是采样一批 minimal PnP tuples，标记：

- 是否 all-inlier
- 是否几何退化
- 是否最终进入高 consensus pose
- 是否容易被 PROSAC 早期采到

这一步会直接给你一个新诊断：

> support count 不掉，但 viable tuple coverage 掉了。

只要这张图出来，你的论文主问题就立住了。

---

## 第二阶段：从 unary selector 升级成 pairwise / set-aware selector

你需要学的不是单一 $s_i$，而是两类量：

### 1）cluster-conditioned node utility
$$
u_i^{(c)}
$$
表示 landmark $i$ 对 hard-query cluster $c$ 的 solver utility。

### 2）replacement admissibility / redundancy
$$
r_{i \rightarrow j}^{(c)}, \quad \rho_{ij}^{(c)}
$$

- $r_{i \rightarrow j}^{(c)}$：在 cluster $c$ 中，用 $i$ 替换 native landmark $j$ 是否安全且有利；
- $\rho_{ij}^{(c)}$：$i,j$ 是否高度冗余 / 冲突 / 高相关。

这是从“打分”到“替换决策”的跃迁。

### 可行实现
不用一上来搞很重的 GNN。先做 factorized model 就够：

- node MLP：预测 $u_i^{(c)}$
- pair MLP：预测 $\rho_{ij}^{(c)}$
- replacement classifier：预测 $r_{i\rightarrow j}^{(c)}$

输入用你已经有的东西就够强：

- selector score / native score
- self-map support signature
- descriptor similarity
- 3DGS composition overlap
- visibility histogram
- anisotropy / scale / opacity
- hard-negative rate
- pose leverage statistics

---

## 第三阶段：安全骨架 + 条件增量库

我不建议你直接“全局换地图”。更强、更稳的路线是：

### 1）Safe core
先从 native map 中抽一个小而稳的 backbone $B_0$，它专门覆盖 hardest clusters。

约束不是 count，而是：

$$
U_c(B_0) \ge U_c(S_{\text{native}})-\delta_c
$$

对 hardest clusters 尽量取 $\delta_c=0$。

### 2）Conditional delta banks
再对每个 query cluster $c$ 学一个增量子集 $\Delta_c$，让

$$
S_c = B_0 \cup \Delta_c
$$

如果你只想先做 paper-safe 的 static 版本，那就先把多个 $\Delta_c$ 合并压缩成一个 same-budget 全局集合。  
如果 static 版本仍只有小改进，那就必须走 **query-conditioned activation**：

- query 先做轻量 coarse routing
- 预测属于哪个 visibility / appearance / viewpoint cluster
- 激活 $B_0 + \Delta_c$
- 然后仍然走一条固定的 matching + PnP + dense refinement 路径

这不是 branch selection，而是**query-adaptive landmark activation**，本质上更接近 appearance-dependent landmark selection、adaptive retrieval。([arXiv](https://arxiv.org/abs/1808.02656))

我判断：**如果你想从当前“小幅提升”走到“明显提升 + 速度收益”，最终多半要走到这一步。**

---

## 第四阶段：把速度问题变成 solver-aware budgeted localization

你前面已经发现：少 landmarks 不一定更快。这个判断是对的。

真正的速度收益不该来自“盲目减点”，而应该来自：

1. **更少的 bad correspondences**
2. **更高质量的 early minimal sets**
3. **更快的 PROSAC / RANSAC convergence**
4. **更少的 dense refinement 污染区域**

所以速度主张应该写成：

> same accuracy, fewer PnP iterations / lower latency  
> or same latency, higher strict recall

不要写成“real-time”。
现阶段更适合写 **accuracy-latency Pareto**。

### 速度模块怎么做
你可以在不改 evaluator 的前提下加两个东西：

#### 1）solver-aware ordering
把 selected landmarks 的顺序按预计 tuple utility / inlier utility 排序，用于 PROSAC 输入顺序。

#### 2）dense reliability mask
由 selected Gaussian 子集渲染一个 reliability mask，在 dense refinement 时屏蔽高 floater / 高 ambiguity / 高 artifact 的区域。

STDLoc 和 ULF-Loc 都已经表明 dense stage 的 artifact / floater 会伤害后续 refinement，所以这里是有真实物理问题支撑的，不是纯工程 trick。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.pdf))

---

## 第五阶段：实验设计要从“涨点”换成“机制验证”

顶刊要的不是更多表格，而是**机制闭环**。

你至少要做这四组证据。

### A. 现象证据
证明：

- support count 与最终 pose success 相关性弱；
- viable tuple coverage / logdet(H) / ambiguity risk 与 pose success 强相关；
- 这在 OldHospital / StMarysChurch 等 hard scenes 上尤其明显。

### B. 方法证据
比较：

- native
- unary selector
- support-count guard
- tuple-aware selector
- tuple-aware + CVaR
- tuple-aware + safe core
- tuple-aware + conditional bank

### C. 速度证据
比较：

- total latency
- sparse matching time
- PnP time
- PnP iterations
- dense time
- same-budget vs same-accuracy Pareto

### D. 泛化证据
至少要有：

- Cambridge
- 7-Scenes / 12 Scenes 之一
- 最好再加一个 outdoor / illumination-shift / real captured set

因为现在 ULF-Loc、SplitGS-Loc、SplatHLoc 这些工作都已经把精度、效率、泛化往前推了；如果你只在一个数据集上给出小提升，很难撑起顶刊主 claim。([arXiv](https://arxiv.org/html/2605.04730v1))

---

## 一个务实但有冲击力的论文主线

我建议你把最终题目收束成这一句：

> **From Matchability to Solvability: Risk-Aware Minimal-Set Selection for 3DGS Visual Localization**

中文可以写成：

> **从可匹配性到可解性：面向 3DGS 视觉定位的风险控制最小集特征选择**

它的三个 claim 可以是：

1. **问题层面**：  
   3DGS feature selection 不是 unary ranking，而是 query-conditioned、set-valued 的 solver optimization。

2. **理论层面**：  
   support count preservation 不能保证 pose solvability；应以 minimal-set coverage、pose information 和 ambiguity risk 作为 selection objective。

3. **方法层面**：  
   在不改变 native descriptor 与几何 PnP 后端的前提下，提出 safe-core + admissible replacement + conditional activation 的 sampling field，显著改善 hard-query robustness 和 accuracy-latency Pareto。

---

## 你现在最应该立刻做的事

不是再试一个 qcov / churn / retention 权重。

而是：

> **先把 “support count 为什么不够” 这件事做成一张硬证据图。**

具体就是：

- 为 self-map queries 建 minimal-set tuple bank；
- 计算每个 query 的 viable tuple coverage；
- 计算每个 query 的 logdet(H)；
- 对比 native、supportguardall、ret95 这类失败候选；
- 在 OldHospital / StMarysChurch 上画出：
  - support count 几乎不变
  - 但 viable tuple coverage / logdet(H) 明显下降
  - 对应 strict recall / median 退化

只要这张图出来，你的论文就从“我试了很多 selector 但涨点不大”变成了：

> **我发现了 3DGS feature selection 里的错误优化对象，并提出了 solver-centric 的替代目标。**

这才是顶刊级问题的起点。

如果你愿意，我下一步可以直接把这套方案进一步压成一个**论文级方法定义 + 代码模块清单 + 三个月实验计划**。

---

**Sources:**

- [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_From_Sparse_to_Dense_Camera_Relocalization_with_Scene-Specific_Detector_from_CVPR_2025_paper.pdf)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/Loc-GS/main/README.md)
- [[1808.02656] Appearance-Based Landmark Selection for Efficient Long-Term Visual Localization](https://arxiv.org/abs/1808.02656)

## Prompt:
2026/5/18 14:32:04

你说的我大部分认可，那baseline应该选什么呢，如果只是稀疏定位的话，高斯primitives可以直接选，但如果是基于渲染的稠密定位方法呢，特征选择应该是什么形式呢

## Response:
2026/5/18 14:32:05

> **寻找最新的3DGS定位基准与渲染方法**
>
> 用户问到与基线选择和密集渲染定位特性相关的问题。为了得出一个最新的答案，我想搜索相关的3DGS定位基准，如STDLoc、SplatLoc、ULF-Loc等方法。可能还需要关注像iNeRF、Loc-NeRF等密集渲染方法。我会尽量寻找相关的论文来更新信息。
>
> **浏览3DGS视觉定位领域的最新基准和方法**
>
> [Popular 3D models](https://sketchfab.com/3d-models/popular)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://zju3dv.github.io/splatloc/)
> [Ulf | Wiki of Westeros - Fandom](https://gameofthrones.fandom.com/wiki/Ulf)
> [Generalizable Visual Localization for Gaussian Splatting ...](https://gsvisloc.github.io/)
> [Printables.com: 3D models database](https://www.printables.com/)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://arxiv.org/abs/2409.14067)
> [Ulf White - A Wiki of Ice and Fire](https://awoiaf.westeros.org/index.php/Ulf_White)
> [3D modeling](https://en.wikipedia.org/wiki/3D_modeling)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://www.cad.zju.edu.cn/home/gfzhang/papers/splatloc/splatloc.pdf)
> [I find Ulf to be very unlikable and have been annoyed in ...](https://www.reddit.com/r/HouseOfTheDragon/comments/1eezzzb/i_find_ulf_to_be_very_unlikable_and_have_been/)
> [GSVisLoc: Generalizable Visual Localization for Gaussian ...](https://research.nvidia.com/labs/par/publication/gsvisloc.html)
> [3D Systems: 3D Printers, Software, Manufacturing & Digital ...](https://www.3dsystems.com/)
> [GSplatLoc: Grounding Keypoint Descriptors into 3D ...](https://github.com/haksorus/gsplatloc)
> [Who Are Rhaenyra's Dragonriders, Ulf, Hugh Hammer, and ...](https://www.esquire.com/entertainment/tv/a61727765/house-of-the-dragon-ulf-the-white-hugh-hammer/)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://www.computer.org/csdl/journal/tg/5555/01/10919003/251xRhaBPpe)
> [SfM-Free 3D Gaussian Splatting via Hierarchical Training](https://openaccess.thecvf.com/content/CVPR2025/html/Ji_SfM-Free_3D_Gaussian_Splatting_via_Hierarchical_Training_CVPR_2025_paper.html)
> [Tinkercad](https://www.tinkercad.com/)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization for ...](https://www.youtube.com/watch?v=UJowrWOI-Oc)
> [[Literature Review] SplatLoc: 3D Gaussian Splatting-based ...](https://www.themoonlight.io/en/review/splatloc-3d-gaussian-splatting-based-visual-localization-for-augmented-reality)
> [Ulf the White | Pure Evil Wiki - Fandom](https://pure-evil-villains.fandom.com/wiki/Ulf_the_White)
> [LEGS: Visual Localization Enhanced by 3D Gaussian ...](https://www.mdpi.com/2313-433X/12/2/84)
> ['House of the Dragon': How Ulf Dies in 'Fire and Blood'](https://www.businessinsider.com/who-is-ulf-white-house-of-the-dragon-silverwing-dragonrider-2024-7)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://www.researchgate.net/publication/384267334_SplatLoc_3D_Gaussian_Splatting-based_Visual_Localization_for_Augmented_Reality)
> [Ulf](https://open.spotify.com/artist/3aLiHzkydHv9P8n3fgPPmS)
>
> **查找特定密集渲染定位方法**
>
> 我现在需要查找一些特定的密集渲染定位方法，比如GSLoc、GS-CPR和iComMa。为了提供最新的相关信息，我会进行搜索，看看这些方法的最新进展和应用。
>
> **搜索关于GSLoc、GS-CPR、iComMa定位相关信息**
>
> [GSLoc: Visual Localization with 3D Gaussian Splatting](https://arxiv.org/abs/2410.06165)
> [GSLoc: Visual Localization with 3D Gaussian Splatting](https://www.researchgate.net/publication/384769976_GSLoc_Visual_Localization_with_3D_Gaussian_Splatting)
> [GS-CPR: Efficient Camera Pose Refinement via 3D ...](https://github.com/XRIM-Lab/GS-CPR)
> [GSLoc: Visual Localization with 3D Gaussian Splatting](https://www.semanticscholar.org/paper/GSLoc%3A-Visual-Localization-with-3D-Gaussian-Botashev-Pyatov/48d535b0734a7b6d9e92d141af06d41a9f48a23f)
> [Efficient Camera Pose Refinement via 3D Gaussian Splatting](https://openreview.net/forum?id=mP7uV59iJM)
> [GS-CPR: EFFICIENT CAMERA POSE REFINEMENT VIA ...](https://proceedings.iclr.cc/paper_files/paper/2025/file/bf11e0bb868b6735afcdb35ff39ea831-Paper-Conference.pdf)
> [GS-CPR](https://xrim-lab.github.io/GS-CPR/)
> [iComMa: Camera pose estimation by comparing and ...](https://www.tarogoing.uk/2024/05/04/icomma-camera-pose-estimation-by-comparing-and-matching-inverted-3d-gaussian-splash/)
> [Enhancing Structure-from-Motion: Re-aligning images with 3D ...](https://academic.oup.com/jcde/article/13/1/303/8325207)
> [GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian ...](https://www.youtube.com/watch?v=9xhcpLLu7Kg)
> [Recent advances in 3D Gaussian splatting](https://www.sciopen.com/article/10.1007/s41095-024-0436-y)
> [Generalizable Visual Localization for Gaussian Splatting ...](https://gsvisloc.github.io/)
> [MrNeRF's Awesome-3D-Gaussian-Splatting-Paper-List](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
> [GS-CPR: EFFICIENT CAMERA POSE REFINEMENT VIA 3D ...](https://researchportal.hkust.edu.hk/en/publications/gs-cpr-efficient-camera-pose-refinement-via-3d-gaussian-splatting)
> [3D Gaussian Splatting aided Localization for Large and ...](https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/1283/2025/isprs-archives-XLVIII-G-2025-1283-2025.pdf)
> [LEGS: Visual Localization Enhanced by 3D Gaussian Splatting](https://pmc.ncbi.nlm.nih.gov/articles/PMC12941419/)
> [Visual Localization with 3D Gaussian Splatting](https://www.themoonlight.io/es/review/gsloc-visual-localization-with-3d-gaussian-splatting)
> [ICLR 2025](https://iclr.cc/media/iclr-2025/Slides/28467_LEQRMgp.pdf)
> [Gaussian splatting with known camera poses?](https://www.reddit.com/r/GaussianSplatting/comments/1f0357r/gaussian_splatting_with_known_camera_poses/)
>
> **查找ULF-Loc的详细信息**
>
> 我需要查找ULF-Loc的具体信息，了解其当前的研究和应用。这样可以确保我提供最准确的和最新的答案。
>
> **搜索与ULF-Loc相关的视觉定位方法**
>
> [Ulf | Wiki of Westeros - Fandom](https://gameofthrones.fandom.com/wiki/Ulf)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://arxiv.org/abs/2409.14067)
> [Ulf White - A Wiki of Ice and Fire](https://awoiaf.westeros.org/index.php/Ulf_White)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://zju3dv.github.io/splatloc/)
> [I find Ulf to be very unlikable and have been annoyed in ...](https://www.reddit.com/r/HouseOfTheDragon/comments/1eezzzb/i_find_ulf_to_be_very_unlikable_and_have_been/)
> [(PDF) GSFeatLoc: Visual Localization Using Feature ...](https://www.researchgate.net/publication/391282084_GSFeatLoc_Visual_Localization_Using_Feature_Correspondence_on_3D_Gaussian_Splatting)
> [GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian ...](https://gsplatloc.github.io/)
> [Who Are Rhaenyra's Dragonriders, Ulf, Hugh Hammer, and ...](https://www.esquire.com/entertainment/tv/a61727765/house-of-the-dragon-ulf-the-white-hugh-hammer/)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization for ...](https://www.youtube.com/watch?v=UJowrWOI-Oc)
> [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://www.cad.zju.edu.cn/home/gfzhang/papers/splatloc/splatloc.pdf)
> [Generalizable Visual Localization for Gaussian Splatting ...](https://www.computer.org/csdl/proceedings-article/iccvw/2025/898800a178/2eldjI0F2XS)
> [LEGS: Visual Localization Enhanced by 3D Gaussian ...](https://www.mdpi.com/2313-433X/12/2/84)
> [GSplatLoc: Grounding Keypoint Descriptors into 3D ...](https://github.com/haksorus/gsplatloc)
> [Ulf the White | Pure Evil Wiki - Fandom](https://pure-evil-villains.fandom.com/wiki/Ulf_the_White)
> [GSLoc: Visual Localization with 3D Gaussian Splatting - ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv241006165B/abstract)
> [3D Gaussian Splatting for Map Representation and Visual ...](https://scholars.library.tamu.edu/vivo/display/n1438307SE)
> ['House of the Dragon': How Ulf Dies in 'Fire and Blood'](https://www.businessinsider.com/who-is-ulf-white-house-of-the-dragon-silverwing-dragonrider-2024-7)
> [GSLoc: Visual Localization with 3D Gaussian Splatting](https://www.semanticscholar.org/paper/GSLoc%3A-Visual-Localization-with-3D-Gaussian-Botashev-Pyatov/48d535b0734a7b6d9e92d141af06d41a9f48a23f)
> [3DGS-Loc: 3D Gaussian Splatting for Map Representation ...](https://discovery.researcher.life/article/3dgsloc-3d-gaussian-splatting-for-map-representation-and-visual-localization/323aa9d63d6e3487bc3b372eafaf9d26)
> [Ulf](https://open.spotify.com/artist/3aLiHzkydHv9P8n3fgPPmS)
> [Who is Ulf White in House of the Dragon and who plays him?](https://www.radiotimes.com/tv/fantasy/house-of-the-dragon-ulf-white-tom-bennett-explained/)
> [Disambiguating 2D-3D Correspondences in Gaussian Splatting ...](https://bytez.com/docs/arxiv/2605.07351/paper)
>
> **选择合适的基线方法**
>
> 我需要挑选适合的基线方法，根据用户问到的内容，首先考虑稀疏特征方法，比如HLoc、PixLoc等；其次，3DGS稀疏特征方法更具体，如STDLoc、SplatLoc、GSVisLoc、ULF-Loc等。如果涉及密集渲染定位，应考虑像iComMa、GSLoc等方法，应依据可见性、权重等选择特征。
>
> **分析稀疏与密集定位方法基准**
>
> [
    SplatLoc
  ](https://zju3dv.github.io/splatloc/)
> [Generalizable Visual Localization for Gaussian Splatting Scene Representations](https://gsvisloc.github.io/)
> [[2410.06165] GSLoc: Visual Localization with 3D Gaussian Splatting](https://arxiv.org/abs/2410.06165)
>
> **查找STDLoc详细信息**
>
> 我将查找关于STDLoc的具体信息，看看是否能够找到更多关于其更新和应用的资源。需要打开相关的来源来确保提供最准确的信息。
>
> **搜索STDLoc相关信息**
>
> [From Sparse to Dense: Camera Relocalization with Scene ...](https://zju3dv.github.io/STDLoc/)
> [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](https://arxiv.org/abs/2503.19358)
> [StdLoc — SeisComP Release documentation](https://www.seiscomp.de/doc/apps/global_stdloc.html)
> [CVPR2025 - STDLoc, a new camera relocalization method ...](https://www.youtube.com/watch?v=auodnGC51tw)
>
> **基线选择与密集渲染特征选择**
>
> 针对用户问题，选取基线依赖于稀疏、密集或混合方式的定位目标。对于3DGS，选择STDLoc作为主基线，适用于稀疏到密集的特征选择。密集渲染方法应包含iComMa、GSLoc等，若使用PnP+密集优化，应与STDLoc或GSVisLoc进行比较，同时避免过多引用来源。
>
> 已思考 35s

可以把 baseline 分成两层：**论文主 baseline** 和 **机制验证 baseline**。你的主线如果是“3DGS 定位中的特征选择 / sampling field”，主 baseline 不应该只选稀疏定位方法，也不应该只选渲染优化方法，而应覆盖 **sparse 2D-3D、sparse-to-dense、render-based refinement** 三类。

## 1. 主 baseline 应该怎么选

### 第一优先级：STDLoc native

**STDLoc 必须是第一主 baseline。**  
原因很直接：你的方法目前继承它的 Feature Gaussian、sampling、detector、PnP 和 dense refinement 语境。STDLoc 本身就是 sparse-to-dense pipeline：先用 matching-oriented Gaussian sampling 和 scene-specific detector 做 sparse matching + PnP，再基于初始位姿渲染 dense feature/depth map 做 coarse-to-fine dense matching refinement。它不是纯稀疏方法，而是你最直接的对手。([DiffWind](https://zju3dv.github.io/STDLoc/?utm_source=chatgpt.com))

所以主表至少要有：

```text
STDLoc native sparse
STDLoc native dense
Loc-GS selection field sparse
Loc-GS selection field dense
```

其中 **STDLoc native dense** 才是最关键比较对象。你如果只打 sparse stage，审稿人会说你的方法没有超过完整 STDLoc pipeline。

---

### 第二优先级：SplatLoc / GSVisLoc / PlugGS-Loc 或 SplitGS-Loc

如果你的 claim 是“3DGS landmark / primitive selection”，那么还应比较至少一到两个 3DGS localization 方法：

```text
SplatLoc
GSVisLoc
PlugGS-Loc / SplitGS-Loc
GSplatLoc
```

SplatLoc 直接有“salient 3D landmark selection”，使用 learned unbiased descriptor fields 和 primitive subset selection，所以它是 feature-selection 维度上的自然 baseline。([DiffWind](https://zju3dv.github.io/splatloc/))

GSVisLoc 是 generalizable 3DGS localization，做 coarse-to-fine 3D-2D matching，再 PnP+RANSAC 和 3DGS-based pose refinement；它更像“训练范式不同但目标接近”的强对比。([GSVisLoc](https://gsvisloc.github.io/))

PlugGS-Loc / SplitGS-Loc 很重要，因为它们指出 rendering-based feature optimization 会造成 feature blurring / many-to-one pixel-Gaussian ambiguity，并用 composition weights / Gaussian decomposition 来改善 2D-3D correspondence 和 PnP convergence。这个方向和你现在的“选择哪些 Gaussian 才适合定位”高度相关。([arXiv](https://arxiv.org/html/2605.07351v1))

如果代码不可用，可以只做 paper-level 对比或引用其公开结果，但至少要在 related work 和实验讨论中正面对齐。

---

### 第三优先级：传统强 localization baseline

这类不是你的主要竞争对象，但能说明你不是只在 3DGS 小圈子里有效：

```text
HLoc, SuperPoint + SuperGlue / LightGlue
LoFTR / RoMa retrieval + PnP
ACE / DSAC* / NeuMap
PixLoc, NeRFMatch 等
```

STDLoc 论文自己也在 7-Scenes、Cambridge 上和 HLoc、DSAC*、ACE、NeRFMatch 等比较，并报告 STDLoc 在多个指标上超过或接近这些方法。([arXiv](https://arxiv.org/html/2503.19358v1?utm_source=chatgpt.com))

但这些 baseline 不一定都要你重跑。你可以按重要性分：

```text
必须重跑：STDLoc native
尽量重跑：HLoc / ACE / SplatLoc 中可复现者
引用公开结果：其他训练成本过高或代码不可用方法
```

---

## 2. 如果论文强调 dense rendering localization，baseline 怎么选

如果你的方法要进入“基于渲染的稠密定位”这个语境，那 baseline 应该包括：

```text
iComMa
GSLoc
GS-CPR
STDLoc dense stage
GSVisLoc pose refinement
SplatHLoc iterative rendered matching
```

GSLoc 是典型 dense camera alignment：用 3DGS 渲染图像，然后对 pose 反传梯度，最小化 query 与 rendered image 的 residual，并用 coarse-to-fine blur kernel 缓解非凸性。([arXiv](https://arxiv.org/abs/2410.06165))

GS-CPR 是 test-time camera pose refinement：用 3DGS 渲染 RGB/depth，借助 MASt3R 建立 2D matching，然后做 one-shot pose refinement；它本质上是“给定 coarse pose 的渲染辅助 refinement baseline”。([arXiv](https://arxiv.org/abs/2408.11085))

SplatHLoc 则是另一类 rendered matching baseline：先检索，再渲染近邻虚拟视角，然后用 hybrid feature matching 做初始和迭代 refined relocalization；它还明确指出 rendered features 更适合 coarse stage，image features 更适合 fine stage。([arXiv](https://arxiv.org/html/2603.29185v1))

注意公平性：**render-based dense methods 通常需要初始 pose**。所以不要把它们和 pose-prior-free 的 STDLoc 全流程直接混比。合理协议是：

```text
Full pipeline comparison:
  query image -> final pose
  比 STDLoc / Loc-GS / HLoc / SplatLoc / GSVisLoc

Refinement-only comparison:
  给同一个 initial pose
  比 STDLoc dense refinement / GSLoc / iComMa / GS-CPR / your dense-selection refinement
```

---

## 3. 稀疏定位里的 feature selection 是什么

稀疏定位中，selection object 很清楚：

```text
从 3D Gaussian primitives / landmarks 中选一个子集 S
用于 2D-3D matching + PnP
```

具体形式是：

```text
Gaussian primitive selection
landmark sampling
sampled_idx.pkl
sampled_scores.pkl
PROSAC ordering
keypoint detector prior
```

目标是让 PnP 得到更好的 correspondence set：

```text
更多 inliers
更低 hard-negative ratio
更好的空间分布
更好的 pose information
更少 RANSAC iterations
```

所以稀疏路线里的核心实验是：

```text
same landmark budget 下更准
same accuracy 下更少 landmarks
same latency 下更高 recall
```

---

## 4. 稠密渲染定位里的 feature selection 应该是什么形式

这是你问题里最关键的一点。

对于 dense rendering localization，**不能简单地说“选 Gaussian primitives”就结束了**。因为 dense 渲染里的一个像素/patch 往往是很多 Gaussians 经 alpha compositing 混合的结果。此时 selection 应该发生在四个层面：

---

### 4.1 Primitive-level selection：选哪些 Gaussians 参与渲染或特征渲染

这是最接近 sparse selection 的形式：

```text
每个 Gaussian 有一个 reliability / utility score r_i
渲染 dense feature map 时，对 Gaussian contribution 加权或筛除
```

原始渲染是：

$$
F(u)=\sum_i w_i(u) f_i
$$

你可以改成：

$$
F_{\text{sel}}(u)=
\frac{
\sum_i r_i \, w_i(u) f_i
}{
\sum_i r_i \, w_i(u)+\epsilon
}
$$

其中：

```text
w_i(u): Gaussian i 对 pixel u 的 alpha-composition weight
f_i: Gaussian feature
r_i: learned localization reliability / selection score
```

这就是 dense 里的 **primitive selection field**。

但它有风险：如果直接屏蔽 Gaussian，可能破坏渲染连续性或导致 feature holes。所以更合理的是 soft weighting，而不是 hard deletion。

---

### 4.2 Pixel / ray-level selection：选哪些 rendered pixels 或 rays 参与 pose optimization

这可能是 dense localization 中最自然的 feature selection 形式。

dense refinement 原来通常最小化：

$$
\sum_u \rho\left(
\phi_q(u)-\phi_r(u;T)
\right)
$$

你可以改成：

$$
\sum_u m(u;T)\,
\rho\left(
\phi_q(u)-\phi_r(u;T)
\right)
$$

其中 $m(u;T)$ 是 rendered reliability mask。

这个 mask 可以由以下信号组成：

```text
composition entropy
top-1 Gaussian dominance
depth uncertainty
feature disagreement
view-angle stability
opacity / floater risk
self-localization success frequency
local descriptor ambiguity
pose Jacobian leverage
```

例如：

$$
m(u)=
\underbrace{\max_i w_i(u)}_{\text{dominance}}
\cdot
\underbrace{(1-H(w(u)))}_{\text{low mixture entropy}}
\cdot
\underbrace{R_{\text{view}}(u)}_{\text{view consistency}}
\cdot
\underbrace{R_{\text{feat}}(u)}_{\text{feature reliability}}
$$

这个形式很适合解释 3DGS dense localization 的痛点：**不是所有渲染出来的像素都该参与 pose refinement**。有些像素由很多 Gaussians 混合、受 floater/遮挡影响、或者 feature ambiguity 高，把它们放进 dense alignment 反而会拉坏 pose。

---

### 4.3 Patch / correspondence-level selection：选哪些 dense matches 进入 refinement

STDLoc dense stage 和 SplatHLoc 这种 rendered matching 方法，本质上都不是直接全图 photometric optimization，而是 dense/semi-dense feature matching。STDLoc 渲染 feature map 和 depth map，再做 coarse-to-fine dense feature matching；SplatHLoc 也观察到 rendered features 更适合 coarse matching，fine stage 更适合 image features。([DiffWind](https://zju3dv.github.io/STDLoc/?utm_source=chatgpt.com))

因此你可以把 feature selection 定义成：

```text
dense correspondence selection
```

即从 dense matches 中选择：

```text
高置信
低歧义
低深度不确定性
低 composition entropy
高 pose leverage
空间分布均匀
```

形式上：

$$
\mathcal{M}_{dense}^{*}
=
\operatorname{TopK}_{u}
\left[
s_{match}(u)
+
\lambda_1 m_{render}(u)
+
\lambda_2 \log\det(J_u^\top J_u)
-
\lambda_3 a(u)
\right]
$$

然后只用这些 dense matches 做 pose update。

这个比“选 Gaussian”更贴合 dense matching pipeline。

---

### 4.4 Residual / gradient-level selection：选哪些 residual 对 pose 优化有权重

如果 baseline 是 GSLoc / iComMa 这种 differentiable rendering pose optimization，那么 feature selection 不是 landmark selection，而是：

```text
residual weighting
gradient gating
robust optimization mask
```

即：

$$
\min_T
\sum_u
m(u;T)\,
\rho
\left(
I_q(u)-I_r(u;T)
\right)
$$

或者 feature 版本：

$$
\min_T
\sum_u
m(u;T)\,
\rho
\left(
\phi_q(u)-\phi_r(u;T)
\right)
$$

这里的 selection field 是一个 **rendered confidence / optimization weight map**，不是 sampled landmark list。

它的理论目标是：

```text
保留对 pose Jacobian 有信息量的 residual
去掉 view-dependent artifact residual
去掉 floater / occlusion / specular / dynamic object residual
去掉 low-gradient 或 textureless residual
避免优化陷入错误局部极小
```

这就是 dense 渲染定位中最合理的“特征选择”形式。

---

## 5. 所以 dense 方法里的 selection 应该定义成什么

我建议统一定义为：

> **rendered localization support selection**

它不是单一对象，而是一个三层 field：

```text
Gaussian reliability field r_i
  控制哪些 primitives 的特征可信

Ray / pixel reliability field m(u)
  控制哪些 rendered pixels / rays 进入 dense matching 或 optimization

Correspondence / residual utility field c(u)
  控制哪些 dense matches / residuals 对 pose update 有贡献
```

论文里可以这样写：

```text
For sparse localization, feature selection is landmark subset selection.
For dense render-based localization, feature selection becomes residual support selection:
a selected feature is not necessarily a Gaussian primitive, but a reliable rendered ray / patch / correspondence whose underlying Gaussian mixture is unambiguous and whose residual provides stable pose information.
```

中文就是：

> 对稀疏定位，特征选择是选 Gaussian landmarks；对基于渲染的稠密定位，特征选择应定义为选择可靠的渲染射线、像素块或 dense correspondences，而不是简单选择 primitive。因为 dense feature 是 Gaussian mixture 的结果，真正进入位姿优化的是 residual / correspondence，而不是单个 Gaussian。

---

## 6. 推荐你的方法同时覆盖 sparse 和 dense，但主攻一个统一对象

你可以把方法设计成统一的 **Localization Support Field**：

```text
per-Gaussian support score r_i
↓ differentiable / rasterized aggregation
per-pixel support mask m(u)
↓ dense matching / residual weighting
per-correspondence support score c(u)
↓ sparse PnP / dense refinement
```

具体实现：

### Sparse stage

```text
r_i -> landmark sampling / sampled_idx
r_i -> sampled_scores / PROSAC ordering
r_i -> detector prior
```

### Dense stage

```text
r_i + composition weights -> rendered reliability mask m(u)
m(u) -> dense match filtering
m(u) -> residual weighting
m(u) -> dense refinement early rejection / robust weighting
```

这样你的方法就不再只是“sampling field”，而是：

> **从 Gaussian primitive 到 rendered residual 的 localization support field。**

这个表述比“feature selection”更统一，也能覆盖 sparse 与 dense。

---

## 7. 对应 baseline 应该怎样组织实验

我建议最终实验表分三块。

### 表 1：Full visual localization

用于证明整体 pipeline 有效：

```text
HLoc / ACE / DSAC*
SplatLoc
GSVisLoc
STDLoc native
Loc-GS sparse-only
Loc-GS sparse+dense
```

主比较：

```text
Loc-GS sparse+dense vs STDLoc native dense
```

---

### 表 2：Sparse feature selection / sampling

用于证明 primitive selection 有效：

```text
STDLoc sampling
random Gaussian sampling
opacity / scale / saliency sampling
SplatLoc-style saliency sampling
your unary selector
your support-aware selector
your solver-aware selector
```

指标：

```text
same budget accuracy
RANSAC iterations
inlier ratio
minimal-set coverage
pose information logdet
hard-query failure rate
```

---

### 表 3：Dense rendered support selection

用于证明 dense 里的 selection 不是空话：

```text
STDLoc dense refinement no mask
composition-dominance mask
low-entropy ray mask
feature-confidence mask
pose-Jacobian mask
your localization support mask
oracle successful-residual mask
```

如果你接 GSLoc / iComMa / GS-CPR，就做 refinement-only 表：

```text
same initial pose
same max iterations
same rendered resolution
same photometric / feature loss type when possible
compare final error and runtime
```

---

## 8. 最推荐的 baseline 组合

如果资源有限，我建议最小但有说服力的组合是：

```text
Main:
1. STDLoc native sparse
2. STDLoc native dense
3. Loc-GS sparse support field
4. Loc-GS sparse+dense support field

3DGS selection baselines:
5. random / FPS / opacity
6. STDLoc matching-oriented sampling
7. SplatLoc saliency-style sampling if implementable
8. PlugGS/SplitGS-style composition dominance proxy if implementable

Dense refinement baselines:
9. STDLoc dense no mask
10. composition entropy mask
11. feature confidence mask
12. your rendered support mask

External context:
13. HLoc
14. ACE / DSAC*
15. GSVisLoc / SplatLoc published or reproduced results
```

其中最关键的是 1-4 和 9-12。  
如果这两组成立，你的论文就有主线了。

---

## 9. 你应该避免的 baseline 错误

不要只比：

```text
Loc-GS selector sparse vs STDLoc sparse
```

因为 STDLoc 强在 sparse-to-dense full pipeline。

不要把 GSLoc / iComMa 直接当 full localization baseline，除非你给它们同等 coarse initialization。它们更多是 pose refinement / dense alignment 方法，而不是完全 pose-prior-free relocalization pipeline。GSLoc 和 iComMa 都依赖 initial pose 的 basin；GSLoc 还使用 coarse-to-fine blur 来缓解非凸性。([arXiv](https://arxiv.org/abs/2410.06165))

不要只报告精度。feature selection 论文必须报告：

```text
selected landmark count
dense valid pixel ratio
inlier ratio
minimal-set coverage
PnP iterations
dense refinement time
latency
failure recovery / failure introduction
```

---

## 10. 最终建议

你的 baseline 选择应围绕两句话：

> **稀疏定位中，feature selection 是 Gaussian landmark subset selection。**  
> **稠密渲染定位中，feature selection 是 rendered residual / ray / correspondence support selection。**

因此你的方法最好不是只输出一个 `sampled_idx.pkl`，而是输出一套统一 field：

```text
Gaussian support field
→ sparse landmark sampling
→ rendered reliability mask
→ dense correspondence / residual weighting
```

如果你能证明：

```text
1. sparse stage：它比 STDLoc / saliency / random / composition proxy 选出更可解的 landmark set；
2. dense stage：它比 no-mask / entropy-mask / confidence-mask 更可靠地过滤坏 residual；
3. full pipeline：它在 STDLoc native dense 之上稳定提升或形成更好的 accuracy-speed Pareto；
```

这就会比“我们给 Gaussian 打了一个更好的分”强很多，也更能成为一篇围绕 3DGS 定位特征选择的完整工作。

---

**Sources:**

- [From Sparse to Dense: Camera Relocalization with Scene ...](https://zju3dv.github.io/STDLoc/?utm_source=chatgpt.com)
- [Generalizable Visual Localization for Gaussian Splatting Scene Representations](https://gsvisloc.github.io/)
- [Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization](https://arxiv.org/html/2605.07351v1)

## Prompt:
2026/5/18 14:50:29

那接下来我该如何在已有的git仓库基础上具体推进本项目呢，请列一个明确的执行计划

## Response:
2026/5/18 14:50:30

我会把计划写成可直接开分支推进的执行路线：先冻结研究目标，再安排代码模块、Stage A/B/C 实验闸门、失败归档和最终投稿证据链。

## Response:
2026/5/18 14:50:43

接下来不再扩展 qcov/churn/weight 网格，而先建立 solver-level 诊断层，解释 support count 保住但定位仍坏的机制。

## Response:
2026/5/18 14:55:30

可以。接下来我建议你把项目推进方式改成 **“诊断先行、方法后置、少量候选、严格闸门”**。你现在仓库里已经有 selector-only、resampling、support guard、profile wrapper、audit/report 体系；继续调 qcov、retention、hard-negative weight 这类标量不会快，反而会继续消耗实验预算。当前应该做一个更结构化的版本：**solver-centric localization support field**。

我先给总判断：

> 下一阶段主线不是“再做一个更好的 selector 分数”，而是证明并利用一个新事实：**3DGS 定位中的 feature selection 不是 support count preservation，也不是 unary Gaussian ranking，而是面向 PnP 可解性、hard-query tail risk 和 dense residual reliability 的 set-level selection 问题。**

你的 README 已经把主线写成 native STDLoc descriptor + self-localization distilled selector / sampling field + STDLoc-compatible export + single-path PROSAC PnP + dense refinement，并且明确当前 selector-only 正结果仍然小、下一步要让 learned field 改变 `sampled_idx.pkl` 而不只是 nudging `sampled_scores.pkl`。这说明项目方向已经收束，但还缺 solver-level 理论和证据层。([GitHub](https://github.com/Arthurshen926/Loc-GS))

---

# 一、立即冻结当前主线与实验边界

先开一个新分支，不要在现有实验分支继续混改：

```bash
git checkout -b solver-centric-lsf
```

然后做三件事。

## 1. 更新主线文档

新增：

```text
docs/solver_centric_lsf_plan_20260518.md
```

里面写清楚：

```text
Main problem:
3DGS localization feature selection should optimize solver-level pose solvability,
not unary matchability or support count.

Main representation:
Localization Support Field, LSF.

Sparse form:
per-Gaussian / per-landmark selection for PnP.

Dense form:
per-ray / per-pixel / per-correspondence reliability for dense refinement.

Non-goals:
- no descriptor replacement as main claim
- no test-time branch selection
- no per-scene test-tuned recipe
- no full/test promotion before Stage A pass
```

你现在仓库里的 `sampling_field_budget_20260516.md` 已经有一条正确的单路径部署定义：native descriptor map → self-localization distilled selector / sampling field → sampled_idx + sampled_scores + locability → descriptor matching + OpenCV PROSAC PnP → dense refinement。接下来所有文档和代码都应该围绕这条 pipeline，不要再回 residual descriptor 主线。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/sampling_field_budget_20260516.md))

## 2. 把最近负结果明确归档

新增或更新：

```text
docs/negative_results_solver_selection_20260518.md
```

把以下内容标为 **diagnostic / rejected as main candidate**：

```text
supportguardcount_ret90
supportguardall_ret90
OldHospital ret95 / ret975
StMarysChurch ret95 / hn100
qcov / churn / retention / hard-negative scalar tuning
inlier-only early exit
native subset low-budget speed probe
```

每条必须写：

```text
hypothesis
result
why rejected
what it teaches
```

这一步不是形式主义。你这轮最有价值的结论是：

> support count loss 修掉了，但 pose accuracy 仍退化。

这说明下一步必须转向 **minimal-set / pose-information / ambiguity-aware selection**，而不是继续调 support count guard。

## 3. 禁止 full/test 继续消耗

文档里写死：

```text
No full Cambridge / official test promotion until:
1. train-q80 pass
2. shifted train-s3/q75 pass
3. split audit pass
4. query-level lost/recovered audit pass
5. at least one solver-level diagnostic metric improves
```

你之前的 full 结果已经出现“旧诊断有小正向但 audit unknown”的情况；这种结果只能作为方向信号，不能进 paper-safe 主表。README 也明确说当前 selector005 full improvement positive but small，只能支持 selector/sampling-field 方向，不能支持 strong SOTA claim。([GitHub](https://github.com/Arthurshen926/Loc-GS))

---

# 二、第一阶段：先做 solver-level 诊断，不改方法

这一阶段的目标不是涨点，而是建立论文的核心证据：

> support count 不足以解释 localization success；viable PnP minimal-set coverage、pose information、ambiguity risk 才是更好的 feature selection 目标。

这是接下来两周最重要的任务。

## 1. 新增模块

建议新增：

```text
loc_gs/diagnostics/
  __init__.py
  solver_tuple_bank.py
  pose_information.py
  ambiguity_metrics.py
  support_metrics.py
  query_cluster.py
  report_solver_diagnostics.py

loc_gs/scripts/
  build_solver_tuple_bank.py
  compare_solver_tuple_banks.py
  summarize_solver_selection_failure.py

tests/
  test_solver_tuple_bank.py
  test_pose_information_metrics.py
  test_ambiguity_metrics.py
```

不要动 evaluator，不要动 STDLoc metric，不要动 full/test。

---

## 2. `solver_tuple_bank.py` 要做什么

输入来自已有 self-map / train episode cache、selector-resampled map、PnP logs、match records。每个 query 保存：

```python
@dataclass
class SolverCandidate:
    query_id: str
    landmark_id: int
    gaussian_id: int | None
    xy: tuple[float, float]
    xyz: tuple[float, float, float]
    desc_score: float
    selector_score: float | None
    native_score: float | None
    is_native_sampled: bool
    is_candidate_sampled: bool
    pnp_inlier: bool | None
    reproj_error_px: float | None
    visibility: float | None
    hard_negative_risk: float | None
    ambiguity_score: float | None
```

然后对每个 query 采样 PnP minimal tuples。实际可以用 4-tuple，不必纠结 P3P 的最小点数，因为这里是 diagnostic / surrogate，不是替代 OpenCV PnP：

```python
@dataclass
class SolverTuple:
    query_id: str
    landmark_ids: tuple[int, int, int, int]
    all_inlier_prob: float
    geometry_logdet: float
    geometry_condition: float
    spatial_spread_2d: float
    spatial_spread_3d: float
    ambiguity_penalty: float
    viable: bool
```

`viable=True` 的条件建议先用保守版本：

```text
1. tuple 内点的 inlier probability 或 inlier label 足够高；
2. reprojection error 低；
3. 2D 点不是过度集中；
4. 3D 点不是过度共线 / 共面 / 深度跨度过低；
5. pose Jacobian logdet 高于场景内分位数阈值；
6. ambiguity penalty 低于阈值。
```

核心输出是每个 query 的四个指标：

```text
support_count
weighted_support
viable_tuple_mass
pose_information_logdet
ambiguity_risk
```

你这轮 supportguard 的失败，正好要用这些指标解释：support count 没掉，但 viable tuple mass / logdet / ambiguity 变差。

---

## 3. `pose_information.py` 要做什么

对每个 2D-3D correspondence 计算近似投影 Jacobian：

$$
H_q(S)=\sum_{i\in S} p_{qi}J_{qi}^{\top}\Omega_{qi}J_{qi}
$$

输出：

```text
trace_H
logdet_H
condition_number_H
min_eigenvalue_H
spatial_coverage_score
depth_spread_score
```

这一步不需要 differentiable PnP。你只需要一个稳定 surrogate，用于比较 native、supportguard、qcov、future solver-aware map。

---

## 4. `ambiguity_metrics.py` 要做什么

每个 query 对 selected landmarks 计算 ambiguity：

```text
descriptor near-duplicate rate
3D proximity with high descriptor similarity
co-visibility redundancy
same-pixel / same-ray competition if available
hard-negative collision rate
```

如果目前拿不到 composition weights，可以先做三种轻量 proxy：

```text
1. descriptor cosine similarity > threshold
2. 3D distance < threshold but descriptor too similar / too different
3. same query top-k candidates 中多个 landmark 竞争同一 keypoint
```

后续再接 3DGS raster composition entropy。

ULF-Loc 和 SplitGS-Loc 都把 3DGS feature bias、many-to-one pixel-Gaussian ambiguity、stable PnP convergence 作为核心问题，这说明你把 selection 目标从“matchability”升级到“solver ambiguity / PnP solvability”是有前沿依据的。([arXiv](https://arxiv.org/html/2605.04730v1?utm_source=chatgpt.com))

---

## 5. 第一阶段验收

新增脚本：

```bash
python -m loc_gs.scripts.build_solver_tuple_bank \
  --scene OldHospital \
  --native_map output/unified_lff_v2/native_rebuilt_detector_source_20260516/OldHospital \
  --candidate_map <supportguardall_ret90_map> \
  --episode_cache <audited_selfmap_cache> \
  --output_dir output/solver_diagnostics/oldhospital_supportguardall_ret90

python -m loc_gs.scripts.compare_solver_tuple_banks \
  --native output/solver_diagnostics/oldhospital_native/solver_tuple_bank.json \
  --candidate output/solver_diagnostics/oldhospital_supportguardall_ret90/solver_tuple_bank.json \
  --output_report docs/solver_selection_failure_oldhospital_202605XX.md
```

单元测试：

```bash
pytest -q tests/test_solver_tuple_bank.py tests/test_pose_information_metrics.py tests/test_ambiguity_metrics.py
python -m compileall loc_gs
```

必须产出一张机制图：

```text
x-axis: support_count_delta
y-axis: final_pose_success_delta

x-axis: viable_tuple_mass_delta
y-axis: final_pose_success_delta

x-axis: logdet_H_delta
y-axis: final_pose_success_delta

x-axis: ambiguity_risk_delta
y-axis: final_pose_success_delta
```

如果结果显示 viable tuple / logdet / ambiguity 比 support count 更能解释 OldHospital / StMarysChurch 的退化，论文主问题就立住了。

如果这些指标也解释不了退化，说明还缺 dense residual / rendering artifact 诊断，应先转 dense support，而不是继续 sparse map editing。

---

# 三、第二阶段：实现 replacement admissibility，而不是继续 support guard

当前 support guard 的问题是：它只约束 count。下一步要约束 **替换是否 admissible**。

## 1. 新增模块

```text
loc_gs/stdloc_native/
  solver_admissibility.py
  solver_aware_resampling.py

loc_gs/scripts/
  export_solver_aware_map.py

tests/
  test_solver_admissibility.py
  test_solver_aware_resampling.py
```

---

## 2. 核心算法

从 native set $S_0$ 出发，而不是从 all-Gaussian top-k 出发。

```text
Input:
  S0 = native sampled landmarks
  C = candidate pool from selector / all-Gaussian / qcov
  Q = train/self-map query set
  H = hard-query subset
  B = landmark budget

Output:
  S* = same-budget solver-aware sampled set
```

### Step A：定义 hard-query set

从 native baseline 中找：

```text
R5/R2 failure queries
high reprojection error queries
low inlier-ratio queries
dense refinement worsened queries
OldHospital / StMarysChurch hard clusters
```

然后按 pose / viewpoint / image retrieval / covisibility 聚类：

```text
hard_cluster_id
query_ids
native_support_count
native_viable_tuple_mass
native_logdet_H
native_ambiguity
```

### Step B：构造 safe core

保留 native 中对 hard queries 的关键 support：

```text
safe_core = landmarks that participate in top viable tuples
            or high pose-information support
            or unique hard-cluster coverage
```

这比 `ret90/ret95` 更合理，因为不是保留随机比例，而是保留 hard-query solver support。

### Step C：定义 replacement admissibility

一个非 native candidate $i$ 替换 native landmark $j$，必须满足：

```text
1. j 不属于 safe_core，或 i 能在所有 hard clusters 中补偿 j 的 solver support；
2. 对 hard queries，viable_tuple_mass 不下降超过 δ_v；
3. 对 hard queries，logdet_H 不下降超过 δ_h；
4. ambiguity_risk 不上升超过 δ_a；
5. candidate i 有足够 positive evidence；
6. candidate i 的 hard-negative risk 低于阈值；
7. same-budget 约束严格成立。
```

形式上：

$$
\Delta_{i\leftarrow j}(q)
=
\lambda_v\Delta V_q
+
\lambda_h\Delta\log\det H_q
-
\lambda_a\Delta A_q
-
\lambda_n\Delta N_q
$$

对 hard queries：

$$
\Delta V_q \ge -\delta_v,\quad
\Delta\log\det H_q \ge -\delta_h,\quad
\Delta A_q \le \delta_a
$$

对普通 queries 可以用 CVaR / worst-20% 约束，而不是所有 query 都零损失。你这轮已经证明 “all query support loss = 0” 仍然会失败，所以不能再只用 loss count。

### Step D：贪心局部编辑

伪代码：

```python
S = set(native_sampled_ids)
safe_core = build_safe_core(native_tuple_bank, hard_queries)

for step in range(max_edits):
    best_edit = None
    best_gain = 0.0

    for i in candidate_pool - S:
        if not has_positive_evidence(i):
            continue
        for j in removable_native_ids(S, safe_core):
            if not is_admissible_replace(i, j, query_metrics):
                continue
            gain = replacement_gain(i, j, query_metrics)
            if gain > best_gain:
                best_gain = gain
                best_edit = (i, j)

    if best_edit is None:
        break

    add i, remove j
    update incremental query metrics
```

输出必须包括：

```text
native_kept_count
native_dropped_count
safe_core_size
safe_core_dropped_count
added_non_native_count
num_admissible_replacements
num_rejected_by_hard_query
num_rejected_by_logdet
num_rejected_by_ambiguity
num_rejected_by_hard_negative
per-query V/logdet/ambiguity delta
worst-10% query delta
```

这会让每次 map edit 可解释，而不是只看 final Cambridge metric。

---

## 3. 第二阶段验收

新增导出命令：

```bash
python -m loc_gs.scripts.export_solver_aware_map \
  --scene OldHospital \
  --source_map output/unified_lff_v2/native_rebuilt_detector_source_20260516/OldHospital \
  --checkpoint <selector_checkpoint> \
  --tuple_bank output/solver_diagnostics/OldHospital/native_tuple_bank.json \
  --candidate_pool all_gaussian \
  --budget same_as_source \
  --safe_core_mode viable_tuple \
  --admissibility_mode hard_cvar \
  --max_edits 256 \
  --output_map output/unified_lff_v2/solver_aware_resample_202605XX/OldHospital
```

测试：

```bash
pytest -q tests/test_solver_admissibility.py tests/test_solver_aware_resampling.py tests/test_selector_resampling.py
python -m compileall loc_gs
```

必须先只跑 OldHospital / StMarysChurch，因为它们是当前方法最容易暴露问题的场景。通过这两个 hard scenes 后，再跑 q80 macro。

---

# 四、第三阶段：Stage A 实验 funnel

不要再一口气跑很多 candidate。每轮最多三个候选。

## 1. 固定 baseline

每次 Stage A 都必须包含：

```text
native clean 16k
native clean 12k
selector005 native descriptor old diagnostic, if audit allows
supportguardall_ret90 diagnostic
solver-aware candidate
```

你之前的 q80 budget sweep 和 full validation已经说明，12k selector cov8 有一些中位数正向但 R5 很弱，rank 版本会有更明显的不稳定性；这正好作为新方法必须超过的弱 baseline。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/sampling_field_budget_20260516.md))

## 2. 固定 Stage A split

只用：

```text
train-q80
shifted train-s3/q75
```

不要用 official full/test 做 recipe selection。

## 3. Go / no-go 标准

### Accuracy candidate 通过线

同时满足：

```text
macro median_te <= native - 0.15 cm
macro R5 >= native + 0.75 pp
macro R2 >= native + 0.25 pp
no scene R5 loss > 1.0 pp
no scene median_te regression > 0.4 cm
OldHospital and StMarysChurch not worse than native on both median and R5
```

如果你觉得这太严，可以第一轮设成观察线，但 paper-facing 候选必须逐步达到这个量级。当前你已有的 `TE -0.00287 cm, R5 +0.0025` 这种量级不值得继续推进。

### Mechanism candidate 通过线

即使 accuracy 还没过，也可以继续一轮，但必须满足：

```text
hard-query viable_tuple_mass improves
hard-query logdet_H does not drop
ambiguity_risk decreases
support_count not used as the only positive evidence
```

如果机制指标也不改善，直接归档。

---

# 五、第四阶段：速度只在 accuracy 通过后做

你现在已经有 profile/timing wrapper，也有 q20 timing diagnostics。文档记录了一些 fast probe：同一 q20 设置、同一 native wrapper、一轮 dense iteration，并且列出了 mean/median ms、sparse/dense 时间、inlier 数等；同时也显示减少 landmark 数不一定带来速度收益，某些 4k/2k 反而 sparse 时间更高。([GitHub](https://github.com/Arthurshen926/Loc-GS/blob/main/docs/audited_sampling_fast_probe_20260517.md))

所以速度实验不要提前做。只有 solver-aware map 在 Stage A 过线后，再做。

## 1. 速度评估目标

不要写“real-time”。写：

```text
accuracy-latency Pareto
same accuracy, lower latency
same latency, higher recall
same PnP cap, better recall
```

## 2. 速度 candidate

只测三组：

```text
native same budget
solver-aware same budget
solver-aware same budget + fixed lower PROSAC/dense cap
```

不要再扫很多 landmark budget。

## 3. Go / no-go 标准

```text
accuracy no worse than native tolerance:
  median_te <= native + 0.1cm
  R5/R2 no loss > 0.5pp

latency:
  solo median latency >= 10% faster
  or sparse PnP time >= 15% faster
  or same fixed cap under native recall drop but solver-aware stable
```

如果速度收益没有达到 10% 级别，就不要主推速度，只把它作为辅助指标。

---

# 六、第五阶段：dense feature selection 只做一个轻量、可解释版本

不要一开始就大改 STDLoc dense refinement。先把 dense selection 作为 **rendered residual support mask** 做成 diagnostic，然后看是否有价值。

## 1. 新增模块

```text
loc_gs/dense_support/
  __init__.py
  gaussian_reliability_raster.py
  rendered_support_mask.py
  dense_match_filter.py

loc_gs/scripts/
  export_dense_support_mask.py
  eval_dense_support_filter.py

tests/
  test_rendered_support_mask.py
  test_dense_match_filter.py
```

## 2. Dense selection 的形式

如果 sparse stage 是选 Gaussian landmarks，那么 dense stage 不是简单选 primitives，而是选：

```text
ray / pixel / patch / dense correspondence / residual
```

定义：

$$
m(u)=
\frac{\sum_i w_i(u)r_i}{\sum_i w_i(u)+\epsilon}
\cdot
\text{dominance}(u)
\cdot
(1-\text{entropy}(u))
\cdot
\text{depth\_stability}(u)
$$

其中：

```text
r_i: Gaussian localization reliability
w_i(u): Gaussian-to-pixel contribution
dominance(u): top-1 Gaussian contribution ratio
entropy(u): composition entropy
```

先用 soft mask 过滤 dense matches：

```text
keep dense match if:
  m(u) > threshold
  pose leverage high enough
  local ambiguity low enough
```

不要直接删渲染 Gaussian；先在 dense match / residual 层做 gating，更安全。

## 3. Dense 阶段实验

只比较：

```text
STDLoc dense no mask
composition dominance mask
entropy mask
selector reliability mask
solver-aware reliability mask
```

指标：

```text
dense refinement success rate
dense worsened-query count
R5/R2
dense time
valid dense correspondence count
```

如果 dense mask 只减少耗时但掉 recall，不进主线。  
如果 dense mask 能减少 “sparse 已经对、dense 反而拉坏” 的 query，它就能成为第二贡献。

---

# 七、最终 paper-facing 方法结构

等 sparse solver-aware 和 dense support mask 都有结果后，论文方法可以写成三层。

## Layer 1：Self-localization feedback bank

从 train / rendered / shifted self-localization episodes 得到：

```text
per-query candidate matches
inlier / hard-negative labels
pose Jacobian
viable PnP tuple bank
ambiguity risk
dense refinement success/failure
```

## Layer 2：Sparse solver-aware support field

输出：

```text
sampled_idx.pkl
sampled_scores.pkl
locability
PROSAC ordering
```

核心目标：

$$
\max_S
\mathbb{E}_q[
\lambda_v V_q(S)
+
\lambda_h \log\det H_q(S)
-
\lambda_a A_q(S)
-
\lambda_n N_q(S)
]
+
\beta \operatorname{CVaR}_{\alpha}(U_q(S))
$$

其中：

```text
V_q: viable minimal-set coverage
H_q: pose information
A_q: ambiguity / redundancy
N_q: hard-negative risk
CVaR: hard-query tail risk
```

## Layer 3：Dense rendered support mask

输出：

```text
per-pixel / per-correspondence reliability mask
```

用于 dense refinement 过滤或加权。

---

# 八、实验矩阵

## 1. Sparse feature selection 表

```text
Method                         Budget   Median t/r   R10   R5   R2   TupleMass   logdetH   Ambiguity
STDLoc native                  16k
STDLoc native                  12k
random / opacity / visibility  12k
selector005 score-only         12k/16k
supportguardall                12k/16k
solver-aware no CVaR           12k/16k
solver-aware + CVaR            12k/16k
solver-aware + ordering        12k/16k
```

## 2. Dense support 表

```text
Method                   Dense valid %   Dense worsened #   Median t/r   R5   R2   Dense ms
STDLoc dense no mask
entropy mask
dominance mask
selector reliability mask
solver-aware mask
```

## 3. Full pipeline 表

主比较：

```text
STDLoc native dense
Loc-GS sparse LSF only
Loc-GS sparse+dense LSF
```

STDLoc 必须是第一主 baseline，因为它就是 Feature Gaussian + matching-oriented Gaussian sampling + scene-specific detector + sparse-to-dense relocalization pipeline。([arXiv](https://arxiv.org/abs/2503.19358?utm_source=chatgpt.com))

如果资源允许，再加：

```text
SplatLoc / SplatLoc-style saliency selection
GSVisLoc / GSplatLoc / SplitGS-Loc public or reproduced comparison
HLoc / ACE external context
```

SplatLoc 也有 learned descriptor field 和 salient 3D landmark selection，所以它是 feature-selection 方向上的自然对照。([DiffWind](https://zju3dv.github.io/splatloc/?utm_source=chatgpt.com))

---

# 九、具体两周执行安排

## Day 1：冻结与归档

完成：

```text
docs/solver_centric_lsf_plan_20260518.md
docs/negative_results_solver_selection_20260518.md
```

命令：

```bash
pytest -q tests/test_selector_resampling.py
python -m compileall loc_gs
git diff --check
```

目标：停止 qcov / ret / hn weight 小修补路线。

---

## Day 2-4：实现 solver tuple bank

完成：

```text
loc_gs/diagnostics/solver_tuple_bank.py
loc_gs/diagnostics/pose_information.py
loc_gs/diagnostics/ambiguity_metrics.py
loc_gs/scripts/build_solver_tuple_bank.py
```

测试：

```bash
pytest -q tests/test_solver_tuple_bank.py tests/test_pose_information_metrics.py tests/test_ambiguity_metrics.py
```

先只跑：

```text
OldHospital native vs supportguardall
StMarysChurch native vs supportguardall
```

产出：

```text
docs/solver_selection_failure_oldhospital_202605XX.md
docs/solver_selection_failure_stmarys_202605XX.md
```

---

## Day 5-6：机制报告

做图和表：

```text
support_count_delta vs pose_delta
viable_tuple_mass_delta vs pose_delta
logdet_H_delta vs pose_delta
ambiguity_delta vs pose_delta
```

如果这一步不能解释失败，暂停 sparse resampling，转 dense support artifact 诊断。  
如果能解释失败，继续 Day 7。

---

## Day 7-10：实现 replacement admissibility

完成：

```text
loc_gs/stdloc_native/solver_admissibility.py
loc_gs/stdloc_native/solver_aware_resampling.py
loc_gs/scripts/export_solver_aware_map.py
tests/test_solver_admissibility.py
tests/test_solver_aware_resampling.py
```

只实现三种配置：

```text
solver_adm_safe_core
solver_adm_cvar
solver_adm_cvar_ordered
```

不要再开十几个参数组合。

---

## Day 11-12：hard-scene Stage A

先跑：

```text
OldHospital train-q80
StMarysChurch train-q80
OldHospital shifted train-s3/q75
StMarysChurch shifted train-s3/q75
```

不跑 full/test。

通过条件：

```text
median 不明显退化
R5/R2 不掉
viable_tuple_mass/logdet_H 正向
ambiguity_risk 不升
```

不过就回到 admissibility 约束，不调 qcov/retention。

---

## Day 13-14：macro Stage A

只在 hard scenes 通过后跑五场景 Stage A：

```text
GreatCourt
KingsCollege
OldHospital
ShopFacade
StMarysChurch
```

输出：

```text
output/solver_lsf_stagea_202605XX/reports/stagea_train_q80_summary.json
output/solver_lsf_stagea_202605XX/reports/stagea_shifted_q75_summary.json
docs/solver_lsf_stagea_202605XX.md
```

只有通过 go/no-go，才进入 solo timing 或 full/test。

---

# 十、项目加速规则

为了避免继续“推进很多轮但没有 paper-safe 主结果”，建议以后每个候选都按下面规则处理。

## 每轮最多 3 个 candidate

例如：

```text
A: solver_adm_safe_core
B: solver_adm_cvar
C: solver_adm_cvar_ordered
```

不要再同时跑：

```text
ret90/95/975
hq005/025/05
hn05/1/100
qcov01/05/12
churn64/128/256
```

这种网格会拖慢项目，也不利于论文叙事。

## 每个 candidate 必须有 hypothesis

格式：

```text
Hypothesis:
supportguard failed because it preserved count but lost viable tuple geometry.

Expected:
solver_adm_cvar should improve OldHospital/StMarysChurch viable_tuple_mass
without increasing ambiguity risk.

Pass:
Stage A metrics and solver diagnostics both improve.

Fail:
Archive if either accuracy or solver diagnostics fail.
```

## 没有机制指标，不允许继续调参

只看 R5/R2 太容易被随机波动误导。接下来每个实验都必须同时报告：

```text
pose metrics
support count
viable tuple mass
logdet_H
ambiguity risk
hard-negative risk
native kept / dropped
safe core dropped
```

---

# 十一、当前不该做的事情

明确暂停：

```text
1. 不再推进 qcov/churn/retention scalar tuning。
2. 不再把 residual descriptor 拉回主线。
3. 不再把 speed 当作独立主线。
4. 不再跑 q20 test diagnostic 指导 recipe。
5. 不再用 full/test 做候选选择。
6. 不再只用 support count guard 作为安全约束。
7. 不再因为 R5 +0.0025 级别微涨推进候选。
```

---

# 十二、最终目标形态

如果按这套计划推进，一个理想的 paper-safe 主结果应该长这样：

```text
Method:
Loc-GS / LSF: Solver-Centric Localization Support Field

Sparse:
same native descriptor, same STDLoc backend,
but sampled landmarks are selected by viable tuple coverage,
pose information, ambiguity suppression, and CVaR hard-query risk.

Dense:
rendered residual support mask filters unreliable dense correspondences.

Claim:
not “we tune STDLoc better”,
but “we show 3DGS localization feature selection must optimize pose solvability,
and implement a solver-aware support field that improves hard-query robustness
and accuracy-latency Pareto.”
```

最小可投稿线：

```text
1. Full Cambridge paper-safe:
   R5 ≥ +1.0pp, R2 ≥ +0.5pp, median translation non-worse,
   at least 3/5 scenes positive, OldHospital/StMarysChurch not degraded.

2. Mechanism evidence:
   support count fails to predict success,
   viable tuple mass / logdet_H / ambiguity explain failures.

3. Speed evidence:
   either ≥10% solo latency improvement at same accuracy,
   or same latency with higher strict recall.

4. No audit weakness:
   split audit passed, no evaluator drift, no test-tuned recipe.
```

较强顶刊线：

```text
1. Cambridge + one extra dataset positive.
2. Sparse and dense selection both show interpretable improvement.
3. Failure taxonomy shows hard-query recovery, not just macro metric gain.
4. Comparisons include STDLoc native dense, saliency/visibility/opacity sampling, and at least one 3DGS localization selection baseline or proxy.
```

你的下一步不是“再提升一点精度”，而是先做出这条证据链：

```text
support count guard 失败
→ viable minimal-set / pose-information diagnostic 解释失败
→ replacement admissibility 修复失败机制
→ Stage A hard scenes 通过
→ macro Stage A 通过
→ solo timing
→ paper-safe full/test
```

这条链一旦打通，本项目才会从“很多小改进尝试”变成一个有理论对象、有机制证据、有工程闭环的投稿级工作。

---

**Sources:**

- [GitHub - Arthurshen926/Loc-GS · GitHub](https://github.com/Arthurshen926/Loc-GS)
- [Unbiased Landmark Feature for Robust Visual Localization ...](https://arxiv.org/html/2605.04730v1?utm_source=chatgpt.com)
- [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://zju3dv.github.io/splatloc/?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)