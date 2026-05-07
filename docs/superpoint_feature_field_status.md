# SuperPoint Feature Field 当前方案总结

> 更新时间：2026-04-20  
> 目标：把当前已经完成的 `SuperPoint + 3DGS/Hybrid Feature Field` 方案、实现入口、实验结果、可视化输出与后续方向整理成一份可直接交接和继续推进的技术文档。

## 1. 当前目标

当前这条线已经不再使用 RADIO 1280 维 foundation feature 作为监督目标，而是改为：

1. 用 `SuperPoint descriptor (256d)` 作为稠密描述子监督。
2. 用 `SuperPoint detector logits (65ch)` 作为关键点检测监督。
3. 把这两类纹理/局部匹配特征蒸馏进一个 `3D Gaussian Splatting` 场景表示中。
4. 在新视角下渲染出可直接用于视觉定位的 `descriptor + detector + depth`。

一句话概括：

**当前方案是在固定 3DGS 几何骨架上，训练一个 hybrid feature field 去重建 SuperPoint 的局部匹配表征，并已经跑通下游定位评估。**

## 2. 当前方案的结构

### 2.1 总体结构

```text
Per-Gaussian latent [N, 48]
  -> gsplat rasterize -> latent_map [B, 48, 60, 80]
  -> FineDecoder -> fine_feat [B, 192, 60, 80]

Depth -> unproject -> world positions [B, 3, 60, 80]
  -> SpatialHashField -> hash_feat [B, 96, 60, 80]
  -> CoarseDecoder -> coarse_feat [B, 192, 60, 80]

FusionHead -> fused [B, 384, 60, 80]
  -> SuperPointOutputHead
       -> descriptor [B, 256, 60, 80]
       -> detector   [B, 65, 60, 80]
```

### 2.2 核心设计选择

当前主线设计已经固定为：

1. 同时重建 `descriptor + detector`，不是只做单头描述子。
2. 去掉 HCD codec，不再走 1280d 压缩/解码链路。
3. 直接在 SuperPoint 空间做监督。
4. 采用 `hybrid` 架构，而不是纯 explicit。
5. 训练目标偏向 descriptor 保真，detector 作为较轻权重辅助头。

## 3. 数据与实验对象

### 3.1 场景

当前已完成并验证的 Replica 场景：

1. `room_0`
2. `room_2`

### 3.2 3DGS 几何骨架

- `room_0`: `output/3dgs_models/room_0/v8_fixed_poses_3dgs/point_cloud/iteration_30000/point_cloud.ply`
- `room_2`: `output/3dgs_models/room_2/v8_fixed_poses_3dgs/point_cloud/iteration_30000/point_cloud.ply`

### 3.3 教师特征

SuperPoint 特征已经提取完成：

- `output/superpoint_features/room_0/Sequence_1/`
- `output/superpoint_features/room_0/Sequence_2/`
- `output/superpoint_features/room_2/Sequence_1/`
- `output/superpoint_features/room_2/Sequence_2/`

每帧包含：

1. `descriptor/rgb_xxx.pt`
2. `detector/rgb_xxx.pt`

### 3.4 SuperPoint 参考资源

- 权重：`third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth`
- 参考实现：`third_party/stdloc/encoders/sp_encoder/export_image_embeddings.py`

## 4. 关键实现入口

### 4.1 特征提取

- `loc_gs/scripts/extract_superpoint_features.py`

作用：

1. 从 Replica RGB 提取 SuperPoint descriptor 与 detector。
2. 以训练脚本可直接读取的 `.pt` 格式落盘。

### 4.2 模型结构

- `loc_gs/models/hybrid_gaussian.py`

新增/关键部分：

1. `SuperPointOutputHead`
2. `_ResBlock`
3. 原有 HybridFeatureGaussian 的 fine/coarse/fusion 路径
4. `unproject_depth_to_positions()` 反投影逻辑

### 4.3 训练主脚本

- `loc_gs/scripts/train_feature_field.py`

当前在 SuperPoint 模式下已支持：

1. 读取 `descriptor + detector`
2. 跳过 codec 路径
3. descriptor 的 `L2 + cosine/focal cosine` 损失
4. detector 的 `KL divergence` 损失
5. checkpoint 中保存 `sp_output_head_state_dict`
6. mixed split 训练/验证

### 4.4 渲染器

- `loc_gs/rendering/feature_renderer.py`

关键修复：

1. `gsplat 1.4.0` 中 `info["depths"]` 不是 per-pixel depth map
2. 现已改为通过 `render_mode="RGB+ED"` 正确渲染出屏幕空间 depth

### 4.5 评估脚本

- `loc_gs/scripts/eval_superpoint.py`
  用于 descriptor/detector 重建质量评估

- `loc_gs/scripts/eval_localization.py`
  用于下游定位评估

- `loc_gs/scripts/visualize_superpoint_results.py`
  用于本轮新增的 qualitative 重建/匹配可视化

## 5. 配置与当前最优实验

### 5.1 room_0 当前最佳配置

- `configs/superpoint_hybrid_room_0_v3.yaml`

### 5.2 room_2 当前最佳配置

- `configs/superpoint_hybrid_room_2_v3.yaml`

### 5.3 v3 的关键超参

| 项 | 值 |
|---|---:|
| `hybrid_latent_dim` | 48 |
| `hash_output_dim` | 96 |
| `fine_dim` | 192 |
| `coarse_dim` | 192 |
| `hybrid_output_dim` | 384 |
| `sp_head_hidden_dim` | 512 |
| `sp_head_num_res_blocks` | 2 |
| `sp_head_use_3x3` | true |
| `descriptor_dim` | 256 |
| `detector_dim` | 65 |
| `detector_loss_weight` | 0.01 |
| `focal_cosine_gamma` | 2.0 |
| `feature_height × width` | 60 × 80 |

## 6. 已修复的关键问题

### 6.1 depth 渲染 bug

问题：

`gsplat` 返回的 `info["depths"]` 不是屏幕空间 per-pixel depth map，而是 packed 的 per-Gaussian 投影深度。

修复：

用 `render_mode="RGB+ED"` 输出 expected depth 作为真正可用的 depth map。

### 6.2 config 静默丢字段 bug

问题：

`sp_head_hidden_dim`、`sp_head_num_res_blocks`、`sp_head_use_3x3`、`focal_cosine_gamma` 没有声明在 dataclass 中，导致 YAML 配置被静默忽略。

影响：

v2 里写了更大 head，实际却一直在用默认值。

修复：

这些字段已加入 `loc_gs/config.py` 的 `LocGSConfig`。

### 6.3 eval device bug

问题：

评估脚本曾经把 `cuda:2` 和 `map_location` 写死，在 `CUDA_VISIBLE_DEVICES` 下会出错。

修复：

已经改成 `cuda:0` + `map_location="cpu"`。

## 7. 目前的结果

## 7.1 特征重建结果

### room_0 v3

| 指标 | 结果 |
|---|---:|
| Mean cosine similarity | **0.9147** |
| PSNR | **31.76 dB** |
| Exact match rate | **94.2%** |
| Match@3px | **98.5%** |
| Match@5px | **99.1%** |
| Mean NN cosine | **0.9166** |
| Median match distance | **0.00 px** |

### room_2 v3

| 指标 | 结果 |
|---|---:|
| Mean cosine similarity | **0.9108** |
| PSNR | **31.57 dB** |
| Exact match rate | **89.9%** |
| Match@3px | **95.0%** |
| Match@5px | **96.1%** |
| Mean NN cosine | **0.9150** |
| Median match distance | **0.00 px** |

结论：

1. 两个场景都已经稳定达到 `>0.91` 的 descriptor cosine。
2. room_0 的像素级匹配更接近 GT。
3. room_2 略弱，但仍然足够支撑定位。

## 7.2 下游定位结果

### room_0 v3

100 帧评估，4 个噪声等级，全部 `100%` 成功：

| Noise | Success | Rot Median | Trans Median |
|---|---:|---:|---:|
| small | 100% | 0.791° | 1.60cm |
| medium | 100% | 0.743° | 1.74cm |
| large | 100% | 0.711° | 2.18cm |
| xlarge | 100% | 0.722° | 2.10cm |

### room_2 v3

100 帧评估，4 个噪声等级，全部 `100%` 成功：

| Noise | Success | Rot Median | Trans Median |
|---|---:|---:|---:|
| small | 100% | 0.875° | 2.22cm |
| medium | 100% | 0.812° | 2.02cm |
| large | 100% | 0.895° | 2.51cm |
| xlarge | 100% | 0.903° | 3.05cm |

结论：

1. 当前 feature field 已经足以支撑鲁棒的下游定位。
2. 即使是 `±20cm / ±10°` 的初始位姿扰动，也能稳定回到低厘米、亚度级误差。
3. room_0 略优于 room_2，但两者都已经处于“可以作为主结果汇报”的状态。

## 8. 可视化输出

### 8.1 已有重建评估图

- `output/sp_gs/room0_hybrid_v3/eval/`
- `output/sp_gs/room2_hybrid_v3/eval/`

包含：

1. `sample_000.png ... sample_009.png`
2. `cosine_distribution.png`
3. `eval_summary.txt`

### 8.2 已有定位评估结果

- `output/sp_gs/room0_hybrid_v3/localization/`
- `output/sp_gs/room2_hybrid_v3/localization/`

包含：

1. `localization_results.json`
2. `localization_detail.json`
3. `localization_summary.png`
4. `localization_summary.txt`

### 8.3 新增 qualitative 可视化脚本

- `loc_gs/scripts/visualize_superpoint_results.py`

输出两类图：

1. `reconstruction/`
   - `GT RGB`
   - `GT descriptor PCA`
   - `Rendered descriptor PCA`
   - `Cosine map`
   - `GT detector heatmap`
   - `Rendered detector heatmap`

2. `matching/`
   - `GT RGB + matched keypoints`
   - `perturbed pose render + matched points`
   - `descriptor match canvas`
   - `GT / rendered detector heatmap`
   - `rendered depth`

并额外生成：

1. `reconstruction_grid.png`
2. `matching_grid.png`
3. `visualization_manifest.json`

## 9. 推荐命令

### 9.1 生成 room_0 可视化

```bash
CUDA_VISIBLE_DEVICES=5 python -m loc_gs.scripts.visualize_superpoint_results \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
  --output_dir output/sp_gs/room0_hybrid_v3/qualitative_superpoint \
  --num_reconstruction 6 \
  --num_matching 8 \
  --device cuda:0
```

### 9.2 生成 room_2 可视化

```bash
CUDA_VISIBLE_DEVICES=3 python -m loc_gs.scripts.visualize_superpoint_results \
  --config configs/superpoint_hybrid_room_2_v3.yaml \
  --checkpoint output/sp_gs/room2_hybrid_v3/checkpoints/best.pth \
  --output_dir output/sp_gs/room2_hybrid_v3/qualitative_superpoint \
  --num_reconstruction 6 \
  --num_matching 8 \
  --device cuda:0
```

## 10. 当前方案的价值判断

从目前结果看，这条线已经完成了从“能训练”到“能用于定位”的关键跨越：

1. descriptor 重建精度足够高。
2. detector 头能提供稳定关键点分布。
3. depth 渲染和反投影链路已打通。
4. dense matching + PnP 在两个场景上都表现出强鲁棒性。

所以当前方案不再只是 feature reconstruction baseline，而是：

**一个已经具备实际定位可用性的 3D Gaussian local feature field。**

## 11. 下一步建议

如果继续往论文或更完整实验推进，优先级建议如下：

1. 加入与 2D SuperPoint baseline 的直接定位对比。
2. 扩展更大规模 Cambridge 场景上的定位导向选择实验。
3. 评估不同 keypoint 阈值、ratio test、PnP 阈值对结果的敏感性。
4. 增加更大视角偏移和跨序列视角外推实验。
5. 形成统一表格与论文图版，把重建质量和定位质量放到同一页中展示。

## 12. 一句话总结

当前 `SP-GS` 已经实现并验证了一条完整的 `SuperPoint teacher -> hybrid Gaussian feature field -> novel-view descriptor/detector rendering -> dense matching -> PnP localization` 管线，并在 `room_0` 与 `room_2` 上取得了高质量重建和 `100%` 成功率的定位结果。
