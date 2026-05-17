# Selector Causality Ablation

This ablation verifies that Loc-GS Sampling Field gains come from
self-localization feedback, not from arbitrary detector-score perturbation.

## Fixed Conditions

- Native STDLoc descriptors: `descriptor_mode=native`.
- Same STDLoc-compatible evaluator and single OpenCV PROSAC PnP path.
- Same Cambridge query split for evaluation only.
- No per-scene recipe, no per-query branch selection, no evaluator changes.

## Ablations

| Ablation | Transform | Detector scores | PLY locability | Purpose |
| --- | --- | --- | --- | --- |
| native | disabled | no | no | Baseline control. |
| selector | identity | yes | yes | Main selector-score update. |
| uniform | mean gate | yes | yes | Tests whether any constant score shift helps. |
| permuted | seeded permutation | yes | yes | Preserves score distribution but destroys landmark identity. |
| inverted | `1 - gate` | yes | yes | Tests whether high feedback utility is directionally meaningful. |
| detector_only | identity | yes | no | Isolates sparse landmark ranking / PROSAC path. |
| locability_only | identity | no | yes | Isolates PLY locability / dense refinement prior. |
| both | identity | yes | yes | Alias for the main selector update in batch exports. |

If the learned selector beats uniform, permuted, and inverted controls under the
same global blend, the evidence supports a localization-utility signal rather
than random score nudging.

## Export Command

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.export_selector_ablation_maps \
  --source_map output/stdloc/map_cambridge_spgs/ShopFacade \
  --checkpoint_path output/unified_lff_v2/train_20260516_rehearsal60k_decoupled_gate1_fp0/ShopFacade/unified_lff_v2.pt \
  --output_root output/unified_lff_v2/selector_ablation_20260516/ShopFacade \
  --gate_locability_blend 0.05 \
  --descriptor_mode native \
  --ablation native,selector,uniform,permuted,inverted,detector_only,locability_only,both \
  --seed 0
```

The export writes STDLoc-compatible maps only. It does not run Cambridge
evaluation or modify metrics.
