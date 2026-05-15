# Listwise SceneMatch-v3 Rank-Gap Results, 2026-05-14

This note records the follow-up to `listwise_scenematch_v2_20260514`.
The goal was to test one expert-plan suggestion without adding test-time pose
branches: expose top-K candidate context to the selector and keep a single
PROSAC solve.

## Implemented

- Added `query_score_rank_gap` listwise scalar mode:
  - query detector score, broadcast to all candidates;
  - normalized candidate rank inside the per-query top-K list;
  - cosine gap to the best candidate in that list.
- Stored the feature mode in SceneMatch checkpoints as
  `config["listwise_extra_features"]`.
- Kept backward compatibility with v2 checkpoints, where the default scalar
  mode remains `query_score`.
- Training uses the existing natural train/rendered top16 listwise caches; no
  per-scene hyperparameters were introduced.

## Training

Five Cambridge scenes were trained with the same command shape:

```bash
python -m loc_gs.scripts.train_scene_matcher \
  --pair_files output/stdloc_hybrid/listwise_v2_20260514/pairs/<scene>/scene_match_pairs.pt \
  --output_path output/stdloc_hybrid/listwise_v3_rankgap_20260514/scenematch/<scene>/best.pt \
  --listwise \
  --listwise_extra_features query_score_rank_gap \
  --epochs 12 \
  --batch_size 16384 \
  --hidden_dim 256 \
  --num_layers 3 \
  --dropout 0.05 \
  --device cuda:0
```

The checkpoints all use `scalar_dim=7`.

## Full Cambridge Results

Macro averages use dense final pose over the five Cambridge scenes.

| Variant | Median cm | Median deg | R10 | R5 | R2 | Sparse R10 | Sparse R5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| covisibility PROSAC baseline | 12.603 | 0.159 | 0.485 | 0.278 | 0.086 | 0.402 | 0.195 |
| LFF residual PROSAC | 12.456 | 0.156 | 0.485 | 0.284 | 0.085 | 0.406 | 0.196 |
| listwise-v2 top16, dustbin score | 12.575 | 0.156 | 0.491 | 0.285 | 0.082 | 0.410 | 0.209 |
| listwise-v3 rank-gap top16 | 12.631 | 0.157 | 0.486 | **0.287** | 0.085 | 0.405 | 0.201 |
| listwise-v3 rank-gap top32 | 12.550 | 0.159 | 0.488 | 0.286 | **0.091** | **0.416** | 0.203 |
| residual + listwise-v3 rank-gap top16 | 12.641 | 0.157 | 0.483 | 0.285 | 0.082 | 0.400 | 0.207 |
| top16 candidate oracle | **11.183** | **0.138** | **0.526** | **0.310** | **0.100** | **0.600** | **0.324** |

## Conclusion

Rank/gap context is useful, but not sufficient. It improves strict recall:
top16 gives the best learned dense R5 so far, while top32 gives the best learned
dense R2 and sparse R10 in this listwise family. However, it does not close the
large gap to the top16 oracle, and it does not combine cleanly with the current
residual descriptor field.

The paper-facing conclusion is therefore sharper:

- LFF residual remains the best median-error component.
- Listwise candidate selection supports the localization-feedback feature
  selection claim through recall gains, especially at R5/R2.
- The remaining SOTA gap is not an eval-time multi-pose issue; it is candidate
  supervision and feature-bank quality. The next high-value step is training the
  field or selector with stronger hard-negative/verifier labels, rather than
  adding more pose branches.
