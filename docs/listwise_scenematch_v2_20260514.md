# Listwise SceneMatch-v2 Results, 2026-05-14

This note records the first full Cambridge pass for the expert-plan P2 update:
query-level top-K candidate selection instead of independent pair reranking.

## Implemented

- `SceneMatchListwiseNet`: candidate MLP + query-level mean/max context, producing `K + dustbin` logits.
- Listwise self-localization cache from calibration:
  - train views + rendered/perturbed rehearsal views.
  - topK = 16 candidates per query keypoint.
  - natural topK group distribution; no forced 1:1 positive/dustbin sampling.
  - automatic binary-balanced CE weights at training time to avoid all-dustbin collapse.
- Eval support:
  - listwise checkpoint auto-detected by `model_type=listwise`.
  - `scene_matcher_listwise_dustbin=score` keeps the best candidate and uses `candidate_logit - dustbin_logit` as confidence.
  - `scene_matcher_listwise_dustbin=drop` is retained as a hard rejection diagnostic.

## Pair Cache Coverage

| Scene | Samples | Positive Top16 Ratio |
|---|---:|---:|
| GreatCourt | 120k | 0.1775 |
| KingsCollege | 120k | 0.0308 |
| OldHospital | 120k | 0.0897 |
| ShopFacade | 120k | 0.1767 |
| StMarysChurch | 120k | 0.1288 |

KingsCollege remains the hardest scene: even top16 candidates contain a correct landmark
for only about 3% of collected query groups, so selector learning is heavily limited by
candidate-pool quality.

## Full Cambridge Results

Macro averages use dense final pose over the five Cambridge scenes.

| Variant | Median cm | Median deg | R10 | R5 | R2 |
|---|---:|---:|---:|---:|---:|
| covisibility PROSAC baseline | 12.603 | 0.159 | 0.485 | 0.278 | 0.086 |
| LFF residual PROSAC | 12.456 | 0.156 | 0.485 | 0.284 | 0.085 |
| listwise top16, hard dustbin drop | 12.592 | 0.158 | 0.480 | 0.281 | 0.080 |
| listwise top16, dustbin score | 12.575 | 0.156 | 0.491 | 0.285 | 0.082 |
| residual + listwise dustbin score | 12.612 | 0.159 | 0.483 | 0.285 | 0.086 |
| top16 candidate oracle | 11.183 | 0.138 | 0.526 | 0.310 | 0.100 |

## Conclusion

Listwise scoring is the first learned selector variant that improves strict recall over the
plain covisibility baseline on the full split, but the gain is small and does not close the
gap to the top16 candidate oracle. Hard dustbin rejection is not suitable as the mainline
because it sacrifices coverage. The best paper-facing evidence from this round is:

- top16 candidate oracle confirms meaningful headroom in candidate selection rather than
  only PROSAC ordering.
- listwise dustbin-score selection moves R10/R5 in the right direction without branch
  selection or multi-pose test-time paths.
- residual LFF remains the better median-error component; it does not combine cleanly with
  the current listwise selector.

The next precision push should therefore improve candidate-pool supervision, not add more
eval-time scoring tricks: include stronger hard negatives, candidate rank/geometry context,
and external matcher/MASt3R/RIPE-style teacher labels for topK candidate labels.
