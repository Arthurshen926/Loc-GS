# Clean Render Phase Gate Assessment 20260530

This assessment checks whether Phase0-Phase3 pass their current gates. It uses
existing train/self-map diagnostic artifacts and unit tests only. It does not
promote any result to paper-facing Cambridge test status.

## Environment Note

`locgsctl list-scenes` reports the current default Cambridge native sampled
maps as `8192`, while the expected native count is `16384`. Therefore this
assessment is limited to diagnostic phase readiness and train/self-map evidence.
It is not a paper-facing native STDLoc parity claim.

## Gate Summary

| Phase | Gate | Status | Evidence |
| --- | --- | --- | --- |
| Phase0 hard/normal benchmark | fixed train/self-map benchmark, all four case types present, audit marks no official test | PASS | 157 cases: A=7, B=50, C=50, D=50; split train=119/val=38; `official_test_used=false`, `source_splits=["train"]` |
| Phase1 clean-render validation | render-health metrics should distinguish improvement from regression before final pose use | FAIL for main path, PASS as diagnostic | improvement median visible=0.9423 / feature=0.5448, regression median visible=0.9486 / feature=0.6213; 4 regression cases have Reg20 |
| Phase2 patch dense candidates | standalone candidate generator emits patch/global diagnostics without selecting pose | PASS diagnostic/unit only | `tests/test_patch_dense_candidates.py`: 3 passed |
| Phase3 sparse-anchor residual | standalone sparse-anchor residual group and no-anchor/concat/separate comparison | PASS diagnostic/unit only | `tests/test_sparse_anchor_residual.py`: 3 passed |
| Full train/self-map SLCDP transition | dense worsened count must not increase, reg20 should be reduced, median/tail should not regress | FAIL | train gate says GreatCourt dense-worsened increased and raw Reg20 remains nonzero; macro median `9.8882 -> 9.9047`; dense worsened `237 -> 242`; Reg20=4 |

## Detailed Reading

### Phase0

Phase0 passes its current purpose: it creates a fixed train/self-map diagnostic
benchmark with four case classes:

- `A_sparse_good_dense_bad`: sparse is good but dense degrades.
- `B_sparse_marginal_dense_recoverable`: sparse is marginal and dense can recover.
- `C_sparse_catastrophic`: sparse is already far outside the reliable basin.
- `D_normal_dense_good`: native dense is normal.

This phase is safe for train/self-map diagnostics because the generated audit
marks `official_test_used=false` and `source_splits=["train"]`.

### Phase1

Phase1 does not pass as a main-method acceptance gate. The key failure is that
the current render-health signals cannot separate positive and negative clean
render candidates:

- Improvement cases: median visible `0.9423`, median feature cosine `0.5448`.
- Regression cases: median visible `0.9486`, median feature cosine `0.6213`.

The regression group looks at least as healthy under these metrics, yet produces
Reg20 regressions. This means anchor visibility, near-occluder ratio, and feature
cosine are not enough to decide whether a clean render should enter final dense
refinement.

Phase1 is still useful as a diagnostic layer: it exposes exactly which signal is
missing.

### Phase2

Phase2 passes its current unit/diagnostic gate. It now provides a patch dense
candidate generator with:

- local match quality;
- anchor consistency;
- off-patch consistency;
- patch pose-cluster consistency;
- ambiguity score;
- patch residual weight.

It does not yet pass a Cambridge data gate, because it has not been run over the
Phase0 train/val benchmark or integrated with clean-render candidates.

### Phase3

Phase3 passes its current unit/diagnostic gate. It provides sparse-anchor
residual diagnostics and confirms the intended objective distinction:

- `concat_matches` can dilute sparse-anchor influence when dense matches are
  numerous;
- `separate_residual_group` preserves an explicit sparse-anchor contribution.

It does not yet pass a final-method gate, because it has not been connected to a
real dense optimizer or full train/self-map evaluation.

## Overall Status

The overall sparse-conditioned clean-render transition does not pass the current
train/self-map full gate. Existing gate summary:

- Macro median TE: `9.8882 -> 9.9047`, slightly worse.
- Macro P90 TE: `33.3367 -> 33.2989`, slightly better.
- Macro P95 TE: `62.9950 -> 59.9610`, better tail.
- Dense worsened count: `237 -> 242`, worse.
- Reg20 total: `4`.
- GreatCourt dense worsened increased.

So the correct status is:

> Phase0, Phase2, and Phase3 pass diagnostic readiness. Phase1 fails as a
> main-path acceptance gate. The combined SLCDP clean-render transition fails the
> train/self-map full gate and must not proceed to official Cambridge test.

## Required Next Gate

Before any official test or paper-facing claim, the next step should build a
single no-GT transition safety score using Phase1-3 signals:

- native dense stability: sparse-anchor retention, small pose update, dense
  inlier ratio;
- Phase2 patch/global consistency: ambiguity, off-anchor consistency, spatial
  coverage, patch weight;
- Phase3 sparse-anchor separate residual: anchor group loss and anchor loss
  share;
- clean-render health from Phase1: visibility, depth consistency, feature
  agreement, near-occluder ratio.

That safety score should be validated first on Phase0 train/val splits. Passing
criteria should be:

- D normal cases: no Reg20 and no non-base action unless native preflight is
  clearly invalid.
- A sparse-good/dense-bad cases: recover at least one case without introducing
  D-case regressions.
- C sparse-catastrophic cases: do not trust sparse-anchor clean render unless
  sparse confidence is explicitly true.
- full train/self-map: dense worsened count does not increase, Reg20 decreases,
  macro median and P90/P95 do not regress.
