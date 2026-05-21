# Alpha-Composition Reliability Plan, 2026-05

## Claim Boundary

Loc-GS should not claim that it solves alpha-blending feature bias or learns
unbiased per-Gaussian descriptors. The paper-safe statement is:

```text
ULF-Loc avoids alpha-blending feature optimization.
Loc-GS instead estimates which alpha-composited landmarks, rendered rays, and
dense residuals remain reliable enough for the geometric localization backend.
```

## Reliability Signals

Exact raster composition weights are preferred when available. Until then, the
implemented proxy supports:

```text
composition dominance
normalized composition entropy
per-Gaussian LSF support aggregation
optional opacity
optional depth stability
```

The cache explicitly records:

```text
composition_proxy=true
exact_raster_weights=false
```

## Implemented Modules

```text
loc_gs/diagnostics/alpha_composition.py
loc_gs/diagnostics/rendered_reliability.py
loc_gs/scripts/build_alpha_reliability_cache.py
tests/test_alpha_composition.py
tests/test_rendered_reliability.py
tests/test_build_alpha_reliability_cache.py
```

## Intended Use

Sparse stage:

```text
Gaussian support can be combined with solver tuple diagnostics before
admitting non-native landmark replacements.
```

Dense stage:

```text
Rendered support mask filters or weights dense correspondences/residuals:
m(u) = LSF_support(u) * dominance(u) * (1 - entropy(u)) * depth_stability(u)
```

## CLI

```bash
python -m loc_gs.scripts.build_alpha_reliability_cache \
  --contributor_ids <ids.pt> \
  --contribution_weights <weights.pt> \
  --gaussian_support <support.pt> \
  --threshold 0.5 \
  --output_dir output/alpha_reliability/<scene>
```

This first version is a cache builder and diagnostic primitive. It does not yet
modify STDLoc dense refinement.

