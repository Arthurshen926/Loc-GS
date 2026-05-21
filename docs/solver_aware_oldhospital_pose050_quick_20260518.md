# Solver Tuple Diagnostic: OldHospital

This report compares native/source and candidate sampled landmark sets with solver-level diagnostics.
It is a mechanism diagnostic, not an official pose metric.

## Audit

- split audit: `passed`
- query grouping: `synthetic_chunks_512`
- support count delta: `+14`
- viable tuple mass delta: `+1.550953`
- mean logdet(H) delta: `-0.065598`
- ambiguity risk delta: `+0.000000`

## Summary

| Set | Support count | Tuples | Viable tuples | Viable mass | Mean logdet(H) | Ambiguity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source | 72 | 209 | 209 | 4.570092 | 13.350074 | 0.000000 |
| candidate | 86 | 222 | 222 | 6.121045 | 13.284476 | 0.000000 |

## Limitation

The input cache has no per-row `query_id`, so rows were grouped into synthetic chunks.
Use this as an early diagnostic only; paper-facing tuple banks should store real query/image ids.
