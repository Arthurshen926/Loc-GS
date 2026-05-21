# Solver Tuple Diagnostic: StMarysChurch

This report compares native/source and candidate sampled landmark sets with solver-level diagnostics.
It is a mechanism diagnostic, not an official pose metric.

## Audit

- split audit: `passed`
- query grouping: `synthetic_chunks_512`
- support count delta: `+81`
- viable tuple mass delta: `+5.056733`
- mean logdet(H) delta: `-2.081454`
- ambiguity risk delta: `+0.000000`

## Summary

| Set | Support count | Tuples | Viable tuples | Viable mass | Mean logdet(H) | Ambiguity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source | 110 | 253 | 251 | 2.106330 | 18.833152 | 0.000000 |
| candidate | 191 | 433 | 427 | 7.163063 | 16.751698 | 0.000000 |

## Limitation

The input cache has no per-row `query_id`, so rows were grouped into synthetic chunks.
Use this as an early diagnostic only; paper-facing tuple banks should store real query/image ids.
