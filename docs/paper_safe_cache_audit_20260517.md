# Paper-Safe Cache Audit Update, 2026-05-17

## Purpose

The older 60k rehearsal/listwise caches record `phase_counts` but do not record
the source image ids used to build train/rendered labels. Those artifacts must
remain `audit_status=unknown` and out of paper-facing main tables.

This update patches future `calibrate_landmark_matchability` pair/listwise
caches to carry enough split metadata for a mechanical audit:

```text
source_split_name
feedback_bank_split_name
phase_source_image_ids
test_image_ids
split_audit
```

Rendered rehearsal entries are recorded as `rendered_from:{train_image_id}` and
are audited against the underlying train image id. No test query image ids are
used for labels, calibration, model selection, or hard-negative mining.

## Smoke Verification

A tiny ShopFacade smoke cache was generated from one train image:

```text
output/unified_lff_v2/audit_smoke_20260517/ShopFacade/listwise.pt
output/unified_lff_v2/audit_smoke_20260517/ShopFacade/listwise.json
```

The cache metadata records:

```text
phase_source_image_ids.train = ["seq2/frame00001.png"]
phase_source_image_ids.rendered = []
test_image_ids count = 103
split_audit.audit_status = passed
```

A one-epoch selector-only smoke train from that cache also preserved the passed
audit:

```text
output/unified_lff_v2/audit_smoke_20260517/ShopFacade/selector_only_smoke.pt
output/unified_lff_v2/audit_smoke_20260517/ShopFacade/manifest.json
output/unified_lff_v2/audit_smoke_20260517/ShopFacade/split_audit.json
```

The training sidecar reports:

```text
split = selfmap_train_rendered
audit_status = passed
```

## Boundary

This is an audit pipeline smoke, not a localization result. It does not improve
pose accuracy, recall, or runtime. Its value is that future full rehearsal caches
can be made paper-safe without weakening the split rules.

## Next Use

Regenerate the 60k rehearsal/listwise caches with the patched writer before
using selector-only training or feedback-bank labels as paper-facing evidence.
Old 2026-05-16 caches stay diagnostic unless their source image ids can be
recovered from independent logs.
