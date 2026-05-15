# STDLoc Native Parity Design

## Goal

Create a first-class STDLoc baseline path inside this repository before any Loc-GS improvement is evaluated. The baseline must run the original sparse-to-dense STDLoc localization flow and training flow without mixing in Loc-GS matchability, SceneMatcher, calibrated selectors, multi-hypothesis tricks, or learned residual descriptor branches.

## Architecture

The first implementation uses `third_party/stdloc` as the source-of-truth execution engine and wraps it with `loc_gs` scripts. This avoids reintroducing approximation errors while giving the project a clean, auditable STDLoc control path. Later Loc-GS changes can compare against this baseline through the same result format and parity audit.

## Components

- `loc_gs/stdloc_native/commands.py`: typed command builders for official STDLoc training and evaluation.
- `loc_gs/stdloc_native/results.py`: summary/result loading and metric normalization for STDLoc JSON outputs.
- `loc_gs/scripts/eval_stdloc_native.py`: run one scene through official STDLoc evaluation from the project CLI.
- `loc_gs/scripts/train_stdloc_native.py`: run one scene through official STDLoc feature-GS and detector training from the project CLI.
- `loc_gs/scripts/launch_stdloc_native_cambridge.py`: dispatch Cambridge scene training/evaluation across available GPUs.
- `loc_gs/scripts/audit_stdloc_native_parity.py`: compare official STDLoc output and any local/native output at query level.

## Data Flow

Evaluation reads a dataset scene, a STDLoc map folder, a STDLoc YAML config, and optionally a query subset. It invokes `third_party/stdloc/stdloc.py`, writes results into an explicit output directory, then normalizes the output summary so downstream comparison code can consume the same fields.

Training reads a Cambridge scene and writes STDLoc map artifacts to a map root. It invokes `third_party/stdloc/train.py` with the official Cambridge defaults, including feature Gaussian training, matching-oriented Gaussian sampling, and scene-specific detector training.

## Success Criteria

- A single-scene native evaluation command can be generated deterministically and run without importing Loc-GS localization modules.
- A Cambridge launcher can fill multiple GPUs by assigning scenes round-robin.
- JSON summaries expose sparse and dense median pose errors and recall fields in a normalized format.
- Existing tests continue to pass.
- Full metric parity is judged by running the wrappers against current STDLoc outputs and auditing query-level deltas.
