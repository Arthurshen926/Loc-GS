# STDLoc Integration

`third_party/stdloc` is a full vendored copy of the current `/root/STDLoc`
workspace. Cambridge maps, logs, and result folders needed to reproduce the
localization-guided reconstruction experiments are stored canonically under
`output/stdloc/`.

Run STDLoc commands from the vendored directory:

```bash
cd /root/Loc-GS/third_party/stdloc
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests
```

Large artifact directories are intentionally present on disk but ignored by git:

- `output/stdloc/map_cambridge_spgs/`
- `output/stdloc/map_cambridge_smoke/`
- `output/stdloc/results/`
- `output/stdloc/logs/`

For compatibility with existing STDLoc scripts, the corresponding paths under
`third_party/stdloc/` are relative symlinks back to `output/stdloc/`.

The vendored project keeps its original MIT license at
`third_party/stdloc/LICENSE`.
