import json

from loc_gs.reporting.frozen_profile_coverage import build_frozen_profile_coverage
from loc_gs.scripts.write_frozen_profile_coverage import main


def _write_profile(root, name, *, scene, map_path, split="train", queries=5, peak_gpu_mb=1234.0):
    run = root / name / scene
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "split": split,
                "map_path": map_path,
                "hyperparameters": {"max_test_cameras": queries, "warmup_cameras": 1},
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "eval_split": split,
                "max_test_cameras": queries,
                "landmark_count": 8192,
                "peak_gpu_mb": peak_gpu_mb,
            }
        ),
        encoding="utf-8",
    )
    (run / "timing_profile.json").write_text(
        json.dumps(
            {
                "queries": queries,
                "latency_ms": {"total": {"mean": 100.0, "median": 90.0, "p95": 130.0}},
                "memory": {"peak_gpu_mb": peak_gpu_mb},
                "peak_gpu_mb": peak_gpu_mb,
            }
        ),
        encoding="utf-8",
    )
    (run / "split_audit.json").write_text(json.dumps({"audit_status": "passed"}), encoding="utf-8")
    return run


def _frozen_manifest():
    return {
        "freeze_id": "freeze-test",
        "splits": {
            "q80": {
                "selected_runs": {
                    "ShopFacade": {
                        "scene": "ShopFacade",
                        "selected": "v3",
                        "map_path": "output/maps/shop_v3",
                    },
                    "KingsCollege": {
                        "scene": "KingsCollege",
                        "selected": "native",
                        "map_path": "output/maps/kings_native",
                    },
                }
            },
            "q160": {
                "selected_runs": {
                    "ShopFacade": {
                        "scene": "ShopFacade",
                        "selected": "v3",
                        "map_path": "output/maps/shop_v3",
                    }
                }
            },
        },
    }


def test_build_frozen_profile_coverage_deduplicates_selected_maps_and_matches_profiles(tmp_path):
    profile_root = tmp_path / "profiles"
    _write_profile(profile_root, "shop_lsf_v3", scene="ShopFacade", map_path="output/maps/shop_v3")
    _write_profile(profile_root, "kings_native", scene="KingsCollege", map_path="output/maps/kings_native")

    report = build_frozen_profile_coverage(
        _frozen_manifest(),
        profile_roots=[profile_root],
        required_queries=5,
    )

    assert report["format"] == "loc_gs_frozen_profile_coverage_v1"
    assert report["checks"]["all_unique_selected_maps_profiled"] is True
    assert report["checks"]["all_profiles_have_runtime"] is True
    assert report["checks"]["all_profiles_have_memory"] is True
    assert report["checks"]["no_test_split_profiles"] is True
    assert report["unique_selected_map_count"] == 2

    shop_entry = next(entry for entry in report["entries"] if entry["scene"] == "ShopFacade")
    assert shop_entry["recipe_splits"] == ["q160", "q80"]
    assert shop_entry["profile"]["queries"] == 5
    assert shop_entry["profile"]["peak_gpu_mb"] == 1234.0


def test_build_frozen_profile_coverage_flags_missing_and_test_split_profiles(tmp_path):
    profile_root = tmp_path / "profiles"
    _write_profile(profile_root, "shop_lsf_v3", scene="ShopFacade", map_path="output/maps/shop_v3", split="test")

    report = build_frozen_profile_coverage(
        _frozen_manifest(),
        profile_roots=[profile_root],
        required_queries=5,
    )

    assert report["checks"]["all_unique_selected_maps_profiled"] is False
    assert report["checks"]["no_test_split_profiles"] is False
    assert report["checks"]["missing_profile_count"] == 1
    assert report["checks"]["test_split_profile_count"] == 1

    missing = next(entry for entry in report["entries"] if entry["scene"] == "KingsCollege")
    assert missing["profile_status"] == "missing"


def test_write_frozen_profile_coverage_cli(tmp_path):
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(_frozen_manifest()), encoding="utf-8")
    profile_root = tmp_path / "profiles"
    _write_profile(profile_root, "shop_lsf_v3", scene="ShopFacade", map_path="output/maps/shop_v3")
    _write_profile(profile_root, "kings_native", scene="KingsCollege", map_path="output/maps/kings_native")
    output = tmp_path / "coverage.json"

    assert (
        main(
            [
                "--frozen-recipe",
                str(frozen_path),
                "--profile-roots",
                str(profile_root),
                "--required-queries",
                "5",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["frozen_recipe_path"] == str(frozen_path)
    assert payload["checks"]["profile_coverage_ready"] is True
