from pathlib import Path

from loc_gs.scripts import audit_stdloc_native_parity


def test_audit_stdloc_native_parity_forwards_to_cambridge_audit(monkeypatch, tmp_path, capsys):
    called = {}

    def fake_audit_cambridge_parity(**kwargs):
        called.update(kwargs)
        return {"scene": kwargs["scene"], "common_queries": 3}

    monkeypatch.setattr(
        audit_stdloc_native_parity,
        "audit_cambridge_parity",
        fake_audit_cambridge_parity,
    )
    args = audit_stdloc_native_parity.build_argparser().parse_args(
        [
            "--native_dir",
            str(tmp_path / "native"),
            "--stdloc_dir",
            str(tmp_path / "official"),
            "--output_dir",
            str(tmp_path / "audit"),
            "--scene",
            "ShopFacade",
        ]
    )

    audit_stdloc_native_parity.main(args)

    assert called == {
        "native_dir": tmp_path / "native",
        "stdloc_dir": tmp_path / "official",
        "output_dir": tmp_path / "audit",
        "scene": "ShopFacade",
    }
    assert '"common_queries": 3' in capsys.readouterr().out
