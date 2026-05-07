import importlib
import sys
import types

import torch


def test_gaussian_renderer_imports_when_2dgs_backend_is_missing(monkeypatch):
    gsplat_stub = types.ModuleType("gsplat")
    gsplat_stub.rasterization = object()
    monkeypatch.setitem(sys.modules, "gsplat", gsplat_stub)
    sys.modules.pop("gaussian_renderer", None)

    renderer = importlib.import_module("gaussian_renderer")

    assert renderer.rasterization is gsplat_stub.rasterization
    assert renderer.rasterization_2dgs is None


def test_visible_locability_values_follow_visible_mask(monkeypatch):
    gsplat_stub = types.ModuleType("gsplat")
    gsplat_stub.rasterization = object()
    monkeypatch.setitem(sys.modules, "gsplat", gsplat_stub)
    sys.modules.pop("gaussian_renderer", None)
    renderer = importlib.import_module("gaussian_renderer")

    class DummyGaussians:
        @property
        def get_locability(self):
            return torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)

    values = renderer.get_visible_locability_values(
        DummyGaussians(), torch.tensor([True, False, True])
    )

    assert torch.equal(values.squeeze(-1), torch.tensor([0.1, 0.9]))
