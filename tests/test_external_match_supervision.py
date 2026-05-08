import numpy as np
import torch
import torch.nn.functional as F

from loc_gs.data.external_match_cache import ExternalMatchCache
from loc_gs.losses.external_match import external_match_supervision_loss
from loc_gs.scripts.train_cambridge_hybrid import load_external_match_training_batch


def test_external_match_cache_round_trips_features_and_matches(tmp_path):
    cache = ExternalMatchCache(tmp_path, scene="ShopFacade", pipeline="loftr", split="train")
    keypoints = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    descriptors = np.eye(2, dtype=np.float32)
    scores = np.array([0.8, 0.9], dtype=np.float32)

    cache.save_features(
        "seq1/frame00001.png",
        keypoints_xy=keypoints,
        descriptors=descriptors,
        scores=scores,
        image_size=(360, 640),
    )
    feat = cache.load_features("seq1/frame00001.png")

    assert feat is not None
    assert feat.pipeline == "loftr"
    assert feat.image_size == (360, 640)
    assert np.allclose(feat.keypoints_xy, keypoints)
    assert np.allclose(feat.descriptors, descriptors)
    assert np.allclose(feat.scores, scores)

    cache.save_matches(
        "seq1/frame00001.png",
        "seq1/frame00002.png",
        kpts_a_xy=keypoints,
        kpts_b_xy=keypoints + 1.0,
        scores=scores,
        geom_inlier_mask=np.array([True, False]),
        stats={"overlap": 0.42, "num_raw_matches": 2},
    )
    match = cache.load_matches("seq1/frame00001.png", "seq1/frame00002.png")

    assert match is not None
    assert match.pipeline == "loftr"
    assert match.image_a == "seq1/frame00001.png"
    assert match.image_b == "seq1/frame00002.png"
    assert match.geom_inlier_mask.tolist() == [True, False]
    assert match.stats["overlap"] == 0.42


def test_load_external_match_training_batch_scales_xy_to_feature_yx(tmp_path):
    cache = ExternalMatchCache(tmp_path, scene="ShopFacade", pipeline="loftr", split="train")
    cache.save_matches(
        "a.png",
        "b.png",
        kpts_a_xy=np.array([[80.0, 40.0], [160.0, 80.0]], dtype=np.float32),
        kpts_b_xy=np.array([[88.0, 48.0], [200.0, 120.0]], dtype=np.float32),
        scores=np.array([0.9, 0.1], dtype=np.float32),
        geom_inlier_mask=np.array([True, False]),
        stats={},
    )

    batch = load_external_match_training_batch(
        cache,
        ["a.png"],
        ["b.png"],
        image_height=160,
        image_width=320,
        feature_height=20,
        feature_width=40,
        max_matches=8,
        device=torch.device("cpu"),
    )

    assert batch is not None
    assert torch.allclose(batch["kpts_a_yx"][0, 0], torch.tensor([5.0, 10.0]))
    assert torch.allclose(batch["kpts_b_yx"][0, 0], torch.tensor([6.0, 11.0]))
    assert batch["valid"].tolist() == [[True]]
    assert torch.allclose(batch["negative_kpts_b_yx"][0, 0], torch.tensor([15.0, 25.0]))


def test_external_match_supervision_loss_pulls_positive_descriptors_together():
    desc_a_good = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_a_bad = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_b = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_a_good.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_a_bad.data[:, :, 0, 0] = torch.tensor([0.0, 1.0])
    desc_b.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_b.data[:, :, 1, 1] = torch.tensor([0.0, 1.0])
    desc_a_good = F.normalize(desc_a_good, dim=1)
    desc_a_bad = F.normalize(desc_a_bad, dim=1)
    desc_b = F.normalize(desc_b, dim=1)
    kpt = torch.tensor([[[0.0, 0.0]]])

    good = external_match_supervision_loss(
        desc_a_good,
        desc_b,
        kpts_a_yx=kpt,
        kpts_b_yx=kpt,
        scores=torch.ones(1, 1),
        temperature=0.1,
    )
    bad = external_match_supervision_loss(
        desc_a_bad,
        desc_b,
        kpts_a_yx=kpt,
        kpts_b_yx=kpt,
        scores=torch.ones(1, 1),
        temperature=0.1,
    )

    assert good["total"] < bad["total"]
    assert good["valid_matches"].item() == 1
    good["total"].backward()
    assert desc_a_good.grad is not None


def test_external_match_supervision_loss_penalizes_hard_negative_similarity():
    desc_a = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_b = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_a.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_b.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_b.data[:, :, 1, 1] = torch.tensor([1.0, 0.0])
    desc_a = F.normalize(desc_a, dim=1)
    desc_b = F.normalize(desc_b, dim=1)

    out = external_match_supervision_loss(
        desc_a,
        desc_b,
        kpts_a_yx=torch.tensor([[[0.0, 0.0]]]),
        kpts_b_yx=torch.tensor([[[0.0, 0.0]]]),
        scores=torch.ones(1, 1),
        negative_kpts_b_yx=torch.tensor([[[1.0, 1.0]]]),
        hard_negative_weight=1.0,
        hard_negative_margin=0.1,
        temperature=0.1,
    )

    assert out["hard_negative"] > 0
    out["total"].backward()
    assert desc_a.grad is not None
    assert desc_b.grad is not None
