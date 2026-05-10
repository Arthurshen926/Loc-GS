import numpy as np
import torch

from loc_gs.data.superpoint_cache import SuperPointTeacherCache, superpoint_score_map_from_logits


def test_superpoint_cache_round_trips_dense_outputs_and_metadata(tmp_path):
    cache = SuperPointTeacherCache(tmp_path, scene="ShopFacade", split="train")
    desc = torch.randn(256, 3, 4)
    logits = torch.randn(65, 3, 4)
    keypoints = torch.tensor([[1.0, 2.0]])
    keypoint_desc = torch.randn(1, 256)

    cache.save("seq/image001.png", desc, logits, keypoints=keypoints, keypoint_descriptors=keypoint_desc)
    loaded = cache.load("seq/image001.png")

    assert loaded is not None
    assert torch.allclose(loaded.descriptor, desc)
    assert torch.allclose(loaded.detector_logits, logits)
    assert cache.score_path("seq/image001.png").exists()
    assert cache.metadata_path("seq/image001.png").exists()

    meta = np.load(cache.metadata_path("seq/image001.png"))
    assert meta["keypoints"].shape == (1, 2)
    assert meta["descriptors"].shape == (1, 256)


def test_superpoint_cache_save_metadata_does_not_overwrite_dense_outputs(tmp_path):
    cache = SuperPointTeacherCache(tmp_path, scene="ShopFacade", split="train_original")
    desc = torch.randn(256, 9, 11)
    logits = torch.randn(65, 9, 11)
    keypoints = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    keypoint_desc = torch.randn(2, 256)

    cache.save("seq/image001.png", desc, logits)
    before = cache.descriptor_path("seq/image001.png").stat().st_size
    cache.save_metadata("seq/image001.png", keypoints=keypoints, keypoint_descriptors=keypoint_desc)
    after = cache.descriptor_path("seq/image001.png").stat().st_size
    loaded = cache.load("seq/image001.png")

    assert before == after
    assert loaded is not None
    assert loaded.descriptor.shape == (256, 9, 11)
    assert loaded.detector_logits.shape == (65, 9, 11)
    assert torch.allclose(loaded.descriptor, desc)
    assert torch.allclose(loaded.detector_logits, logits)


def test_superpoint_score_map_from_logits_pixel_shuffles_detector():
    logits = torch.zeros(65, 2, 3)
    logits[0] = 4.0
    score = superpoint_score_map_from_logits(logits)

    assert score.shape == (16, 24)
    assert float(score.max()) > float(score.min())


def test_superpoint_cache_treats_corrupt_descriptor_as_miss(tmp_path):
    cache = SuperPointTeacherCache(tmp_path, scene="ShopFacade", split="train")
    path = cache.descriptor_path("seq/image001.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"partial-write")

    assert cache.load("seq/image001.png") is None
