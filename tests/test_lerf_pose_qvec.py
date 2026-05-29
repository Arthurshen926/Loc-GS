import numpy as np

from loc_gs.data.lerf_dataset import _qvec_to_rotmat


def test_lerf_colmap_qvec_to_rotmat_normalizes_non_unit_quaternion() -> None:
    qvec = np.asarray([0.003359, 0.017963, -1.018482, 0.234648], dtype=np.float32)

    rotation = _qvec_to_rotmat(qvec)

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
