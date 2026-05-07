import numpy as np
from PIL import Image

from scene.dataset_readers import CameraInfo
from utils.camera_utils import loadCam


class Args:
    resolution = 1
    data_device = "cpu"


def test_load_cam_reads_image_path_when_camera_info_is_lazy(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8)).save(image_path)
    cam_info = CameraInfo(
        uid=1,
        R=np.eye(3),
        T=np.zeros(3),
        FovY=1.0,
        FovX=1.0,
        image=None,
        image_path=str(image_path),
        image_name="frame.png",
        width=4,
        height=3,
    )

    camera = loadCam(Args(), 0, cam_info, 1.0)

    assert camera.original_image.shape == (3, 3, 4)
