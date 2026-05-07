from train import (
    get_next_train_camera,
    get_train_image_names,
    select_report_train_cameras,
)
from scene import Scene


class StreamingScene:
    preload_cameras = False

    def __init__(self):
        self.calls = 0

    def getTrainCameras(self):
        self.calls += 1
        return iter([f"cam-{self.calls}-0", f"cam-{self.calls}-1"])


def test_streaming_train_camera_helper_restarts_after_epoch():
    scene = StreamingScene()
    viewpoint_stack = None
    viewpoint_iter = None

    cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
        scene, viewpoint_stack, viewpoint_iter
    )
    assert cam == "cam-1-0"

    cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
        scene, viewpoint_stack, viewpoint_iter
    )
    assert cam == "cam-1-1"

    cam, viewpoint_stack, viewpoint_iter = get_next_train_camera(
        scene, viewpoint_stack, viewpoint_iter
    )
    assert cam == "cam-2-0"


def test_report_camera_selection_accepts_streaming_iterable():
    scene = StreamingScene()

    cameras = select_report_train_cameras(scene, count=2)

    assert cameras == ["cam-1-0", "cam-1-1"]


def test_train_image_names_exclude_cambridge_test_split(tmp_path):
    scene_root = tmp_path
    image_root = scene_root / "processed"
    (image_root / "seq1").mkdir(parents=True)
    (image_root / "seq1" / "train.png").touch()
    (image_root / "seq1" / "test.png").touch()
    (scene_root / "dataset_test.txt").write_text("seq1/test.png 0 0 0 1 0 0 0\n")

    names = get_train_image_names(str(scene_root), "processed")

    assert names == ["seq1/train.png"]


def test_streaming_empty_test_split_returns_empty_list():
    scene = Scene.__new__(Scene)
    scene.preload_cameras = False
    scene.scene_info = type("SceneInfo", (), {"test_cameras": [], "train_cameras": []})()
    scene.pin_memory = False
    scene.dataloader_num_workers = 0
    scene.shuffle = True

    assert scene.getTestCameras() == []
