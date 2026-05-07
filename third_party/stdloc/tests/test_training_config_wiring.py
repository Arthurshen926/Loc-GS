import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse(path):
    return ast.parse((ROOT / path).read_text())


def test_train_passes_detector_iteration_cli_value():
    tree = _parse("train.py")
    training_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "training"
    )
    arg_names = [arg.arg for arg in training_fn.args.args]

    assert "train_detector_iterations" in arg_names

    detector_calls = [
        node
        for node in ast.walk(training_fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "training_detector"
    ]
    assert len(detector_calls) == 1
    train_iteration_kw = next(
        kw for kw in detector_calls[0].keywords if kw.arg == "train_iteration"
    )
    assert isinstance(train_iteration_kw.value, ast.Name)
    assert train_iteration_kw.value.id == "train_detector_iterations"


def test_detector_training_persists_locability_to_loaded_map_iteration():
    detector_tree = _parse("train_detector.py")
    training_detector_fn = next(
        node
        for node in ast.walk(detector_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "training_detector"
    )
    arg_names = [arg.arg for arg in training_detector_fn.args.args]

    assert "locability_save_iteration" in arg_names
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scene"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "locability_save_iteration"
        for node in ast.walk(training_detector_fn)
    )

    train_tree = _parse("train.py")
    training_fn = next(
        node
        for node in ast.walk(train_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "training"
    )
    detector_call = next(
        node
        for node in ast.walk(training_fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "training_detector"
    )
    save_iter_kw = next(
        kw for kw in detector_call.keywords if kw.arg == "locability_save_iteration"
    )
    assert isinstance(save_iter_kw.value, ast.Attribute)
    assert isinstance(save_iter_kw.value.value, ast.Name)
    assert save_iter_kw.value.value.id == "opt"
    assert save_iter_kw.value.attr == "iterations"


def test_streaming_training_disables_multiprocess_pin_memory_loader():
    train_tree = _parse("train.py")
    training_fn = next(
        node
        for node in ast.walk(train_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "training"
    )
    scene_call = next(
        node
        for node in ast.walk(training_fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "Scene"
    )
    keyword_names = {kw.arg for kw in scene_call.keywords}

    assert "dataloader_num_workers" in keyword_names
    assert "pin_memory" in keyword_names


def test_train_can_load_existing_point_cloud_iteration_for_finetuning():
    tree = _parse("train.py")
    training_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "training"
    )
    arg_names = [arg.arg for arg in training_fn.args.args]

    assert "load_iteration" in arg_names

    scene_call = next(
        node
        for node in ast.walk(training_fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "Scene"
    )
    load_iter_kw = next(kw for kw in scene_call.keywords if kw.arg == "load_iteration")
    assert isinstance(load_iter_kw.value, ast.Name)
    assert load_iter_kw.value.id == "load_iteration"


def test_gaussian_restore_keeps_old_checkpoint_optimizer_optional():
    tree = _parse("scene/gaussian_model.py")
    gaussian_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
    )
    restore_fn = next(
        node
        for node in gaussian_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "restore"
    )
    catches_value_error = any(
        isinstance(node, ast.Try)
        and any(
            getattr(handler.type, "id", None) == "ValueError"
            for handler in node.handlers
        )
        for node in ast.walk(restore_fn)
    )

    assert catches_value_error


def test_training_exposes_geometry_anchor_regularization_options():
    tree = _parse("arguments/__init__.py")
    opt_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OptimizationParams"
    )
    assigned_names = {
        target.attr
        for node in ast.walk(opt_class)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }

    assert "geometry_anchor_weight" in assigned_names
    assert "geometry_scale_anchor_weight" in assigned_names


def test_training_exposes_selective_reconstruction_options():
    tree = _parse("arguments/__init__.py")
    opt_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OptimizationParams"
    )
    assigned_names = {
        target.attr
        for node in ast.walk(opt_class)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }

    assert "selective_recon_weight" in assigned_names
    assert "selective_recon_min_weight" in assigned_names
    assert "selective_recon_gamma" in assigned_names
    assert "selective_recon_top_ratio" in assigned_names


def test_training_logs_selective_reconstruction_statistics():
    train_source = Path("train.py").read_text()

    assert "train_loss_patches/selective_recon_loss" in train_source
    assert "train_loss_patches/selective_recon_selected_fraction" in train_source
