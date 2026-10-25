from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pad_config.yaml"


def test_config_has_required_sections():
    cfg = yaml.safe_load(CONFIG.read_text())
    for key in ("config_version", "model", "decision", "preprocessing", "logging"):
        assert key in cfg


def test_privacy_default_is_no_face_images():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["logging"]["store_face_images"] is False


def test_threshold_in_valid_range():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert 0.0 < cfg["decision"]["threshold"] < 1.0
