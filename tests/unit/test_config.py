"""配置管理单元测试。"""

import os
import tempfile

import yaml
import pytest

from src.config.config_manager import (
    ConfigManager,
    deep_merge,
    get_nested,
    parse_overrides,
    set_nested,
    _auto_cast,
)
from src.config.schema import (
    CEAISConfig,
    EncoderConfig,
    CEWMConfig,
    SteeringConfig,
    BilateralGatingConfig,
    TrainingConfig,
    EvaluationConfig,
    LoggingConfig,
    ProjectConfig,
)


# ── deep_merge ──────────────────────────────────────────────


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}, "y": 10}
        override = {"x": {"b": 99, "c": 3}}
        result = deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}, "y": 10}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        override = {"x": {"b": 2}}
        deep_merge(base, override)
        assert base == {"x": {"a": 1}}
        assert override == {"x": {"b": 2}}

    def test_override_dict_with_scalar(self):
        base = {"x": {"a": 1}}
        override = {"x": 42}
        result = deep_merge(base, override)
        assert result == {"x": 42}

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}


# ── set_nested / get_nested ─────────────────────────────────


class TestNestedAccess:
    def test_set_and_get(self):
        cfg = {}
        set_nested(cfg, "a.b.c", 42)
        assert get_nested(cfg, "a.b.c") == 42

    def test_get_default(self):
        assert get_nested({}, "missing.key", "default") == "default"

    def test_set_overwrites_existing(self):
        cfg = {"a": {"b": 1}}
        set_nested(cfg, "a.b", 2)
        assert cfg["a"]["b"] == 2

    def test_set_creates_intermediate_dicts(self):
        cfg = {}
        set_nested(cfg, "x.y.z", "val")
        assert cfg == {"x": {"y": {"z": "val"}}}


# ── _auto_cast ──────────────────────────────────────────────


class TestAutoCast:
    def test_int(self):
        assert _auto_cast("42") == 42

    def test_float(self):
        assert _auto_cast("3.14") == 3.14

    def test_bool_true(self):
        assert _auto_cast("true") is True
        assert _auto_cast("True") is True

    def test_bool_false(self):
        assert _auto_cast("false") is False

    def test_string(self):
        assert _auto_cast("hello") == "hello"


# ── parse_overrides ─────────────────────────────────────────


class TestParseOverrides:
    def test_basic(self):
        args = ["--ce_wm.d_model=256", "--training.batch_size=32"]
        result = parse_overrides(args)
        assert result == {"ce_wm.d_model": 256, "training.batch_size": 32}

    def test_without_dashes(self):
        args = ["ce_wm.d_model=256"]
        result = parse_overrides(args)
        assert result == {"ce_wm.d_model": 256}

    def test_bool_override(self):
        args = ["--training.amp=false"]
        result = parse_overrides(args)
        assert result == {"training.amp": False}

    def test_skips_invalid(self):
        args = ["--no-equals", "--valid.key=1"]
        result = parse_overrides(args)
        assert result == {"valid.key": 1}


# ── ConfigManager with YAML files ──────────────────────────


class TestConfigManager:
    def _write_yaml(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    def test_load_simple_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write_yaml(cfg_file, {"project": {"name": "test", "seed": 123}})
        cm = ConfigManager(str(cfg_file))
        assert cm.get("project.name") == "test"
        assert cm.get("project.seed") == 123

    def test_inherit_from(self, tmp_path):
        base_file = tmp_path / "base.yaml"
        child_file = tmp_path / "child.yaml"
        self._write_yaml(base_file, {
            "project": {"name": "base", "seed": 42},
            "encoder": {"latent_dim": 128},
        })
        self._write_yaml(child_file, {
            "inherit_from": str(base_file),
            "project": {"name": "child"},
            "encoder": {"latent_dim": 256},
        })
        cm = ConfigManager(str(child_file))
        # child overrides
        assert cm.get("project.name") == "child"
        assert cm.get("encoder.latent_dim") == 256
        # base preserved
        assert cm.get("project.seed") == 42

    def test_inherit_from_relative_path(self, tmp_path):
        base_file = tmp_path / "base.yaml"
        child_file = tmp_path / "child.yaml"
        self._write_yaml(base_file, {"a": 1, "b": 2})
        self._write_yaml(child_file, {
            "inherit_from": "base.yaml",
            "b": 99,
        })
        cm = ConfigManager(str(child_file))
        assert cm.get("a") == 1
        assert cm.get("b") == 99

    def test_chained_inheritance(self, tmp_path):
        grandparent = tmp_path / "gp.yaml"
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(grandparent, {"x": 1, "y": 2, "z": 3})
        self._write_yaml(parent, {"inherit_from": "gp.yaml", "y": 20})
        self._write_yaml(child, {"inherit_from": "parent.yaml", "z": 300})
        cm = ConfigManager(str(child))
        assert cm.get("x") == 1
        assert cm.get("y") == 20
        assert cm.get("z") == 300

    def test_overrides(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write_yaml(cfg_file, {"training": {"batch_size": 64}})
        cm = ConfigManager(str(cfg_file), overrides={"training.batch_size": 32})
        assert cm.get("training.batch_size") == 32

    def test_save_snapshot(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write_yaml(cfg_file, {"project": {"name": "snap"}})
        cm = ConfigManager(str(cfg_file))
        log_dir = str(tmp_path / "logs")
        snap_path = cm.save_snapshot(log_dir)
        assert os.path.exists(snap_path)
        with open(snap_path, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["project"]["name"] == "snap"

    def test_save_snapshot_custom_filename(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write_yaml(cfg_file, {"a": 1})
        cm = ConfigManager(str(cfg_file))
        log_dir = str(tmp_path / "logs")
        snap_path = cm.save_snapshot(log_dir, filename="my_snapshot.yaml")
        assert snap_path.endswith("my_snapshot.yaml")
        assert os.path.exists(snap_path)

    def test_to_dict_returns_copy(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        self._write_yaml(cfg_file, {"a": {"b": 1}})
        cm = ConfigManager(str(cfg_file))
        d = cm.to_dict()
        d["a"]["b"] = 999
        assert cm.get("a.b") == 1  # original unchanged

    def test_no_config_path(self):
        cm = ConfigManager()
        assert cm.config == {}

    def test_load_base_yaml(self):
        """Test loading the actual project base.yaml."""
        base_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "base.yaml"
        )
        if os.path.exists(base_path):
            cm = ConfigManager(base_path)
            assert cm.get("project.name") == "ce-ais"
            assert cm.get("encoder.latent_dim") == 128
            assert cm.get("ce_wm.n_layers") == 32

    def test_inherit_from_is_removed_from_config(self, tmp_path):
        base_file = tmp_path / "base.yaml"
        child_file = tmp_path / "child.yaml"
        self._write_yaml(base_file, {"a": 1})
        self._write_yaml(child_file, {"inherit_from": "base.yaml", "b": 2})
        cm = ConfigManager(str(child_file))
        assert "inherit_from" not in cm.config


# ── Schema dataclass tests ──────────────────────────────────


class TestSchemaDataclasses:
    def test_default_ceais_config(self):
        cfg = CEAISConfig()
        assert cfg.project.name == "ce-ais"
        assert cfg.project.seed == 42
        assert cfg.encoder.backbone_type == "resnet18"
        assert cfg.ce_wm.d_model == 640
        assert cfg.steering.n_steps == 5
        assert cfg.bilateral_gating.lambda_max == 1.0
        assert cfg.training.batch_size == 64
        assert cfg.evaluation.protocol == "ABC_to_D"
        assert cfg.logging.tensorboard is True

    def test_encoder_config_custom(self):
        cfg = EncoderConfig(backbone_type="vit_small", latent_dim=256)
        assert cfg.backbone_type == "vit_small"
        assert cfg.latent_dim == 256
        assert cfg.pose_dim == 7  # default preserved

    def test_evaluation_nested_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.ood_perturbations.physics.mass_scale == [0.5, 2.0]
        assert cfg.ood_perturbations.visual.brightness_jitter == 0.5
