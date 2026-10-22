"""The render geometry must come from the checkpoint that will answer the items.

`infer` refuses a render/checkpoint mismatch -- "scoring would ask a different question from the one
trained on" -- and the two laya checkpoints in play genuinely differ: the fine-tuned pilot answers at
8192/2048, the shipped `laya-base` at 1024/256. The `items` step used to take its width from the CLI
defaults, so rendering for one model and scoring with the other failed *after* burning the render.
"""

import json
from pathlib import Path

import pytest

from sckevtype.laya.score import ScoreError, checkpoint_render_config

WIDE = {"max_len": 8192, "head_max_len": 2048}
SHIPPED = {"max_len": 1024, "head_max_len": 256}


def _ckpt(tmp_path: Path, cfg: dict) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return root


def test_the_geometry_is_taken_from_the_checkpoint(tmp_path: Path) -> None:
    cfg = checkpoint_render_config(str(_ckpt(tmp_path, SHIPPED)))
    assert (cfg.max_len, cfg.head_max_len) == (1024, 256)


def test_the_finetuned_geometry_is_read_the_same_way(tmp_path: Path) -> None:
    cfg = checkpoint_render_config(str(_ckpt(tmp_path, WIDE)))
    assert (cfg.max_len, cfg.head_max_len) == (8192, 2048)


def test_a_directory_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-checkpoint"
    empty.mkdir()
    with pytest.raises(ScoreError, match="rl_agent_config.json"):
        checkpoint_render_config(str(empty))
