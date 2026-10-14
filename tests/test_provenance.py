import json
from pathlib import Path

import pytest

from tsfm.provenance import config_hash, sha256_file, write_result


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_write_result_stamps_meta(tmp_path: Path):
    p = write_result(tmp_path / "r.json", {"v": 1.5}, split="val", label="t")
    doc = json.loads(p.read_text())
    assert doc["v"] == 1.5
    assert doc["_meta"]["split"] == "val"
    assert {"git_sha", "git_dirty", "created_utc", "env"} <= doc["_meta"].keys()


def test_write_result_rejects_unknown_split(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown split"):
        write_result(tmp_path / "r.json", {}, split="train", label="t")
