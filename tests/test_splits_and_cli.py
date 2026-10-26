import pytest

from elicitor_distill.cli import main
from elicitor_distill.splits import EvalSplitError, guard, load_split


def test_guard_blocks_eval_personas_by_default(data_dir):
    with pytest.raises(EvalSplitError):
        load_split(data_dir, "eval")
    assert len(load_split(data_dir, "eval", allow_eval=True)) == 48


def test_train_and_dev_splits_need_no_flag(data_dir):
    assert len(load_split(data_dir, "train")) == 160
    dev = load_split(data_dir, "dev")
    assert len(dev) == 12 and all(p.split == "dev" for p in dev)


def test_guard_passes_non_eval_personas_through(data_dir):
    personas = load_split(data_dir, "dev")
    assert guard(personas) == personas


def test_unknown_split_is_rejected(data_dir):
    with pytest.raises(ValueError):
        load_split(data_dir, "test")


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("distill 0.")


def test_cli_make_personas_check(data_dir, capsys):
    assert main(["make-personas", "--check", "--data-dir", str(data_dir)]) == 0
    assert "up to date" in capsys.readouterr().out


def test_cli_make_personas_writes_to_a_fresh_folder(tmp_path, data_dir, capsys):
    # The domain copy must sit next to the output; link it in by copying.
    import shutil

    shutil.copytree(data_dir / "travel", tmp_path / "travel")
    assert main(["make-personas", "--data-dir", str(tmp_path), "--n", "8", "--seed", "3"]) == 0
    assert (tmp_path / "personas_train.jsonl").exists()
    assert (
        main(["make-personas", "--check", "--data-dir", str(tmp_path), "--n", "8", "--seed", "3"])
        == 0
    )
    assert (
        main(["make-personas", "--check", "--data-dir", str(tmp_path), "--n", "8", "--seed", "4"])
        == 1
    )
    assert "stale" in capsys.readouterr().out
