from types import SimpleNamespace

from drumblender.callbacks import SaveConfigCallbackWanb


def test_wandb_config_secondary_rank_does_not_touch_run_directory(mocker, tmp_path):
    class FakeWandbLogger:
        @property
        def experiment(self):
            raise AssertionError("Only rank zero can access the live run")

    mocker.patch("drumblender.callbacks.WandbLogger", FakeWandbLogger)
    parent = mocker.patch("drumblender.callbacks.SaveConfigCallback.setup")
    callback = SaveConfigCallbackWanb.__new__(SaveConfigCallbackWanb)
    callback.already_saved = False
    trainer = SimpleNamespace(
        logger=FakeWandbLogger(), is_global_zero=False, log_dir=str(tmp_path)
    )
    callback.setup(trainer, None, "fit")
    parent.assert_called_once()


def test_wandb_config_setup_can_be_called_again(mocker, tmp_path):
    class FakeWandbLogger:
        experiment = SimpleNamespace(dir=str(tmp_path / "run"))

    mocker.patch("drumblender.callbacks.WandbLogger", FakeWandbLogger)
    callback = SaveConfigCallbackWanb.__new__(SaveConfigCallbackWanb)
    callback.already_saved = False

    def save(*args):
        (tmp_path / "config.yaml").write_text("seed: 1")
        callback.already_saved = True

    parent = mocker.patch(
        "drumblender.callbacks.SaveConfigCallback.setup", side_effect=save
    )
    trainer = SimpleNamespace(
        logger=FakeWandbLogger(), is_global_zero=True, log_dir=str(tmp_path)
    )
    callback.setup(trainer, None, "fit")
    callback.setup(trainer, None, "validate")
    parent.assert_called_once()
    assert (tmp_path / "run" / "model-config.yaml").read_text() == "seed: 1"
