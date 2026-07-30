import pytest
import torch
from unittest.mock import MagicMock, patch
from train.scheduler import Trainer


@patch("train.scheduler.ConfigLoader")
@patch("train.scheduler.ConvNextLoRAModel")
@patch("train.scheduler.WeatherFetcher")
def test_trainer_initialization(mock_weather, mock_model_cls, mock_config):
    """Verify that the trainer initializes correctly."""
    mock_config.return_value.metar_station = "KSEA"
    mock_config.return_value.lora_settings = {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["fc1"],
    }
    mock_config.return_value.checkpoint_dir = "checkpoints"

    mock_model = MagicMock()
    mock_model.model_dict.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    mock_model_cls.return_value = mock_model

    Trainer("mountain.toml")

    assert mock_model_cls.called
    assert mock_weather.called


@patch("train.scheduler.WebcamStream")
@patch("train.scheduler.WeatherFetcher")
def test_run_single_cycle_execution(mock_weather_cls, mock_webcam):
    """Verify that run_single_cycle captures from single source and performs a training step."""
    with patch("train.scheduler.ConfigLoader") as mock_config:
        mock_config.return_value.webcam_url = "http://cam.jpg"
        mock_config.return_value.metar_station = "KSEA"
        mock_config.return_value.lora_settings = {
            "rank": 8,
            "alpha": 16,
            "target_modules": ["fc1"],
        }
        mock_config.return_value.checkpoint_dir = "checkpoints"

        with patch("train.scheduler.ConvNextLoRAModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.model_dict.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1))
            ]
            mock_model.train_step.return_value = 0.5
            mock_model_cls.return_value = mock_model
            mock_weather = MagicMock()
            mock_weather_vector = torch.tensor([0.8, 0.9])
            mock_weather.get_weather_vector.return_value = mock_weather_vector
            mock_weather_cls.return_value = mock_weather

            trainer = Trainer("mountain.toml")
            trainer.optimizer = MagicMock()

            mock_stream = MagicMock()
            mock_tensor = torch.randn(1, 3, 224, 224)
            mock_stream.capture_to_tensor.return_value = mock_tensor
            mock_webcam.return_value = mock_stream

            trainer.run_single_cycle(label=1)

            assert mock_stream.capture_to_tensor.call_count == 1
            args, _ = mock_model.train_step.call_args
            image_batch, weather_batch, label_batch, _ = args
            assert image_batch.shape == (1, 3, 224, 224)
            assert weather_batch.shape == (1, 2)
            assert label_batch.shape == (1,)


@patch("train.scheduler.WebcamStream")
@patch("train.scheduler.WeatherFetcher")
@patch("time.sleep", side_effect=InterruptedError)
def test_live_training_loop_cycle(mock_sleep, mock_weather_cls, mock_webcam):
    """Verify that live_training_loop captures and performs training."""
    with patch("train.scheduler.ConfigLoader") as mock_config:
        mock_config.return_value.webcam_url = "http://cam.jpg"
        mock_config.return_value.metar_station = "KSEA"
        mock_config.return_value.lora_settings = {
            "rank": 8,
            "alpha": 16,
            "target_modules": ["fc1"],
        }
        mock_config.return_value.capture_interval_seconds = 0
        mock_config.return_value.gradient_accumulation_steps = 1
        mock_config.return_value.checkpoint_dir = "checkpoints"

        with patch("train.scheduler.ConvNextLoRAModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.model_dict.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1))
            ]
            mock_model.train_step.return_value = 0.5
            mock_model_cls.return_value = mock_model
            mock_weather = MagicMock()
            mock_weather_vector = torch.tensor([0.8, 0.9])
            mock_weather.get_weather_vector.return_value = mock_weather_vector
            mock_weather_cls.return_value = mock_weather

            trainer = Trainer("mountain.toml")
            trainer.optimizer = MagicMock()

            mock_stream = MagicMock()
            mock_tensor = torch.randn(1, 3, 224, 224)
            mock_stream.capture_to_tensor.return_value = mock_tensor
            mock_webcam.return_value = mock_stream

            with pytest.raises(InterruptedError):
                trainer.live_training_loop(label=1)

            assert mock_stream.capture_to_tensor.call_count == 1


class TestStratifiedSplit:
    """The split is the reason the old val numbers were not trustworthy.

    It was already stratified — but it ran AFTER oversampling, so the same
    minority frame landed in both train and val ~8 times over, and it was
    unseeded, so no two runs' val metrics described the same val set.
    """

    # The measured real balance (2010 R2 labels, 2026-07-30).
    REAL = {
        0: [f"n{i}" for i in range(1735)],
        1: [f"f{i}" for i in range(111)],
        2: [f"p{i}" for i in range(164)],
    }

    def _split(self, by_class=None, seed=1337, fraction=0.15):
        import random

        from train.scheduler import stratified_split

        return stratified_split(
            by_class if by_class is not None else self.REAL,
            val_fraction=fraction,
            rng=random.Random(seed),
        )

    def test_no_item_is_in_both_sides(self):
        """The leak test. This is the whole point."""
        train, val = self._split()
        for cls in train:
            assert not (set(train[cls]) & set(val[cls]))

    def test_every_class_is_represented_in_val(self):
        _, val = self._split()
        assert len(val[0]) == 260
        assert len(val[1]) == 17  # Full — the class that motivated stratifying
        assert len(val[2]) == 25

    def test_split_is_deterministic_for_a_given_seed(self):
        assert self._split(seed=1337) == self._split(seed=1337)

    def test_a_different_seed_draws_a_different_val_set(self):
        _, a = self._split(seed=1337)
        _, b = self._split(seed=7)
        assert a[1] != b[1]

    def test_split_ignores_input_ordering(self):
        shuffled = {cls: list(reversed(items)) for cls, items in self.REAL.items()}
        assert self._split(shuffled) == self._split()

    def test_nothing_is_lost_or_duplicated(self):
        train, val = self._split()
        for cls, items in self.REAL.items():
            assert sorted(train[cls] + val[cls]) == sorted(items)

    def test_a_singleton_class_stays_in_train(self):
        # Spending the only example on val makes the class untrainable AND its
        # recall a coin flip — worst of both.
        train, val = self._split({0: ["a", "b", "c", "d"], 1: ["only"]})
        assert train[1] == ["only"]
        assert val[1] == []

    def test_an_empty_class_is_survivable(self):
        train, val = self._split({0: ["a", "b"], 1: []})
        assert train[1] == [] and val[1] == []

    def test_a_two_item_class_gives_one_to_each_side(self):
        train, val = self._split({1: ["a", "b"]})
        assert len(train[1]) == 1 and len(val[1]) == 1
