"""Verify that OG validation reports the joint model's three tail energies."""

import unittest
from unittest.mock import Mock

import torch
from torch import nn

from drumblender.baseline import VariableLengthDrumBlender, collate_baseline


class Modal(nn.Module):
    def forward(self, params, samples):
        return params.new_ones((len(params), 1, samples))


class Noise(nn.Module):
    def forward(self, params, samples):
        return params.new_full((len(params), samples), 2)


class Transient(nn.Module):
    def forward(self, audio, params):
        return torch.full_like(audio, 3)


def loss(prediction, target, lengths):
    return (prediction - target).square().mean()


def clip(length, amplitude, key):
    return torch.full((1, length), amplitude), torch.zeros(3, 1, 1), key


class BranchEnergyTest(unittest.TestCase):
    def test_validation_logs_normalized_energy_over_valid_tails_only(self):
        model = VariableLengthDrumBlender(
            modal_synth=Modal(),
            loss_fn=loss,
            noise_autoencoder=nn.Identity(),
            noise_autoencoder_accepts_audio=True,
            noise_synth=Noise(),
            transient_synth=Transient(),
            transient_parallel=True,
        ).eval()
        model.log = Mock()
        model.log_dict = Mock()
        model.all_gather = Mock(
            side_effect=lambda local: torch.stack((local, local.new_tensor([12, 48, 108, 1])))
        )
        batch = collate_baseline(
            [clip(5000, 2.0, "short"), clip(6400, 4.0, "long"), clip(4096, 1.0, "no_tail")]
        )

        model.on_validation_epoch_start()
        validation_loss = model.validation_step(batch, 0)
        regular_loss, prediction = model._do_step(batch)
        torch.testing.assert_close(validation_loss, regular_loss)
        torch.testing.assert_close(prediction[0, 0, :5000], torch.full((5000,), 6.0))
        model.on_validation_epoch_end()

        model.log_dict.assert_called_once()
        scores = model.log_dict.call_args.args[0]
        self.assertEqual(
            set(scores),
            {"val_tail_modal_energy", "val_tail_noise_energy", "val_tail_transient_energy"},
        )
        # Local sums are 200/2^2 + 1600/4^2, scaled by each branch's squared amplitude.
        for branch, value in zip(("modal", "noise", "transient"), (54.0, 216.0, 486.0)):
            torch.testing.assert_close(scores[f"val_tail_{branch}_energy"], torch.tensor(value))
        model.all_gather.assert_called_once()


if __name__ == "__main__":
    unittest.main()
