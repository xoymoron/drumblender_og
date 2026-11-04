import torch

from drumblender.baseline import VariableLengthDrumBlender


def test_only_audio_and_modal_features_move_to_device():
    model = VariableLengthDrumBlender(modal_synth=torch.nn.Identity(), loss_fn=torch.nn.Identity())
    audio = torch.zeros(2, 1, 10)
    params = torch.zeros(2, 3, 4, 3)
    lengths = torch.tensor([8, 10])
    frames = torch.tensor([2, 3])
    result = model.transfer_batch_to_device((audio, params, lengths, frames), torch.device('meta'), 0)
    assert result[0].device.type == result[1].device.type == 'meta'
    assert result[2].device.type == result[3].device.type == 'cpu'
    assert torch.equal(result[2], lengths) and torch.equal(result[3], frames)
