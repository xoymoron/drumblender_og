"""Exact greedy CQT tracking with vectorized bookkeeping."""

import numpy as np
import torch

from drumblender.utils.modal_analysis import CQTModalAnalysis, peak_interpolation


def fast_modal_tracking(spec, threshold=-80.0, diff_threshold=0.05):
    """Preserve original track order, greedy matches, ties and silent frames."""
    bins, frames, components = spec.shape
    if components != 2 or frames < 1:
        raise ValueError("Expected nonempty complex CQT [bins,frames,2]")
    reverse = np.flip(spec, axis=1)
    magnitude = np.sqrt(np.sum(reverse**2, axis=-1))
    phase = np.arctan2(reverse[..., 1], reverse[..., 0])
    db = 20 * np.log10(magnitude + 1e-8)
    capacity = 16
    histories = [np.zeros((capacity, frames), dtype=np.float64) for _ in range(3)]
    previous = np.zeros(capacity, dtype=np.float64)
    previous_phase = np.zeros(capacity, dtype=np.float64)
    count = 0
    for frame in range(frames):
        d = db[:, frame]
        peaks = (
            np.flatnonzero(
                (d[1:-1] > threshold) & (d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])
            )
            + 1
        )
        locations, levels, angles = peak_interpolation(
            magnitude[:, frame], phase[:, frame], peaks
        )
        available = np.ones(len(locations), dtype=bool)
        # Pairwise subtraction has the same precision as the reference scalar loop.
        distances = np.abs(locations[None, :] - previous[:count, None])
        for track in range(count):
            if len(locations):
                closest = int(distances[track].argmin())
                if distances[track, closest] < previous[track] * diff_threshold:
                    previous[track] = float(locations[closest])
                    previous_phase[track] = float(angles[closest])
                    histories[1][track, frame] = float(levels[closest])
                    available[closest] = False
                    distances[:, closest] = np.inf
        for history, values in (
            (histories[0], previous),
            (histories[2], previous_phase),
        ):
            history[:count, frame] = values[:count]
        for peak in np.flatnonzero(available):
            if count == capacity:
                histories = [
                    np.concatenate((h, np.zeros_like(h)), axis=0) for h in histories
                ]
                previous = np.pad(previous, (0, capacity))
                previous_phase = np.pad(previous_phase, (0, capacity))
                capacity *= 2
            previous[count] = float(locations[peak])
            previous_phase[count] = float(angles[peak])
            histories[0][count, : frame + 1] = previous[count]
            histories[1][count, frame] = float(levels[peak])
            histories[2][count, : frame + 1] = previous_phase[count]
            count += 1
    return tuple(h[:count] for h in histories)


class BaselineModalAnalysis(CQTModalAnalysis):
    def modal_tracking(self, spec, threshold=-80.0):
        if spec.ndim != 3 or spec.shape[0] != self.n_bins or spec.shape[-1] != 2:
            raise ValueError("Expected complex CQT [bins,frames,2]")
        return fast_modal_tracking(spec, threshold, self.diff_threshold)

    def create_modal_tensors(self, freqs, amps, phases, num_hops, min_length):
        valid = np.flatnonzero(
            np.count_nonzero(np.asarray(amps), axis=-1) >= min_length
        )
        if not len(valid):
            return tuple(torch.zeros(0, num_hops) for _ in range(3))
        return tuple(
            torch.from_numpy(np.ascontiguousarray(np.asarray(v)[valid, ::-1])).float()
            for v in (freqs, amps, phases)
        )
