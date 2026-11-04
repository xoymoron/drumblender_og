"""Timing and track-pairing fixes for the upstream CQT modal analyzer."""

import numpy as np
import torch

from drumblender.utils.modal_analysis import CQTModalAnalysis, peak_interpolation


class BaselineModalAnalysis(CQTModalAnalysis):
    """Keep every analysis frame and one independent track per detected peak."""

    def modal_tracking(self, spec, threshold=-80.0):
        if spec.ndim != 3 or spec.shape[-1] != 2 or spec.shape[0] != self.n_bins:
            raise ValueError("Expected complex CQT [bins, frames, 2]")
        frequencies, amplitudes, phases = [], [], []
        for frame_index, frame in enumerate(np.flip(spec, axis=1).transpose(1, 0, 2)):
            magnitude = np.sqrt(np.sum(frame**2, axis=-1))
            phase = np.arctan2(frame[:, 1], frame[:, 0])
            db = 20 * np.log10(magnitude + 1e-8)
            peaks = (
                np.flatnonzero(
                    (db[1:-1] > threshold) & (db[1:-1] > db[:-2]) & (db[1:-1] > db[2:])
                )
                + 1
            )
            locations, levels, angles = peak_interpolation(magnitude, phase, peaks)
            for track in range(len(frequencies)):
                previous = frequencies[track][-1]
                differences = np.abs(locations - previous)
                if (
                    len(differences)
                    and differences.min() < previous * self.diff_threshold
                ):
                    closest = int(differences.argmin())
                    frequencies[track].append(float(locations[closest]))
                    amplitudes[track].append(float(levels[closest]))
                    phases[track].append(float(angles[closest]))
                    locations, levels, angles = [
                        np.delete(v, closest) for v in (locations, levels, angles)
                    ]
                else:
                    frequencies[track].append(previous)
                    amplitudes[track].append(0.0)
                    phases[track].append(phases[track][-1])
            for location, level, angle in zip(locations, levels, angles):
                frequencies.append([float(location)] * (frame_index + 1))
                amplitudes.append([0.0] * frame_index + [float(level)])
                phases.append([float(angle)] * (frame_index + 1))
        return frequencies, amplitudes, phases

    def create_modal_tensors(self, freqs, amps, phases, num_hops, min_length):
        valid = [i for i, a in enumerate(amps) if np.count_nonzero(a) >= min_length]
        if not valid:
            return tuple(torch.zeros(0, num_hops) for _ in range(3))
        return tuple(
            torch.tensor([values[i][::-1] for i in valid], dtype=torch.float32)
            for values in (freqs, amps, phases)
        )
