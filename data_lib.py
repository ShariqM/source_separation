"""Transform between waveform space, spectrogram, and NN input."""

import numpy as np
import pdb
import scipy.signal as signal

import constants


def stft(x):
  """Compute the STFT."""
  return signal.stft(x, constants.Fs, nperseg=constants.nperseg,
                        noverlap=constants.noverlap)[2]

def istft(X):
  """Compute the iSTFT."""
  return signal.istft(X, constants.Fs, nperseg=constants.nperseg,
                      noverlap=constants.noverlap)[1]

def apply_to_magnitude(X):
  X = np.log(X + 1e-6)
  return X

def unapply_to_magnitude(X):
  return np.exp(X) - 1e-6

def wav_to_nn_representation(wav):
  """Convert the waveform into the representation for the NN."""
  XP = stft(wav)
  X, P = np.float32(np.abs(XP)), np.angle(XP)
  X = apply_to_magnitude(X)
  return X.T, P.T

def mag_phase_to_complex(X, phases):
  return X * np.exp(phases * 1j)

def nn_representation_to_wav_spect(X, phases):
  """Given the NN input, return the spectrogram and waveform representation."""
  X = unapply_to_magnitude(np.transpose(X, axes=[0, 2, 1]))
  P = np.transpose(phases, axes=[0, 2, 1])
  XP = mag_phase_to_complex(X, P)

  waveforms = []
  batch_size = X.shape[0]
  for b in range(batch_size):
    waveforms.append(istft(XP[b, :, :]))

  return np.array(waveforms, dtype=np.float32), X
