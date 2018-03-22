"""Create batches of data for the network to train on."""

import tensorflow as tf
import numpy as np
import glob
import pdb
from scipy.io.wavfile import read
from functools import partial

import rtl_loader
import data_lib
import constants


def make_X_and_mask(hparams, wavs):
  X_sources = np.zeros([hparams.ntimebins, constants.nfreqbins, len(wavs)])
  for (i, wav) in enumerate(wavs):
    X_sources[:, :, i], _ = data_lib.wav_to_nn_representation(wav)

  max_idx = np.argmax(X_sources, axis=2)
  mask = (max_idx[...,None] == np.arange(hparams.num_targets)).astype(int)

  return X_sources, mask

####################
### MAKE MIXTURE ###
####################
def snr_to_weight(snr):
  return 10 ** (snr / 20)

def prepare_wavs(hparams, wav_i, wav_j):
  """Read the wav file, truncate at COLA length, and reweight them."""
  snr = constants.max_input_snr * np.random.random()
  wav_i = snr_to_weight(snr) * wav_i
  wav_j = snr_to_weight(-snr) * wav_j
  return [wav_i, wav_j]

def simulate_mixture(hparams, wav_i, wav_j):
  wavs = prepare_wavs(hparams, wav_i, wav_j)

  mixture = wavs[0] + wavs[1]
  if hparams.add_white_noise:
    l = constants.Fs / 4
    start = int(len(mixture) / 2)
    mixture[start: start + l] += 0.5 * np.max(np.abs(mixture)) * np.random.randn(l)

  X_mixture, phase = data_lib.wav_to_nn_representation(mixture)
  return X_mixture, phase, wavs


######################
### PICK WAV FILES ###
######################
def get_wav(hparams, files):
  """Return a waveform that is long enough using files."""
  wav = np.array([])
  while (len(wav) <= hparams.waveform_size):  # [<=] so randint() (below) works
    wav_file = np.random.choice(files)
    Fs, x = read(wav_file)
    assert Fs == constants.Fs
    wav = np.concatenate((wav, x))

  start = np.random.randint(len(wav) - hparams.waveform_size)
  return wav[start : start + hparams.waveform_size]

def pick_read_wavs(hparams, spk_folders):
  spk_folder_i = np.random.choice(spk_folders)
  spk_folder_j = np.random.choice(spk_folders)

  while spk_folder_i == spk_folder_j:
    spk_folder_j = np.random.choice(spk_folders)

  wav_ext = "wv1" if "WSJ0" in hparams.data_source else "wav"
  wav_i = get_wav(hparams, glob.glob(spk_folder_i + "/*/*.%s" % wav_ext))
  wav_j = get_wav(hparams, glob.glob(spk_folder_j + "/*/*.%s" % wav_ext))
  return wav_i, wav_j


#########################
### DATA CONSTRUCTION ###
#########################
def mp_worker(spk_folders, hparams):
  np.random.seed()  # b/c Multiprocessing units get same seed
  wav_i, wav_j = pick_read_wavs(hparams, spk_folders)

  X_mixture, phase, wavs = simulate_mixture(hparams, wav_i, wav_j)
  X_sources, mask = make_X_and_mask(hparams, wavs)

  sources = np.zeros((hparams.num_targets, hparams.waveform_size))
  sources[0, :] = wavs[0]
  sources[1, :] = wavs[1]

  return X_mixture, phase, mask, sources, X_sources

def get_inference_data(hparams):
  files = glob.glob("data/ours/custom/*2*.wav")
  X_mixtures, phases = [], []
  for i in range(hparams.batch_size):
    mixture = get_wav(hparams, files)
    X_mixture, phase = data_lib.wav_to_nn_representation(mixture)
    X_mixtures.append(X_mixture)
    phases.append(phase)

  return np.stack(X_mixtures), np.stack(phases)

def make_data(pool, hparams, test=False):
  if hparams.data_source == "RTL":
    return rtl_loader.make_data(pool, hparams, test)

  if hparams.data_source == "WSJ0":
    use_dir = "test" if test else "train"
    spk_path = 'data/wsj0/%s/*' % use_dir
    spk_folders = glob.glob(spk_path)
  elif hparams.data_source == "LIBRI":
    use_dir = "test-clean" if test else "train-clean-100"
    spk_path = 'data/LibriSpeech/%s/*' % use_dir
    spk_folders = glob.glob(spk_path)
  else:
    raise Exception("Invalid data source: %s", hparams.data_source)

  r = pool.map_async(partial(mp_worker, hparams=hparams), [spk_folders] * hparams.batch_size)
  results = r.get()

  stacked_results = []
  for result in zip(*results):
    stacked_results.append(np.stack(result))

  return stacked_results
