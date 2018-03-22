"""Create batches of RTL data for the network to train on."""

import numpy as np
import glob
import pdb
from scipy.io.wavfile import read
from functools import partial

import data_lib
import loader
import constants


####################
### MAKE MIXTURE ###
####################
def read_real_wav(wav_file):
  Fs, x = read(wav_file)
  assert (Fs == constants.Fs), "FS was: %d | %s" % (Fs, wav_file)
  return x

def pick_mixed_wavs(hparams, folder):
  mixed_files = glob.glob(folder + "/*mixed*")
  wav_file_mixed = np.random.choice(mixed_files)
  dirs = wav_file_mixed.split('/')
  ID = "_".join(dirs[-1].split('_')[:2])  # 2_701

  files = glob.glob(folder + "/%s_*" % ID)
  files = [f for f in files if "mixed" not in f]
  assert len(files) == 2, (ID, files)

  wav_file_i = files[0]
  wav_file_j = files[1]

  wav_i = read_real_wav(wav_file_i)
  wav_j = read_real_wav(wav_file_j)
  mixture = read_real_wav(wav_file_mixed)
  assert (len(wav_i) == len(wav_j) == len(mixture))

  start = np.random.randint(len(wav_i) - hparams.waveform_size)
  wav_i = wav_i[start: start + hparams.waveform_size]
  wav_j = wav_j[start: start + hparams.waveform_size]
  mixture = mixture[start: start + hparams.waveform_size]

  return wav_i, wav_j, mixture


#########################
### DATA CONSTRUCTION ###
#########################
def mp_worker(folder, hparams):
  np.random.seed()  # b/c Multiprocessing units get same seed
  wav_i, wav_j, mixture = pick_mixed_wavs(hparams, folder)

  X_mixture, phase = data_lib.wav_to_nn_representation(mixture)

  wavs = [wav_i, wav_j]
  X_sources, mask = loader.make_X_and_mask(hparams, wavs)

  sources = np.zeros((hparams.num_targets, hparams.waveform_size))
  sources[0, :] = wavs[0]
  sources[1, :] = wavs[1]

  return X_mixture, phase, mask, sources, X_sources

def make_data(pool, hparams, test=False):
  if test:
    poss_folders = []
    for i in (7, 16):
      poss_folders.append("data/RTL/extracted_%d/libri/*" % i)
  else:  # train
    poss_folders = []
    for i in (2, 10, 12, 14, 15, 17, 18, 19, 20):
      poss_folders.append("data/RTL/extracted_%d/libri/*" % i)

  folders = np.random.choice(poss_folders, hparams.batch_size)
  r = pool.map_async(partial(mp_worker, hparams=hparams), folders)
  results = r.get()

  stacked_results = []
  for result in zip(*results):
    stacked_results.append(np.stack(result))

  return stacked_results
