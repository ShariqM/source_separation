"""Code for setting up scalar, audio, and image summaries."""

import tensorflow as tf
import numpy as np
import pdb
from scipy.io import loadmat
import itertools
import string

import helper
import kmeans
import bss_eval
import mir_bss_eval
import constants
import data_lib

ORACLE_NAME="Oracle"
ATTRACTOR_NAME="Attractor"
KMEANS_NAME="K-Means"
SDR_PROXY_NAME="SDR_proxy.dB"

###########################
### SUMMARIES & HELPERS ###
###########################
def itos(i):
  """Integer to string."""
  letter_count = dict(zip(range(26), string.ascii_uppercase))
  return letter_count[i]

def scale_color_impl(x):
  """Nonlinear color scaling so colors are more saturated."""
  return (1 / np.log(2)) * tf.log(1 + x)

def scale_color(x):
  """Apply twice for more saturation."""
  return scale_color_impl(scale_color_impl(x))

def make_image_summary(X_source_estimates, mask):
  # Use mask to calculate red and blue channels respectively
  hue = mask[:, :, :, 0] * 1 + mask[:, :, :, 1] * (2/3)
  hue = tf.transpose(hue, perm=[0, 2, 1])
  hue = tf.expand_dims(hue, -1)

  # Saturation is determined by power, make it look good
  saturation = sum(X_source_estimates)
  max_for_batch = tf.reduce_max(saturation, axis=[1, 2], keepdims=True)
  saturation = saturation / max_for_batch  # Normalize to [0,1]
  saturation = scale_color(saturation)  # Make colors more saturated
  saturation /= 0.95
  saturation += 0.05
  saturation = tf.expand_dims(saturation, -1)

  value = 0.3 * tf.ones(hue.shape)

  hsv_image = tf.squeeze(tf.stack([hue, saturation, value], axis=3), axis=[4])
  hsv_image = tf.reverse(hsv_image, axis=[1])
  image = tf.image.hsv_to_rgb(hsv_image)
  return tf.summary.image("Mixture-Color", image)

def make_audio_summary(name, signal):
  normalizer = tf.reduce_max(tf.abs(signal), axis=1, keepdims=True)
  return tf.summary.audio(name, signal / normalizer, constants.Fs)


######################################
### CALCULATIONS - ESTIMATES & SDR ###
######################################
def get_source_estimates(hparams, mask, tmp_X_source_estimates, phases):
  """Take the X_source_estimates (spectrogram) and obtain the raw waveform
     representation as well as the original spectorgram (not log magnitude)."""
  X_source_estimates, source_estimates = [], []

  for i in range(hparams.num_targets):
    source_estimate, X_source_estimate = (
      tf.py_func(data_lib.nn_representation_to_wav_spect,
        [tmp_X_source_estimates[:, :, :, i], phases], [tf.float32, tf.float32]))

    source_estimate.set_shape((hparams.batch_size, hparams.waveform_size))
    X_source_estimate.set_shape((hparams.batch_size, constants.nfreqbins, hparams.ntimebins))

    X_source_estimates.append(X_source_estimate)
    source_estimates.append(source_estimate)

  return X_source_estimates, source_estimates

def compute_source_estimates(hparams, X_mixtures, mask, phases):
  X_source_estimates = tf.expand_dims(X_mixtures, -1) * mask
  return get_source_estimates(hparams, mask, X_source_estimates, phases)

def compute_results(hparams, X_mixtures, mask, sources, phases):
  """Take the mask and apply it to the input mixture to obtain quantifiable
     results. Since K-means may permute the results we permute to the
     correct one."""

  X_source_estimates, source_estimates = (
    compute_source_estimates(hparams, X_mixtures, mask, phases))

  stacked_source_estimates = tf.stack(source_estimates, axis=1)
  proxy_SDR = bss_eval.eval_proxy_SDR(sources, stacked_source_estimates)

  return X_source_estimates, source_estimates, proxy_SDR

def compute_SDR_impl(savedir):
  savedir = savedir.decode("utf-8")  # bytes to string
  filename = 'waveforms/%s/example_%d.mat'
  SDRs = []

  print ("\t-- EVALUATING SDR --")
  n = constants.AVG_SDR_ON_N_BATCHES
  for i in range(n):
    if i == n-1 or (i and i % 20 == 0):
      print ("\tExample %d/%d" % (i, n))
    try:
      data = loadmat(filename % (savedir, i))
      sources = data['sources']
      source_estimates = np.squeeze(data['source_estimates'])

      for b in range(sources.shape[0]):
        (sdr, sir, sar, perm) = (
          mir_bss_eval.bss_eval_sources(sources[b, :, :], source_estimates[b, :, :]))
        SDRs.append(np.mean(sdr))
    except Exception as e:
      print ("\t(%d) Exception in SDR calculation:" % i, e)

  SDRs = np.array(SDRs, dtype=np.float32)
  return np.mean(SDRs) if len(SDRs) > 0 else 0.0

def compute_SDR(hparams):
  if not hparams.save_estimate_waveforms:
    return tf.constant(0)
  name = helper.get_kmeans_waveform_savedir(hparams)
  return tf.py_func(compute_SDR_impl, [tf.constant(name)], tf.float32)

###############
### MASKING ###
###############
def get_mask(hparams, embeddings, centers):
  mask_before_squash = tf.einsum("bik,bck->bic", embeddings, centers)
  argmax = tf.argmax(mask_before_squash, axis=2)
  mask = tf.one_hot(argmax, depth=hparams.num_targets, axis=2)
  return tf.reshape(mask, [hparams.batch_size, hparams.ntimebins,
                           constants.nfreqbins, hparams.num_targets])

def get_attractor_mask(hparams, embeddings, attractors):
  return get_mask(hparams, embeddings, attractors)

def permute_mask(X_mixtures, X_sources, mask):
  """Permute the order of the masks so it matches the sources best.
     We need to do this because k-means can permute the clusters."""
  X_source_estimates = np.expand_dims(X_mixtures, axis=X_mixtures.ndim) * mask
  batch_size, _, __, num_targets = X_source_estimates.shape
  mask_permuted = np.zeros_like(mask, dtype=np.float32)

  permutations = list(itertools.permutations(range(num_targets)))
  for b in range(batch_size):
    permutation_errors = []

    # Calculate the error for all possible permutations
    for permutation in permutations:
      reconstruction_error = 0
      for i in range(num_targets):
        diff = X_source_estimates[b, :, :, permutation[i]] - X_sources[b, :, :, i]
        reconstruction_error += np.mean(np.square(diff))
      permutation_errors.append(reconstruction_error / num_targets)

    # Pick the permutation with the smallest error and permute
    argmin = np.argmin(permutation_errors)
    best_permutation = permutations[argmin]

    for i in range(num_targets):
      mask_permuted[b, :, :, i] = mask[b, :, :, best_permutation[i]]

  return mask_permuted

def get_kmeans_mask_impl(hparams, X_mixtures, embeddings, threshold_mask):
  centers = kmeans.get_centers(hparams, embeddings, threshold_mask)
  return get_mask(hparams, embeddings, centers)

def get_kmeans_mask(hparams, X_mixtures, X_sources, embeddings, threshold_mask):
  mask = get_kmeans_mask_impl(hparams, X_mixtures, embeddings, threshold_mask)
  return tf.py_func(permute_mask, [X_mixtures, X_sources, mask], tf.float32)

############################
### SETUP (ENTRY POINTS) ###
############################
def setup_eval_result_summary(hparams, X_mixtures, phases, kmeans_mask):
  # Put ops back in their original shape
  X_mixtures = helper.uncollapse_freq_into_time(hparams, X_mixtures)

  mixture, _ = (tf.py_func(data_lib.nn_representation_to_wav_spect,
        [X_mixtures, phases], [tf.float32, tf.float32]))

  X_source_estimates, source_estimates = (
    compute_source_estimates(hparams, X_mixtures, kmeans_mask, phases))

  summaries = []
  # Generate an audio summary for the input (mixture)
  with tf.name_scope(KMEANS_NAME):
    summaries.append(make_audio_summary("Mixture", mixture))
    for (i, source_estimate) in enumerate(source_estimates):
      name = "Speaker_%s" % itos(i)
      summaries.append(make_audio_summary(name, source_estimate))
      summaries.append(make_image_summary(X_source_estimates, kmeans_mask))

  return summaries

def setup_inference_summary(hparams, threshold_mask, X_mixtures, phases, embeddings):
  kmeans_mask = get_kmeans_mask_impl(hparams, X_mixtures, embeddings, threshold_mask)
  return setup_eval_result_summary(hparams, X_mixtures, phases, kmeans_mask)


def create_audio_image_summaries(mask_name, mask, train_test_summ_ops,
                                 mixture, X_source_estimates, source_estimates):
  """Generate summaries for audio and images of the spectrogram."""
  names = ("train-%s" % mask_name, "test-%s" % mask_name)
  for (name, ops) in zip(names, train_test_summ_ops):
    with tf.name_scope(name):
      ops.append(make_audio_summary("Mixture", mixture))
      for (i, source_estimate) in enumerate(source_estimates):
        name = "Speaker_%s" % itos(i)
        ops.append(make_audio_summary(name, source_estimate))
      ops.append(make_image_summary(X_source_estimates, mask))

def create_summaries(hparams, threshold_mask, attractors,
                     X_mixtures, phases, oracle_mask, sources, X_sources, embeddings):
  # Setup summary lists
  train_summ_ops, test_summ_ops, oracle_summ_ops, SDR_summ_ops = [], [], [], []

  # Separate the TF axis into the two original axis: T, F
  X_sources   = helper.uncollapse_freq_into_time(hparams, X_sources)
  X_mixtures  = helper.uncollapse_freq_into_time(hparams, X_mixtures)
  oracle_mask = helper.uncollapse_freq_into_time(hparams, oracle_mask)

  # Generate an audio summary for the input (mixture)
  mixture, _ = (tf.py_func(data_lib.nn_representation_to_wav_spect,
        [X_mixtures, phases], [tf.float32, tf.float32]))

  train_test_summ_ops = (train_summ_ops, test_summ_ops)
  # Oracle summaries
  X_source_estimates, source_estimates, proxy_SDR = (
    compute_results(hparams, X_mixtures, oracle_mask, sources, phases))
  create_audio_image_summaries(ORACLE_NAME, oracle_mask, train_test_summ_ops,
                               mixture, X_source_estimates, source_estimates)
  with tf.name_scope(ORACLE_NAME):
    oracle_summ_ops.append(tf.summary.scalar(SDR_PROXY_NAME, proxy_SDR))
  tf.add_to_collection(constants.ORACLE_SRC_EST_NAME, tf.stack(source_estimates, axis=1))

  # Attractor Summaries (train & test)
  attractor_mask = get_attractor_mask(hparams, embeddings, attractors)
  X_source_estimates, source_estimates, proxy_SDR = (
    compute_results(hparams, X_mixtures, attractor_mask, sources, phases))
  create_audio_image_summaries(ATTRACTOR_NAME, attractor_mask, train_test_summ_ops,
                               mixture, X_source_estimates, source_estimates)
  with tf.name_scope(ATTRACTOR_NAME):
    # train & test on same graph
    proxy_sdr_summary = tf.summary.scalar(SDR_PROXY_NAME, proxy_SDR)
    train_summ_ops.append(proxy_sdr_summary)
    test_summ_ops.append(proxy_sdr_summary)

  # Oracle Sumaries (train & test)
  kmeans_mask = get_kmeans_mask(hparams, X_mixtures, X_sources, embeddings,
                                threshold_mask)
  X_source_estimates, source_estimates, proxy_SDR = (
    compute_results(hparams, X_mixtures, kmeans_mask, sources, phases))
  create_audio_image_summaries(KMEANS_NAME, attractor_mask, train_test_summ_ops,
                               mixture, X_source_estimates, source_estimates)
  SDR = compute_SDR(hparams)
  with tf.name_scope(KMEANS_NAME):
    # train & test on same graph
    proxy_sdr_summary = tf.summary.scalar(SDR_PROXY_NAME, proxy_SDR)
    train_summ_ops.append(proxy_sdr_summary)
    test_summ_ops.append(proxy_sdr_summary)
    SDR_summ_ops.append(tf.summary.scalar("SDR.dB", SDR))

  tf.add_to_collection(constants.KMEANS_SRC_EST_NAME, tf.stack(source_estimates, axis=1))

  return train_summ_ops, test_summ_ops, oracle_summ_ops, SDR_summ_ops
