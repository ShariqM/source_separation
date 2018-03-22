"""Various non-core functions."""

import tensorflow as tf
import numpy as np
import pdb

import constants


####################
### THRESHOLDING ###
####################
def get_threshold_mask(hparams, x):
  """Threshold the mixtures to 1 or 0 for each TF bin.
  Input:
    X_mixtures: B x T x F
  Output:
    X_mixtures: B x T x F \in {0,1}
  """

  axis = list(range(1, x.shape.ndims))
  min_val = tf.reduce_min(x, axis=axis, keepdims=True)
  max_val = tf.reduce_max(x, axis=axis, keepdims=True)
  thresh = min_val + hparams.threshold_factor * (max_val - min_val)
  cond = tf.less(x, thresh)
  return tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

def np_get_threshold_mask(hparams, x):
  min_val = np.min(x)
  max_val = np.max(x)
  thresh = min_val + hparams.threshold_factor * (max_val - min_val)
  return (x > thresh).astype(np.int32)

def get_attractors(hparams, threshold_mask, embeddings, oracle_mask):
  """Calculate the attractors of the embeddings.

  Input:
    threshold_mask: BxN - Binary Mask indicating non-thresholded TF bins
    embeddings: BxNxK - All N K-dimensional embeddings
    oracle_mask: BxNxC - Binary Mask indicating classification of each TF bin

  Output:
    attractors: BxCxK - C attractor points in the embedding space
  """

  threshold_mask = tf.expand_dims(threshold_mask, -1) * oracle_mask
  bin_count = tf.reduce_sum(threshold_mask, axis=1)  # Count of non-threshold TF bins
  bin_count = tf.expand_dims(bin_count, -1)

  unnormalized_attractors = tf.einsum("bik,bic->bck", embeddings, threshold_mask)
  attractors = tf.divide(unnormalized_attractors, bin_count + 1e-6)  # Dont' divide by 0

  return attractors

############
### MISC ###
############
def np_collapse_freq_into_time(x):
  """Collapse the freq and time dimensions."""
  if x.ndim == 4:
    return np.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], -1])
  return np.reshape(x, [x.shape[0], x.shape[1] * x.shape[2]])

def collapse_freq_into_time(x):
  """Collapse the freq and time dimensions."""
  if x.shape.ndims == 4:
    return tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], -1])
  return tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2]])

def uncollapse_freq_into_time(hparams, x):
  """UNCollapse the freq and time dimensions."""
  if x.shape.ndims == 3:
    return tf.reshape(x, [x.shape[0], hparams.ntimebins, constants.nfreqbins, -1])
  return tf.reshape(x, [x.shape[0], hparams.ntimebins, constants.nfreqbins])

def collapse_time_into_batch(x):
  """Collapse the batch and time dimensions."""
  return tf.reshape(x, [-1] + x.shape.as_list()[2:])

def uncollapse_time_from_batch(hparams, x):
  """Separate the batch and time dimensions."""
  return tf.reshape(x, [hparams.batch_size, -1] + x.shape.as_list()[1:])

def model_is_recurrent(model):
  return "lstm" in model.lower()

def model_is_convolutional(model):
  return "cnn" in model.lower()

def get_oracle_waveform_savedir(hparams):
  return "ORACLE_%s" % hparams.data_source

def get_kmeans_waveform_savedir(hparams):
    if model_is_convolutional(hparams.model):
      name = "%s_%d_c%d_%s_%d" % (hparams.model, hparams.filter_shape[1],
        hparams.channels[0], hparams.data_source, hparams.ntimebins)
    else:
      name = "%s_%s_%d" % (hparams.model, hparams.data_source, hparams.ntimebins)

    if hparams.add_white_noise:
      name = "white_noise_" + name
    return name

def flush(*args):
  for arg in args:
    arg.flush()

