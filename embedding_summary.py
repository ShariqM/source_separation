"""Functions for setting up embedding visualization summary."""

import numpy as np
import os
import pdb
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import helper


class Embedding():
  def __init__(self, embedding_config, embedding_assign, show_embeddings=True):
    self.show_embeddings = show_embeddings
    self.embedding_config = embedding_config
    self.embedding_assign = embedding_assign

  def get_assign_op(self):
    return self.embedding_assign

  def visualize_embeddings(self, train_writer):
    if self.show_embeddings:
      projector.visualize_embeddings(train_writer, self.embedding_config)

def handle_embedding(hparams, embeddings):
  if not hparams.show_embeddings:
    # Embedding of noops
    return Embedding(tf.constant(0), tf.constant(0), False)

  with tf.name_scope("Embedding"):
    embedding_act = embeddings[0, :, :]

    init_value = np.zeros(embedding_act.shape, dtype=np.float32)
    embedding_var = tf.Variable(init_value, name="Variable")
    embedding_assign = tf.assign(embedding_var, embedding_act, name="Assign")

  # Embedding
  embedding_config = projector.ProjectorConfig()
  embedding_obj = embedding_config.embeddings.add()
  embedding_obj.tensor_name = embedding_var.name

  embedding_obj.metadata_path = os.path.abspath('.') + '/' + hparams.logdir + '/train/metadata.tsv'
  embedding_info = Embedding(embedding_config, embedding_assign)
  return embedding_info

color_to_label = {"blue": 0,
                  "yellow": 1,
                  "red": 2,
                  "purple": 3,
                  "pink": 4,
                  "grey": 5,
                  "turqoise": 6,
                  "blue-grey": 7,
                  "green": 8,
                  "orange": 9}

def get_label(hparams, thresholded, max_idx, i):
  assert hparams.num_targets == 2, "More colors unsupported yet"
  # Need these so we can make thresholded values grey
  if i == 0:
    return color_to_label["yellow"]
  if i == 1:
    return color_to_label["purple"]
  if i == 2:
    return color_to_label["pink"]

  if thresholded == 0:
    return color_to_label["grey"]
  if max_idx == 0:
    return color_to_label["red"]  # Spk A
  return color_to_label["blue"]  # Spk B

def write_tsv(hparams, X_mixtures, masks):
  if not hparams.show_embeddings:
    return

  X_mixtures = helper.np_collapse_freq_into_time(X_mixtures)
  masks = helper.np_collapse_freq_into_time(masks)
  with open(hparams.logdir + "/train/metadata.tsv", "w") as f:
    # Write data from first batch mixture
    X_mixture = X_mixtures[0, :]
    mask = masks[0, :, :]

    threshold_mask = helper.np_get_threshold_mask(hparams, X_mixture)
    max_idx = np.argmax(mask, axis=1)
    for i in range(masks.shape[1]):
      label = get_label(hparams, threshold_mask[i], max_idx[i], i)
      f.write("%d\n" % label)

