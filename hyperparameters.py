"""Hyperparameters for the entire graph, model parameters are in modelparameters.py"""

import tensorflow as tf


params = tf.contrib.training.HParams(
  model = "CNN",  # "BLSTM" or "CNN"
  data_source = "RTL",  # "WSJ0", "LIBRI", or "RTL"

  num_targets = 2,  # 2 speakers

  threshold_factor = 0.6,  # Threshold TF bins below 0.6 * MAX
  embedding_size = 20,
  ntimebins = 400,

  #####################
  ### MISCELLAENOUS ###
  #####################
  add_white_noise = False,
  run_inference_test = False,  # Test the model long speech data
  save_estimate_waveforms = True,
  save_oracle_waveforms = False,
  show_embeddings = False,
  summary_frequency = 50,
  save_frequency = 100,
  test_frequency = 50)
