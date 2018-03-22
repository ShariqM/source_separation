"""Build the entire TensorFlow graph: network, loss, optimizer, summaries."""

import tensorflow as tf
import pdb
import numpy as np

import summaries
import embedding_summary
import helper
import constants


##############
### INPUTS ###
##############
def get_input_shapes(hparams):
  X_mixtures_shape =  [hparams.batch_size, hparams.ntimebins, constants.nfreqbins]
  phases_shape =      [hparams.batch_size, hparams.ntimebins, constants.nfreqbins]
  oracle_mask_shape = [hparams.batch_size, hparams.ntimebins, constants.nfreqbins,
                       hparams.num_targets]
  sources_shape =     [hparams.batch_size, hparams.num_targets, hparams.waveform_size]
  X_sources_shape =   [hparams.batch_size, hparams.ntimebins, constants.nfreqbins,
                       hparams.num_targets]

  return (X_mixtures_shape, phases_shape, oracle_mask_shape, sources_shape,
          X_sources_shape)

def build_input_placeholders(hparams):
  place_holders = []
  for shape in get_input_shapes(hparams):
    place_holders.append(tf.placeholder(tf.float32, shape=shape))
  return place_holders


#################
### RNN MODEL ###
#################
def make_multi_rnn_cell(hparams):
  cells = []
  for _ in range(hparams.num_layers):
    cells.append(tf.contrib.rnn.BasicLSTMCell(hparams.layer_size))
  return tf.contrib.rnn.MultiRNNCell(cells)

def make_rnn_net(hparams, X_mixtures):
  both_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
      make_multi_rnn_cell(hparams),
      make_multi_rnn_cell(hparams),
      X_mixtures,
      dtype=tf.float32)
  outputs = tf.concat(both_outputs, 2)

  outputs = helper.collapse_time_into_batch(outputs)
  output_size = constants.nfreqbins * hparams.embedding_size
  embeddings = tf.contrib.layers.linear(outputs, output_size)
  embeddings = tf.reshape(embeddings,
      [embeddings.shape[0].value, -1, hparams.embedding_size])

  return helper.uncollapse_time_from_batch(hparams, embeddings)


#################
### CNN MODEL ###
#################
def conv2d(hparams, x, i):
  return tf.layers.conv2d(x,
                          hparams.channels[i],
                          hparams.filter_shape,
                          use_bias=True,
                          dilation_rate=[hparams.dilation_heights[i], hparams.dilation_widths[i]],
                          padding=hparams.padding,
                          kernel_initializer=None)

def make_cnn_net(hparams, X_mixtures):
  """Make the Dilated convolultional architecture."""
  num_layers = len(hparams.channels)
  prev_layer = 0.
  layer = tf.expand_dims(X_mixtures, -1)
  for i in range(num_layers):
    layer = conv2d(hparams, layer, i)
    if i == num_layers - 1:  # Last layer
      break

    if hparams.use_batch_normalization:
      layer = tf.layers.batch_normalization(layer, axis=3, training=True)
    if hparams.use_residual[i] and prev_layer != 0:
      layer += prev_layer

    layer = tf.nn.relu(layer)

    if hparams.use_residual[i]:
      prev_layer = layer

  return layer

def print_num_parameters():
  """Print the number of parameters in the model."""
  num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
  print ("Model has %d parameters" % num)

def make_net(hparams, X_mixtures):
  if helper.model_is_convolutional(hparams.model):
    embeddings = make_cnn_net(hparams, X_mixtures)
  elif helper.model_is_recurrent(hparams.model):
    embeddings = make_rnn_net(hparams, X_mixtures)
  else:
    raise Exception("Unknown model: %s" % hparams.model)
  print_num_parameters()

  return embeddings


############
### LOSS ###
############
def mse_loss(hparams, attractors, X_mixtures, embeddings, X_sources):
  mask_estimate = tf.einsum("bik,bck->bic", embeddings, attractors)
  mask_estimate = tf.nn.softmax(mask_estimate)
  X_source_estimates = tf.expand_dims(X_mixtures, -1) * mask_estimate

  mse_loss = tf.reduce_mean(tf.square(X_sources - X_source_estimates))
  return mse_loss

#################
### OPTIMIZER ###
#################
def build_optimizer(hparams, loss):
  if not hparams.training:
    tf.add_to_collection(constants.TRAIN_OP_NAME, tf.constant(0, tf.float32))
    return [tf.constant(0, tf.float32)] * 2
  global_step = tf.Variable(0, trainable=False)

  if hparams.use_exponential_decay:
    learning_rate = tf.train.exponential_decay(hparams.learning_rate, global_step,
        hparams.decay_steps, hparams.decay_rate, staircase=True)
  else:
    values = list(hparams.learning_rate * np.array(hparams.rate_factors))
    learning_rate = tf.train.piecewise_constant(global_step, hparams.boundaries, values)

  optimizer = hparams.optimizer.func(learning_rate)
  variables = tf.trainable_variables()
  gradients = tf.gradients(loss, variables)

  if helper.model_is_recurrent(hparams.model) and hparams.clip_gradient_norm > -1:
     gradients = [None if gradient is None else
                  tf.clip_by_norm(gradient, hparams.clip_gradient_norm)
                  for gradient in gradients]

  train_op = optimizer.apply_gradients(zip(gradients, variables),
                                       global_step=global_step)
  tf.add_to_collection(constants.TRAIN_OP_NAME, train_op)

  return global_step, learning_rate


#############
### GRAPH ###
#############
def build_inference_graph(hparams):
  X_mixtures_shape = [hparams.batch_size, hparams.ntimebins, constants.nfreqbins]
  phases_shape =     [hparams.batch_size, hparams.ntimebins, constants.nfreqbins]

  X_mixtures = tf.placeholder(tf.float32, shape=X_mixtures_shape)
  phases = tf.placeholder(tf.float32, shape=phases_shape)

  threshold_mask = helper.get_threshold_mask(hparams, X_mixtures)
  embeddings = make_net(hparams, X_mixtures)
  embeddings = tf.nn.l2_normalize(embeddings, axis=embeddings.shape.ndims - 1)

  X_mixtures_rs  = helper.collapse_freq_into_time(X_mixtures)
  embeddings     = helper.collapse_freq_into_time(embeddings)
  threshold_mask = helper.collapse_freq_into_time(threshold_mask)

  inference_summaries = summaries.setup_inference_summary(hparams, threshold_mask,
                                              X_mixtures_rs, phases, embeddings)

  inference_summaries = tf.summary.merge(inference_summaries)
  return X_mixtures, phases, inference_summaries

def build_train_graph(hparams):
  inputs = build_input_placeholders(hparams)
  X_mixtures, phases, oracle_mask, sources, X_sources = inputs
  threshold_mask = helper.get_threshold_mask(hparams, X_mixtures)

  embeddings = make_net(hparams, X_mixtures)
  embeddings = tf.nn.l2_normalize(embeddings, axis=embeddings.shape.ndims - 1)

  # Put time-frequency in the same axis for clustering embeddings
  X_mixtures  = helper.collapse_freq_into_time(X_mixtures)
  oracle_mask = helper.collapse_freq_into_time(oracle_mask)
  X_sources   = helper.collapse_freq_into_time(X_sources)
  embeddings  = helper.collapse_freq_into_time(embeddings)
  threshold_mask = helper.collapse_freq_into_time(threshold_mask)

  attractors = helper.get_attractors(hparams, threshold_mask, embeddings, oracle_mask)
  loss = mse_loss(hparams, attractors, X_mixtures, embeddings, X_sources)
  global_step, learning_rate = build_optimizer(hparams, loss)

  loss_summary = tf.summary.scalar("Loss", loss)

  # Summaries
  learning_rate_summary = tf.summary.scalar("Learning Rate", learning_rate)
  extra_data_summary = [learning_rate_summary]

  train_summary, test_summary, oracle_summary, SDR_summary = (
      summaries.create_summaries(hparams, threshold_mask, attractors,
         X_mixtures, phases, oracle_mask, sources, X_sources, embeddings))

  loss_summaries = [loss_summary]
  train_summary  = tf.summary.merge(loss_summaries + extra_data_summary + train_summary)
  test_summary   = tf.summary.merge(loss_summaries + test_summary)
  oracle_summary = tf.summary.merge(oracle_summary)
  SDR_summary    = tf.summary.merge(SDR_summary)

  embedding_info = embedding_summary.handle_embedding(hparams, embeddings)

  return (inputs, embedding_info, loss, loss_summary,
          [train_summary, test_summary, oracle_summary, SDR_summary])
