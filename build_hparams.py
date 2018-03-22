"""Combine FLAGS, *MODEL*parameters.py, and hyperparameters together."""

import hyperparameters
import RNNparameters
import CNNparameters
import pdb

import constants
import helper

def transfer_variables(A, hparams):
  """Transfer variables from model parameters to hparams"""
  for (k,v) in vars(A).items():
    if not k.startswith('__'):
      hparams.add_hparam(k, v)
  return hparams

def find(listy, x):
  """Return index of x in listy, and None if it doesn't exist"""
  return listy.index(x) if x in listy else None

def set_model_type(hparams, FLAGS):
  """Set hparams.model to FLAGS.hparams[model] if it is specified there.
     - We need to do this to load the correct hparams."""
  if not FLAGS.hparams:
    return

  keyword = "model="
  model_pos = find(FLAGS.hparams, keyword)
  if model_pos is None:
    return

  model_name_pos = model_pos + len(keyword)
  end_pos = find(FLAGS.hparams[model_name_pos:], ",")
  if end_pos is None:
    end_pos = len(FLAGS.hparams)

  hparams.model = FLAGS.hparams[model_name_pos:end_pos]

def add_model_parameters(hparams, FLAGS):
  """Take the parameters in [MODEL]parameters.py and add them to hparams."""
  set_model_type(hparams, FLAGS)

  if helper.model_is_recurrent(hparams.model):
    return transfer_variables(RNNparameters.RNNParameters, hparams)
  elif helper.model_is_convolutional(hparams.model):
    hparams = transfer_variables(CNNparameters.CNNParameters, hparams)
    hparams.channels[-1] = hparams.embedding_size
    return hparams
  raise Exception("Invalid Model: %s" % hparams.model)

def build_hparams(FLAGS):
  """Build all hyperparameters associated with the core computation."""
  hparams = add_model_parameters(hyperparameters.params, FLAGS)
  hparams.training = True
  if FLAGS.hparams:
    hparams.parse(FLAGS.hparams)
  if FLAGS.eval_model:
    hparams.summary_frequency = 1
    hparams.test_frequency = 1
    hparams.save_frequency = 5
    hparams.training = False

  hparams.sdr_frequency = hparams.test_frequency * constants.AVG_SDR_ON_N_BATCHES
  # See STFT scipy doc
  hparams.waveform_size = (hparams.ntimebins - 1) * constants.ndiff

  return hparams

