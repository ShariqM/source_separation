"""Model parameters for the RNN."""

import constants
import tuples


class RNNParameters():
  ##########################
  ### Network Parameters ###
  ##########################
  num_layers = 4
  layer_size = 500 # For FWD and BWD, so hidden layer has 2 * layer_size units

  ###########################
  ### Training Parameters ###
  ###########################
  max_steps = int(1e7)
  batch_size = 8
  learning_rate = 1e-3
  clip_gradient_norm = 200

    ## Opitmizer and LR decay functions
  optimizer = tuples.RMSProp
  use_exponential_decay = True  # otherwise piecewise

    ## Exponential Decay parameters
  decay_steps = 2000
  decay_rate = 0.95
