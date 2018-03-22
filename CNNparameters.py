"""Model parameters for the RNN."""
import constants
import tuples

class CNNParameters():
  ##########################
  ### NETWORK PARAMETERS ###
  ##########################
  filter_shape = (3, 3)  # (Time, Freq)

  dilation_heights = ([1, 2, 4] + [8, 16, 32] +
                      [1, 2, 4] + [8, 16, 32] + [1])
  dilation_widths = dilation_heights
  n_c = 128
  channels = ([n_c, n_c, n_c] + [n_c, n_c, n_c] +
              [n_c, n_c, n_c] + [n_c, n_c, n_c] + [-1])  # -1 replaced by embeding_size
  use_residual = ([False, True, False] + [True, False, True] +
                  [False, True, False] + [True, False, True] + [False])

  padding = "SAME"

  ###########################
  ### TRAINING PARAMETERS ###
  ###########################
  max_steps = int(1e7)
  batch_size = 8
  learning_rate = 1e-3

    ## Opitmizer and LR decay functions
  optimizer = tuples.Adam
  use_exponential_decay = False  # Use piecewise decay
  use_batch_normalization = True

    ## Piecewise decay parameters
  boundaries =   [10000, 50000, 100000]
  rate_factors = [1.0, 0.50, 0.10, 0.01]

