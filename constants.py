"""Constants for STFT and misc items."""

import socket


#######################
### STFT PARAMETERS ###
#######################
nperseg = 256
noverlap = int(nperseg * 3/4)
ndiff = nperseg - noverlap
nfreqbins = int((nperseg / 2)) + 1

#####################
### MISCELLAENOUS ###
#####################
Fs = 8000
max_input_snr = 5

kmeans_max_iters = 50  # Max iterations when running k-means

if 'fattire' in socket.gethostname():
  use_port = 54122  # Brian
else:
  use_port = 54621

TRAIN_OP_NAME="train_op"
ORACLE_SRC_EST_NAME="oracle_source_estimates"
KMEANS_SRC_EST_NAME="kmeans_source_estimates"

LAST_EXPERIMENT_NUM_FILE = "pylogs/last_experiment_num.log"

AVG_SDR_ON_N_BATCHES = 50
