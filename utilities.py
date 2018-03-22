"""Keep track of the experiment number and save all hyperparameters to new files."""

import subprocess
import pdb
import socket
import constants


def get_offset():
  # Use a different offset for other machines
  if 'fattire' in socket.gethostname():
    return 200000000  # Brian
  return 100000000

def prepare_experiment():
  """Copy all hyperparameter files for this experiment to a different folder.
    Return the current experiment num"""

  f = open(constants.LAST_EXPERIMENT_NUM_FILE, 'r')
  experiment_num = int(f.readline()[:-1]) + 1
  f.close()

  offset_experiment_num = get_offset() + experiment_num

  fnames = ("hyperparameters", "RNNparameters", "CNNparameters", "constants")
  for fname in fnames:
    subprocess.Popen(['cp %s.py hparams_logs/%d_%s.py' %
                      (fname, offset_experiment_num, fname)], shell=True)

  f = open(constants.LAST_EXPERIMENT_NUM_FILE, 'w')
  f.write(str(experiment_num) + '\n')
  f.close()
  return offset_experiment_num
