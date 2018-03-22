"""Start TensorBoard with the specified logs."""

import subprocess
import constants
import argparse
import utilities

parser = argparse.ArgumentParser(description='Train the Keller Net..')
parser.add_argument("-l", type=int, dest="load_experiment_num", default=0,
                    help="Herro")
parser.add_argument("-p", type=int, dest="port", default=0,
                    help="Herro")
FLAGS = parser.parse_args()


if FLAGS.load_experiment_num:
  experiment_num = FLAGS.load_experiment_num
  offset_experiment_num = experiment_num + utilities.get_offset()
  log_file = "impt_logs/%d" % offset_experiment_num
else:
  f = open(constants.LAST_EXPERIMENT_NUM_FILE, 'r')
  experiment_num = int(f.readline()[:-1])
  offset_experiment_num = experiment_num + utilities.get_offset()
  log_file = "logs/%d" % offset_experiment_num
  f.close()


port = constants.use_port
if FLAGS.port:
  port = FLAGS.port

print ("Tensorboard for", log_file)
subprocess.call("tensorboard --logdir=%s --port=%d" %
                (log_file, port), shell=True)
