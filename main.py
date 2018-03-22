"""Build the graph and train it."""

import tensorflow as tf
import numpy as np
from optparse import OptionParser
import argparse
import pdb

import graph
import train
import inference
import utilities
import build_hparams


parser = argparse.ArgumentParser(description='Train the Keller Net..')
parser.add_argument("-l", type=int, dest="load_experiment_num", default=0,
                    help="Load weights from Model, don't load on 0")
parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
parser.add_argument('-e', action='store_true', default=False, dest='eval_model',
                    help='Run summary and test alot to evaluate results.')
FLAGS = parser.parse_args()

def initialize():
  """Build hyperparameters, setup loading/saving"""
  hparams = build_hparams.build_hparams(FLAGS)
  hparams.add_hparam('load_experiment_num', FLAGS.load_experiment_num)

  experiment_num = utilities.prepare_experiment()
  hparams.add_hparam('logdir', "logs/%d" % (experiment_num))

  hparams.add_hparam('loaddir', "impt_logs/%d/" % (hparams.load_experiment_num))
  hparams.add_hparam('savedir', hparams.logdir + "/model.ckpt")

  return hparams

def main():
  """Build hparams, the graph, and train it."""
  hparams = initialize()

  if hparams.run_inference_test:
    hparams.batch_size = 2
    X_mixtures, phases, inference_summaries = graph.build_inference_graph(hparams)
    inference.run_inference(hparams, X_mixtures, phases, inference_summaries)
  else:
    inputs, embedding_info, loss, loss_summary, summaries = (
        graph.build_train_graph(hparams))

    train.run_train(hparams, inputs, embedding_info,
        loss, loss_summary, summaries)

if __name__ == '__main__':
  main()
