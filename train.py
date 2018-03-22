"""Train the parameters of the graph."""

import tensorflow as tf
import time
import numpy as np
import multiprocessing
import os
import pdb
from scipy.io import savemat

import loader
import helper
import embedding_summary
import constants


def save_waveforms(hparams, step, sources, oracle_src_ests, kmeans_src_ests):
  example_num = (step // hparams.test_frequency) % constants.AVG_SDR_ON_N_BATCHES

  if hparams.save_oracle_waveforms:
    oracle_name = helper.get_oracle_waveform_savedir(hparams)
    os.makedirs("waveforms/%s" % oracle_name, exist_ok=True)
    savemat("waveforms/%s/example_%d" % (oracle_name, example_num),
            {"sources": sources, "source_estimates": oracle_src_ests})

  if hparams.save_estimate_waveforms:
    kmeans_name = helper.get_kmeans_waveform_savedir(hparams)
    os.makedirs("waveforms/%s" % kmeans_name, exist_ok=True)
    savemat("waveforms/%s/example_%d" % (kmeans_name, example_num),
            {"sources": sources, "source_estimates": kmeans_src_ests})

def make_feed_dict(hparams, pool, inputs, test=False):
  mixtures, phases, masks, sources, X_sources = (
      loader.make_data(pool, hparams, test))

  return {inputs[0]: mixtures, inputs[1]: phases, inputs[2]: masks,
          inputs[3]: sources, inputs[4]: X_sources}

def setup(hparams, sess, saver):
  """Initialize variables, build writers, restore model if needed."""
  sess.run(tf.global_variables_initializer())

  train_writer = tf.summary.FileWriter(hparams.logdir + "/train", sess.graph)
  test_writer = tf.summary.FileWriter(hparams.logdir + "/test", sess.graph)
  oracle_writer = tf.summary.FileWriter(hparams.logdir + "/oracle", sess.graph)

  if hparams.load_experiment_num:
    print ("Loading: %s" % hparams.loaddir)
    saver.restore(sess, tf.train.latest_checkpoint(hparams.loaddir))

  return train_writer, test_writer, oracle_writer

def get_saved_ops():
  train = tf.get_collection(constants.TRAIN_OP_NAME)
  oracle_src_ests_op = tf.get_collection(constants.ORACLE_SRC_EST_NAME)
  kmeans_src_ests_op = tf.get_collection(constants.KMEANS_SRC_EST_NAME)
  return train, oracle_src_ests_op, kmeans_src_ests_op

def run_train(hparams, inputs, embedding_info, loss, loss_summary, summaries):
  mixtures, phases, masks, sources, X_sources = inputs
  train_summary, test_summary, oracle_summary, SDR_summary = summaries

  embedding_assign = embedding_info.get_assign_op()
  train, oracle_src_ests_op, kmeans_src_ests_op = get_saved_ops()
  pool = multiprocessing.Pool(multiprocessing.cpu_count())
  saver = tf.train.Saver()

  with tf.Session() as sess:
    train_writer, test_writer, oracle_writer = setup(hparams, sess, saver)

    for step in range(hparams.max_steps):
      start = time.time()
      feed_dict = make_feed_dict(hparams, pool, inputs)

      # Gradient step on training set
      if step % hparams.summary_frequency == 0:
        result = sess.run([loss, embedding_assign, train_summary, oracle_summary,
                           train], feed_dict=feed_dict)
        raw_loss, _, raw_summary, raw_oracle_summary, _  = result

        oracle_writer.add_summary(raw_oracle_summary, step)
        embedding_info.visualize_embeddings(train_writer)
      else:
        result = sess.run([loss, loss_summary, train], feed_dict=feed_dict)
        raw_loss, raw_summary, _ = result

      train_writer.add_summary(raw_summary, step)

      # Evaluate on Test Set
      if step and step % hparams.test_frequency == 0:
        test_feed_dict = make_feed_dict(hparams, pool, inputs, test=True)
        result = sess.run([loss, test_summary, oracle_src_ests_op,
                           kmeans_src_ests_op], feed_dict=test_feed_dict)

        raw_loss, raw_summary, oracle_src_ests, kmeans_src_ests = result
        test_writer.add_summary(raw_summary, step)
        save_waveforms(hparams, step, test_feed_dict[sources], oracle_src_ests, kmeans_src_ests)
        print ("\t%d) EVAL-Loss: %.3f" % (step, raw_loss))

      if step and step % hparams.sdr_frequency == 0:
        raw_summary = sess.run([SDR_summary])[0]
        test_writer.add_summary(raw_summary, step)

      # Save Model
      if step % hparams.save_frequency == 0:
        # The embeddings are read from saved weights, make sure the TSV matches
        embedding_summary.write_tsv(hparams, feed_dict[mixtures], feed_dict[masks])
        saver.save(sess, hparams.savedir, step)

        print ("Model Saved.")

      helper.flush(train_writer, test_writer, oracle_writer)
      print ("\t%d) Loss: %.3f || Elapsed = %.3f secs" % (step, raw_loss,
             (time.time() - start)))

