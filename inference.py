"""Only run inference, no training, loss or labels."""

import tensorflow as tf
import pdb
import time

import loader

def setup_inference(hparams, sess, init):
  sess.run(init)
  inference_writer = tf.summary.FileWriter(hparams.logdir + "/inference", sess.graph)

  if hparams.load_experiment_num:
    print ("Loading: %s" % hparams.loaddir)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(hparams.loaddir))

  return inference_writer

def run_inference(hparams, X_mixtures_ph, phases_ph, inference_summaries):
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    inference_writer = setup_inference(hparams, sess, init)

    for step in range(hparams.max_steps):
      start = time.time()
      X_mixtures, phases = loader.get_inference_data(hparams)
      feed_dict = {X_mixtures_ph: X_mixtures, phases_ph: phases}

      raw_summary = sess.run(inference_summaries, feed_dict=feed_dict)

      if step % 25 == 0:
        # Write every 10 b/c TB seems to mix up audio segments...
        inference_writer.add_summary(raw_summary, step)
      print ("\t%d) Elapsed = %.3f secs" % (step, (time.time() - start)))
