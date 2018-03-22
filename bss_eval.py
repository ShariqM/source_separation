"""TensorFlow implementation of Blind Source Separation metric*

   This code implements BSS using the method from II.B in the paper*
   which runs significantly faster than the time distortion one (III.B). We
   run this version during training to reduce computational load.

   * [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.
"""

import tensorflow as tf
import pdb as pdb
import numpy as np
import mir_bss_eval


def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def tf_square_norm(x, axis=None):
  return tf.square(tf.norm(x, axis=2))

def compute_proxy_SDR(st, ei, ea):
  return 10 * tf_log10(tf_square_norm(st) / tf_square_norm(ei + ea))

def compute_targets(sources, source_estimates):
  dotProd = tf.einsum("btw,btw->bt", sources, source_estimates)
  normalizer = tf.square(tf.norm(sources, axis=2))
  targets = sources * tf.expand_dims(dotProd / normalizer, -1)
  return targets

def compute_c(sources, source_estimates):
  gram = tf.einsum("btw,buw->btu", sources, sources)
  Ginv = tf.matrix_inverse(gram)
  products = tf.einsum("btw,buw->btu", source_estimates, sources)
  return tf.einsum("btu,buv->btv", Ginv, products)

def compute_components(sources, source_estimates):
  source_targets = compute_targets(sources, source_estimates)

  c = compute_c(sources, source_estimates)
  subspace_projection = tf.einsum("btu,btw->buw", c, sources)
  interferences = subspace_projection - source_targets

  artifacts = source_estimates - subspace_projection

  return source_targets, interferences, artifacts

def eval_proxy_SDR(sources, source_estimates):
  source_targets, interferences, artifacts = (
    compute_components(sources, source_estimates))

  proxy_SDR = compute_proxy_SDR(source_targets, interferences, artifacts)
  return tf.reduce_mean(proxy_SDR)
