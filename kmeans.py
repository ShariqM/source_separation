"""K-Means implementation."""

import numpy as np
import tensorflow as tf

import constants


def init_centers(embeddings, num_targets):
  """Pick the initial cluster positions for these embeddings.

  Args:
    embeddings: is a NxK matrix with K dimensional embeddings.

  Use a non-stochastic variant of the k-means++ initialization algorithm.

  https://en.wikipedia.org/wiki/K-means%2B%2B

  1) Pick a random data point to be the first cluster point.
  2) Mark each data point, x, with a distance, D(x) which is equal to
  the distance from x to the nearest cluster (minimum distance)
  3) The next cluster is argmax_x (D(x))

  """
  n_embeddings, embedding_size = embeddings.shape

  centers = np.zeros((num_targets, embedding_size))
  distances = np.zeros((num_targets, n_embeddings))

  rand_idx = np.random.randint(embeddings.shape[0])
  centers[0, :] = embeddings[rand_idx, :]

  for i in range(1, num_targets):
    distances[i, :] = np.sum(np.square(centers[i-1, :] - embeddings), axis=1)
    distances[i, :] = np.min(distances[:(i+1), :], axis=0)  # Smallest distances
    idx = np.argmax(distances[i, :])
    centers[i, :] = embeddings[idx, :]

  return centers

def assign(centers, embeddings):
  """Assign embeddings to the closest cluster."""
  num_targets = centers.shape[0]
  embeddings = np.expand_dims(embeddings, axis=2)
  centers = np.expand_dims(centers.T, axis=0)

  distances = np.linalg.norm(embeddings - centers, axis=1)
  assignments = np.argmin(distances, axis=1)
  return np.eye(num_targets)[assignments]

def update_centers(assignments, embeddings):
  """Given the assignments of embeddings update the centers."""
  normalizer = np.sum(assignments, axis=0, keepdims=True).T
  centers =  np.dot(assignments.T, embeddings) / (normalizer + 1e-6)
  return centers

def get_centers_impl(embeddings, threshold_mask, num_targets):
  """Run K-means on the embeddings and output the K-means centers.

  Arguments:
    embeddings: BxNxK - Embeddings for each batch (N index the TF bins)
    threshold_mask: BxN - Binary matrix indicating if this TF-bin was not thresholded
    num_targets: integer - hparams.num_targets
  """

  batch_size, n_embeddings, embedding_size = embeddings.shape
  kmeans_centers = np.zeros((batch_size, num_targets, embedding_size),
                            dtype=np.float32)

  converged_at = []
  for b in range(batch_size):
    indices = np.where(threshold_mask[b, :] == 1.0)[0]
    # Only use embeddings that passed the threshold
    b_embeddings = embeddings[b, indices, :]
    centers = init_centers(b_embeddings, num_targets)

    converged = False
    for i in range(constants.kmeans_max_iters):
      assignments = assign(centers, b_embeddings)

      new_centers = update_centers(assignments, b_embeddings)
      if np.allclose(new_centers, centers):
        converged = True
        break
      centers = new_centers
    converged_at.append(i)

    kmeans_centers[b, :, :] = centers
  print ("K-Means converged in %.2f iters on average" % (np.mean(converged_at)))

  return kmeans_centers

def get_centers(hparams, embeddings, threshold_mask):
  kmeans_centers = tf.py_func(get_centers_impl,
      [embeddings, threshold_mask, hparams.num_targets], tf.float32)

  kmeans_centers.set_shape((embeddings.shape[0].value, hparams.num_targets,
                            hparams.embedding_size))

  return kmeans_centers
