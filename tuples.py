import collections
import tensorflow as tf

Optimizer = collections.namedtuple('Optimizer', 'func')
Adam = Optimizer(func=tf.train.AdamOptimizer)
RMSProp = Optimizer(func=tf.train.RMSPropOptimizer)
SGD = Optimizer(func=tf.train.GradientDescentOptimizer)
