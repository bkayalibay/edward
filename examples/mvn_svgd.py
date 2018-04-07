import tqdm
import numpy as np
import tensorflow as tf

import edward as ed
from edward.models import MultivariateNormalFullCovariance, Empirical

n_iter = 5000
n_particles = 10

A = np.array([[0.2260, 0.1652], [0.1652, 0.6779]], dtype="float32")
loc = np.array([-0.6871, 0.8010], dtype="float32")

x = MultivariateNormalFullCovariance(loc, A)
qx = Empirical(params=tf.Variable(tf.random_normal([n_particles, 2])))

latent_vars = {x: qx}
data = {}

inference = ed.SVGD(latent_vars, data, ed.rbf)
inference.initialize(optimizer=tf.train.AdamOptimizer(0.001))

sess = ed.get_session()
sess.run(tf.global_variables_initializer())

train_loop = tqdm.trange(n_iter)
for _ in train_loop:
    info_dict = inference.update({})
    mean_log_p = np.mean(info_dict['p_log_lik'])
    train_loop.set_description("ave. log p = {}".format(mean_log_p))

print("True loc: {} | SVGD: {}".format(loc, sess.run(qx.mean())))
