import tqdm
import numpy as np
import tensorflow as tf

import edward as ed
from edward.models import MultivariateNormalFullCovariance

n_iter = 5000
n_particles = 10

A = np.array([[0.2260, 0.1652], [0.1652, 0.6779]], dtype="float32")
mu = np.array([-0.6871, 0.8010], dtype="float32")

x = MultivariateNormalFullCovariance(mu, A)
particles = [tf.Variable(tf.random_normal([2])) for _ in range(n_particles)]

latent_vars = {x: particles}
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

print("Ground truth: {}".format(mu))
print("SVGD: {}".format(np.mean(sess.run(particles), axis=0)))
