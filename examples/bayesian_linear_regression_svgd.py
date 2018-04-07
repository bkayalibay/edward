import tqdm
import tensorflow as tf
import numpy as np

import edward as ed
from edward.models import Normal


N = 40
D = 1
P = 10
T = 10000
noise_std = 0.5

def to_float(*xs):
    return [x.astype("float32") for x in xs]


def build_toy_dataset(N, noise_std=0.5):
    X = np.random.randn(N, D)
    W = np.random.randn(D)
    b = np.random.randn(1)
    y = X.dot(W) + b + np.random.normal(0., noise_std)
    return to_float(X, y, W, b)


X_train, y_train, true_W, true_b = build_toy_dataset(N, noise_std=noise_std)

X = tf.placeholder(tf.float32, [N, D])
W = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y_logscale = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, W) + b, scale=tf.nn.softplus(y_logscale)+1e-8)

particles_W = [tf.Variable(tf.random_normal([D]))
               for _ in range(P)]
particles_b = [tf.Variable(tf.random_normal([1]))
               for _ in range(P)]
particles_y_logscale = [tf.Variable(tf.random_normal([1]))
                        for _ in range(P)]

latent_vars = {
    W: particles_W,
    b: particles_b,
    y_logscale: particles_y_logscale
}

data = {y: y_train}

inference = ed.SVGD(latent_vars, data, ed.rbf)
inference.initialize(optimizer=tf.train.AdamOptimizer(0.001))

sess = ed.get_session()
sess.run(tf.global_variables_initializer())

train_loop = tqdm.trange(T)
for _ in train_loop:
    info_dict = inference.update({X: X_train})
    mean_log_p = np.mean(info_dict["p_log_lik"])
    train_loop.set_description("ave. log p = {}".format(mean_log_p))

mean_est_W = np.mean(sess.run(particles_W), 0)
mean_est_b = np.mean(sess.run(particles_b), 0)
mean_est_y_logscale = np.mean(
    sess.run(
        [tf.nn.softplus(p_y_logscale)
         for p_y_logscale in particles_y_logscale]), 0)

print("true W: {} | SVGD: {}".format(true_W, mean_est_W))
print("true b: {} | SVGD: {}".format(true_b, mean_est_b))
print("true noise std: {} | SVGD {}".format(noise_std, mean_est_y_logscale))
