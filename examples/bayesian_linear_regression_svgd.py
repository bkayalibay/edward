import tqdm
import tensorflow as tf
import numpy as np

import edward as ed
from edward.models import Normal, Empirical


N = 40
D = 2
P = 10
T = 10000
noise_std = 0.5


def to_float(*xs):
    return [x.astype("float32") for x in xs]


def build_toy_dataset(N, noise_std=0.5):
    X = np.random.randn(N, D)
    w = np.random.randn(D)
    b = np.random.randn(1)
    y = X.dot(w) + b + np.random.normal(0., noise_std)
    return to_float(X, y, w, b)


X_train, y_train, true_w, true_b = build_toy_dataset(N, noise_std=noise_std)

X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y_logscale = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.nn.softplus(y_logscale)+1e-8)

qw = Empirical(params=tf.Variable(tf.random_normal([P, D])))
qb = Empirical(params=tf.Variable(tf.random_normal([P])))
qy_logscale = Empirical(params=tf.Variable(tf.random_normal([P])))

latent_vars = {w: qw, b: qb, y_logscale: qy_logscale}
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

print("true w: {} | SVGD: {}".format(true_w, sess.run(qw.mean())))
print("true b: {} | SVGD: {}".format(true_b, sess.run(qb.mean())))
print("true noise std: {} | SVGD {}".format(noise_std,
                                            sess.run(tf.nn.softplus(
                                                qy_logscale.mean()))))
