import tqdm
import tensorflow as tf
import numpy as np

import edward as ed
from edward.models import Normal, Empirical, Gamma


N = 40
D = 2
P = 10
T = 10000
noise_std = 1.0


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
y_ph = tf.placeholder(tf.float32, [N])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y_scale = Gamma(tf.ones(1), tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=y_scale+1e-8)

qw = Empirical(params=tf.Variable(tf.random_normal([P, D])))
qb = Empirical(params=tf.Variable(tf.random_normal([P])))
qy_scale = Empirical(params=tf.nn.softplus(
    tf.Variable(tf.random_normal([P]))))

latent_vars = {w: qw, b: qb, y_scale: qy_scale}
data = {y: y_ph}

inference = ed.SVGD(latent_vars, data, ed.rbf)
inference.initialize(optimizer=tf.train.AdamOptimizer(0.001))

sess = ed.get_session()
sess.run(tf.global_variables_initializer())

train_loop = tqdm.trange(T)
for _ in train_loop:
    info_dict = inference.update({X: X_train, y_ph: y_train})
    loss = np.mean(info_dict["loss"])
    train_loop.set_description("ave. - log p = {}".format(loss))

print("true w: {} | SVGD: {}".format(true_w, sess.run(qw.mean())))
print("true b: {} | SVGD: {}".format(true_b, sess.run(qb.mean())))
print("true noise std: {} | SVGD {}".format(noise_std,
                                            sess.run(qy_scale.mean())))
