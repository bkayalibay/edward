import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import edward as ed
from edward.models import Normal, Empirical, Gamma


sns.set_style("darkgrid")

N = 1000
D = 1
P = 10
T = 10000
noise_std = 1.0


def build_toy_dataset(N, noise_std):
    X = np.random.randn(N, D).astype("float32")
    w = np.random.randn(D).astype("float32")
    b = np.random.randn(1).astype("float32")
    y = X.dot(w) + b + np.random.normal(0., noise_std, size=(N,))
    return X, y.astype("float32"), w, b


X_train, y_train, true_w, true_b = build_toy_dataset(N, noise_std=noise_std)

X = tf.placeholder(tf.float32, [N, D])
y_ph = tf.placeholder(tf.float32, [N])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y_scale = Gamma(tf.ones(1), tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=y_scale+1e-8)

qw = Empirical(params=tf.Variable(tf.random_normal([P, D]), name="qw"))
qb = Empirical(params=tf.Variable(tf.random_normal([P]), name="qb"))
qy_scale = Empirical(params=tf.nn.softplus(
    tf.Variable(tf.random_normal([P]), name="qy_scale")))

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

y_post = ed.copy(y, latent_vars, scope="post")
y_pred = sess.run(y_post, feed_dict={X: X_train})

fig, ax = plt.subplots(figsize=(8, 8))

ax = sns.kdeplot(X_train[:, 0], y_train, cmap="Greens")
ax = sns.kdeplot(X_train[:, 0], y_pred, cmap="Blues")

plt.show()
