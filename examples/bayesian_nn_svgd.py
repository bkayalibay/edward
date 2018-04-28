import tqdm
import edward as ed
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from edward.models import Normal, Empirical


sns.set_style('darkgrid')

N = 100
D = 1
H = 10
P = 20  # num particles
T = 10000  # num grad updates


def build_toy_dataset(N=40, noise_std=0.1):
    D = 1
    X = np.concatenate([np.linspace(0, 2, num=N / 2),
                        np.linspace(6, 8, num=N / 2)])
    y = np.cos(X) + np.random.normal(0, noise_std, size=N)
    X = (X - 4.0) / 4.0
    X = X.reshape((N, D))
    return X, y


def neural_network(X):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])


ed.set_seed(42)

# DATA
X_train, y_train = build_toy_dataset(N)

# MODEL
with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1]), name="W_1")
    b_0 = Normal(loc=tf.zeros(H), scale=tf.ones(H), name="b_0")
    b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_1")

    X = tf.placeholder(tf.float32, [None, D], name="X")
    y_ph = tf.placeholder(tf.float32, [None], name="y_ph")
    y = Normal(loc=neural_network(X), scale=0.1, name="y")

# INFERENCE
with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
        # qW_0 = Empirical(params=tf.Variable(
        #     tf.tile(tf.random_normal([1, D, H]), [P, 1, 1]),
        #     name="qW_0"))
        qW_0 = Empirical(params=tf.Variable(
             tf.random_normal([P, D, H]),
             name="qW_0"))
    with tf.variable_scope("qW_1"):
        # qW_1 = Empirical(params=tf.Variable(
        #     tf.tile(tf.random_normal([1, H, 1]), [P, 1, 1]),
        #     name="qW_1"))
        qW_1 = Empirical(params=tf.Variable(
             tf.random_normal([P, H, 1]),
             name="qW_1"))
    with tf.variable_scope("qb_0"):
        qb_0 = Empirical(params=tf.Variable(
            tf.zeros([P, H]), name="qb_0"))
    with tf.variable_scope("qb_1"):
        qb_1 = Empirical(params=tf.Variable(
            tf.zeros([P, 1]), name="qb_1"))

latent_vars = {W_0: qW_0, b_0: qb_0,
               W_1: qW_1, b_1: qb_1}
# scale = {z: 0.0 for z in latent_vars}
scale = {}


def get_latent_vars(i):
    latent_dict = {}
    for z, qz in latent_vars.items():
        latent_dict[z] = qz.params[i]
    return latent_dict


y_post_particles = [ed.copy(y, get_latent_vars(i), scope='post_particles_' + str(i))
                    for i in range(P)]


def median(x):
    x = tf.reshape(x, (-1,))
    half = int(x.shape[0]) // 2
    top_half, _ = tf.nn.top_k(x, half)
    return top_half[-1]


def rbf(x, lengthscale=-1.0, variance=1.0):
    """rbf(x, y) = variance * exp(-0.5*((x-y)/lengthscale)**2)"""
    if lengthscale == -1:
        rep_x = tf.tile(x[:, :, tf.newaxis], [1, 1, x.shape[0]])
        rep_x = tf.transpose(rep_x, (1, 0, 2))
        rep_x2 = tf.transpose(rep_x, (0, 2, 1))
        pdist = tf.sqrt(tf.reduce_sum(tf.square(rep_x - rep_x2), axis=0))

        med = tf.stop_gradient(median(pdist))
        lengthscale = tf.sqrt(0.5 * (med / tf.log(tf.cast(tf.shape(pdist)[0]+1, tf.float32))))

    return ed.rbf(x, lengthscale=lengthscale, variance=variance)


def kernel(x):
    return ed.rbf(x, lengthscale=1., variance=1.)


inference = ed.SVGD(latent_vars, data={y: y_ph}, kernel_fn=rbf)
inference.initialize(optimizer=tf.train.AdamOptimizer(0.001),
                     scale=scale,
                     logdir='logs')

sess = ed.get_session()
sess.run(tf.global_variables_initializer())

train_loop = tqdm.trange(T)
try:
    for _ in train_loop:
        info_dict = inference.update({X: X_train, y_ph: y_train})
        train_loop.set_description("ave. -log p = {}".format(info_dict["loss"]))
except KeyboardInterrupt:
    pass

interval = np.linspace(X_train.min()-0.5, X_train.max()+0.5, 100)

_, axs = plt.subplots((P+2) // 4 + (P+2) % 4, 4)
axs = axs.flatten()

post_samples = []
for i in range(P):
    ax = axs[i]
    y_post_p_i = y_post_particles[i]
    values = sess.run(y_post_p_i, feed_dict={X: interval[:, np.newaxis]})
    post_samples.append(values)
    ax.scatter(X_train, y_train, c='purple', alpha=0.6)
    ax.scatter(interval, values, alpha=0.6)
    ax.set_title("Prediction of particle {}".format(i+1))

axs[P].scatter(X_train, y_train, c='purple', alpha=0.6)
axs[P].scatter(interval, np.mean(post_samples, axis=0), alpha=0.6)
axs[P].set_title("Mean of particle predictions")


def get_mean_dict():
    latent_dict = {}
    for z, qz in latent_vars.items():
        latent_dict[z] = qz.mean()
    return latent_dict


y_post_particles_mean = ed.copy(y, get_mean_dict(), scope="y_post_particles_mean")
particles_mean_prediction = sess.run(y_post_particles_mean,
                                     feed_dict={X: interval[:, np.newaxis]})

axs[P+1].scatter(X_train, y_train, c='purple', alpha=0.6)
axs[P+1].scatter(interval, particles_mean_prediction, alpha=0.6)
axs[P+1].set_title("Prediction of particles mean")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# plt.tight_layout()
plt.show()
