import tqdm
import numpy as np
import tensorflow as tf
import edward as ed

from edward.models import Normal, Gamma, Empirical
from observations import boston_housing


D = 13
H = 32
L = 1
P = 20
B = 32
T = 100


def load_data():
    x_train, _ = boston_housing("./data")
    return x_train[:, :-1], x_train[:, -1]


def get_random_batch():
    indices = np.random.choice(range(len(data)), size=(B,))
    return data[indices], labels[indices]


def mse(preds, labels):
    diff = preds - labels
    return np.mean(np.square(diff))


data, labels = load_data()

x_ph = tf.placeholder(tf.float32, [None, D], name="data")
y_ph = tf.placeholder(tf.float32, [None, L], name="labels")

# Priors:
W1 = Normal(tf.zeros([D, H]), tf.ones([D, H]))
b1 = Normal(tf.zeros([H]), tf.ones([H]))
W2 = Normal(tf.zeros([H, L]), tf.ones([H, L]))
b2 = Normal(tf.zeros([L]), tf.ones([L]))
y_scale = Gamma(1.0, 1.0)

# Approximate posteriors:
qW1 = Empirical(tf.Variable(tf.random_normal([P, D, H]),
                            name="qW1"))
qb1 = Empirical(tf.Variable(tf.random_normal([P, H]),
                            name="qb1"))
qW2 = Empirical(tf.Variable(tf.random_normal([P, H, L]),
                            name="qW2"))
qb2 = Empirical(tf.Variable(tf.random_normal([P, L]),
                            name="qb2"))
qy_scale = Empirical(
    tf.nn.softplus(tf.Variable(tf.random_normal([P]),
                               name="qy_scale")))


def build_nn(weights, biases, activation=tf.nn.relu):
    h = x_ph
    for W, b in zip(weights, biases):
        h = activation(tf.matmul(h, W) + b)
    return h


y_mean = build_nn([W1, W2], [b1, b2])
y = Normal(y_mean, y_scale + 1e-8)

latent_vars = {
    W1: qW1,
    b1: qb1,
    W2: qW2,
    b2: qb2,
    y_scale: qy_scale
}
data = {y: y_ph}

inference = ed.SVGD(latent_vars, data, ed.rbf)
inference.initialize(optimizer=tf.train.AdamOptimizer(0.001))

sess = ed.get_session()
sess.run(tf.global_variables_initializer())

train_loop = tqdm.trange(T)
for _ in train_loop:
    x_batch, y_batch = get_random_batch()
    info_dict = inference.update({x_ph: x_batch, y_ph: y_batch})
    train_loop.set_description("ave. - log p = {}".format(info_dict['loss']))

y_post = ed.copy(y, latent_vars, scope="post")
preds = sess.run(y_post.mean(), {x_ph: data})

error = mse(preds, labels)
print("Mean squared error: {}".format(error))
