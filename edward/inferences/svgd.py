from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import numpy as np

import edward as ed
from edward.util import copy


class SVGD:
    """Stein Variational Gradient Descent."""

    def __init__(self, latent_vars, data, kernel_fn):
        """

        Parameters
        ----------

        particles: dict of ed.RandomVariable: tf.Variable (the particles)

        data: dict of ed.RandomVariable: tf.Tensor/ed.RandomVariable

        kernel_fn: callable, must accept particles and return a covariance
                   matrix K.
        """
        self.latent_vars = latent_vars
        self.data = data
        self.kernel_fn = kernel_fn

    def initialize(self, optimizer):
        self.optimizer = optimizer
        self.build_loss_and_gradients()

    def update(self, feed_dict):
        sess = ed.get_session()
        p_log_lik, _ = sess.run([self.p_log_lik, self.train_op],
                                feed_dict=feed_dict)
        return {'p_log_lik': p_log_lik}

    def build_loss_and_gradients(self):
        n_particles = len(list(six.itervalues(self.latent_vars))[0])

        p_log_lik = [0.0] * n_particles
        for i in range(n_particles):
            dict_swap = {}

            for z, particles in six.iteritems(self.latent_vars):
                qz = particles[i]
                dict_swap[z] = qz
                p_log_lik[i] += tf.reduce_sum(z.log_prob(qz))

            for x, qx in six.iteritems(self.data):
                x_copy = copy(x, dict_swap, scope="particles_{}".format(i))
                p_log_lik[i] += tf.reduce_sum(x_copy.log_prob(qx))

        particles = [
            particle_set
            for particle_set in six.itervalues(self.latent_vars)
        ]

        # (n_latent_vars, n_particles, latent_dim) -> (n_particles, -1)
        particles = list(zip(*particles))

        def flatten(xs):
            return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

        def unflatten(flat_grads_set):
            sample_set = particles[0]  # look at any particle set
            var_sizes = [int(np.prod(var.shape)) for var in sample_set]
            var_shapes = [var.shape for var in sample_set]

            unflat_grads_set = []
            for fg_set in tf.split(flat_grads_set,
                                   flat_grads_set.shape[0],
                                   axis=0):
                fg_set = tf.squeeze(fg_set)
                flat_grads = tf.split(fg_set, var_sizes)
                unflat_grads = [tf.reshape(flat_grad, var_shape)
                                for flat_grad, var_shape
                                in zip(flat_grads, var_shapes)]
                unflat_grads_set.append(unflat_grads)
            return unflat_grads_set

        self.p_log_lik = p_log_lik

        flattened_particles = tf.stack(particles)
        flattened_particles = tf.reshape(particles, [len(particles), -1])
        cov = self.kernel_fn(flattened_particles)  # (n_particles, n_particles)

        log_p_gradients = tf.stack(
            [flatten(tf.gradients(log_p, particle_set))
             for log_p, particle_set in zip(p_log_lik, particles)]
        )  # (n_particles, particle_dim)

        # Data term:
        weighted_log_p = tf.matmul(cov, log_p_gradients)  # (n_particles, particles_dim)

        # Entropy term:
        total_ks = tf.reduce_sum(cov, axis=1)
        k_grads = []
        for i, particle_set in enumerate(particles):
            k_grads.append(tf.gradients(total_ks[i], particle_set))  # (n_particles, particles_dim)
        k_grads = tf.reshape(tf.stack(k_grads), [len(particles), -1])

        particle_updates = - (weighted_log_p + k_grads)
        particle_updates = unflatten(particle_updates)
        particle_updates = [var_update for update_set in particle_updates
                            for var_update in update_set]
        particle_vars = [var for var_set in particles
                         for var in var_set]

        optimizer = self.optimizer
        self.train_op = optimizer.apply_gradients(zip(particle_updates,
                                                      particle_vars))
