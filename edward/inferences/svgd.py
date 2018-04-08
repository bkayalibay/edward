from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import numpy as np

import edward as ed
from edward.util import copy

from collections import OrderedDict


class SVGD:
    """Stein Variational Gradient Descent."""

    def __init__(self, latent_vars, data, kernel_fn):
        """

        Parameters
        ----------

        particles: dict of ed.RandomVariable: ed.models.Empirical

        data: dict of ed.RandomVariable: tf.Tensor/ed.RandomVariable

        kernel_fn: callable, must accept particles and return a covariance
                   matrix K.
        """
        self.latent_vars = latent_vars
        self.data = data
        self.kernel_fn = kernel_fn

    def initialize(self, optimizer):
        self.optimizer = optimizer
        self.loss, grads = self.build_loss_and_gradients()

        variables = [var
                     for particle_set
                     in self._all_particles
                     for var in ed.get_variables(particle_set)]

        optimizer = self.optimizer
        self.train = optimizer.apply_gradients(zip(grads, variables))

    def update(self, feed_dict):
        sess = ed.get_session()
        loss, _ = sess.run([self.loss, self.train],
                           feed_dict=feed_dict)
        return {'loss': loss}

    def build_loss_and_gradients(self):
        # We want a fixed order of iteration in the loop over particles:
        latent_vars = list(six.iteritems(self.latent_vars))
        if not isinstance(latent_vars, list):
            latent_vars = [latent_vars]

        # Obtain number of particles used:
        sample_set = latent_vars[0][1]
        n_particles = int(sample_set.params.shape[0])

        _, all_particles = list(zip(*latent_vars))

        p_log_lik = [0.0] * n_particles
        dict_swaps = []
        for i in range(n_particles):
            dict_swap = OrderedDict()

            particle_set = []
            for z, particles in latent_vars:
                qz = particles.params[i]
                dict_swap[z] = qz
                p_log_lik[i] += tf.reduce_sum(z.log_prob(qz))

            for x, qx in six.iteritems(self.data):
                x_copy = copy(x, dict_swap, scope="particles_{}".format(i))
                p_log_lik[i] += tf.reduce_sum(x_copy.log_prob(qx))

            dict_swaps.append(dict_swap)

        used_particles = [[particle for particle in six.itervalues(dict_swap)]
                          for dict_swap in dict_swaps]

        def flatten(xs):
            return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

        def unflatten(flat_grads_set):
            sample_set = used_particles[0]  # look at any particle set
            var_sizes = [int(np.prod(var.shape)) for var in sample_set]
            var_shapes = [var.shape for var in sample_set]

            unflat_grads_set = []
            for fg_set in tf.split(flat_grads_set, n_particles):
                fg_set = tf.squeeze(fg_set)
                flat_grads = tf.split(fg_set, var_sizes)
                unflat_grads = [tf.reshape(flat_grad, var_shape)
                                for flat_grad, var_shape
                                in zip(flat_grads, var_shapes)]
                unflat_grads_set.append(unflat_grads)
            return unflat_grads_set

        def stitch_updates(particle_updates):
            particle_updates = list(zip(*particle_updates))
            return [tf.stack(ps) for ps in particle_updates]

        flattened_particles = tf.stack([flatten(ps) for ps in used_particles])
        cov = self.kernel_fn(flattened_particles)  # (n_particles, n_particles)

        log_p_gradients = tf.stack(
            [flatten(tf.gradients(log_p, particle_set))
             for log_p, particle_set in zip(p_log_lik, used_particles)]
        )  # (n_particles, particle_dim)

        # Data term:
        weighted_log_p = tf.matmul(cov, log_p_gradients)  # (n_particles, particles_dim)

        # Entropy term:
        total_ks = tf.reduce_sum(cov, axis=1)
        k_grads = []
        for i, particle_set in enumerate(used_particles):
            k_grads.append(tf.gradients(total_ks[i], particle_set))  # (n_particles, particles_dim)
        k_grads = tf.stack([flatten(k_grad) for k_grad in k_grads])

        particle_updates = - (weighted_log_p + k_grads)
        particle_updates = unflatten(particle_updates)  # (n_particles, latent_vars, latent_dim)
        particle_updates = stitch_updates(particle_updates)

        particle_tensors = [particle_set.params
                            for particle_set
                            in all_particles]

        particle_vars = [var
                         for particle_set
                         in all_particles
                         for var in ed.get_variables(particle_set)]

        # In general, qz.params will not itself be directly connected
        # to one tf.Variable but will possibly be a result of some
        # computation involving some tf.Variable's, e.g. tf.exp(tf.Variable)
        # to enforce non-negativity. In more complicated cases, the particles
        # might even be generated by a neural network. Thus, we must backpropagate
        # the particle updates to the variables that determine them:
        grads = tf.gradients(particle_tensors,
                             particle_vars,
                             grad_ys=particle_updates)

        loss = - tf.reduce_mean(p_log_lik)

        self._all_particles = all_particles

        return loss, grads
