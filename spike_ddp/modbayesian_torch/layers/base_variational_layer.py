# Copyright (C) 2021 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ===============================================================================================
import torch
import torch.nn as nn
import torch.distributions as distributions

# ########
# def log_variational_posterior(self):
#     """Log posterior using a Gaussian and reparameterization trick"""
#     sigma = tf.math.log1p(tf.exp(self.W_rho))
#     gaussian = tf.distributions.Normal(self.W_mu, sigma)
#     log_prob_W = tf.reduce_sum(gaussian.log_prob(self.W))
#
#     sigma = tf.math.log1p(tf.exp(self.b_rho))
#     gaussian = tf.distributions.Normal(self.b_mu, sigma)
#     log_prob_b = tf.reduce_sum(gaussian.log_prob(self.b))
#     return log_prob_W + log_prob_b
#
# def log_mixture_prior(self):
#     """Log scale mixture of two Gaussian densities for the prior"""
#     gaussian1 = tf.distributions.Normal(0., self.prior_sigma1)
#     gaussian2 = tf.distributions.Normal(0., self.prior_sigma2)
#     log_prob_b = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.b) +
#                                                (1 - self.prior_pi) * gaussian2.prob(self.b)))
#     log_prob_W = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.W) +
#                                                (1 - self.prior_pi) * gaussian2.prob(self.W)))
#
#     return log_prob_b + log_prob_W
#
#  def log_variational_posterior(self):
#         """Log posterior using a Gaussian and reparameterization trick for a single layer"""
#         log_prob = 0
#         for layer in self.layers:
#             log_prob += layer.log_variational_posterior()
#         return log_prob
#
# def log_mixture(self):
#     """Log scale mixture of two Gaussian densities for the prior for a single layer"""
#     log_prob = 0
#     for layer in self.layers:
#         log_prob += layer.log_mixture_prior()
#     return log_prob
#
# def kl_loss(self, y, yhat, N):
#     """KL loss (ELBO) that we optimize"""
#     # loss = (self.log_variational_posterior() - self.log_mixture())/steps - self.log_likelihood(y, yhat)
#     loss = (self.log_variational_posterior() - self.log_mixture()) * tf.cast(tf.shape(y)[0], tf.float32) / N - \
#            self.log_likelihood(y, yhat)
#     return loss
#
# def kl_loss_bound(self, yhat, N):
#     """KL loss (ELBO) that we optimize"""
#     # loss = (self.log_variational_posterior() - self.log_mixture())/steps - self.log_likelihood(y, yhat)
#     loss = (self.log_variational_posterior() - self.log_mixture()) * tf.cast(tf.shape(yhat)[0], tf.float32) / N + \
#            tf.nn.relu(self.yhat - self.ub)
#     return loss
#
# #########
class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()
