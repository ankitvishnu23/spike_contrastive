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
#
# Linear Reparameterization Layers with reparameterization estimator to perform
# variational inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
# required to compute Evidence Lower Bound (ELBO).
#
# @authors: Ranganath Krishnan
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from ..base_variational_layer import BaseVariationalLayer_
import math


class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 save_buffer_sd = False):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparameterization, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))

        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=save_buffer_sd)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=save_buffer_sd)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=save_buffer_sd)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=save_buffer_sd)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=save_buffer_sd)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=save_buffer_sd)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=save_buffer_sd)
            self.register_buffer('prior_bias_sigma', None, persistent=save_buffer_sd)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=save_buffer_sd)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + \
            (sigma_weight * self.eps_weight.data.normal_())
        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)
        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out

class ReparamMLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 save_buffer_sd = False,
                 mlp_depth=1,
                 mlp_hidden_dim=512):
        super(ReparamMLP, self).__init__()
        self.out_features = out_features
        self.mlp_depth = mlp_depth
        # if mlp_depth != 2:
        #     raise
        # self.layer1 = LinearReparameterization(in_features=in_features, out_features=mlp_hidden_dim,
        #                         save_buffer_sd=save_buffer_sd,
        #                         prior_mean=prior_mean,prior_variance=prior_variance)
        # self.layer2 = LinearReparameterization(in_features=mlp_hidden_dim, out_features=out_features,
        #                             save_buffer_sd=save_buffer_sd,
        #                             prior_mean=prior_mean,prior_variance=prior_variance)

        # self.list_layers = [LinearReparameterization(in_features=in_features, out_features=mlp_hidden_dim,
        #                             save_buffer_sd=save_buffer_sd,
        #                             prior_mean=prior_mean,prior_variance=prior_variance)]
        # for _ in range(mlp_depth-2):
        #     self.list_layers += [LinearReparameterization(in_features=mlp_hidden_dim, out_features=mlp_hidden_dim,
        #                                 save_buffer_sd=save_buffer_sd,
        #                                 prior_mean=prior_mean,prior_variance=prior_variance)]
        # self.list_layers += [LinearReparameterization(in_features=mlp_hidden_dim, out_features=out_features,
        #                             save_buffer_sd=save_buffer_sd,
        #                             prior_mean=prior_mean,prior_variance=prior_variance)]
        # self.mlp= nn.Sequential(*list_layers)
        self.layers = nn.ModuleDict()
        for i in range(mlp_depth):
            if i ==0:
                self.layers.update({'layer'+str(i):  LinearReparameterization(in_features=in_features, out_features=mlp_hidden_dim,
                                        save_buffer_sd=save_buffer_sd,
                                        prior_mean=prior_mean,prior_variance=prior_variance) })
            elif 0 < i < mlp_depth-1:
                self.layers.update({'layer'+str(i):  LinearReparameterization(in_features=mlp_hidden_dim, out_features=mlp_hidden_dim,
                                                save_buffer_sd=save_buffer_sd,
                                                prior_mean=prior_mean,prior_variance=prior_variance) })

            elif i == mlp_depth-1:
                self.layers.update({'layer'+str(i):  LinearReparameterization(in_features=mlp_hidden_dim, out_features=out_features,
                                            save_buffer_sd=save_buffer_sd,
                                            prior_mean=prior_mean,prior_variance=prior_variance) })


    def forward(self, x, return_kl = True):
        kl_sum = 0
        for i in range(self.mlp_depth):
            x, kl = self.layers['layer'+str(i)](x)
            kl_sum += kl
            if i < self.mlp_depth-1:
                x = F.relu(x)
        if return_kl:
            return x, kl_sum
        else:
            return x
        # out, kl = self.layer2(out)
        # kl_sum += kl
        # # i = 0
        # # print(x.device)
        # # for layer in self.list_layers:
        # #     print(layer.device)
        # #     out, kl = layer(x)
        # #     kl_sum += kl
        # #     if i < len(self.list_layers):
        # #         out = F.relu(out)
        # #         print(i, "relu-ed")
        # #     i += 1
        # return out, kl_sum



class GaussianPosterior(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        # self.normal = torch.distributions.Normal(0,1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    # def sample(self):
    #     epsilon = self.normal.sample(self.rho.size())
    #     return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).mean()

class MixturePrior(object):
    def __init__(self, pi, mean, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(mean,sigma1)
        self.gaussian2 = torch.distributions.Normal(mean,sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).mean()


class ReparamMixtureLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_sigma1=1,
                 prior_sigma2=0.01,
                 prior_pi=0.5,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 save_buffer_sd = False):
        """

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(ReparamMixtureLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.posterior_mu_init = posterior_mu_init  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features).normal_(mean=self.posterior_mu_init, std=0.1))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features).normal_(mean=self.posterior_rho_init, std=0.1))
        self.mu_bias = Parameter(torch.Tensor(out_features).normal_(mean=self.posterior_mu_init, std=0.1))
        self.rho_bias = Parameter(torch.Tensor(out_features).normal_(mean=self.posterior_rho_init, std=0.1))

        self.posterior_weight = GaussianPosterior(self.mu_weight, self.rho_weight) # Gaussian posterior
        self.posterior_bias = GaussianPosterior(self.mu_bias, self.rho_bias)

        # define prior
        self.prior_mean = prior_mean
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi
        self.prior_weight = MixturePrior(self.prior_pi, self.prior_mean, self.prior_sigma1, self.prior_sigma2)
        self.prior_bias = MixturePrior(self.prior_pi, self.prior_mean, self.prior_sigma1, self.prior_sigma2)

        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features),
                             persistent=save_buffer_sd)
        # self.register_buffer('prior_weight_mu', torch.Tensor(out_features, in_features),
        #                      persistent=save_buffer_sd)
        # self.register_buffer('prior_weight_sigma1', torch.Tensor(out_features, in_features),
        #                      persistent=save_buffer_sd)
        # self.register_buffer('prior_weight_sigma2', torch.Tensor(out_features, in_features),
        #                      persistent=save_buffer_sd)
        self.register_buffer('eps_bias',torch.Tensor(out_features),persistent=save_buffer_sd)
        # self.register_buffer('prior_bias_mu',torch.Tensor(out_features),persistent=save_buffer_sd)
        # self.register_buffer('prior_bias_sigma1',torch.Tensor(out_features),persistent=save_buffer_sd)
        # self.register_buffer('prior_bias_sigma2',torch.Tensor(out_features),persistent=save_buffer_sd)

        # self.prior_weight_mu.fill_(self.prior_mean)
        # self.prior_weight_sigma1.fill_(self.prior_sigma1)
        # self.prior_weight_sigma2.fill_(self.prior_sigma2)
        # self.prior_bias_sigma1.fill_(self.prior_sigma1)
        # self.prior_bias_sigma2.fill_(self.prior_sigma2)
        #
        # self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        # self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        # self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        # self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],std=0.1)

    def forward(self, input):
        sample_weight = self.posterior_weight.mu + \
            (self.posterior_weight.sigma * self.eps_weight.data.normal_())
        sample_bias = self.posterior_bias.mu + \
            (self.posterior_bias.sigma * self.eps_bias.data.normal_())
        # sample_weight = self.posterior_weight.sample()
        # sample_bias = self.posterior_bias.sample()
        if self.training:
            self.log_prior = self.prior_weight.log_prob(sample_weight) + self.prior_bias.log_prob(sample_bias)
            self.log_variational_posterior = self.posterior_weight.log_prob(sample_weight) + self.posterior_bias.log_prob(sample_bias)
        else:
            # sample_weight = self.posterior_weight.mu
            # sample_bias = self.posterior_bias.mu
            self.log_prior = 0.
            self.log_variational_posterior = 0.

        out = F.linear(input, sample_weight, sample_bias)
        return out

class ReparamMixtureMLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_sigma1=1,
                 prior_sigma2=0.1,
                 prior_pi=0.5,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True,
                 save_buffer_sd = False,
                 mlp_depth=1,
                 mlp_hidden_dim=512):
        """

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(ReparamMixtureMLP, self).__init__()
        self.out_features = out_features
        if mlp_depth > 1:
            self.list_layers = [ReparamMixtureLayer(in_features=in_features, out_features=mlp_hidden_dim,
                                            prior_mean=prior_mean, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2,prior_pi=prior_pi,
                                            posterior_mu_init=posterior_mu_init,posterior_rho_init=posterior_rho_init,save_buffer_sd = save_buffer_sd),
                                            nn.ReLU(inplace=True)]
            for _ in range(mlp_depth-2):
                self.list_layers += [ReparamMixtureLayer(in_features=mlp_hidden_dim, out_features=mlp_hidden_dim,
                                                prior_mean=prior_mean, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2,prior_pi=prior_pi,
                                                posterior_mu_init=posterior_mu_init,posterior_rho_init=posterior_rho_init,save_buffer_sd = save_buffer_sd),
                                                nn.ReLU(inplace=True)]
            self.list_layers += [ReparamMixtureLayer(in_features=mlp_hidden_dim, out_features=out_features,
                                            prior_mean=prior_mean, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2,prior_pi=prior_pi,
                                            posterior_mu_init=posterior_mu_init,posterior_rho_init=posterior_rho_init,save_buffer_sd = save_buffer_sd)]
        else:
            self.list_layers = [ReparamMixtureLayer(in_features=in_features, out_features=out_features,
                                            prior_mean=prior_mean, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2,prior_pi=prior_pi,
                                            posterior_mu_init=posterior_mu_init,posterior_rho_init=posterior_rho_init,save_buffer_sd = save_buffer_sd)]

        self.mlp = nn.Sequential(*self.list_layers)

    def log_prior(self):
        log_prob = 0.
        for layer in self.list_layers:
            log_prob += layer.log_prior
        return log_prob

    def log_variational_posterior(self):
        log_prob = 0.
        for layer in self.list_layers:
            log_prob += layer.log_variational_posterior
        return log_prob

    def forward(self, input, targets= None, mc_samples=10,return_kl=True, mean_likelihood=False):
        """ optimize ELBO """
        batch_size = input.shape[0]
        outputs = torch.zeros(mc_samples, batch_size, self.out_features).to(input.device)
        log_priors = torch.zeros(mc_samples).to(input.device)
        log_variational_posteriors = torch.zeros(mc_samples).to(input.device)
        for i in range(mc_samples):
            outputs[i] = self.mlp(input) # posterior is sampled here
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        if return_kl:
            log_prior = log_priors.mean() # mean over mc samples
            log_variational_posterior = log_variational_posteriors.mean()
            nll = F.nll_loss(outputs.log_softmax(dim=-1).mean(0), targets) # if set size_average=False, log_prob = sum() not mean()

            # kl_loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood
            kl_loss = (log_variational_posterior - log_prior)
            return outputs.mean(0), kl_loss, nll
        else:
            # inference
            if outputs.shape[0] != 1:
                raise "set mc_samples = 1 during inference!"

            return outputs[0]
