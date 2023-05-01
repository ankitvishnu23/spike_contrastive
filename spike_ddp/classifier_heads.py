import torch
import torch.nn as nn
from modbayesian_torch.layers import LinearFlipout, LinearReparameterization, ReparamMixtureMLP, ReparamMLP
# from ensem import LinearEnsem, MimoBackbone

def get_classifier(rep_dim=2048,bayes=False,bayes_mode='reparam',ensem=False,mlp_depth=1, mlp_hidden_dim=2028,\
                    prior_mu=0,prior_sigma=1.,prior_sigma1=1.,prior_sigma2=0.01,prior_pi=0.5,\
                    ensem_samples=1, mimo=False, add_bn=False, num_classes=10):
    if mlp_depth == 1:
        if bayes:
            if bayes_mode == 'flipout':
                print(f"Using Flipout Bayes linear classifier mu={prior_mu}, sig={prior_sigma} [{rep_dim},{num_classes}]")
                classifier = LinearFlipout(in_features=rep_dim, out_features=num_classes,
                                            save_buffer_sd=False,
                                            prior_mean=prior_mu,prior_variance=prior_sigma)
            elif bayes_mode == 'reparam':
                print(f"Using Reparam Bayes linear classifier mu={prior_mu}, sig={prior_sigma} [{rep_dim},{num_classes}]")
                classifier = LinearReparameterization(in_features=rep_dim, out_features=num_classes,
                                            save_buffer_sd=False,
                                            prior_mean=prior_mu,prior_variance=prior_sigma)
            elif bayes_mode == 'mixture':
                print(f"Using Reparam Mixture Bayes linear classifier mu={prior_mu}, sig1={prior_sigma1}, sig2={prior_sigma2}, pi={prior_pi} ")
                classifier = ReparamMixtureMLP(in_features=rep_dim, out_features=num_classes,
                                            prior_mean=prior_mu,prior_sigma1=prior_sigma1,prior_sigma2=prior_sigma2,
                                            prior_pi=prior_pi,mlp_depth=1)
        # elif ensem or mimo:
        #     print(f"Using linear ensemble with {ensem_samples} members [{rep_dim},10]")
        #     classifier = LinearEnsem(M=ensem_samples, in_dim=rep_dim, num_classes=10)
        else:
            print(f"Using linear classifier [{rep_dim},{num_classes}]")
            classifier = nn.Linear(rep_dim, num_classes)

    elif mlp_depth == -1: # for one layer linear, one layer bayes
        print(f"Using 2 deep mlp with 1 linear and 1 bayes")

        if not bayes: raise ValueError("set bayes Flag!")
        list_layers = [nn.Linear(rep_dim, mlp_hidden_dim), nn.ReLU(inplace=True)]
        list_layers += [LinearReparameterization(in_features=mlp_hidden_dim, out_features=num_classes,
                                        save_buffer_sd=False,
                                        prior_mean=prior_mu,prior_variance=prior_sigma)]
        classifier= nn.Sequential(*list_layers)

    else:
        # if ensem or mimo:
            # print(f"Using {mlp_depth} deep ensemble with {ensem_samples} members ")
            # classifier = LinearEnsem(M=ensem_samples, in_dim=rep_dim, num_classes=10, mlp_depth=mlp_depth, mlp_hidden_dim=mlp_hidden_dim)
        if bayes:
            if bayes_mode == 'mixture':
                print(f"Using depth={mlp_depth} Reparam Mixture Bayes linear classifier mu={prior_mu}, sig1={prior_sigma1}, sig2={prior_sigma2}, pi={prior_pi} ")

                classifier = ReparamMixtureMLP(in_features=rep_dim, out_features=num_classes,
                                            prior_mean=prior_mu,prior_sigma1=prior_sigma1,prior_sigma2=prior_sigma2,
                                            prior_pi=prior_pi,mlp_depth=mlp_depth)
            elif bayes_mode == 'flipout':
                raise

            elif bayes_mode == 'reparam':
                print(f"Using {mlp_depth} deep Reparam Bayes linear classifier mu={prior_mu}, sig={prior_sigma} ")
                classifier = ReparamMLP(in_features=rep_dim, out_features=num_classes,
                                            save_buffer_sd=False,
                                            prior_mean=prior_mu,prior_variance=prior_sigma,
                                            mlp_depth=mlp_depth)

        else:
            print(f"Using {mlp_depth} deep classifier")
            list_layers = [nn.Linear(rep_dim, mlp_hidden_dim)]
            if add_bn:
                raise "using batchnorm for mlp classifier. No BN implemented for bayes!"
                list_layers += [nn.BatchNorm1d(mlp_hidden_dim)]

            list_layers += [nn.ReLU(inplace=True)]
            for _ in range(mlp_depth-2):
                list_layers += [nn.Linear(mlp_hidden_dim, mlp_hidden_dim)]
                if add_bn:
                    list_layers += [nn.BatchNorm1d(mlp_hidden_dim)]
                list_layers += [nn.ReLU(inplace=True)]
            list_layers += [nn.Linear(mlp_hidden_dim, num_classes)]
            classifier= nn.Sequential(*list_layers)
    print("Total classifier params: {:.2f}K".format(
                sum(p.numel() for p in classifier.parameters())/1e3))
    return classifier
