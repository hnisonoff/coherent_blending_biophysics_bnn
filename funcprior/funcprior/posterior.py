from .imports import torch

def functional_posterior_normal(w_post, prior):
    f_post_mean = ((w_post.mean * (1 / w_post.variance)) + (prior.mean * (1 / prior.variance))) / ((1 / w_post.variance) + (1 / prior.variance))
    f_post_var = 1 / ((1 / w_post.variance) + (1 / prior.variance))
    f_post = torch.distributions.Normal(f_post_mean, torch.sqrt(f_post_var))
    return f_post


