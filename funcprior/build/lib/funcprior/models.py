from .imports import *
from .utils import get_gaussian_nll
from abc import ABC, abstractmethod


class SimpleMeanVar(ConditionalDensityEstimator, nn.Module):
    def __init__(self, input_sz, hidden=50, feature_dim=8, use_dropout=False, p=0.2):
        super(SimpleMeanVar, self).__init__()
        self.use_dropout = use_dropout
        self.p = p
        self.fc1 = nn.Linear(input_sz, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, feature_dim, bias=True)
        self.fc3 = nn.Linear(feature_dim, 2, bias=True)
        self.d1 = nn.Dropout(p=self.p)
        self.d2 = nn.Dropout(p=self.p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.d1(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.d2(x)
        x = self.fc3(x)
        mean = x[:, 0].reshape(-1)
        var = (F.softplus(x[:, 1]).reshape(-1) + 1e-6)
        return mean, var

    def nll(self, x, y):
        mean, var = self(x)
        return get_gaussian_nll(y, mean, var)

    def mse(self, x, y):
        mean, var = self(x)
        mse = F.mse_loss(mean, y)
        return mse


class EnsembleNN(nn.Module):
    def __init__(self, model_generator, n_models=10):
        super(EnsembleNN, self).__init__()
        self.n_models = n_models
        self.models = torch.nn.ModuleList([model_generator() for _ in range(n_models)])

    def forward(self, x):
        means, variances = list(zip(*[self.models[i](x) for i in range(self.n_models)]))
        means, variances = torch.stack(means), torch.stack(variances)
        mean, variance = self.combine_means_variances(means, variances)
        return mean, variance

    def means_vars_per_model(self, x):
        means, variances = list(zip(*[self.models[i](x) for i in range(self.n_models)]))
        means, variances = torch.stack(means), torch.stack(variances)
        means, variances = means.transpose(0,1), variances.transpose(0,1)
        return means, variances


    @staticmethod
    def combine_means_variances(means, variances):
        mean = means.mean(dim=0)
        variance = (variances + (means ** 2)).mean(dim=0) - (mean ** 2)
        return mean, variance

    def nll(self, x, y):
        mean, var = self(x)
        return get_gaussian_nll(y, mean, var)

    def mse(self, x, y):
        mean, var = self(x)
        mse = F.mse_loss(mean, y)
        return mse
