from abc import ABC, abstractmethod
from ..imports import *
from ..utils import get_gaussian_nll
#from ..quadrature import get_points_weights_jacobian_gausslegendre
import math
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from torch.nn import BatchNorm1d
from torch.distributions import Normal, Categorical, Bernoulli
from sklearn import cluster


#from joint_dkl.imports import *
#from joint_dkl.utils import get_gaussian_nll
def initial_values_for_GP(train_dataset, feature_extractor, n_inducing_points):
    '''
    Taken from DUE github code (thank you!)
    '''
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points)
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_inducing_points,
                                     batch_size=n_inducing_points * 10)
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()


class ConditionalDensityEstimator(ABC, nn.Module):

    @abstractmethod
    def posterior(self, x):
        return

    @abstractmethod
    def posterior_from_forward(self, model_output):
        return


class DKL(ConditionalDensityEstimator, gpytorch.Module):

    def __init__(self, feature_extractor, gp, likelihood, elbo):
        super(DKL, self).__init__()

        self.feature_extractor = feature_extractor
        self.gp = gp
        self.likelihood = likelihood
        self.elbo = elbo

    def forward(self, x):
        features = self.feature_extractor(x)

        return self.gp(features)

    def posterior(self, x):
        if type(self.likelihood) is GaussianLikelihood:
            gpytorch_normal = self.likelihood(self(x))
            posterior = Normal(loc=gpytorch_normal.mean,
                               scale=gpytorch_normal.stddev)
        elif type(self.likelihood) is SoftmaxLikelihood:
            # SoftmaxLikelihood returns probs with shape:
            # (num_likelihood_samples, batch_size, num_categories)
            probs = self.likelihood(self(x)).probs.mean(0)
            posterior = Categorical(probs=probs)
        elif type(self.likelihood) is BernoulliLikelihood:
            probs = self.likelihood(self(x)).probs
            posterior = Bernoulli(probs)
        return posterior

    def posterior_from_forward(self, gp_pred):
        if type(self.likelihood) is GaussianLikelihood:
            gpytorch_normal = self.likelihood(gp_pred)
            posterior = Normal(loc=gpytorch_normal.mean,
                               scale=gpytorch_normal.stddev)
        elif type(self.likelihood) is SoftmaxLikelihood:
            # SoftmaxLikelihood returns probs with shape:
            # (num_likelihood_samples, batch_size, num_categories)
            probs = self.likelihood(gp_pred).probs.mean(0)
            posterior = Categorical(probs=probs)
        elif type(self.likelihood) is BernoulliLikelihood:
            probs = self.likelihood(gp_pred).probs
            posterior = Bernoulli(probs)
        return posterior

    def prediction_loss(self, x, y):
        gp_pred = self(x)
        loss = -(self.elbo(gp_pred, y))
        return loss

    def prediction_loss_from_forward(self, gp_pred, y):
        loss = -(self.elbo(gp_pred, y))
        return loss


class FeedForward(nn.Module):

    def __init__(self, in_dim, hiddens, out_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        dims = [in_dim] + hiddens + [out_dim]
        layers = []
        for i in range(len(hiddens)):
            start = dims[i]
            end = dims[i + 1]
            p = dropout if i < len(dims) - 2 else 0.0
            layer = nn.Linear(start, end)
            if p != 0:
                layers.append(nn.Sequential(layer, nn.ReLU(), nn.Dropout(p=p)))
            else:
                layers.append(nn.Sequential(layer, nn.ReLU()))
        layers.append(nn.Linear(hiddens[-1], out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FeedForwardMeanVar(ConditionalDensityEstimator, nn.Module):

    def __init__(self, in_dim, hiddens, dropout=0.0):
        super(FeedForwardMeanVar, self).__init__()
        self.layers = FeedForward(in_dim, hiddens, 2, dropout=dropout)

    def forward(self, x):
        x = self.layers(x)
        mean = x[:, 0].reshape(-1)
        var = (F.softplus(x[:, 1]).reshape(-1) + 1e-6)
        return mean, var

    def posterior(self, x):
        mean, var = self(x)
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def posterior_from_forward(self, mean_var):
        mean, var = mean_var
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, mean_var, y):
        posterior = self.posterior_from_forward(mean_var)
        nll = -posterior.log_prob(y).mean()
        return nll


class FeedForwardMean(ConditionalDensityEstimator, nn.Module):

    def __init__(self, in_dim, hiddens, dropout=0.0):
        super(FeedForwardMean, self).__init__()
        self.layers = FeedForward(in_dim, hiddens, 1, dropout=dropout)

    def forward(self, x):
        mean = self.layers(x).reshape(-1)
        var = torch.ones_like(mean)
        return mean, var

    def posterior(self, x):
        mean, var = self(x)
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def posterior_from_forward(self, mean_var):
        mean, var = mean_var
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, mean_var, y):
        posterior = self.posterior_from_forward(mean_var)
        nll = -posterior.log_prob(y).mean()
        return nll


class EnsembleMeanVar(ConditionalDensityEstimator):

    def __init__(self, models):
        super(EnsembleMeanVar, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        means, variances = zip(*[model(x) for model in self.models])
        means = torch.stack(means)
        variances = torch.stack(variances)
        mean, variance = self.combine_means_variances(means, variances)
        return mean, variance

    def posterior(self, x):
        mean, variance = self(x)
        posterior = Normal(loc=mean, scale=torch.sqrt(variance))
        return posterior

    def posterior_from_forward(self, mean_var):
        mean, variance = mean_var
        posterior = Normal(loc=mean, scale=torch.sqrt(variance))
        return posterior

    @staticmethod
    def combine_means_variances(means, variances):
        mean = means.mean(dim=0)
        variance = (variances + (means**2)).mean(dim=0) - (mean**2)
        #print(((means ** 2)).mean(dim=0) - (mean ** 2))
        #variance = ((means ** 2)).mean(dim=0) - (mean ** 2)
        return mean, variance

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, mean_var, y):
        posterior = self.posterior_from_forward(mean_var)
        nll = -posterior.log_prob(y).mean()
        return nll


class CategoricalDNN(ConditionalDensityEstimator, nn.Module):

    def __init__(self, model):
        super(CategoricalDNN, self).__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return logits

    def posterior(self, x):
        logits = self(x)
        posterior = Categorical(logits=logits)
        return posterior

    def posterior_from_forward(self, logits):
        posterior = Categorical(logits=logits)
        return posterior

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, logits, y):
        posterior = self.posterior_from_forward(logits)
        nll = -posterior.log_prob(y).mean()
        return nll


class EnsembleCategoricalDNN(CategoricalDNN):

    def __init__(self, models):
        super(EnsembleCategoricalDNN, self).__init__(models)
        self.models = models

    def forward(self, x):
        logits_per_model = [model(x) for model in self.models]
        posterior_per_model = [
            Categorical(logits=logits) for logits in logits_per_model
        ]
        probs_per_model = [
            posterior.probs for posterior in posterior_per_model
        ]
        probs = torch.stack(probs_per_model).mean(dim=0)
        posterior = Categorical(probs=probs)
        return posterior.logits


class VariationalGP(gpytorch.models.ApproximateGP):

    def __init__(self,
                 num_dim,
                 initial_inducing_points,
                 initial_lengthscale,
                 num_outputs=1,
                 use_matern=False,
                 use_zero_mean=False,
                 no_ard=False):
        num_inducing_points = initial_inducing_points.shape[0]
        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])
        #batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points, batch_shape=batch_shape)
        variational_strategy = VariationalStrategy(self,
                                                   initial_inducing_points,
                                                   variational_distribution)
        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs)

        super(VariationalGP, self).__init__(variational_strategy)
        if use_zero_mean:
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.ConstantMean(
                batch_shape=batch_shape)
        ard = None if no_ard else num_dim
        if use_matern:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1 / 2,
                                              ard_num_dims=ard,
                                              batch_shape=batch_shape,
                                              lengthscale_prior=None),
                batch_shape=batch_shape)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ard,
                                           batch_shape=batch_shape,
                                           lengthscale_prior=None),
                batch_shape=batch_shape)
            self.covar_module.base_kernel.lengthscale = initial_lengthscale * torch.ones_like(
                self.covar_module.base_kernel.lengthscale)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ConvFeatureExtractor(nn.Module):

    def __init__(self, feature_dim):
        super(ConvFeatureExtractor, self).__init__()
        hidden = 100
        k_sz = 20
        dropout = 0.0
        self.base_to_onehot = nn.Embedding(4, 4)
        self.base_to_onehot.weight = nn.Parameter(torch.eye(4),
                                                  requires_grad=False)
        self.conv = nn.Conv1d(4, hidden, k_sz)
        self.drop1 = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(250 - k_sz + 1)
        self.drop2 = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, feature_dim)

    def forward(self, x):
        out = self.base_to_onehot(x)
        out = self.conv(out.transpose(1, 2))
        out = self.drop1(out)
        out = self.pool(out).squeeze()
        out = self.drop2(out)
        out = self.lin(out)
        return out

    def forward_from_onehot(self, x):
        out = self.conv(x.transpose(1, 2))
        out = self.drop1(out)
        out = self.pool(out).squeeze()
        out = self.drop2(out)
        out = self.lin(out)
        return out


class FeedForwardSepMeanVar(ConditionalDensityEstimator, nn.Module):

    def __init__(self, in_dim, hiddens, dropout=0.0):
        super(FeedForwardSepMeanVar, self).__init__()
        self.mean_layers = FeedForward(in_dim, hiddens, 1, dropout=dropout)
        self.var_layers = FeedForward(in_dim, hiddens, 1, dropout=dropout)

    def forward(self, x):
        mean = self.mean_layers(x).squeeze()
        var = self.var_layers(x)
        var = (F.softplus(var).reshape(-1) + 1e-6)
        return mean, var

    def posterior(self, x):
        mean, var = self(x)
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def posterior_from_forward(self, mean_var):
        mean, var = mean_var
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, mean_var, y):
        posterior = self.posterior_from_forward(mean_var)
        nll = -posterior.log_prob(y).mean()
        return nll


class MeanVarModel(ConditionalDensityEstimator, nn.Module):

    def __init__(self, model):
        super(MeanVarModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        mean = x[:, 0].reshape(-1)
        var = (F.softplus(x[:, 1]).reshape(-1) + 1e-6)
        return mean, var

    def posterior(self, x):
        mean, var = self(x)
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def posterior_from_forward(self, mean_var):
        mean, var = mean_var
        posterior = Normal(loc=mean, scale=torch.sqrt(var))
        return posterior

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, mean_var, y):
        posterior = self.posterior_from_forward(mean_var)
        nll = -posterior.log_prob(y).mean()
        return nll


class EnsembleMean(ConditionalDensityEstimator):

    def __init__(self, models):
        super(EnsembleMean, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        means, _ = zip(*[model(x) for model in self.models])
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variance = ((means**2)).mean(dim=0) - (mean**2)
        return mean, variance

    def posterior(self, x):
        mean, variance = self(x)
        posterior = Normal(loc=mean, scale=torch.sqrt(variance) + 1e-8)
        return posterior

    def posterior_from_forward(self, mean_var):
        mean, variance = mean_var
        posterior = Normal(loc=mean, scale=torch.sqrt(variance))
        return posterior

    def prediction_loss(self, x, y):
        posterior = self.posterior(x, y)
        nll = -posterior.log_prob(y).mean()
        return nll

    def prediction_loss_from_forward(self, mean_var, y):
        posterior = self.posterior_from_forward(mean_var)
        nll = -posterior.log_prob(y).mean()
        return nll
