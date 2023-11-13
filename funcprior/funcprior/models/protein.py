from . import ConditionalDensityEstimator
from ..imports import *

class ProtNN(nn.Module):
    def __init__(self, hiddens, out_dim, num_emb=20, emb_dim=20, seq_len=237, dropout=0.0):
        super(ProtNN, self).__init__()
        # b x seq_len --> # b x seq_len x emb_dim
        emb = nn.Embedding(num_emb, emb_dim)
        # b x seq_len x emb_dim --> # b x (seq_len x emb_dim)
        flat = nn.Flatten()
        layers = [emb, flat]
        in_dim = seq_len * emb_dim
        dims = [in_dim] + hiddens + [out_dim]
        for i in range(len(hiddens)):
            start = dims[i]
            end = dims[i+1]
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


class ProtMeanVar(ConditionalDensityEstimator, nn.Module):
    def __init__(self, hiddens, num_emb=20, emb_dim=20, seq_len=237, dropout=0.0):
        super(ProtMeanVar, self).__init__()
        self.layers = ProtNN(hiddens, 2, num_emb=num_emb, emb_dim=emb_dim, seq_len=seq_len, dropout=dropout)

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


class ProtMean(ConditionalDensityEstimator, nn.Module):
    def __init__(self, hiddens, num_emb=20, emb_dim=20, seq_len=237, dropout=0.0):
        super(ProtMean, self).__init__()
        self.layers = ProtNN(hiddens, 1, num_emb=num_emb, emb_dim=emb_dim, seq_len=seq_len, dropout=dropout)

    def forward(self, x):
        x = self.layers(x)
        mean = x.reshape(-1)
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
