from .imports import torch, np

def get_gaussian_nll(y, mean, var):
    pi = torch.tensor(3.141592653589793)
    loglik = 0.5 * (((-(y - mean)**2) / var) - torch.log(var) - torch.log(2*pi))
    return -loglik


class MinMaxNormalizer():
    def __init__(self, y):
        self.y = y
        self.y_max = np.max(y)
        self.y_min = np.min(y)

    def normalize(self, y):
        y_norm =(y - self.y_min)/(self.y_max - self.y_min)
        return y_norm

    def unnormalize(self, y_norm):
        assert(np.all((y_norm >= 0) & (y_norm <= 1)))
        return y_norm * (self.y_max - self.y_min) + self.y_min


def split_data(X, y, split=0.8, as_tensor=True, x_dtype=torch.float, y_dtype=torch.float):
    idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    num_train = int(X.shape[0]*split)
    train_idxs = idxs[:num_train]
    test_idxs = idxs[num_train:]
    if as_tensor:
        if not type(X) is torch.Tensor:
            X = torch.tensor(X, dtype=x_dtype)
        if not type(y) is torch.Tensor:
            y = torch.tensor(y, dtype=y_dtype)
    return (X[train_idxs], y[train_idxs]), (X[test_idxs], y[test_idxs]), (train_idxs, test_idxs)


def holdout_top(X, y, perc=0.1, as_tensor=True):
    # first is idx of largest y
    idxs_dec = np.argsort(-y)  
    num_test = int(y.shape[0] * perc)
    test_idxs = idxs_dec[:num_test]
    train_idxs = idxs_dec[num_test:]
    if as_tensor:
        if not type(X) is torch.Tensor:
            X = torch.tensor(X, dtype=torch.float)
        if not type(y) is torch.Tensor:
            y = torch.tensor(y, dtype=torch.float)
    return (X[train_idxs], y[train_idxs]), (X[test_idxs], y[test_idxs])
