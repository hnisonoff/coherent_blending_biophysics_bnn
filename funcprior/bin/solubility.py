import os
import argparse
import sys
from pathlib import Path

SFI_PATH = (Path(__file__).parent / "../../sfi").resolve()
sys.path.append(SFI_PATH.as_posix())
from tdc.single_pred import ADME
from sfi_predictor import SFIPredictor
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score
import torch

from torch.distributions import Normal
import scipy.stats
from funcprior.posterior import functional_posterior_normal
from funcprior.utils import split_data
from funcprior.models import FeedForwardMeanVar

tqdm.pandas()
sns.set()
import deepchem
#import tensorflow as tf

import sys
import os
import deepchem
from deepchem.models import GraphConvModel, WeaveModel

import pandas as pd
from rdkit import Chem
import itertools

DATASETS = (Path(__file__).parent / "../../datasets").resolve()


def get_stacking_features(model, X, prior_preds):
    y_hat_model = model.predict_on_batch(X).reshape(-1)
    X_stack = np.stack((y_hat_model, prior_preds), axis=1)
    return X_stack


def generate_graph_conv_model():
    batch_size = 128
    model = GraphConvModel(1, batch_size=batch_size, mode='regression')
    return model


def get_sfi(smiles, sfi_predictor):
    sfi_score = np.asarray([sfi_predictor.predict_smiles(s) for s in smiles])
    return sfi_score


def get_slope_intercept(sfi, y):
    linear_model = scipy.stats.linregress(sfi, y=y)
    slope, intercept = linear_model.slope, linear_model.intercept
    return slope, intercept


def get_tdc_data():
    data = ADME(name='Solubility_AqSolDB')
    sol_df = data.get_data()
    #splits = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    splits = data.get_split(method="scaffold", frac=[0.6, 0.0, 0.4])
    train_val_smiles = splits['train'].Drug.to_numpy()
    train_val_sol = splits['train'].Y.to_numpy()

    test_smiles = splits['test'].Drug.to_numpy()
    test_sol = splits['test'].Y.to_numpy()
    return (train_val_smiles, train_val_sol), (test_smiles, test_sol)


def get_tdc_data():
    data = ADME(name='Solubility_AqSolDB')
    sol_df = data.get_data()
    #splits = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    splits = data.get_split(method="scaffold", frac=[0.6, 0.0, 0.4])
    train_val_smiles = splits['train'].Drug.to_numpy()
    train_val_sol = splits['train'].Y.to_numpy()

    test_smiles = splits['test'].Drug.to_numpy()
    test_sol = splits['test'].Y.to_numpy()
    return (train_val_smiles, train_val_sol), (test_smiles, test_sol)


def get_bnn_posterior(models, X):
    preds = []
    for model in models:
        y_hat = model.predict_on_batch(X)
        preds.append(y_hat)
    mean_preds = np.asarray(preds).mean(axis=0).reshape(-1)
    std_preds = np.asarray(preds).std(axis=0).reshape(-1)
    orig_dist = torch.distributions.Normal(torch.tensor(mean_preds),
                                           torch.tensor(std_preds))
    return orig_dist


def train_gcn(train_ds, val_ds):
    metric = deepchem.metrics.mean_squared_error
    val_callback = deepchem.models.ValidationCallback(val_ds, 10, metric)
    model = generate_graph_conv_model()
    model.fit(train_ds, 30, callbacks=[val_callback])
    return model


def get_solubility_metrics(dist, y_true):
    y_true = np.asarray(y_true)
    y_hat = dist.mean.cpu()
    log_likelihood = dist.log_prob(torch.tensor(y_true)).mean().item()
    mse = mean_squared_error(y_hat, y_true)
    mae = mean_absolute_error(y_hat, y_true)
    spearman = scipy.stats.spearmanr(y_hat, y_true).correlation
    return (log_likelihood, np.sqrt(mse), mae, spearman)


def main():
    args = parse_args()
    to_keep = args.num_train_val
    results = []
    sfi_fn = SFI_PATH / 'model_logd.txt'
    sfi_predictor = SFIPredictor(sfi_fn)
    for fold in range(10):
        (train_val_smiles, train_val_sol), (test_smiles,
                                            test_sol) = get_tdc_data()

        featurizer = deepchem.feat.ConvMolFeaturizer()
        X_train_val = featurizer.featurize(train_val_smiles)
        y_train_val = train_val_sol
        idxs = np.arange(X_train_val.shape[0])
        idxs = np.random.permutation(idxs)[:to_keep]
        X_train_val = X_train_val[idxs]
        y_train_val = y_train_val[idxs]

        train_val_sfi = get_sfi(train_val_smiles[idxs], sfi_predictor)
        slope, intercept = get_slope_intercept(train_val_sfi, y_train_val)
        (X_train, y_train), (X_val,
                             y_val), (train_idx,
                                      val_idx) = split_data(X_train_val,
                                                            y_train_val,
                                                            as_tensor=False,
                                                            split=0.7)

        train_sfi = train_val_sfi[train_idx]
        val_sfi = train_val_sfi[val_idx]

        X_test = featurizer.featurize(test_smiles)
        y_test = test_sol
        test_sfi = get_sfi(test_smiles, sfi_predictor)
        test_sfi_pred = (test_sfi * slope) + intercept

        val_sfi_pred = (val_sfi * slope) + intercept

        train_ds = deepchem.data.NumpyDataset(X_train, y_train)
        val_ds = deepchem.data.NumpyDataset(X_val, y_val)
        test_ds = deepchem.data.NumpyDataset(X_test, y_test)

        model = train_gcn(train_ds, val_ds)
        y_hat = model.predict_on_batch(X_val).reshape(-1)
        rmse = np.sqrt(np.mean((y_hat - y_val)**2))
        y_hat = model.predict_on_batch(X_test).reshape(-1)
        dist = Normal(loc=torch.tensor(y_hat), scale=torch.tensor(rmse))
        (log_likelihood, rmse, mae,
         spearman) = get_solubility_metrics(dist, y_test)
        results.append(("NN", log_likelihood, rmse, mae, spearman, fold))
        print("NN Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print()

        models = []
        num_ensembles = 5
        for _ in tqdm(range(num_ensembles)):
            model = train_gcn(train_ds, val_ds)
            models.append(model)

        dist = get_bnn_posterior(models, X_val)
        y_hat = dist.mean.numpy()
        rmse = np.sqrt(np.mean((y_hat - y_val)**2))
        bnn_rmse = rmse
        dist = get_bnn_posterior(models, X_test)
        dist = Normal(loc=dist.mean, scale=torch.sqrt(dist.scale**2 + rmse**2))
        (log_likelihood, rmse, mae,
         spearman) = get_solubility_metrics(dist, y_test)
        results.append(("BNN", log_likelihood, rmse, mae, spearman, fold))
        print("BNN Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print()
        dist = get_bnn_posterior(models, X_val)
        y_hat = dist.mean.numpy()
        rmse = np.sqrt(np.mean((y_hat - y_val)**2))
        data = []
        for sigma in np.linspace(0.1, 10, 100):
            prior = torch.distributions.Normal(
                torch.tensor(val_sfi_pred),
                torch.ones_like(dist.mean) * sigma)
            post = functional_posterior_normal(dist, prior)
            post = Normal(loc=post.mean,
                          scale=torch.sqrt(post.scale**2 + rmse**2))
            (log_likelihood, rmse, mae,
             spearman) = get_solubility_metrics(post, y_val)
            data.append((sigma, -log_likelihood))
        best_sigma = sorted(data, key=lambda x: x[1])[0][0]

        rmse = bnn_rmse
        dist = get_bnn_posterior(models, X_test)
        prior = torch.distributions.Normal(
            torch.tensor(test_sfi_pred),
            torch.ones_like(dist.mean) * best_sigma)
        post = functional_posterior_normal(dist, prior)
        post = Normal(loc=post.mean, scale=torch.sqrt(post.scale**2 + rmse**2))
        (log_likelihood, rmse, mae,
         spearman) = get_solubility_metrics(post, y_test)
        results.append(("BNN_SFI", log_likelihood, rmse, mae, spearman, fold))
        print("BNN SFI Prior Results:")
        print("Sigma:", best_sigma)
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print()

        dist = get_bnn_posterior(models, X_val)
        y_hat_model = dist.mean.reshape(-1).numpy()
        prior_preds = val_sfi_pred
        X_val_stack = np.stack((y_hat_model, prior_preds), axis=1)
        stacker = LinearRegression()
        stacker.fit(X_val_stack, y_val)
        y_hat = stacker.predict(X_val_stack)
        rmse = np.sqrt(np.mean((y_hat - y_val)**2))

        dist = get_bnn_posterior(models, X_test)
        y_hat_model = dist.mean.reshape(-1).numpy()
        prior_preds = test_sfi_pred
        X_test_stack = np.stack((y_hat_model, prior_preds), axis=1)
        y_hat = stacker.predict(X_test_stack)
        dist = Normal(loc=torch.tensor(y_hat), scale=torch.tensor(rmse))

        (log_likelihood, rmse, mae,
         spearman) = get_solubility_metrics(dist, y_test)
        results.append(
            ("Stacking Results:", log_likelihood, rmse, mae, spearman, fold))
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Spearman: {spearman: .2f}")

        y_hat = val_sfi_pred
        rmse = np.sqrt(np.mean((y_hat - y_val)**2))

        y_hat = test_sfi_pred
        dist = Normal(loc=torch.tensor(y_hat), scale=torch.tensor(rmse))
        (log_likelihood, rmse, mae,
         spearman) = get_solubility_metrics(dist, y_test)
        results.append(("SFI", log_likelihood, rmse, mae, spearman, fold))
        print("SFI Prior Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print()

    df = pd.DataFrame(
        results,
        columns=["method", "loglik", "rmse", "mae", "spearman", "fold"])
    out_fn = f'solubility_results_{args.num_train_val}.csv'
    df.to_csv(out_fn, index=False)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-val", type=int, default=3000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
