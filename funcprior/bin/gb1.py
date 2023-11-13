import os
import argparse
import funcprior
from funcprior.imports import *
from funcprior.utils import get_gaussian_nll, split_data
from funcprior.datasets.blundell import true_blundell, sample_blundell
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
from funcprior.models import DKL, FeedForward, FeedForwardMeanVar, VariationalGP, EnsembleMeanVar, initial_values_for_GP
from funcprior.plots import plot_blundell
from funcprior.training import train_model
from funcprior.models.protein import ProtNN, ProtMeanVar, ProtMean
import scipy.stats
import sklearn.metrics
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score
from funcprior.posterior import functional_posterior_normal
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

from funcprior.models import EnsembleMean
import seaborn as sns

DATASETS = (Path(__file__).parent / "../../datasets").resolve()

sns.set(font='serif')
sns.set_style(
    "white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
#device = "cuda"
device = "cpu"


def get_stacking_features(model, X, prior_preds):
    dist = posterior(model, X)
    y_hat_model = dist.mean.cpu().numpy()
    X_stack = np.stack((y_hat_model, prior_preds), axis=1)
    return X_stack


def estimate_noise_sd(model, X, y):
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(X.to(device))
    rmse = torch.sqrt(torch.mean((y_hat - y.to(device))**2)).item()
    return rmse


def get_gb1_metrics(dist, y_true):
    y_true = np.asarray(y_true)

    y_hat = dist.mean.cpu()

    log_likelihood = dist.log_prob(torch.tensor(y_true).to(device)).mean().item()
    mse = mean_squared_error(y_hat, y_true)
    mae = mean_absolute_error(y_hat, y_true)
    spearman = scipy.stats.spearmanr(y_hat, y_true).correlation

    y_hat = dist.mean.cpu()
    ndcg = ndcg_score(y_true.reshape(1, -1), y_hat.reshape(1, -1))

    y_hat = y_hat.reshape(-1).numpy()
    y_true = y_true.reshape(-1)
    return (log_likelihood, np.sqrt(mse), mae, spearman, ndcg, y_hat, y_true)


def posterior(model, X):
    model.eval()
    with torch.no_grad():
        dist = model.posterior(X.to(device))
    return dist


def main():
    args = parse_args()
    results = []
    paired_plot_results = []
    for fold in range(10):
        data_fn = Path(DATASETS / 'gb1/gb1-elife-16965-supp1-v4.csv')
        df = pd.read_csv(data_fn)
        df = df.rename(columns={
            'Variants': 'seq',
            'HD': 'n_mut',
            'Fitness': 'fitness'
        })
        n_mut = np.asarray(df.n_mut)

        df = pd.read_csv(DATASETS / 'gb1/AllPredictions.csv')
        df = df.loc[:, ["Variants", "Fitness", "Triad-FixedBb-dG"]]
        df = df.rename(
            columns={
                'Variants': 'seq',
                'HD': 'n_mut',
                'Fitness': 'fitness',
                'Triad-FixedBb-dG': 'triad'
            })
        df["n_mut"] = n_mut

        seqs = df.seq.tolist()
        seqs = [[aa for aa in s] for s in seqs]
        alphabet = sorted(set([aa for s in seqs for aa in s]))
        aa_to_i = {aa: i for i, aa in enumerate(alphabet)}

        seqs_tok = np.asarray([[aa_to_i[aa] for aa in s] for s in seqs])
        fitness = df.fitness.to_numpy()
        n_mut = df.n_mut.to_numpy()
        triad = df.triad.to_numpy()

        num_train_val = args.num_train_val
        train_val_idxs_high = np.random.permutation(
            np.where((fitness > 0.5) & (n_mut <= 2))[0])
        train_val_idxs_low = np.random.permutation(
            np.where((fitness <= 0.5) & (n_mut <= 2))[0])
        train_val_idxs = np.concatenate(
            (train_val_idxs_high[:num_train_val // 2],
             train_val_idxs_low[:num_train_val // 2]))

        #train_val_idxs = np.random.permutation(np.where(n_mut <=2)[0])[:num_train_val]
        X_train_val = seqs_tok[train_val_idxs]
        y_train_val = fitness[train_val_idxs]

        triad_train_val = triad[train_val_idxs]
        n_mut_train_val = n_mut[train_val_idxs]

        #test_idxs = np.asarray([i for i in range(len(fitness)) if i not in train_val_idxs])
        test_idxs = np.concatenate((train_val_idxs_high[num_train_val // 2:],
                                    train_val_idxs_low[num_train_val // 2:]))
        test_idxs = np.concatenate((test_idxs, np.where(n_mut > 2)[0]))

        X_test = seqs_tok[test_idxs]
        y_test = fitness[test_idxs]

        # num_test = 5000
        # test_high = np.random.permutation(np.where(y_test > 0.5)[0])[:num_test//10]
        # test_low = np.random.permutation(np.where(y_test <= 0.5)[0])[:(num_test*9)//10]
        # test_idxs = np.concatenate((test_high, test_low))

        triad_test = triad[test_idxs]
        n_mut_test = n_mut[test_idxs]

        (X_train, y_train), (X_val,
                             y_val), (train_idxs,
                                      val_idxs) = split_data(X_train_val,
                                                             y_train_val,
                                                             x_dtype=torch.int)
        triad_train = triad_train_val[train_idxs]
        triad_val = triad_train_val[val_idxs]

        X_test = torch.tensor(X_test, dtype=torch.int)
        y_test = torch.tensor(y_test, dtype=torch.float)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        test_ds = TensorDataset(X_test, y_test)

        ###########################################################################################################
        bs = 64 * 2
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        model = ProtMean([300], dropout=0.0, emb_dim=20, seq_len=4).to(device)
        lr = 1e-4
        optim = torch.optim.Adam([{
            'params': model.parameters(),
            'weight_decay': 1e-4
        }],
                                 lr=lr)
        model = train_model(model,
                            optim,
                            train_dl,
                            val_ds,
                            max_epochs=2000,
                            early_stopping=10,
                            device=device).eval()
        rmse = estimate_noise_sd(model, X_val, y_val)

        dist = posterior(model, X_test)
        dist = Normal(dist.mean, scale=rmse)

        method = "NN"
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(dist, y_test)
        print("Neural Network Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print()
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot
        ###########################################################################################################

        ###########################################################################################################
        y_train_val_labels = (np.asarray(y_train_val) >= 0.5).astype(int)
        threshs = []
        for thresh in np.linspace(triad_train_val.min(), triad_train_val.max(),
                                  200):
            triad_label = (triad_train_val >= thresh).astype(int)
            acc = accuracy_score(triad_label, y_train_val_labels)
            acc = roc_auc_score(y_train_val_labels, triad_label)
            threshs.append((thresh, acc))
        thresh = sorted(threshs, key=lambda x: -x[1])[0][0]

        method = "BNN"
        num_ensembles = 5
        models = []
        for model_idx in range(num_ensembles):
            model = ProtMean([300], dropout=0.0, emb_dim=20,
                             seq_len=4).to(device)
            lr = 1e-4
            optim = torch.optim.Adam([{
                'params': model.parameters(),
                'weight_decay': 1e-4
            }],
                                     lr=lr)
            model = train_model(model,
                                optim,
                                train_dl,
                                val_ds,
                                max_epochs=2000,
                                early_stopping=10, device=device)
            models.append(model)
        model = EnsembleMean(models).eval()
        rmse = estimate_noise_sd(model, X_val, y_val)
        bnn_rmse = rmse
        dist = posterior(model, X_test)
        dist.scale = torch.sqrt(dist.scale**2 + rmse**2)
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(dist, y_test)
        print("BNN Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot
        ###########################################################################################################

        ###########################################################################################################
        prior_hat_val = np.asarray(
            [1.0 if ros > thresh else 0. for ros in triad_val])
        X_val_stack = get_stacking_features(model, X_val, prior_hat_val)
        stacker = LinearRegression()
        stacker.fit(X_val_stack, y_val)
        y_hat = stacker.predict(X_val_stack)
        rmse = np.sqrt(np.mean((y_hat - y_val.numpy())**2))

        prior_hat_test = np.asarray(
            [1.0 if ros > thresh else 0. for ros in triad_test])
        X_test_stack = get_stacking_features(model, X_test, prior_hat_test)
        y_hat = stacker.predict(X_test_stack)

        dist = Normal(loc=torch.tensor(y_hat).to(device),
                      scale=torch.tensor(rmse).to(device))
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(dist, y_test)
        method = "Stacking_Rosetta"
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot
        print("Stacking Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print()
        ###########################################################################################################

        ###########################################################################################################
        prior_hat_val = np.asarray([0.0 for ros in triad_val])
        X_val_stack = get_stacking_features(model, X_val, prior_hat_val)
        stacker = LinearRegression()
        stacker.fit(X_val_stack, y_val)
        y_hat = stacker.predict(X_val_stack)
        rmse = np.sqrt(np.mean((y_hat - y_val.numpy())**2))

        prior_hat_test = np.asarray(
            [0.0 if ros > thresh else 0. for ros in triad_test])
        X_test_stack = get_stacking_features(model, X_test, prior_hat_test)
        y_hat = stacker.predict(X_test_stack)

        dist = Normal(loc=torch.tensor(y_hat).to(device),
                      scale=torch.tensor(rmse).to(device))
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(dist, y_test)
        method = "Stacking_Uniform"
        results.append(("Stacking_Uniform", log_likelihood, spearman, rmse,
                        mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot
        print("Stacking Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print()
        ###########################################################################################################

        ###########################################################################################################
        rmse = bnn_rmse
        X_val, y_val = val_ds[:]
        dist = posterior(model, X_val)
        data = []
        for sigma in np.linspace(0.1, 10, 100):
            prior_mean = torch.tensor([0 for ros in triad_val]).to(device)
            prior_sigma = torch.tensor(sigma).to(device)
            prior = torch.distributions.Normal(prior_mean, prior_sigma)
            dist = functional_posterior_normal(dist, prior)
            dist = Normal(loc=dist.mean,
                          scale=torch.sqrt(dist.scale**2 + rmse**2))

            loglik = dist.log_prob(y_val.to(device)).mean().item()
            mse = mean_squared_error(dist.mean.detach().cpu().numpy(), y_val)
            #data.append((sigma, mse))
            data.append((sigma, -loglik))
        best_sigma = sorted(data, key=lambda x: x[1])[0][0]
        print(f"Sigma: {best_sigma}")
        print()
        prior_mean = torch.tensor([0 for ros in triad_test]).to(device)
        prior_sigma = torch.tensor(best_sigma).to(device)
        prior = torch.distributions.Normal(prior_mean, prior_sigma)

        prior_mean = torch.tensor([0 for ros in triad_test]).to(device)
        prior_sigma = torch.tensor(best_sigma).to(device)
        prior = torch.distributions.Normal(prior_mean, prior_sigma)
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(prior, y_test)
        method = "Prior_Uniform"
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot

        print("Prior Uniform Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        ###########################################################################################################

        ###########################################################################################################
        rmse = bnn_rmse
        dist = posterior(model, X_test)
        dist = functional_posterior_normal(dist, prior)
        dist = Normal(loc=dist.mean.cpu(),
                      scale=torch.sqrt(dist.scale**2 + rmse**2).cpu())
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(dist, y_test)
        print("BNN Uniform Prior Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        method = "BNN_Uniform"
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot

        best_sigma_1 = best_sigma

        X_val, y_val = val_ds[:]
        dist = posterior(model, X_val)
        data = []
        rmse = bnn_rmse
        for sigma in np.linspace(0.1, 10, 100):
            prior_mean = torch.tensor([0 for ros in triad_val]).to(device)
            prior_sigma = torch.tensor([
                best_sigma_1 if ros > thresh else sigma for ros in triad_val
            ]).to(device)
            prior = torch.distributions.Normal(prior_mean, prior_sigma)
            dist = functional_posterior_normal(dist, prior)
            dist = Normal(loc=dist.mean,
                          scale=torch.sqrt(dist.scale**2 + rmse**2))

            loglik = dist.log_prob(y_val.to(device)).mean().item()
            mse = mean_squared_error(dist.mean.detach().cpu().numpy(), y_val)
            data.append((sigma, -loglik))
        best_sigma = sorted(data, key=lambda x: x[1])[0][0]
        print(f"Sigma: {best_sigma}")
        print()
        prior_mean = torch.tensor([0 for ros in triad_test]).to(device)
        prior_sigma = torch.tensor([
            best_sigma_1 if ros > thresh else best_sigma for ros in triad_test
        ]).to(device)
        prior = torch.distributions.Normal(prior_mean, prior_sigma)
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(prior, y_test)
        print("Prior Triad Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        method = "Prior_Triad"
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot
        ###########################################################################################################

        ###########################################################################################################
        rmse = bnn_rmse
        dist = posterior(model, X_test)
        dist = functional_posterior_normal(dist, prior)
        dist = Normal(loc=dist.mean.cpu(),
                      scale=torch.sqrt(dist.scale**2 + rmse**2).cpu())
        (log_likelihood, rmse, mae, spearman, ndcg, y_hat,
         y_true) = get_gb1_metrics(dist, y_test)
        print("BNN Triad Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        method = "BNN_Triad"
        results.append(
            (method, log_likelihood, spearman, rmse, mae, ndcg, fold))
        data_paired_plot = list(
            zip(y_hat, y_true, [method] * len(y_hat), [fold] * len(y_hat)))
        paired_plot_results += data_paired_plot
        ###########################################################################################################

    df = pd.DataFrame(results,
                      columns=[
                          "method", "loglik", "spearman", "rmse", "mae",
                          "ndcg", "fold"
                      ])
    df.to_csv(f"gb1_results_{args.num_train_val}.csv", index=False)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-val", type=int, default=500)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
