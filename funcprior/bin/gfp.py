import os
import argparse
from sklearn.metrics import ndcg_score
import funcprior
from funcprior.imports import *
from funcprior.utils import get_gaussian_nll, split_data
from funcprior.datasets.blundell import true_blundell, sample_blundell
from sklearn.linear_model import LinearRegression
from funcprior.models import DKL, FeedForward, FeedForwardMeanVar, VariationalGP, EnsembleMeanVar, initial_values_for_GP
from funcprior.plots import plot_blundell
from funcprior.training import train_model
from funcprior.models.protein import ProtNN, ProtMeanVar
from funcprior.models.protein import ProtNN, ProtMeanVar, ProtMean
from funcprior.models import EnsembleMean
import scipy.stats
import sklearn.metrics
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score
from funcprior.posterior import functional_posterior_normal
import random

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

DATASETS = (Path(__file__).parent / "../../datasets").resolve()

import seaborn as sns

sns.set(font='serif')
sns.set_style(
    "white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
device = "cpu"


def get_gfp_metrics(dist, y_true, thresh=-1.25):
    y_true = np.asarray(y_true)
    y_true_labels = (y_true > -1.25).astype(np.int)

    y_hat = dist.mean.cpu()
    y_hat_labels = (y_hat.numpy() > -1.25).astype(np.int)

    log_likelihood = dist.log_prob(torch.tensor(y_true).to(device)).mean().item()
    mse = mean_squared_error(y_hat, y_true)
    mae = mean_absolute_error(y_hat, y_true)
    spearman = scipy.stats.spearmanr(y_hat, y_true).correlation

    accuracy = accuracy_score(y_true_labels, y_hat_labels)
    auc = roc_auc_score(y_true_labels, y_hat_labels)

    ndcg = ndcg_score((y_true + 2.5).reshape(1, -1), y_hat.reshape(1, -1))

    y_hat = y_hat.reshape(-1).numpy()
    y_true = y_true.reshape(-1)
    return (log_likelihood, np.sqrt(mse), mae, spearman, accuracy, auc, ndcg,
            y_hat, y_true)


def get_stacking_features(model, X, prior_preds):
    dist = posterior(model, X)
    y_hat_model = dist.mean.cpu().numpy()
    X_stack = np.stack((y_hat_model, prior_preds), axis=1)
    return X_stack


def posterior(model, X):
    model.eval()
    with torch.no_grad():
        dist = model.posterior(X.to(device))
    return dist


def estimate_noise_sd(model, X, y):
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(X.to(device))
    rmse = torch.sqrt(torch.mean((y_hat - y.to(device))**2)).item()
    return rmse


def main():
    args = parse_args()
    results = []
    for fold in range(10):
        fn = DATASETS / 'gfp/avgfp.csv'
        df = pd.read_csv(fn)

        seqs = df.seq.tolist()
        seqs = [[aa for aa in s] for s in seqs]
        alphabet = sorted(set([aa for s in seqs for aa in s]))
        aa_to_i = {aa: i for i, aa in enumerate(alphabet)}

        seqs_tok = np.asarray([[aa_to_i[aa] for aa in s] for s in seqs])
        fitness = df.log_fitness.to_numpy()
        rosetta = df.rosetta_prediction.to_numpy()
        n_mut = df.n_mut.to_numpy()

        num_train_val = args.num_train_val
        idxs_less_than_2 = np.where(n_mut <= 2)[0]
        idxs_greater_than_2 = np.where(n_mut > 2)[0]
        shuffled_idxs_less_than_2 = np.random.permutation(idxs_less_than_2)
        train_val_idxs = shuffled_idxs_less_than_2[:num_train_val]

        test_idxs = np.concatenate(
            (shuffled_idxs_less_than_2[num_train_val:], idxs_greater_than_2))

        X_train_val = seqs_tok[train_val_idxs]
        y_train_val = fitness[train_val_idxs]
        rosetta_train_val = rosetta[train_val_idxs]
        n_mut_train_val = n_mut[train_val_idxs]

        (X_train, y_train), (X_val,
                             y_val), (train_idxs,
                                      val_idxs) = split_data(X_train_val,
                                                             y_train_val,
                                                             x_dtype=torch.int)
        rosetta_train = rosetta_train_val[train_idxs]
        rosetta_val = rosetta_train_val[val_idxs]
        n_mut_train = n_mut_train_val[train_idxs]
        n_mut_val = n_mut_train_val[val_idxs]

        X_test = seqs_tok[test_idxs]
        y_test = fitness[test_idxs]
        rosetta_test = rosetta[test_idxs]
        n_mut_test = n_mut[test_idxs]

        X_test_tens = torch.tensor(X_test, dtype=torch.int)
        y_test_tens = torch.tensor(y_test, dtype=torch.float)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        test_ds = TensorDataset(X_test_tens, y_test_tens)

        bs = 64 * 2
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

        #model = ProtMean([100,100], dropout=0.0, emb_dim=20).to(device)
        model = ProtMean([100, 100], dropout=0.0, emb_dim=20).to(device)
        lr = 1e-4

        #optim = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': 1e-6}], lr=lr)
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
                            device=device)
        rmse = estimate_noise_sd(model, X_val, y_val)

        dist = posterior(model, X_test_tens)
        dist = Normal(dist.mean, scale=rmse)
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)
        results.append(
            ("NN", log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg))
        print("NN Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")
        print()

        y_train_val_labels = (np.asarray(y_train_val) > -1.5).astype(np.int)
        threshs = []
        for thresh in np.linspace(rosetta_train_val.min(),
                                  rosetta_train_val.max(), 200):
            rosetta_label = (rosetta_train_val >= thresh).astype(np.int)
            acc = accuracy_score(rosetta_label, y_train_val_labels)
            acc = roc_auc_score(y_train_val_labels, rosetta_label)
            threshs.append((thresh, acc))
        thresh = sorted(threshs, key=lambda x: -x[1])[0][0]

        num_ensembles = 5
        models = []
        for model_idx in range(num_ensembles):
            #model = ProtMean([100,100], dropout=0.0, emb_dim=20).to(device)
            model = ProtMean([100, 100], dropout=0.0, emb_dim=20).to(device)
            lr = 1e-4
            optim = torch.optim.Adam([{
                'params': model.parameters(),
                'weight_decay': 1e-4
            }],
                                     lr=lr)
            #optim = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': 1e-6}], lr=lr)
            model = train_model(model,
                                optim,
                                train_dl,
                                val_ds,
                                max_epochs=2000,
                                early_stopping=10,
                                device=device)
            models.append(model)
        model = EnsembleMean(models)
        rmse = estimate_noise_sd(model, X_val, y_val)
        bnn_rmse = rmse
        dist = posterior(model, X_test_tens)
        dist = Normal(loc=dist.mean, scale=torch.sqrt(dist.scale**2 + rmse**2))
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)
        results.append(
            ("BNN", log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg))
        print("BNN Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")
        print()

        prior_hat_val = np.asarray(
            [1.0 if ros > thresh else 0. for ros in rosetta_val])
        X_val_stack = get_stacking_features(model, X_val, prior_hat_val)
        stacker = LinearRegression()
        stacker.fit(X_val_stack, y_val)
        y_hat = stacker.predict(X_val_stack)
        rmse = np.sqrt(np.mean((y_hat - y_val.numpy())**2))

        prior_hat_test = np.asarray(
            [1.0 if ros > thresh else 0. for ros in rosetta_test])
        X_test_stack = get_stacking_features(model, X_test_tens,
                                             prior_hat_test)
        y_hat = stacker.predict(X_test_stack)

        dist = Normal(loc=torch.tensor(y_hat).to(device),
                      scale=torch.tensor(rmse).to(device))
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)
        results.append(("Stacking_Rosetta", log_likelihood, rmse, mae,
                        spearman, accuracy, auc, ndcg))
        print(log_likelihood)
        print("Stacking Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")
        print()

        prior_hat_val = np.asarray([0.0 for ros in rosetta_val])
        X_val_stack = get_stacking_features(model, X_val, prior_hat_val)
        stacker = LinearRegression()
        stacker.fit(X_val_stack, y_val)
        y_hat = stacker.predict(X_val_stack)
        rmse = np.sqrt(np.mean((y_hat - y_val.numpy())**2))

        prior_hat_test = np.asarray([0.0 for ros in rosetta_test])
        X_test_stack = get_stacking_features(model, X_test_tens,
                                             prior_hat_test)
        y_hat = stacker.predict(X_test_stack)

        dist = Normal(loc=torch.tensor(y_hat).to(device),
                      scale=torch.tensor(rmse).to(device))
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)
        results.append(("Stacking_Uniform", log_likelihood, rmse, mae,
                        spearman, accuracy, auc, ndcg))
        print(log_likelihood)
        print("Stacking Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")
        print()

        mode = -2.5
        mode = y_train_val[y_train_val < -1.5].mean()
        X_train, y_train = train_ds[:]
        X_val, y_val = val_ds[:]
        y_train_val = torch.cat((y_train, y_val))
        lower_mode = y_train_val[y_train_val < -1.5].mean().item()
        X_val, y_val = val_ds[:]
        with torch.no_grad():
            model.eval()
            w_post = model.posterior(X_val.to(device))
        data = []
        rmse = bnn_rmse
        for sigma in np.linspace(0.1, 10, 100):
            prior_mean = torch.tensor(mode.item()).to(device)
            prior_sigma = torch.tensor(sigma).to(device)
            prior = torch.distributions.Normal(prior_mean, prior_sigma)
            f_post = functional_posterior_normal(w_post, prior)
            f_post = Normal(loc=f_post.mean,
                            scale=torch.sqrt(f_post.scale**2 + rmse**2))
            loglik = f_post.log_prob(y_val.to(device)).mean().item()
            mse = mean_squared_error(f_post.mean.detach().cpu().numpy(), y_val)
            #data.append((sigma, mse))
            data.append((sigma, -loglik))
        best_sigma = sorted(data, key=lambda x: x[1])[0][0]

        prior_mean = torch.ones_like(y_test_tens) * mode.item()
        prior_sigma = torch.ones_like(y_test_tens) * best_sigma
        prior = torch.distributions.Normal(prior_mean.to(device),
                                           prior_sigma.to(device))

        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(prior, y_test_tens)
        results.append(("Prior_Uniform", log_likelihood, rmse, mae, spearman,
                        accuracy, auc, ndcg))
        print("Prio Uniform Results:")
        print(f"Sigma: {best_sigma}")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")

        prior_mean = torch.tensor(mode.item()).to(device)
        prior_sigma = torch.tensor(best_sigma).to(device)
        prior = torch.distributions.Normal(prior_mean, prior_sigma)

        dist = posterior(model, X_test_tens)
        dist = functional_posterior_normal(dist, prior)
        rmse = bnn_rmse
        dist = Normal(loc=dist.mean, scale=torch.sqrt(dist.scale**2 + rmse**2))
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)
        results.append(("BNN_Prior", log_likelihood, rmse, mae, spearman,
                        accuracy, auc, ndcg))
        print("BNN Uniform Results:")
        print(f"Sigma: {best_sigma}")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")

        sigmas = []
        dist = posterior(model, X_val.to(device))
        sigma_1 = best_sigma
        for sigma_2 in np.linspace(0.1, 10, 100):
            prior_mean = torch.tensor([lower_mode
                                       for ros in rosetta_val]).to(device)
            prior_sigma = torch.tensor([
                sigma_1 if ros > thresh else sigma_2 for ros in rosetta_val
            ]).to(device)
            prior = torch.distributions.Normal(prior_mean, prior_sigma)
            tmp_dist = functional_posterior_normal(dist, prior)
            (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
             y_true) = get_gfp_metrics(tmp_dist, y_val)
            mse = mean_squared_error(tmp_dist.mean.detach().cpu().numpy(),
                                     y_val)
            sigmas.append((sigma_1, sigma_2, mse))
            #sigmas.append((sigma_1, sigma_2, -log_likelihood))
        sigma_1, sigma_2, _ = sorted(sigmas, key=lambda x: x[2])[0]

        ##########################################
        ## TODO Figure out what is happening here
        ##########################################
        prior_mean = torch.tensor([lower_mode for ros in rosetta_test]).to(device)
        prior_sigma = torch.tensor([
            sigma_1 if ros > thresh else sigma_2 for ros in rosetta_test
        ]).to(device)
        prior = torch.distributions.Normal(prior_mean, prior_sigma)
        dist = posterior(model, X_test_tens)
        dist = functional_posterior_normal(dist, prior)
        rmse = bnn_rmse
        dist = Normal(loc=dist.mean, scale=torch.sqrt(dist.scale**2 + rmse**2))
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)

        prior_mean = torch.tensor([lower_mode for ros in rosetta_test]).to(device)
        prior_sigma = torch.tensor([
            sigma_1 if ros > thresh else sigma_2 for ros in rosetta_test
        ]).to(device)
        prior = torch.distributions.Normal(prior_mean, prior_sigma)
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(prior, y_test_tens)
        results.append(("Prior_Rosetta", log_likelihood, rmse, mae, spearman,
                        accuracy, auc, ndcg))
        print("Prior Rosetta Results:")
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")

        dist = posterior(model, X_test_tens)
        dist = functional_posterior_normal(dist, prior)
        rmse = bnn_rmse
        dist = Normal(loc=dist.mean, scale=torch.sqrt(dist.scale**2 + rmse**2))
        (log_likelihood, rmse, mae, spearman, accuracy, auc, ndcg, y_hat,
         y_true) = get_gfp_metrics(dist, y_test_tens)
        results.append(("BNN_Prior_Rosetta", log_likelihood, rmse, mae,
                        spearman, accuracy, auc, ndcg))
        print("BNN Rosetta Results:")
        print(sigma_1, sigma_2)
        print(f"Log Lik:  {log_likelihood: .2f}")
        print(f"Spearman: {spearman: .2f}")
        print(f"RMSE:     {rmse: .2f}")
        print(f"MAE:      {mae: .2f}")
        print(f"Accuracy: {accuracy: .2f}")
        print(f"AUC:      {auc: .2f}")

    df = pd.DataFrame(results,
                      columns=[
                          "model", "loglik", "rmse", "mae", "spearman",
                          "accuracy", "auc", "ndcg"
                      ])
    out_fn = f'gfp_results_{args.num_train_val}.csv'
    df.to_csv(out_fn, index=False)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-val", type=int, default=3000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
