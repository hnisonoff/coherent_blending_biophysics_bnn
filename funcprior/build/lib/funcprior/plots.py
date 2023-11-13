from .imports import *
from .datasets.blundell import true_blundell
from .posterior import functional_posterior_normal

plt.rcParams['text.usetex'] = True

def plot_blundell(model,
                  X_train,
                  y_train,
                  prior=None,
                  train=None,
                  xlim=(-2., 2.), 
                  ylim=(-2., 2.)):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set(ylim=ylim, xlim=xlim)
    # plot true curve
    x,y = true_blundell(xlim=xlim)
    sns.lineplot(x=x, y=y, ax=ax, style=True, dashes=[(2,2)], legend=False, color='grey', label="Ground Truth")
    # plot training data
    ax.scatter(X_train.reshape(-1), y_train, color='red', s=10, label="Train Points")
    if train:
        return fig, ax

    model.eval()
    xmin, xmax = xlim
    ymin, ymax = ylim
    # get predictions over full input space
    X_test = np.linspace(xmin, xmax, 1000)
    X_test = torch.tensor(X_test, dtype=torch.float).reshape(-1, 1).cuda()
    with torch.no_grad():
        posterior = model.posterior(X_test)
        if prior:
            posterior = functional_posterior_normal(posterior, prior)
        means = posterior.mean.cpu()
        variances = posterior.variance.cpu()
        sigmas = np.sqrt(variances)

    # plot means
    X_test = X_test.cpu()
    ax.plot(X_test, means, color='#1f77b4', label=r"$\hat{\mu}$")
    # plot variances
    color='skyblue'
    ax.fill_between(X_test.reshape(-1), means-sigmas, means+sigmas, alpha=0.5, color=color, label=r'$\pm \hat{\sigma}$')
    ax.fill_between(X_test.reshape(-1), means-(2*sigmas), means+(2*sigmas), alpha=0.5, color=color, label=r'$\pm 2 \hat{\sigma}$')
    return fig, ax
