from .imports import *

def train_model(model, optim, train_dl, val_ds, max_epochs, early_stopping=100, device="cuda"):
    epoch_iter = tqdm(range(max_epochs), total=max_epochs, position=0)
    iter = 0
    train_losses = np.zeros(max_epochs)
    val_losses = np.zeros(max_epochs)
    test_losses = np.zeros(max_epochs)
    best_metric = np.inf
    best_model = None
    best_epoch = 0
    num_epochs_since_best = 0
    model.to(device)


    for epoch in epoch_iter:
        epoch_loss = 0
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            posterior_params = model(x)
            loss = model.prediction_loss_from_forward(posterior_params, y)
            posterior = model.posterior_from_forward(posterior_params)

            y_hat = posterior.mean
            mse = F.mse_loss(y_hat, y).item()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * x.shape[0]
        epoch_loss /= len(train_dl.dataset)
        #######################################
        ############# Validation ##############
        #######################################
        model.eval()
        with torch.no_grad():
            x, y = val_ds[:]
            x, y = x.to(device), y.to(device)
            val_posterior = model.posterior(x)
            # compute negative log likelihood
            val_metric = -val_posterior.log_prob(y).mean().item()

        val_losses[epoch] = val_metric
        if val_metric < best_metric:
            best_metric = val_metric
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            num_epochs_since_best = 0
        else:
            num_epochs_since_best += 1            
        if early_stopping is not None and num_epochs_since_best > early_stopping:
            break
        epoch_iter.set_postfix(epoch_loss = epoch_loss, best_metric=best_metric, val_metric=val_metric)
    model.load_state_dict(best_model)
    return model
