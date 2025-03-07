import numpy as np
import torch
from math import comb, sqrt


# helper functions
def get_exp_combinations(D, M):
    # returns the list of all possible combinations of exponents for the polynomial features
    if D == 1:
        return [
            (M,)
        ]  # base case: return only the highest order for the single variable
    exp = []
    for i in range(M + 1):
        for j in get_exp_combinations(
            D - 1, M - i
        ):  # return all possible combinations for the remaining variables
            exp.append((i,) + j)
    return exp


def polynomial_features(x, M):
    # returns the polynomial features of x
    D = x.shape[0]

    all_exp = []
    for m in range(M + 1):
        all_exp.extend(get_exp_combinations(D, m))

    features = []
    for exp in all_exp:
        # convert tuple of exp to tensor
        exp = torch.tensor(exp, dtype=x.dtype, device=x.device)
        # compute each term of the polynomial
        term = torch.prod(x**exp)
        features.append(term)
    return torch.stack(features)


def logistic_fun(w, M, x):
    # Dimension means num variables, M is the polynomial order
    f = polynomial_features(x, M)
    f = w @ f
    # returns the functional value representing prob of y
    y = torch.sigmoid(f)
    return y


## Loss functions
class MyCrossEntropy:
    # optinal param: epsilon for stability
    def __call__(self, y_pred, y_true, epsilon=1e-7):
        # prevent log(0) by clipping the values
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        # cross-entropy loss
        loss = -torch.mean(
            y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
        )
        return loss


class MyRootMeanSquare:
    # compute the root-mean-square error between predicted class probabilities and targets.
    def __call__(self, y_pred, y_true):
        # square error
        squared_error = torch.square(y_pred - y_true)
        # mean square error
        mse = torch.mean(squared_error)
        # root mean square error
        loss = torch.sqrt(mse)
        return loss


def fit_logistic_sgd(
    x_train, t_train, M, loss_fn, learning_rate=0.001, minibatch_size=32, epochs=100
):
    """
    compute the optimum weight vector W~ using the training set, by
    minimising the cross-entropy loss. Considering three hyperparameter values, M ele of {1,2,3}, for
    each M, compute the predicted target values y~ for all x in both the training and test sets.
    """
    N, D = x_train.shape
    # using combination formula to calculate the number of polynomial terms
    p = sum(comb(D + m - 1, m) for m in range(M + 1))
    # initialize the weight vector
    w = torch.randn(p, requires_grad=True)

    for epoch in range(epochs):
        # shuffle training data
        perm = torch.randperm(N)
        x_train_shuffle = x_train[perm]
        t_train_shuffle = t_train[perm]

        for i in range(0, N, minibatch_size):
            x_batch = x_train_shuffle[i : i + minibatch_size]
            t_batch = t_train_shuffle[i : i + minibatch_size]
            if w.grad is not None:
                w.grad.zero_()
            # compute preds
            y_preds = []
            for x in x_batch:
                y_pred = logistic_fun(w, M, x)
                y_preds.append(y_pred)
            y_preds = torch.stack(y_preds)
            loss = loss_fn(y_preds, t_batch)
            loss.backward()
            with torch.no_grad():
                w -= learning_rate * w.grad
        # loss reporting
        if epoch % 10 == 0:
            y_preds = []
            for x in x_batch:
                y_pred = logistic_fun(w, M, x)
                y_preds.append(y_pred)
            y_preds = torch.stack(y_preds)
            train_loss = loss_fn(y_preds, t_batch)
            print(f"Epoch: {epoch}, Loss: {train_loss.item()}")
    return w.detach()


def generate_sample(w, M, D):
    """
    Generate one (x, t) pair.

    x is uniformly sampled from [-5, 5]^D.
    y is computed using logistic_fun(w, M, x).
    Then a random Gaussian noise (std=1.0) is added to y and thresholded at 0.5 to produce t.
    """
    # Sample x uniformly from [-5, 5]^D:
    x = torch.empty(D).uniform_(-5.0, 5.0)

    # Compute the logistic function probability for x:
    y = logistic_fun(w, M, x)

    # Add Gaussian noise (mean 0, std 1.0)
    noise = torch.randn(1).item()  # single scalar noise
    y_noisy = y.item() + noise

    # Threshold at 0.5 to get binary target:
    t = 1.0 if y_noisy >= 0.5 else 0.0
    return x, t


def main():
    # setting the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # generate synthetic data
    M = 2
    D = 5
    p = sum(comb(D + m - 1, m) for m in range(M + 1))

    # Generate weights
    w_list = [((-1) ** (p - i)) * (sqrt(p - i) / p) for i in range(p)]
    W = torch.tensor(w_list, dtype=torch.float32)

    print("Underlying weight vector w_true:")
    print(W)

    # Generate 200 training samples
    N_train = 200
    train_samples = [generate_sample(W, M, D) for _ in range(N_train)]
    x_train = torch.stack([s[0] for s in train_samples])  # shape: (200, D)
    t_train = torch.tensor(
        [s[1] for s in train_samples], dtype=torch.float32
    )  # shape: (200,)

    # Generate 100 test samples:
    N_test = 100
    test_samples = [generate_sample(W, M, D) for _ in range(N_test)]
    x_test = torch.stack([s[0] for s in test_samples])  # shape: (100, D)
    t_test = torch.tensor(
        [s[1] for s in test_samples], dtype=torch.float32
    )  # shape: (100,)

    assert x_train.shape == (N_train, D)
    assert t_train.shape == (N_train,)
    assert x_test.shape == (N_test, D)
    assert t_test.shape == (N_test,)

    print("Training set x_train shape:", x_train.shape)
    print("Training set t_train shape:", t_train.shape)
    print("Test set x_test shape:", x_test.shape)
    print("Test set t_test shape:", t_test.shape)

    # --- Optimization and Prediction for Different M Values with cross entropy and root mean square---
    loss_fns = [MyCrossEntropy(), MyRootMeanSquare()]

    for loss_fn in loss_fns:
        print("\n --------------------------------------------------")
        print(f"Using loss function: {loss_fn.__class__.__name__}")
        for M_val in [1, 2, 3]:
            print("--------------------------------------------------")
            print(f"Training with polynomial order M = {M_val}")
            w = fit_logistic_sgd(x_train, t_train, M_val, loss_fn)
            print("Optimized weight vector (first 5 elements):", w[:5])

            # Compute predicted target values (probabilities) for the training set.
            y_train_hat = torch.stack([logistic_fun(w, M_val, x) for x in x_train])
            # Compute predicted target values (probabilities) for the test set.
            y_test_hat = torch.stack([logistic_fun(w, M_val, x) for x in x_test])

            print("Train predictions (first 10):", y_train_hat[:10])
            print("Test predictions (first 10):", y_test_hat[:10])


if __name__ == "__main__":
    main()
