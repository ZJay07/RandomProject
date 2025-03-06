import numpy as np
import torch
from math import comb


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
            loss = loss_fn(y_preds, t_batch)
            loss.backward()
            with torch.no_grad():
                w -= learning_rate * w.grad
        # loss reporting
        if epoch % 10 == 0:
            preds = []
            for x in x_train:
                y_pred = logistic_fun(w, M, x)
                preds.append(y_pred)
            preds = torch.stack(preds)
            train_loss = loss_fn(preds, t_train)
            print(f"Epoch: {epoch}, Loss: {train_loss.item()}")
    return w.detach()


def main():
    # setting the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # generate synthetic data
    M = 2
    D = 5
    exp = get_exp_combinations(D, M)

    # Generate weights
    W_np = np.array([np.sqrt(len(exp) - i) for i in range(len(exp))])
    W = torch.tensor(W_np, dtype=torch.float32)

    # generate training and test datasets.
    N_train = 200
    N_test = 100
    x_train = torch.FloatTensor(N_train, D).uniform_(0.0, 10.0)
    x_test = torch.FloatTensor(N_test, D).uniform_(0.0, 10.0)

    # use `logistic_fun` to generate training and test set
    with torch.no_grad():
        y_train_clean = logistic_fun(W, M, x_train)
        y_test_clean = logistic_fun(W, M, x_test)

    # adding random Gaussian noice
    noise_train = torch.randn(N_train)
    noise_test = torch.randn(N_test)
    y_train_noisy = y_train_clean + noise_train
    y_test_noisy = y_test_clean + noise_test
    # threshold cutoff of 0.5
    t_train = (y_train_noisy >= 0.5).float()
    t_test = (y_test_noisy >= 0.5).float()

    # normalise the data
    x_train = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
    x_test = (x_test - x_test.mean(dim=0)) / x_test.std(dim=0)

    # define the loss functions
    loss_functions = {"cross_entropy": MyCrossEntropy(), "rms": MyRootMeanSquare()}

    for loss_fn_name, loss_fn in loss_functions.items():
        print(f"Training with {loss_fn_name} loss function:")
        for M_val in [1, 2, 3]:
            print(f"Training with Polynomial Order M={M_val}")
            # use `fit_logistic_sgd` to compute optimum weight vector W~ using the training set
            w = fit_logistic_sgd(x_train, t_train, M_val, loss_fn)

            with torch.no_grad():
                y_train_pred = logistic_fun(w, M_val, x_train)
                y_test_pred = logistic_fun(w, M_val, x_test)
                train_pred = (y_train_pred >= 0.5).float()
                train_acc = torch.mean((train_pred == t_train).float())

                test_pred = (y_test_pred >= 0.5).float()
                test_acc = torch.mean((test_pred == t_test).float())
                loss = loss_fn(y_train_pred, t_train)
            print(
                f"M={M_val}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Loss: {loss:.4f}"
            )


if __name__ == "__main__":
    main()
    ## placeholder
    # print("Comment: Accuracy is chosen as the metric because it directly reflects the percentage of correctly classified samples, making it a more interpretable measure for classification compared to the raw loss values.")
