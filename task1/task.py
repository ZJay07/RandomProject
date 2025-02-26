import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


# helper functions
def get_exp_combinations(D, M):
    # returns all combinations of exponents for the polynomial features
    stack = [([], 0, M)]
    results = []

    while stack:
        cur, idx, remaining = stack.pop()

        if idx == D:
            if remaining == 0:
                results.append(cur)
            continue
        if idx < D:
            for i in range(remaining + 1):
                new_cur = cur + [i]
                stack.append((new_cur, idx + 1, remaining - i))
    return results


def logistic_fun(w, m, x):
    def get_poly_features(x, M):
        # returns the polynomial features of x
        N, D = x.shape
        exp = get_exp_combinations(D, M)
        poly_features = []
        for e in exp:
            exp_tensor = torch.tensor(e, dtype=torch.float32, device=x.device)
            term = torch.prod(x**exp_tensor, dim=1, keepdim=True)
            poly_features.append(term)
        return torch.cat(poly_features, dim=1)

    f = get_poly_features(x, m)
    f = torch.matmul(f, w.unsqueeze(1)).squeeze(1)

    # returns the functional value representing prob of y
    y = torch.sigmoid(f)
    return y


## Loss functions
class MyCrossEntropy:
    # optinal: epsilon for stability
    def __call__(self, y, t, epsilon=1e-7):
        loss = -torch.mean(
            t * torch.log(y + epsilon) + (1 - t) * torch.log(1 - y + epsilon)
        )
        return loss


class MyRootMeanSquare:
    # compute the root-mean-square error between predicted class probabilities and targets.
    def __call__(self, y, t):
        loss = torch.sqrt(torch.mean((y - t) ** 2))
        return loss


def fit_logistic_sgt(
    x_train, t_train, M, loss_fn, learning_rate=0.001, minibatch_size=32, epochs=100
):
    """
    compute the optimum weight vector W~ using the training set, by
    minimising the cross-entropy loss. Considering three hyperparameter values, M ele of {1,2,3}, for
    each M, compute the predicted target values y~ for all x in both the training and test sets.
    """
    N, D = x_train.shape
    exp = get_exp_combinations(D, M)
    w = torch.nn.Parameter(torch.randn(len(exp), device=x_train.device) * 0.1)

    dataset = TensorDataset(x_train, t_train)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)
    optimizer = optim.AdamW([w], lr=learning_rate)

    for epoch in range(epochs):
        for x, t in loader:
            optimizer.zero_grad()
            y_pred = logistic_fun(w, M, x)
            loss = loss_fn(y_pred, t)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return w.detach()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # generate synthetic data
    M = 2
    D = 16
    exp = get_exp_combinations(D, M)
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
            # use `fit_logistic_sgt` to compute optimum weight vector W~ using the training set
            w = fit_logistic_sgt(x_train, t_train, M_val, loss_fn)

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
