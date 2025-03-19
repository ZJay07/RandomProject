import numpy as np
import torch
from math import comb, sqrt


## helper function for logistic_fun
def get_exp_combinations(D, M):
    """
    Returns all possible combinations of exponents for polynomial features

    Args:
        D (int): Number of dimensions/variables
        M (int): Maximum polynomial degree/order

    Returns:
        list: List of tuples, where each tuple contains D integers representing
             exponent combinations that sum to at most M
    """
    if D == 1:
        # base case: return only the highest order for the single variable
        return [(M,)]
    exp = []
    for i in range(M + 1):
        for j in get_exp_combinations(D - 1, M - i):
            # return all possible combinations for the remaining variables
            exp.append((i,) + j)
    return exp


def polynomial_features(x, M):
    """
    Computes polynomial features of input tensor up to degree M

    Args:
        x (torch.Tensor): Input tensor of shape (D,) where D is the number of dimensions
        M (int): Max polynomial degree

    Returns:
        torch.Tensor: Tensor of polynomial features with shape (N,) where N is the
                      number of polynomial terms up to degree M
    """
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
    """
    Applies logistic function using polynomial features

    Args:
        w (torch.Tensor): Weight vector of shape (p,) where p is the number of polynomial features
        M (int): Polynomial order/degree
        x (torch.Tensor): Input data of shape (D,) where D is the number of dimensions

    Returns:
        torch.Tensor: Probability value (sigmoid output) as a scalar tensor
    """
    # Dimension means num variables, M is the polynomial order
    f = polynomial_features(x, M)
    f = w @ f
    # returns the functional value representing prob of y
    y = torch.sigmoid(f)
    return y


## Loss functions
class MyCrossEntropy:
    """
    Implements cross-entropy loss for binary classification.

    The loss is calculated as:
    L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    """

    def __call__(self, y_pred, y_true, epsilon=1e-7):
        """
        Compute cross-entropy loss between predictions and targets

        Args:
            y_pred (torch.Tensor): Predicted probabilities, shape (batch_size,)
            y_true (torch.Tensor): Target values (0 or 1), shape (batch_size,)
            epsilon (float, optional): Small constant for numerical stability with a default value of `1e-7`

        Returns:
            torch.Tensor: Scalar tensor containing the mean cross-entropy loss
        """
        # prevent log(0) by clipping the values
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        # cross-entropy loss
        loss = -torch.mean(
            y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
        )
        return loss


class MyRootMeanSquare:
    """
    Computes root mean square error between predictions and targets

    The loss is calculated as sqrt(mean((y_pred - y_true)**2))
    """

    def __call__(self, y_pred, y_true):
        """
        Compute root mean square error between predictions and targets.

        Args:
            y_pred (torch.Tensor): Predicted probabilities, shape (batch_size,)
            y_true (torch.Tensor): Target values (0 or 1), shape (batch_size,)

        Returns:
            torch.Tensor: Scalar tensor containing the root mean square error
        """
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
    Computes the optimum weight vector using stochastic gradient descent by minimising the specified loss function

    Args:
        x_train (torch.Tensor): Training features of shape (N, D) where N is the number of samples
                               and D is the number of dimensions
        t_train (torch.Tensor): Training targets of shape (N,) with values 0 or 1
        M (int): Polynomial order for feature expansion
        loss_fn (callable): Loss function to minimise, defined above
        learning_rate (float, optional): Learning rate for SGD with a default value of `0.001`
        minibatch_size (int, optional): Size of mini-batches for SGD a default value of `32`
        epochs (int, optional): Number of training epochsa default value of `100`

    Returns:
        torch.Tensor: Optimised weight vector w of shape (p,) where p is the number of polynomial features
    """
    N, D = x_train.shape
    # using combination formula to calculate the number of polynomial terms
    p = sum(comb(D + m - 1, m) for m in range(M + 1))
    # init the weight vector
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
    Generate one (x, t) pair of input features and target.

    Args:
        w (torch.Tensor): Weight vector of shape (p,) where p is the number of polynomial features
        M (int): Polynomial order/degree
        D (int): Number of dimensions for the feature vector

    Returns:
        tuple: A pair (x, t) where:
            - x (torch.Tensor): Randomly sampled feature vector of shape (D,)
            - t (float): Binary target (0.0 or 1.0) generated by adding noise to the logistic function
                        output and thresholding at 0.5
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


## helper functions for main()
def compute_accuracy(predictions, ground_truth):
    """
    Threshold predictions at 0.5 and compute accuracy.

    Args:
        predictions (torch.Tensor): Predicted probabilities, shape (batch_size,)
        ground_truth (torch.Tensor): Target values (0 or 1), shape (batch_size,)

    Returns:
        float: Classification accuracy as a value between 0 and 1
    """
    preds = (predictions >= 0.5).float()
    return (preds == ground_truth).float().mean().item()


def compute_true_labels(x_data, w_true, M_true):
    """
    Compute true binary labels using the underlying true model (without noise)

    Args:
        x_data (torch.Tensor): Input features of shape (N, D) where N is the number of samples
                              and D is the number of dimensions
        w_true (torch.Tensor): True weight vector of shape (p,) where p is the number of polynomial features
        M_true (int): True polynomial order/degree of the underlying model

    Returns:
        torch.Tensor: Binary labels (0.0 or 1.0) of shape (N,) by applying the logistic function
                     with the true weights and thresholding at 0.5
    """
    labels = []
    for x in x_data:
        y = logistic_fun(w_true, M_true, x)
        labels.append(1.0 if y.item() >= 0.5 else 0.0)
    return torch.tensor(labels, dtype=torch.float32)


## Main loop
def main():
    """
    Main function to run the logistic regression experiment

    Generates synthetic data, trains logistic regression models with different
    polynomial orders (M=1,2,3) and loss functions, and evaluates their performance
    """
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
    x_train = torch.stack([s[0] for s in train_samples])
    t_train = torch.tensor([s[1] for s in train_samples], dtype=torch.float32)

    # Generate 100 test samples:
    N_test = 100
    test_samples = [generate_sample(W, M, D) for _ in range(N_test)]
    x_test = torch.stack([s[0] for s in test_samples])

    # Compute true labels
    true_train = compute_true_labels(x_train, W, 2)
    true_test = compute_true_labels(x_test, W, 2)
    observed_acc_train = compute_accuracy(t_train, true_train)

    # The observed training data compared to the true classes
    print(f"Observed training data accuracy: {observed_acc_train * 100:.2f}%")

    # Optimisation and Prediction for Different M Values with cross entropy and root mean square
    loss_fns = [MyCrossEntropy(), MyRootMeanSquare()]

    for loss_fn in loss_fns:
        print("\n==================================================")
        print(f"Using loss function: {loss_fn.__class__.__name__}")
        for M_val in [1, 2, 3]:
            print("\n==================================================")
            print(f"Training with polynomial order M = {M_val}")
            w = fit_logistic_sgd(x_train, t_train, M_val, loss_fn)
            print("Optimised weight vector (first 5 elements):", w[:5])

            # Compute predicted target values (probabilities) for the training set.
            y_train_hat = torch.stack([logistic_fun(w, M_val, x) for x in x_train])
            # Compute predicted target values (probabilities) for the test set.
            y_test_hat = torch.stack([logistic_fun(w, M_val, x) for x in x_test])

            # Compute accuracy using the underlying true labels.
            model_acc_train = compute_accuracy(y_train_hat, true_train)
            model_acc_test = compute_accuracy(y_test_hat, true_test)

            print("\nAccuracy analysis:")
            # The model predictions compared to true classes
            print(f"Model prediction accuracy on training set: {model_acc_train * 100:.2f}%")
            print(f"Model prediction accuracy on test set: {model_acc_test * 100:.2f}%")


if __name__ == "__main__":
    print("Other metric appropriate for classificationn task: Accuracy")
    print(
        "Accuracy measures the proportion of correct predictions. It is intuitive, widely used for classification tasks, and effectively evaluates how well our logistic regression model assigns binary labels."
    )
    print("=== Task 1 ===")
    main()
    print(
        "Comment: Accuracy from model predictions shows how well the fitted model recovers the true classes, while accuracy on observed training data reveals the effect of label noise. A significant gap between them indicates that noise substantially distorts the observed labels compared to the true underlying classes."
    )
