import torch
from math import comb, sqrt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task import (
    MyCrossEntropy,
    MyRootMeanSquare,
    generate_sample,
    compute_accuracy,
    compute_true_labels,
    polynomial_features,
)
import numpy as np


## Adaptation of polynomial_features and logistic_fun to work with continuous M
def continuous_polynomial_features(x, M):
    """
    Convert floating point M to int and use polynomial_features defined in task.py

    Args:
        x (torch.Tensor): Input tensor of shape (D,) where D is the number of dimensions
        M (float or torch.Tensor): Polynomial degree, can be float or tensor

    Returns:
        torch.Tensor: Tensor of polynomial features with shape (N,) where N is the number of
                      polynomial terms up to degree int(round(M))
    """
    M_int = int(round(M.item() if isinstance(M, torch.Tensor) else M))
    return polynomial_features(x, M_int)


def continuous_logistic_fun(w, M, x):
    """
    Adapting logistic function to work with continuous M parameter

    Args:
        w (torch.Tensor): Weight vector of shape (p_max,) where p_max is the maximum
                          number of polynomial features possible
        M (float or torch.Tensor): Polynomial order/degree, can be float
        x (torch.Tensor): Input data of shape (D,) where D is the number of dimensions

    Returns:
        torch.Tensor: Probability value (sigmoid output) as a scalar tensor
    """
    f = continuous_polynomial_features(x, M)
    w_used = w[: f.shape[0]]
    f_value = w_used @ f
    y = torch.sigmoid(f_value)
    return y


def fit_logistic_sgd_with_learnable_m(
    x_train,
    t_train,
    loss_fn,
    learning_rate=0.001,
    minibatch_size=32,
    epochs=100,
    m_init=2.0,
):
    """
    Train a logistic regression model with M as a learnable parameter

    Args:
        x_train (torch.Tensor): Training features of shape (N, D) where N is number of samples
                                and D is number of dimensions
        t_train (torch.Tensor): Training targets of shape (N,) with values 0 or 1
        loss_fn (callable): Loss function to minimise
        learning_rate (float, optional): Learning rate for Adam optimiser with a default value of `0.001`
        minibatch_size (int, optional): Size of mini-batches with a default value of `32`
        epochs (int, optional): Number of training epochs with a default value of `100`
        m_init (float, optional): Initial value for the learnable M parameter with a default value of `2.0`

    Returns:
        tuple: A pair (w, M) where:
            - w (torch.Tensor): Optimised weight vector
            - M (float): Optimised polynomial order as a float value
    """
    N, D = x_train.shape

    # Maximum possible M value to determine weight vector size, fixed at 5
    M_max = 5

    # Calculate maximum number of features for largest possible M
    p_max = sum(comb(D + m - 1, m) for m in range(M_max + 1))

    # Init params
    w = torch.zeros(p_max, requires_grad=True)
    M = torch.tensor(m_init, requires_grad=True)

    # Adam optimiser for better convergence
    optimizer = torch.optim.Adam([w, M], lr=learning_rate)

    for epoch in range(epochs):
        # Shuffle training data
        perm = torch.randperm(N)
        x_train_shuffle = x_train[perm]
        t_train_shuffle = t_train[perm]

        batch_losses = []

        for i in range(0, N, minibatch_size):
            x_batch = x_train_shuffle[i : min(i + minibatch_size, N)]
            t_batch = t_train_shuffle[i : min(i + minibatch_size, N)]

            optimizer.zero_grad()

            # Forward pass
            y_preds = []
            for x in x_batch:
                y_pred = continuous_logistic_fun(w, M, x)
                y_preds.append(y_pred)

            y_preds = torch.stack(y_preds)

            # Compute loss
            batch_loss = loss_fn(y_preds, t_batch)
            batch_losses.append(batch_loss.item())

            # small regularisation to prevent extreme M values
            m_reg = 0.01 * M**2
            total_loss = batch_loss + m_reg

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # make sure M stays within reasonable bounds
            with torch.no_grad():
                M.clamp_(0.5, float(M_max) - 0.5)

        # Reporting loss and accuracy
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0

            # Compute predictions for a sample to check accuracy
            sample_size = min(100, N)
            indices = torch.randperm(N)[:sample_size]
            x_sample = x_train[indices]
            t_sample = t_train[indices]

            y_sample_preds = []
            for x in x_sample:
                y_pred = continuous_logistic_fun(w, M, x)
                y_sample_preds.append(y_pred)

            y_sample_preds = torch.stack(y_sample_preds)
            sample_acc = compute_accuracy(y_sample_preds, t_sample)

            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Current M = {M.item():.4f}")
            print(f"Sample accuracy: {sample_acc * 100:.2f}%")

    # Final evaluation
    print("\n========== Final Results ==========")
    print(f"Optimised M value: {M.item():.4f}")

    # Compute full training set accuracy
    y_train_preds = []
    for x in x_train:
        y_pred = continuous_logistic_fun(w, M, x)
        y_train_preds.append(y_pred)

    y_train_preds = torch.stack(y_train_preds)
    train_acc = compute_accuracy(y_train_preds, t_train)
    print(f"Final training accuracy: {train_acc * 100:.2f}%")

    return w.detach(), M.item()


def evaluate_learned_m_model(w, M, x_test, t_test, true_test=None):
    """
    Evaluate the model with learned M value on test data

    Args:
        w (torch.Tensor): Optimised weight vector
        M (float): Optimised polynomial order
        x_test (torch.Tensor): Test features of shape (N_test, D)
        t_test (torch.Tensor): Observed test labels of shape (N_test,)
        true_test (torch.Tensor, optional): True test labels (without noise) of shape (N_test,) with a default value of None

    Returns:
        float: Test accuracy against observed labels (percentage)
    """
    # Compute predictions for test set
    y_test_preds = []
    for x in x_test:
        y_pred = continuous_logistic_fun(w, M, x)
        y_test_preds.append(y_pred)

    y_test_preds = torch.stack(y_test_preds)

    # Compute accuracy against observed test labels
    test_acc = compute_accuracy(y_test_preds, t_test)
    print(f"Test accuracy (vs. observed labels): {test_acc * 100:.2f}%")

    # If true test labels are provided, compute accuracy against them
    if true_test is not None:
        true_acc = compute_accuracy(y_test_preds, true_test)
        print(f"Test accuracy (vs. true labels): {true_acc * 100:.2f}%")

    return test_acc


## Learnable M experiment
def experiment_with_learnable_m():
    """
    Main experiment function for learnable M

    Generates synthetic data with a known true model (in this case M=2), then trains
    multiple models with different loss functions and M initialisations
    The goal is to recover the true model order M through optimisation

    Returns:
        None
    """
    # setting the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # generate synthetic data
    M_true = 2
    D = 5
    p = sum(comb(D + m - 1, m) for m in range(M_true + 1))

    # Generate weights
    w_list = [((-1) ** (p - i)) * (sqrt(p - i) / p) for i in range(p)]
    W = torch.tensor(w_list, dtype=torch.float32)

    print("Underlying weight vector w_true:")
    print(W)
    print(f"True M value: {M_true}")

    # Generate 200 training samples
    N_train = 200
    train_samples = [generate_sample(W, M_true, D) for _ in range(N_train)]
    x_train = torch.stack([s[0] for s in train_samples])
    t_train = torch.tensor([s[1] for s in train_samples], dtype=torch.float32)

    # Generate 100 test samples
    N_test = 100
    test_samples = [generate_sample(W, M_true, D) for _ in range(N_test)]
    x_test = torch.stack([s[0] for s in test_samples])
    t_test = torch.tensor([s[1] for s in test_samples], dtype=torch.float32)

    # Compute true labels (without noise)
    true_train = compute_true_labels(x_train, W, M_true)
    true_test = compute_true_labels(x_test, W, M_true)

    # Observed accuracy with noisy labels
    observed_acc_train = compute_accuracy(t_train, true_train)
    print(f"Observed training data accuracy: {observed_acc_train * 100:.2f}%")

    # Trying both loss functions
    loss_fns = [MyCrossEntropy(), MyRootMeanSquare()]
    initial_m_values = [1.0, 2.0, 3.0]

    results = []

    for loss_fn in loss_fns:
        print("\n==================================================")
        print(f"Using loss function: {loss_fn.__class__.__name__}")

        for m_init in initial_m_values:
            print("\n--------------------------------------------------")
            print(f"Training with initial M = {m_init:.1f}")

            # Train model with learnable M
            w, optimised_m = fit_logistic_sgd_with_learnable_m(
                x_train, t_train, loss_fn, m_init=m_init, epochs=100
            )

            # Evaluate on test set
            test_acc = evaluate_learned_m_model(
                w, optimised_m, x_test, t_test, true_test
            )

            # Store results
            results.append(
                {
                    "loss_fn": loss_fn.__class__.__name__,
                    "m_init": m_init,
                    "optimised_m": optimised_m,
                    "test_acc": test_acc,
                }
            )

    # Print summary of results
    print("\n==================================================")
    print("Summary of Results:")
    for result in sorted(results, key=lambda x: -x["test_acc"]):
        print(
            f"Loss: {result['loss_fn']}, Init M: {result['m_init']}, "
            f"Optimised M: {result['optimised_m']:.4f}, "
            f"Test Acc: {result['test_acc'] * 100:.2f}%"
        )


if __name__ == "__main__":
    print("=== Learning M as a model parameter ===")
    experiment_with_learnable_m()
