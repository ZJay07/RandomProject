import torch


def logistic_fun(w, m, x):
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

    f = get_poly_features(m, x)
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


def fit_logistic_sgt():
    """
    compute the optimum weight vector W~ using the training set, by
    minimising the cross-entropy loss. Considering three hyperparameter values, M ele of {1,2,3}, for
    each M, compute the predicted target values y~ for all x in both the training and test sets.
    """
    pass
