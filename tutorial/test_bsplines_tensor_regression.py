"""
This script is to test MLPClassifier and MLPRegressor vs. TensorBSplines classifier and regressors

__________________________________________________
Basis Regression
https://www.dbs.ifi.lmu.de/Lehre/MaschLernen/SS2016/Skript/BasisFunctions2016.pdf

Sample Dataset
https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py


Sample Classification code
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

Loss Function in PyTorch Models
https://machinelearningmastery.com/loss-functions-in-pytorch-models/
"""
import pandas as pd
import torch.nn
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from RectifiedFlow.tutorial.bsplines_basis import BSplinesBasis
import numpy as np

SEED = 42

random.seed(SEED)  # python random generator
np.random.seed(SEED)  # numpy random generator
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_mlp_classifier_numel(mlp):
    W = mlp.coefs_
    B = mlp.intercepts_
    numel = 0
    for w in W:
        w_cnt = np.prod(w.shape)
        numel += w_cnt
    for b in B:
        b_cnt = np.prod(b.shape)
        numel += b_cnt
    return numel


def get_data(dataset_name, n_samples):
    if dataset_name == "circles":
        X, y = make_circles(noise=0.2, factor=0.5, random_state=SEED, n_samples=n_samples)

    elif dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=SEED)
    else:
        raise ValueError(f"dataset_name : {dataset_name} is not supported")
    return X, y


class TensorBSplinesRegressor(torch.nn.Module):
    def __init__(self, data_dim, basis_dim, x_low, x_high, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.A = torch.nn.Parameter(
            torch.distributions.Uniform(low=-0.1, high=0.1).sample(sample_shape=torch.Size([data_dim, basis_dim])))
        self.bsp = BSplinesBasis(x_low=x_low, x_high=x_high, n_knots=basis_dim, degree=3)

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        x_list = x.tolist()
        basis_list = list(map(lambda e: self.bsp.calculate_basis_vector(e), x_list))
        basis_tensor = torch.tensor(basis_list)
        y = torch.einsum('bij,ij->b', basis_tensor, self.A)
        return y


class TensorBSplinesClassifier(torch.nn.Module):
    def __init__(self, data_dim, basis_dim, x_low, x_high, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = torch.nn.Parameter(
            torch.distributions.Uniform(low=-0.1, high=0.1).sample(sample_shape=torch.Size([data_dim, basis_dim])))
        self.bsp = BSplinesBasis(x_low=x_low, x_high=x_high, n_knots=basis_dim, degree=3)

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        x_list = x.tolist()
        basis_list = list(map(lambda e: self.bsp.calculate_basis_vector(e), x_list))
        basis_tensor = torch.tensor(basis_list)
        y = torch.einsum('bij,ij->b', basis_tensor, self.A)
        out = torch.nn.Sigmoid()(y)
        return out

    def numel(self):
        return torch.numel(self.A)


def test_classifier():
    print("##########################################################")
    print("############ Test Classification #########################")
    print("##########################################################")

    n_samples = 32
    dataset_name = "moons"
    tol = 1e-4
    train_iter = 5000
    lr = 0.05
    ##
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1, max_iter=train_iter, random_state=SEED, verbose=True,
                        early_stopping=False,
                        learning_rate_init=lr, learning_rate="adaptive")

    X, y = get_data(dataset_name=dataset_name, n_samples=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=SEED)
    mlp.fit(X_train, y_train)
    mlp_numel = get_mlp_classifier_numel(mlp)
    print(f"MLP Classifier numel = {mlp_numel}")
    y_hat = mlp.predict(X_test)
    print(f"MLP Classifier Accuracy for dataset {dataset_name}= {accuracy_score(y_pred=y_hat, y_true=y_test)}")

    # Full-Tensor-BSplines Classifier
    tns_clf = TensorBSplinesClassifier(data_dim=2, basis_dim=11, x_low=-1, x_high=1)
    print(f"TensorBSplinesClassifier numel {tns_clf.numel()}")
    #
    optimizer = torch.optim.Adam(tns_clf.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    alpha = 0.1
    n_epochs = 5000
    si = None
    loss_curve_itr_index = []
    loss_curve_values = []
    acc_i = 0.0

    for i in tqdm(range(int(n_epochs) + 1), desc="training"):
        optimizer.zero_grad()
        X, y = get_data(dataset_name=dataset_name, n_samples=n_samples)
        X = torch.tensor(X)
        y = torch.tensor(y).double()
        y_hat = tns_clf(X)
        y_hat_copy = torch.clone(y_hat)
        li = loss_fn(y_hat, y)

        if si is None:
            si = li
        else:
            si = alpha * li + (1 - alpha) * si

        if i % 100 == 0:
            loss_curve_itr_index.append(i)
            loss_curve_values.append(si.item())
            y_hat_copy.detach().apply_(lambda v: 1 if v >= 0.5 else 0)
            acc = accuracy_score(y_true=y, y_pred=y_hat_copy.detach().numpy())
            print(f"iter = {i} , si = {si.item()}, acc = {acc}")
            acc_new = alpha * acc + (1 - alpha) * acc_i
            if np.abs(acc_new - acc_i) < tol:
                print("Accuracy has converged, exiting")
                break
            acc_i = acc_new
        li.backward()
        optimizer.step()
    plt.plot(loss_curve_itr_index, loss_curve_values)
    plt.savefig("tensor_bsplines_classifier_loss_curve.png")
    # Test data
    X, y = get_data(dataset_name=dataset_name, n_samples=n_samples)
    y_hat = tns_clf(torch.tensor(X))
    y_hat.detach().apply_(lambda v: 1 if v >= 0.5 else 0)
    print(
        f"TensorBSplines Classifier Accuracy for dataset "
        f"{dataset_name} ={accuracy_score(y_true=y, y_pred=y_hat.detach().numpy())}")


if __name__ == '__main__':
    test_classifier()
