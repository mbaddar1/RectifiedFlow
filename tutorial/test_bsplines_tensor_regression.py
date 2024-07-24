"""
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

from RectifiedFlow.tutorial.bsplines_basis import BSplinesBasis


def get_data(dataset_name, n_samples):
    if dataset_name == "circles":
        X, y = make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=n_samples)

    elif dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=170)
    else:
        raise ValueError(f"dataset_name : {dataset_name} is not supported")
    return X, y


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


if __name__ == '__main__':

    n_samples = 256
    dataset_name = "circles"
    ##
    mlp = MLPClassifier(alpha=1, max_iter=1000, random_state=42)
    X, y = get_data(dataset_name=dataset_name, n_samples=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    mlp.fit(X_train, y_train)
    y_hat = mlp.predict(X_test)
    acc_score_value = accuracy_score(y_pred=y_hat, y_true=y_test)
    # score = mlp.score(X_test, y_test)
    # print(score)
    print(acc_score_value)
    tns_clf = TensorBSplinesClassifier(data_dim=2, basis_dim=51, x_low=-1, x_high=1)

    #
    optimizer = torch.optim.Adam(tns_clf.parameters(), lr=0.05)
    loss_fn = torch.nn.BCELoss()
    alpha = 0.1
    si = None
    loss_curve_itr_index = []
    loss_curve_values = []
    for i in tqdm(range(int(1000) + 1), desc="training"):
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
        li.backward()
        optimizer.step()
    plt.plot(loss_curve_itr_index, loss_curve_values)
    plt.savefig("loss_curve.png")

    # Test data
    X, y = get_data(dataset_name=dataset_name, n_samples=n_samples)
    y_hat = tns_clf(torch.tensor(X))
    y_hat.detach().apply_(lambda v: 1 if v >= 0.5 else 0)
    df = pd.DataFrame({'y': y, 'y_hat': y_hat.detach().numpy()})
    print(accuracy_score(y_true=y, y_pred=y_hat.detach().numpy()))
