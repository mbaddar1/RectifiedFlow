"""
This script is to test MLPClassifier and MLPRegressor vs. TensorBSplines Classifier and Regressors
______
Summary of Results
========================
For Classification
---
I) moons dataset
1) MLP
MLP Classifier training time for dataset : moons= 104164 micro-secs
MLP Classifier numel = 401
MLP Classifier Accuracy for dataset moons= 0.8776041666666666

2) TensorBSplines training time for dataset : moons = 635049 micro-secs
TensorBSplinesClassifier numel 22
TensorBSplines Classifier Accuracy for dataset moons =1.0
***
II) circles dataset
1) MLP
MLP Classifier training time for dataset : circles= 166374 micro-secs
MLP Classifier numel = 401
MLP Classifier Accuracy for dataset circles= 1.0

2) TensorBSplines training time for dataset : circles = 301859 micro-secs
TensorBSplinesClassifier numel 22
TensorBSplines Classifier Accuracy for dataset circles =0.9921875
=======================
For Regression
------
b_splines_degree = 2
basis_dim = 51
dataset_name = "trig"
MLP Regression training time for dataset trig = 377168 microseconds
numel for MLPRegressor for dataset trig = 10501
RMSE for MLP regressor for dataset trig = 0.004759512690025826


numel for TensorBSplines-Regressor in dataset trig= 102
TensorBSplines Regressor training time for dataset trig= 296568 microseconds
RMSE for TensorBSplines regressor for dataset trig = 0.007229183679341802

******************************************************************************************************************************

b_splines_degree = 2
basis_dim = 11
dataset_name = "diabetes"
MLP Regression training time for dataset diabetes = 185401 microseconds
numel for MLPRegressor for dataset diabetes = 11301
RMSE for MLP regressor for dataset diabetes = 51.14554004371418

numel for TensorBSplines-Regressor in dataset diabetes= 110
TensorBSplines Regressor training time for dataset diabetes= 934656 microseconds
RMSE for TensorBSplines regressor for dataset diabetes = 54.926424384406886
*********************************************************************************************************************************

__________________________________________________

+++++++++++++++++++++++++++++++
    *** Related Material ***
+++++++++++++++++++++++++++++++
Basis Regression
https://www.dbs.ifi.lmu.de/Lehre/MaschLernen/SS2016/Skript/BasisFunctions2016.pdf

Sample Datasets
-------
https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py


Sample Classification code
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

Loss Function in PyTorch Models
https://machinelearningmastery.com/loss-functions-in-pytorch-models/
"""
import sys

import torch.nn
from sklearn.datasets import make_circles, make_moons, load_diabetes
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from torch.nn import MSELoss
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from RectifiedFlow.tutorial.bsplines_basis import BSplinesBasis
import numpy as np
from datetime import datetime

from RectifiedFlow.tutorial.ttde.torch_rbf import RBFLayer, basis_func_dict

SEED = 42

random.seed(SEED)  # python random generator
np.random.seed(SEED)  # numpy random generator
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RBFN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbf = RBFLayer(in_features=2, out_features=20, basis_func=basis_func_dict()["gaussian"])
        self.linear = torch.nn.Linear(in_features=20,out_features=1)
        self.model = torch.nn.Sequential(self.rbf,self.linear)
    def forward(self,x):
        return self.model(x)

class RBFReg(torch.nn.Module):
    def __init__(self, c: np.array, out_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_centers = len(c)
        self.c = torch.nn.Parameter(torch.tensor(c))
        self.A = torch.nn.Linear(in_features=n_centers, out_features=out_dim)
        self.l = torch.nn.Parameter(torch.tensor([1000.0]))

    def forward(self, x):
        # assert (x.shape[1] == len(self.c))
        rbf_ = RBF(length_scale=self.l.detach().numpy())
        feat = torch.tensor(rbf_(x.detach().numpy(), self.c.detach().numpy()))
        y = self.A(feat)
        return y.T

    @staticmethod
    def get_centers(X: torch.Tensor, n_centers: int):
        x_min = torch.min(X, dim=0).values.detach().numpy()
        x_max = torch.max(X, dim=0).values.detach().numpy()
        return np.linspace(start=x_min, stop=x_max, num=n_centers, endpoint=True)


def trainRBFreg(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, max_iter: int, batch_size: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_fn = MSELoss()
    N = X.shape[0]
    for i in tqdm(range(max_iter), desc="train rbf reg"):
        optimizer.zero_grad()
        indices = torch.randperm(N)[:batch_size]
        X_batch = X[indices, :]
        y_batch = y[indices]
        y_hat = model(X_batch)
        loss = loss_fn(y_batch, y_hat)
        if i % 100 == 0:
            print(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Training RBF Reg finished")


def get_mlp_numel(mlp):
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


def get_clf_data(dataset_name, n_samples):
    if dataset_name == "circles":
        X, y = make_circles(factor=0.3, noise=0.05, random_state=SEED, n_samples=n_samples)

    elif dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=SEED)
    else:
        raise ValueError(f"dataset_name : {dataset_name} is not supported")
    return X, y


def get_reg_data(dataset_name, n_samples):
    if dataset_name == "diabetes":
        data_ = load_diabetes()
        X, y = data_.data, data_.target
        return torch.tensor(X), torch.tensor(y)
    elif dataset_name == "trig":
        mvn_1 = torch.distributions.Normal(loc=0, scale=0.1)
        mvn_2 = torch.distributions.MultivariateNormal(loc=torch.tensor([0.0, 0.0]), covariance_matrix=torch.eye(2))
        X = mvn_2.sample(sample_shape=torch.Size([n_samples]))
        y = 0.2 + torch.sin(X[:, 0]) + 0.8 * torch.cos(X[:, 1])
        y += mvn_1.sample(sample_shape=torch.Size([n_samples]))
        return X, y
    else:
        raise ValueError(f"unknown dataset_name = {dataset_name}")


class TensorBSplinesModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, basis_dim, x_range, degree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.A_list = torch.nn.ParameterList([torch.nn.Parameter(
            torch.distributions.Uniform(low=-0.01, high=0.01).sample(sample_shape=torch.Size([input_dim, basis_dim])))
            for
            _ in range(output_dim)])
        self.b = torch.nn.Parameter(
            torch.distributions.Uniform(low=-0.1, high=0.1).sample(sample_shape=torch.Size([output_dim])))
        assert input_dim == len(x_range)
        self.bsp = []
        for d in range(input_dim):
            self.bsp.append(BSplinesBasis(x_low=x_range[d][0], x_high=x_range[d][1], n_knots=basis_dim, degree=degree))

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        D = x.shape[1]
        x_list = x.T.tolist()
        basis_list = list(map(lambda d: self.bsp[d].calculate_basis_vector(x_list[d]), list(range(D))))
        basis_tensor = torch.tensor(basis_list).permute(1, 0, 2)
        yd_list = []
        for d in range(self.output_dim):
            yd = torch.einsum('bij,ij->b', basis_tensor, self.A_list[d]) + self.b[d]
            yd_list.append(yd)
        y_tensor = torch.stack(yd_list, dim=1)
        return y_tensor

    def numel(self):
        return int(np.sum([torch.numel(A) for A in self.A_list]))

    def norm(self):
        return np.sum([torch.linalg.norm(A).item() for A in self.A_list])

    @staticmethod
    def get_data_range(x: torch.Tensor):
        """
        x : 2D tensor of shape N X D
        """
        # N = x.shape[0]
        # D = x.shape[1]
        decimals = 1
        x_max, x_min = (torch.round(torch.max(x, dim=0).values, decimals=decimals),
                        torch.round_(torch.min(x, dim=0).values, decimals=decimals))
        step = 1.0 / 10 ** decimals
        x_max = x_max + step
        x_min = x_min - step
        x_range = torch.stack(tensors=[x_min, x_max], dim=0).T.tolist()
        return x_range


class TensorBSplinesRegressor(TensorBSplinesModel):
    def __init__(self, input_dim, output_dim, basis_dim, x_range, degree, *args, **kwargs):
        super().__init__(input_dim, output_dim, basis_dim, x_range, degree, *args, **kwargs)

    def forward(self, x):
        return super().forward(x)


class TensorBSplinesClassifier(TensorBSplinesModel):
    def __init__(self, input_dim, output_dim, basis_dim, x_range, degree, *args, **kwargs):
        super().__init__(input_dim, output_dim, basis_dim, x_range, degree, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        y = torch.nn.Sigmoid()(out)
        return y


def test_classifier():
    print("##########################################################")
    print("################# Test Classification ####################")
    print("##########################################################")

    batch_size = 32
    train_size = batch_size * 16
    test_size = batch_size * 8
    test_ratio = float(test_size) / train_size
    dataset_name = "circles"
    tol = 1e-4
    train_iter = 5000
    lr = 0.05
    bspline_degree = 2
    basis_dim = 11
    ##
    mlp = MLPClassifier(hidden_layer_sizes=(100,), alpha=1, max_iter=train_iter, random_state=SEED, verbose=True,
                        early_stopping=False,
                        learning_rate_init=lr, learning_rate="adaptive", batch_size=batch_size)

    X, y = get_clf_data(dataset_name=dataset_name, n_samples=train_size + test_size)
    D = X.shape[1]
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.savefig(f"{dataset_name}_data.png")
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=SEED)
    start_time = datetime.now()
    mlp.fit(X_train, y_train)
    end_time = datetime.now()
    print(
        f"MLP Classifier training time for dataset : {dataset_name}= {(end_time - start_time).microseconds} micro-secs")
    mlp_numel = get_mlp_numel(mlp)
    print(f"MLP Classifier numel = {mlp_numel}")
    y_hat = mlp.predict(X_test)
    print(f"MLP Classifier Accuracy for dataset {dataset_name}= {accuracy_score(y_pred=y_hat, y_true=y_test)}")
    # plt predictions vs actual
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("MLP Classifier")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax1.set_title("Test data")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_hat)
    ax2.set_title("Predicted data")
    plt.savefig(f"clf_mlp_actual_predicted_{dataset_name}.png")

    # Full-Tensor-BSplines Classifier
    x_range = TensorBSplinesModel.get_data_range(x=torch.tensor(X_train))
    tns_clf = TensorBSplinesClassifier(input_dim=D, basis_dim=basis_dim, output_dim=1, x_range=x_range,
                                       degree=bspline_degree)

    optimizer = torch.optim.Adam(tns_clf.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    alpha = 0.1
    n_epochs = 5000
    si = None
    loss_curve_itr_index = []
    loss_curve_values = []
    acc_i = 0.0
    start_time = datetime.now()
    for i in tqdm(range(int(n_epochs) + 1), desc="training"):
        optimizer.zero_grad()
        X, y = get_clf_data(dataset_name=dataset_name, n_samples=batch_size)
        X = torch.tensor(X)
        y = torch.tensor(y).double()
        y_hat = tns_clf(X)
        y_hat_copy = torch.clone(y_hat)
        li = loss_fn(y_hat, y.view(-1, 1))

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

    end_time = datetime.now()
    print(
        f"TensorBSplines training time for dataset : {dataset_name} = {(end_time - start_time).microseconds} micro-secs")
    plt.clf()
    plt.plot(loss_curve_itr_index, loss_curve_values)
    plt.title(f"Loss Curve for TensorBsplines Classifier for dataset : {dataset_name}")
    plt.savefig(f"clf_tns_bsplines_loss_curve_{dataset_name}.png")
    # Test data
    X_test, y_test = get_clf_data(dataset_name=dataset_name, n_samples=test_size)
    y_hat = tns_clf(torch.tensor(X_test))
    y_hat.detach().apply_(lambda v: 1 if v >= 0.5 else 0)
    print(f"TensorBSplinesClassifier numel {tns_clf.numel()}")
    print(f"TensorBSplines Classifier Accuracy for dataset "
          f"{dataset_name} ={accuracy_score(y_true=y_test, y_pred=y_hat.detach().numpy())}")

    # plt actual vs predicted
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("TensorBSplines Classifier")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax1.set_title("Test data")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_hat.detach().numpy())
    ax2.set_title("Predicted data")
    plt.savefig(f"clf_tns_bsplines_actual_predicted_{dataset_name}.png")


def test_regression():
    # https://scikit-learn.org/0.16/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
    # MLP Regressor
    dataset_name = "trig"
    iter = 10000
    batch_size = 64

    X, y = get_reg_data(dataset_name=dataset_name, n_samples=10000)
    D = X.shape[1]
    b_splines_degree = 1
    basis_dim = 51
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    N_train = X_train.shape[0]

    ### RBF Reg
    ############## Add RBF Reg ########################
    c = RBFReg.get_centers(X=X, n_centers=10000)
    #m = RBFReg(c=c, out_dim=1)
    m = RBFN()
    # y = m(torch.tensor(X))
    trainRBFreg(model=m, X=X, y=y, max_iter=10000, batch_size=1024)
    print("########## Training MLP Regressor############")
    start_time = datetime.now()
    mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=iter, verbose=True, early_stopping=False,
                           batch_size=batch_size)
    mlp_reg.fit(X_train, y_train)
    end_time = datetime.now()
    y_hat = mlp_reg.predict(X_test)
    mse_ = mean_squared_error(y_true=y_test, y_pred=y_hat)

    print(
        f"MLP Regression training time for dataset {dataset_name} = {(end_time - start_time).microseconds} microseconds")
    nel = get_mlp_numel(mlp_reg)
    print(f"numel for MLPRegressor for dataset {dataset_name} = {nel}")
    print(f"MSE for MLP regressor for dataset {dataset_name} = {mse_}")

    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html#sklearn.preprocessing.SplineTransformer
    model = make_pipeline(SplineTransformer(n_knots=100, degree=3), Ridge(alpha=1e-3))
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mse_ = mean_squared_error(y_true=y_test, y_pred=y_hat)
    print(f"B splines sklearn reg mse {mse_}")
    print("finnn")

    # TensorBSplines
    print("############## Training TensorBSplines Regressor ################")
    x_range = TensorBSplinesModel.get_data_range(X)  # get range based on complete dataset: train and test
    tns_reg = TensorBSplinesRegressor(input_dim=D, output_dim=1, basis_dim=basis_dim, x_range=x_range,
                                      degree=b_splines_degree)

    loss_fn = torch.nn.MSELoss()
    indices = list(np.arange(0, N_train))
    optimizer = torch.optim.Adam(tns_reg.parameters(), lr=0.005)
    si = None
    alpha = 0.1
    start_time = datetime.now()
    for i in tqdm(range(iter), desc="TensorBSplines Regression Training"):
        optimizer.zero_grad()
        batch_idx = random.sample(population=indices, k=batch_size)
        X_batch = X_train[batch_idx, :]
        y_batch = y_train[batch_idx]
        y_hat = tns_reg(X_batch)
        lambda_ = 1e-3
        loss = loss_fn(y_hat, y_batch.view(-1, 1)) + lambda_ * tns_reg.norm()
        if si is None:
            si = loss.item()
        else:
            si = alpha * loss.item() + (1 - alpha) * si
        if i % 100 == 0:
            print(f"i = {i},si = {si}")
        loss.backward()
        optimizer.step()
    end_time = datetime.now()
    nel = tns_reg.numel()
    print(f"numel for TensorBSplines-Regressor in dataset {dataset_name}= {nel}")
    print(
        f"TensorBSplines Regressor training time for dataset {dataset_name}= {(end_time - start_time).microseconds} microseconds")
    y_hat = tns_reg(torch.tensor(X_test))
    rmse_ = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_hat.detach().numpy()))
    print(f"RMSE for TensorBSplines regressor for dataset {dataset_name} = {rmse_}")


if __name__ == '__main__':
    # test_classifier()
    test_regression()
