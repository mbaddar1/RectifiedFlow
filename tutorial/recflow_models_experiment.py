# -*- coding: utf-8 -*-
"""Tutorial: Rectified Flow with Neural Network.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P

# Rectified Flow
This jupyter notebook contains simple tutorial code for Rectified Flow proposed in '[Flow Straight and Fast: Learning
to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)'.

The problem here is to learn an ODE $\dot Z_t = v(Z_t, t) $ to transfer data from $\pi_0$ to $\pi_1$, where both
$\pi_0$ and $\pi_1$ are unknown and empirically observed through a set of points.

The velocity field $v(z,t)$ in rectified flow can be fitted with either kernel method or deep neural networks.
This tutorial illustrates the use of a neural network.

## Generating Distribution $\pi_0$ and $\pi_1$
We generate $\pi_0$ and $\pi_1$ as two Gaussian mixture models with different modes.

We sample 10000 data points from $\pi_0$ and $\pi_1$, respectively,
and store them in ```samples_0```, ```samples_1```.


Material
------------------
Auto Knots selection
https://arxiv.org/pdf/1808.01770
https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf
"""
import pickle
import random
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.datasets import make_swiss_roll, make_circles, make_blobs, make_moons
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from torch.distributions import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from tqdm import tqdm
from RectifiedFlow.tutorial.seed import set_global_seed
from RectifiedFlow.tutorial.splines_models import TensorBSplinesRegressor, TensorBSplinesModel
from functional_tt_fabrique import orthpoly, Extended_TensorTrain
from geomloss import SamplesLoss
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from utils.utils import get_target_samples, filter_tensor
from datetime import datetime


# Set seed
# SEED = 42
# set_global_seed(SEED)


# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

class RBFReg(torch.nn.Module):
    def __init__(self, c: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass

    @staticmethod
    def get_centers(X: torch.Tensor,n_centers : int):
        pass


class DummyReg(torch.nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = torch.nn.Parameter(
            torch.distributions.Uniform(low=-0.1, high=0.1).sample(torch.Size([output_dim, input_dim])))

    def forward(self, x):
        y = torch.matmul(self.A, x.T)
        return y.T


@torch.no_grad()
def draw_plot(recflow_model, z0, z1, N=None, **kwargs):
    print(f"Drawing plot for model of class : {type(recflow_model)}")
    assert isinstance(recflow_model, (RectifiedFlowTT, RectifiedFlowNN, RectifiedFlowRegBsplines))
    fig_title_part = ""
    if isinstance(recflow_model, RectifiedFlowTT):
        suffix = "tt"
        fig_title_part += f"model=tt-recflow\nr={kwargs['r']},\nd={kwargs['d']}"
    elif isinstance(recflow_model, RectifiedFlowNN):
        suffix = "nn"
        fig_title_part += f"model=nn-recflow"
    elif isinstance(recflow_model, RectifiedFlowRegBsplines):
        suffix = "rb"
        fig_title_part += f"model=tensor-bsplines-recflow"
    else:
        raise ValueError(f"Unsupported recflow model type : {type(recflow_model)}")

    traj = recflow_model.sample_ode(z0=z0, N=N)
    # Get sinkhorn value
    samples_loss_ = SamplesLoss()
    z1_gen = traj[-1]
    if isinstance(recflow_model, RectifiedFlowTT):
        z1_gen_filtered = filter_tensor(z1_gen)
    else:
        z1_gen_filtered = z1_gen
    sinkhorn_value = samples_loss_(z1_gen_filtered, z1)
    print(f"Sinkhorn value for the generated samples= {sinkhorn_value}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Actual vs Generated Distribution : {fig_title_part} , \nsinkhorn = {sinkhorn_value}')
    x_lim = (-15, 15)
    y_lim = (-15, 15)

    ax1.set_ylim(y_lim)
    ax1.set_xlim(x_lim)
    ax1.set_title(r'$\pi_0$')
    ax1.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), color='red')

    ax2.set_ylim(y_lim)
    ax2.set_xlim(x_lim)
    ax2.set_title(r'$\pi_1$')
    ax2.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), color='green')

    ax3.set_ylim(y_lim)
    ax3.set_xlim(x_lim)
    ax3.set_title('Generated')
    ax3.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), color='blue')
    plt.tight_layout()
    plt.savefig(f"actual_vs_generated_samples_{suffix}.png")

    plt.clf()

    # Plot trajectories
    traj_particles = torch.stack(traj)
    plt.figure(figsize=(4, 4))
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.axis('equal')
    for i in tqdm(range(30), desc="generating trajectory"):
        x = traj_particles[:, i, 0]
        y = traj_particles[:, i, 1]
        plt.plot(x, y)
    plt.title('Transport Trajectory')
    plt.tight_layout()
    plt.savefig(f"trajectory_{suffix}.png")

    plt.clf()


def get_train_tuple(z0=None, z1=None):
    t = torch.rand((z1.shape[0], 1))
    z_t = t * z1 + (1. - t) * z0
    target = z1 - z0
    return z_t, t, target


def train_rectified_flow_nn(rectified_flow_nn, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    alpha = 0.1
    loss_fn = torch.nn.MSELoss()
    si = None
    for i in tqdm(range(inner_iters + 1), desc="training recflow-nn "):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = get_train_tuple(z0=z0, z1=z1)

        pred = rectified_flow_nn.model(z_t, t)
        loss = loss_fn(pred, target)  # both losses are the same
        # loss = 0.5 * (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1)
        # loss = loss.mean()
        if si is None:
            si = loss.item()
        else:
            si = alpha * loss.item() + (1 - alpha) * si
        if i % 100 == 0:
            print(f"si for loss type {type(loss_fn)} @i = {i} => {si}")

        with torch.no_grad():
            l2 = loss_fn(target, pred)
        loss.backward()

        optimizer.step()
        loss_curve.append(np.log(loss.item()))  # to store the loss curve

    return rectified_flow_nn, loss_curve


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


class RectifiedFlowRegBsplines:
    def __init__(self, input_dim, output_dim, sp):
        # assert len(x_range) == input_dim
        # self.tns_bsp_reg = TensorBSplinesRegressor(input_dim=input_dim, output_dim=out_dim, x_range=x_range,
        #                                            basis_dim=basis_dim, degree=basis_degree)
        self.model = DummyReg(input_dim, output_dim)
        self.sp = sp

    # FIXME , repeated fn , need to make a base class for Recflow
    def sample_ode(self, z0: torch.Tensor, N: int):
        dt = 1. / N
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        for i in tqdm(range(N), desc="generate tensor-b-splines trajectory"):
            t = torch.ones((batchsize, 1)) * i / N
            pred = self.v(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return traj

    # FIXME , repeated fn , need to make a base class for Recflow
    def v(self, zt, t) -> torch.Tensor:
        zt_aug = torch.cat([zt, t], dim=1)
        X = self.sp.fit_transform(zt_aug.detach().numpy())
        pred_vec = self.model(torch.tensor(X))
        return pred_vec


def train_recflow_reg_bsplines(recflow_model: RectifiedFlowRegBsplines, X0: torch.Tensor, X1: torch.Tensor,
                               train_iterations: int, batch_size: int):
    z_t, t, target = get_train_tuple(z0=X0, z1=X1)
    z_t_aug = torch.concat([z_t, t], dim=1)
    X = z_t_aug.detach().numpy()
    y = target.detach().numpy()
    ########## Splines Regression Quick Test #################
    X_feat = recflow_model.sp.fit_transform(X)

    # model = make_pipeline(SplineTransformer(n_knots=128, degree=3, knots="quantile"), Ridge(alpha=1e-3))
    # model.fit(X, y)
    # y_hat = model.predict(X)
    # mse_ = mean_squared_error(y_true=y, y_pred=y_hat)
    # print(mse_)
    #
    # mlpreg = MLPRegressor(verbose=True)
    # mlpreg.fit(X, y)
    # y_hat = mlpreg.predict(X)
    # mse2_ = mean_squared_error(y_true=y, y_pred=y_hat)
    # print("finnnnn")
    ################################################
    loss_fn = torch.nn.MSELoss()

    params = recflow_model.model.parameters()
    optimizer = torch.optim.Adam(params=params, lr=0.1)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    alpha = 0.1
    si = None
    # for j in tqdm(range(sch_itr), desc="Scheduler iterations"):
    for i in tqdm(range(train_iterations), desc=f"Training iterations "):
        optimizer.zero_grad()
        indices = torch.randperm(z_t_aug.shape[0])[:batch_size]
        X_batch = torch.tensor(X_feat)[indices]
        target_batch = target[indices]
        y_hat = recflow_model.model(X_batch)
        loss = loss_fn(y_hat, target_batch)
        if si is None:
            si = loss.item()
        else:
            si = alpha * loss.item() + (1 - alpha) * si
        if i % 100 == 0:
            print(f"si, i = {i} = {si}")
        loss.backward()
        optimizer.step()
        # before_lr = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        # after_lr = optimizer.param_groups[0]["lr"]
        # print("Epoch %d: lr %.4f -> %.4f" % (j, before_lr, after_lr))
    print(f"Final EMA loss (si) over total # iter {train_iterations} = {si}")
    print("Finished training")


class RectifiedFlowTT:
    def __init__(self, basis_degrees, limits, data_dim, ranks):
        # basis_degrees = [basis_degree] * (data_dim + 1)  # hotfix by charles that made the GMM work
        # ranks = [1] + [tt_rank] * data_dim + [1]
        domain = [list(limits) for _ in range(data_dim)] + [[0, 1]]
        print("Generating Orthopoly Func.(This might take a couple of secs)")
        op = orthpoly(basis_degrees, domain)
        self.ETTs = [Extended_TensorTrain(op, ranks) for i in range(data_dim)]

    # FIXME , repeated fn , need to make a base class for Recflow
    def sample_ode(self, z0: torch.Tensor, N: int):
        dt = 1. / N
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        for i in tqdm(range(N), desc="generate tt-recflow trajectory"):
            t = torch.ones((batchsize, 1)) * i / N
            pred = self.v(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return traj

    # FIXME , repeated fn , need to make a base class for Recflow
    def v(self, zt, t) -> torch.Tensor:
        data_dim = zt.shape[1]
        zt_aug = torch.cat([zt, t], dim=1)
        pred_list = []
        for d in range(data_dim):
            pred_vec = self.ETTs[d](zt_aug).view(-1, 1)
            pred_list.append(pred_vec)
        pred_tensor = torch.cat(tensors=pred_list, dim=1)
        return pred_tensor


class RectifiedFlowNN:
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps

    # FIXME , repeated fn , need to make a base class for Recflow
    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        # NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1)) * i / N
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return traj


def train_rectified_flow_tt(rectified_flow_tt: RectifiedFlowTT, x0, x1, reg_coeff=1e-20, iterations=40, tol=5e-10,
                            rule=None):
    z_t, t, target = get_train_tuple(z0=x0, z1=x1)
    z_t_aug = torch.concat([z_t, t], dim=1)
    for i, ETT in enumerate(rectified_flow_tt.ETTs):
        print(f"for output d = {i}")
        y_i = target[:, i].view(-1, 1)
        ETT.fit(
            x=z_t_aug,
            y=y_i,
            iterations=iterations,
            rule=rule,
            tol=tol,
            verboselevel=1,
            reg_param=reg_coeff,
        )
        ETT.tt.set_core(x0.shape[1])
    print("recflow-tt training finished")
    v_ = rectified_flow_tt.v(z_t, t)
    with torch.no_grad():
        loss_fn = MSELoss()
        k = loss_fn(v_, target)
    return rectified_flow_tt


def hopt_objective(args):
    r = args['r']
    d = args['d']
    print(f"Creating a RecFlow TT object with r={(r, r)}")
    ranks = [1] + [r, r] + [1]
    limits = (-20, 20)
    model = RectifiedFlowTT(ranks=ranks, basis_degrees=[d, d, d], data_dim=data_dim, limits=limits)
    print("training tt-recflow")
    reg_coeff = 1e-10
    n_itr = 20
    tol = 5e-10
    samples_loss_ = SamplesLoss(loss="sinkhorn")
    x0 = args['init_model'].sample(torch.Size([args['N']]))
    x1_train = get_target_samples(dataset_name=args['dataset_name'], n_samples=args['N'])
    x1_test = get_target_samples(dataset_name=args['dataset_name'], n_samples=args['N'])
    checkpoint_sinkhorn = samples_loss_(x1_train, x1_test)
    print(f"Checkpoint sinkhorn between train and test : {checkpoint_sinkhorn}")
    print(f"r={(r, r)}")
    train_rectified_flow_tt(rectified_flow_tt=model, x0=x0, x1=x1_train, reg_coeff=reg_coeff,
                            iterations=n_itr, tol=tol)
    gen_sample = model.sample_ode(z0=args['init_model'].sample(torch.Size([n_samples])), N=2000)[-1]
    gen_sample_filtered = filter_tensor(x=gen_sample)
    gen_sinkhorn = samples_loss_(x1_test, gen_sample_filtered).item()
    print(f"with r = {(r, r)}, d = {d}gen_sinkhorn value = {gen_sinkhorn}")
    return {
        'loss': gen_sinkhorn,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
    }


def tt_recflow_hopt(init_model: Distribution, hopt_max_evals: int, target_dataset_name: str):
    # https://github.com/hyperopt/hyperopt/issues/835
    space = {'r': hp.randint('r', 8, 20),
             'd': hp.randint('d', 5, 20),
             # TODO reg param
             #  Time Descritization
             'init_model': init_model,
             'dataset_name': target_dataset_name,
             'N': 10000}
    trials = Trials()
    best = fmin(fn=hopt_objective, space=space, algo=tpe.suggest, max_evals=hopt_max_evals, trials=trials)
    time_stamp = datetime.now().isoformat()
    print("writing trials")
    with open(f"hopt_trials_{time_stamp}.hopt", "wb") as f:
        pickle.dump(trials, f)
    print(f"Best parameters = {best}")
    print(f"Opt loss = {trials.best_trial['result']['loss']}")


def get_mlp_numel(mlp):
    return np.sum([torch.numel(param) for param in mlp.parameters()])


def train_mlp(X: torch.Tensor, t: torch.Tensor, Y: torch.Tensor, batch_size: int, max_iter: int):
    N = X.shape[0]
    N_train = int(0.8 * N)
    X_train = X[:N_train, :]
    X_test = X[N_train:, :]
    Y_train = Y[:N_train, :]
    Y_test = Y[N_train:, ]
    t_train = t[:N_train, :]
    t_test = t[N_train:, ]
    loss_fn = torch.nn.MSELoss()
    si = None

    in_dim = X.shape[1]
    mlp = MLP(in_dim, hidden_num=100)
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)
    nel = get_mlp_numel(mlp)
    print(f"nel mlp  = {nel}")
    alpha = 0.1
    for i in tqdm(range(max_iter + 1), desc="train mlp "):
        optimizer.zero_grad()
        indices = torch.randperm(N_train)[:batch_size]
        X_batch = X_train[indices, :]
        Y_batch = Y_train[indices, :]
        t_batch = t_train[indices, :]
        pred = mlp(X_batch, t_batch)
        loss = loss_fn(pred, Y_batch)  # both losses are the same
        # loss = 0.5 * (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1)
        # loss = loss.mean()
        if si is None:
            si = loss.item()
        else:
            si = alpha * loss.item() + (1 - alpha) * si
        if i % 100 == 0:
            print(f"si for loss type {type(loss_fn)} @i = {i} => {si}")
        loss.backward()
        optimizer.step()

    y_hat = mlp(X_test, t_test)
    mse_ = loss_fn(y_hat, Y_test)
    print(f"mlo mse test {mse_}")


def train_splines_regression(X: torch.Tensor, t: torch.Tensor, Y: torch.Tensor, batch_size: int, max_iter: int):
    # Poly and Splines Reg
    loss_fn = MSELoss()
    N = X.shape[0]
    N_train = int(0.8 * N)
    X_train = X[:N_train, :]
    X_test = X[N_train:, :]
    Y_train = torch.tensor(Y[:N_train, :])
    Y_test = Y[N_train:, ]
    t_train = t[:N_train, :]
    t_test = t[N_train:, ]
    X_train_aug = torch.concat([X_train, t_train], dim=1)
    X_test_aug = torch.concat([X_test, t_test], dim=1)
    feature_model: SplineTransformer = SplineTransformer(n_knots=4096, degree=3, knots='quantile', include_bias=True,
                                                         extrapolation='linear')
    u = feature_model.fit_transform(X_train_aug.detach().numpy())
    dummy_reg = DummyReg(input_dim=u.shape[1], output_dim=2)
    u = torch.tensor(feature_model.fit_transform(X_train_aug.detach().numpy()))
    optimizer = torch.optim.Adam(dummy_reg.parameters(), lr=1e-1)
    for j in tqdm(range(2000), desc="train dummy reg"):
        optimizer.zero_grad()
        indices = torch.randperm(N_train)[:batch_size]
        X_batch = u[indices, :][:batch_size]
        Y_batch = Y_train[indices, :][:batch_size]
        y_hat = dummy_reg(X_batch)
        lambda_ = 1e-3
        l1 = loss_fn(y_hat, Y_batch)
        l2 = lambda_ * torch.norm(dummy_reg.A)
        loss = l1 + l2
        if j % 100 == 0:
            print(l1.item())

        loss.backward()
        optimizer.step()
    print("train finished")
    u_test = feature_model.fit_transform(X_test_aug.detach().numpy())
    y_hat = dummy_reg(torch.tensor(u_test))
    mse2_ = loss_fn(y_hat, Y_test)
    sys.exit(-1)
    # XX = X_train_aug.detach().numpy()
    # xx_min = np.min(XX, axis=0)
    # xx_max = np.max(XX, axis=0)
    # kk = feature_model._get_base_knot_positions(X=XX, n_knots=512, knots="quantile")
    # reg_model = Ridge(alpha=1e-3)
    # model = make_pipeline(feature_model, reg_model, verbose=True)
    # model.fit(X=X_train_aug.detach().numpy(), y=Y_train.detach().numpy())
    # X_test_aug = torch.concat([X_test, t_test], dim=1)
    # N_test = X_test_aug.shape[0]
    # X_test_list = list(X_test_aug)
    # out_of_range = 0
    # # k = feature_model.knots
    # # for j in tqdm(range(N_test), desc="test out of range"):
    # #     try:
    # #         yy = model.predict(X_test_list[j].reshape(1, -1))
    # #     except ValueError as e:
    # #         print(e)
    # #         out_of_range += 1
    # # print(f"out_of_range = {out_of_range}")
    # y_hat = model.predict(X_test_aug.detach().numpy())
    # mse_ = MSELoss()(torch.tensor(y_hat), Y_test)
    # print("")


def compare_recflow_regression_models(x0: torch.Tensor, x1: torch.Tensor):
    z_t, t, target = get_train_tuple(z0=x0, z1=x1)
    max_iter = 10000
    batch_size = 1024
    # z_t_aug = torch.concat((z_t, t), dim=1)
    ########### MLP ############
    train_mlp(X=z_t, t=t, Y=target, batch_size=batch_size, max_iter=max_iter)
    # print("Finished train mlp")

    ########### Splines Regression ##############
    # train_splines_regression(X=z_t, t=t, Y=target, batch_size=batch_size, max_iter=max_iter)


# Main
if __name__ == '__main__':
    D = 10.
    M = D + 5
    VAR = 0.3
    DOT_SIZE = 4
    COMP = 3
    n_samples = 50000
    data_dim = 2
    model_type = "rb"  # can be nn,tt,tb
    # nn is neural network
    # tt is tensor train with legendre poly
    # tb tensor bsplines
    do_hyperopt = False
    hopt_max_evals = 100
    target_dataset_name = "moons"
    initial_model = MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
    samples_0 = initial_model.sample(torch.Size([n_samples]))
    # samples_1 = get_target_samples(dataset_name=target_dataset_name, n_samples=n_samples)
    samples_1 = torch.tensor(make_moons(shuffle=True, noise=0.01, n_samples=n_samples, random_state=53)[0] * 5)
    plt.figure(figsize=(4, 4))
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.title(r'Samples from $\pi_0$ and $\pi_1$')
    plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_0$')
    plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_1$')
    plt.legend()

    plt.tight_layout()
    plt.savefig("distributions.png")
    plt.clf()

    # Training RecFlow 1
    print("Training Recflow 1")
    x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
    x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
    x0_test = initial_model.sample(torch.Size([2000]))
    x_pairs = torch.stack([x_0, x_1], dim=1)

    # Experimental / Debugging function to compare all possible RecFlow Regression Models
    # compare_recflow_regression_models(x0=x_0, x1=x_1)
    ############
    recflow_model = None
    if model_type == "nn":
        if do_hyperopt:
            raise NotImplementedError(f"hyperopt with nn is not implemented")
        else:
            print("Training nn-recflow")
            iterations = 10000
            batch_size = 2048
            input_dim = 2
            recflow_model = RectifiedFlowNN(model=MLP(input_dim, hidden_num=100), num_steps=100)
            optimizer = torch.optim.Adam(recflow_model.model.parameters(), lr=5e-3)

            recflow_model, loss_curve = train_rectified_flow_nn(recflow_model, optimizer, x_pairs, batch_size,
                                                                iterations)
            draw_plot(recflow_model, z0=x0_test, z1=samples_1.detach().clone(), N=2000)
            plt.plot(np.linspace(0, iterations, iterations + 1), loss_curve[:(iterations + 1)])
            plt.title('Training Loss Curve')
            plt.savefig("loss_curve_recflow_nn_1.png")

    elif model_type == "tt":
        if do_hyperopt:
            tt_recflow_hopt(init_model=initial_model, target_dataset_name=target_dataset_name,
                            hopt_max_evals=hopt_max_evals)
            print("tt recflow hyperopt is finished.\n"
                  "See console for results.exiting.\n""")
            sys.exit(0)
        else:
            basis_degree = [30, 30, 30]
            limits = (-20, 20)
            ranks = [1, 8, 8, 1]
            recflow_model = RectifiedFlowTT(ranks=ranks, basis_degrees=basis_degree, data_dim=2, limits=limits)
            print("training tt-recflow")
            reg_coeff = 1e-3
            iterations = 40
            tol = 5e-10
            train_rectified_flow_tt(rectified_flow_tt=recflow_model, x0=samples_0, x1=samples_1, iterations=iterations,
                                    tol=tol, reg_coeff=reg_coeff)
            # FIXME , the code for drawing and sinkhorn must be the same , i.e. the sinkhorn must be
            #   calculate for the same drawn data
            print("tt recflow training finished , next step is to generate samples ")
            draw_plot(recflow_model, z0=x0_test, z1=samples_1.detach().clone(), N=2000, r=ranks, d=basis_degree)
    elif model_type == "rb":
        # x_range = TensorBSplinesModel.get_data_range(samples_0)
        # x_range.append([0, 1])  # for time
        degree = 3
        nknots = 128
        nfeat = 3
        nfeatall = nfeat * (nknots + degree - 1)
        sp = SplineTransformer(n_knots=nknots, degree=degree)
        recflow_model = RectifiedFlowRegBsplines(input_dim=nfeatall, output_dim=2, sp=sp)
        train_recflow_reg_bsplines(recflow_model=recflow_model, X0=samples_0, X1=samples_1,
                                   train_iterations=1000, batch_size=1024)
        draw_plot(recflow_model, z0=x0_test, z1=samples_1.detach().clone(), N=2000)
        print(f"finished")

    else:
        raise ValueError(f"Unsupported recflow model type : {type(model_type)}")

    assert recflow_model is not None, "recflow_model is not initialized or trained"

    # print("Generating sinkhorn values")
    # samples_loss = SamplesLoss(loss="sinkhorn")
    # samples_11 = get_target_samples(dataset_name=target_dataset_name, n_samples=n_samples)
    # samples_12 = get_target_samples(dataset_name=target_dataset_name, n_samples=n_samples)
    # ref_sinkhorn = samples_loss(samples_11, samples_12)
    # print(f"ref sinkhorn value = {ref_sinkhorn}")

    # generated_sample = recflow_model.sample_ode(z0=x0_test, N=2000)[-1]
    # generated_sample_filtered = filter_tensor(x=generated_sample)
    # gen_sinkhorn_1 = samples_loss(samples_11, generated_sample_filtered)
    # gen_sinkhorn_2 = samples_loss(samples_12, generated_sample_filtered)
    # gen_sinkhorn_avg = (gen_sinkhorn_1 + gen_sinkhorn_2) / 2.0
    # print(f"generated sinkhorn 1 = {gen_sinkhorn_1}")
    # print(f"generated sinkhorn 2 = {gen_sinkhorn_2}")
    # print(f"generated sinkhorn avg = {gen_sinkhorn_avg}")
    print("Finished")

"""
Results Log
------------------
We show Hyperopt and Normal Train run
*********************************************************************

Result Set # 1 : Normal tt-recflow train run with parameters based on a Hyperopt run (See Result Set 2) 
____________________
# Code Snippet and Params #

# With fixed SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
.....
basis_degree = [30, 30, 30]
limits = (-8, 8)
recflow_model = RectifiedFlowTT(ranks=[1, 8, 3, 1], basis_degrees=basis_degree, data_dim=2, limits=limits)
print("training tt-recflow")
reg_coeff = 1e-3
iterations = 40
tol = 5e-10
train_rectified_flow_tt(rectified_flow_tt=recflow_model, x0=samples_0, x1=samples_1, iterations=iterations,
    tol=tol, reg_coeff=reg_coeff)

# Results #
Run 1 : Sinkhorn value = 0.21108624166078424
Run 2 : Sinkhorn value = 0.21108622431951704
Run 3  : Sinkhorn value = 0.21108624043411978        
*************************************************************************************
Results Set 2 : Hyper opt run 
_________________
basis_degree = [30, 30, 30]
Best parameters = {'r1': 8, 'r2': 3}
Opt loss = 0.11993751016478815
tt recflow hyperopt is finished.
See console for results.exiting.
***************************************************************************************

Results Set 3 : Hyperopt run

100%|██████████| 100/100 [3:14:53<00:00, 116.94s/trial, best loss: 0.1319644362387083]
Best parameters = {'d': 30, 'r': 8}
Opt loss = 0.1319644362387083
tt recflow hyperopt is finished.
"""