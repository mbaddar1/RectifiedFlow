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
"""
import pickle
import random
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.datasets import make_swiss_roll, make_circles, make_blobs
from torch.distributions import Normal, Categorical
from torch.distributions import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
from tqdm import tqdm
from functional_tt_fabrique import orthpoly, Extended_TensorTrain
from geomloss import SamplesLoss
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from utils.utils import get_target_samples, filter_tensor


# Set seed
#
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)


@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None):
    assert isinstance(rectified_flow, (RectifiedFlowTT, RectifiedFlowNN))
    if isinstance(rectified_flow, RectifiedFlowTT):
        suffix = "tt"
    elif isinstance(rectified_flow, RectifiedFlowNN):
        suffix = "nn"
    else:
        raise ValueError(f"Unsupported recflow model type : {type(rectified_flow)}")

    print(f"Drawing plot for model of class : {type(rectified_flow)}")
    traj = rectified_flow.sample_ode(z0=z0, N=N)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Actual vs Generated Distribution')
    x_lim = (-10, 10)
    y_lim = (-10, 10)

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


def get_train_tuple(z0=None, z1=None):
    t = torch.rand((z1.shape[0], 1))
    z_t = t * z1 + (1. - t) * z0
    target = z1 - z0
    return z_t, t, target


def train_rectified_flow_nn(rectified_flow_nn, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    for i in tqdm(range(inner_iters + 1), desc="training recflow-nn "):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = get_train_tuple(z0=z0, z1=z1)

        pred = rectified_flow_nn.model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
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


class RectifiedFlowTT:
    def __init__(self, basis_degree, limits, data_dim, ranks):
        basis_degrees = [basis_degree] * (data_dim + 1)  # hotfix by charles that made the GMM work
        # ranks = [1] + [tt_rank] * data_dim + [1]
        domain = [list(limits) for _ in range(data_dim)] + [[0, 1]]
        print("Generating Orthopoly Func.(This might take a couple of secs)")
        op = orthpoly(basis_degrees, domain)
        self.ETTs = [Extended_TensorTrain(op, ranks) for i in range(data_dim)]

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

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
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
    return rectified_flow_tt


def hopt_objective(args):
    r1 = args['r1']
    r2 = args['r2']
    print(f"Creating a RecFlow TT object with r={(r1, r2)}")
    ranks = [1] + [r1, r2] + [1]
    basis_degree = 30
    limits = (-20, 20)
    model = RectifiedFlowTT(ranks=ranks, basis_degree=basis_degree, data_dim=data_dim, limits=limits)
    print("training tt-recflow")
    reg_coeff = 1e-20
    n_itr = 40
    tol = 5e-10
    samples_loss_ = SamplesLoss(loss="sinkhorn")
    x0 = args['init_model'].sample(torch.Size([args['N']]))
    x1_train = get_target_samples(dataset_name=args['dataset_name'], n_samples=args['N'])
    x1_test = get_target_samples(dataset_name=args['dataset_name'], n_samples=args['N'])
    checkpoint_sinkhorn = samples_loss_(x1_train, x1_test)
    print(f"Checkpoint sinkhorn between train and test : {checkpoint_sinkhorn}")
    print(f"r={(r1, r2)}")
    train_rectified_flow_tt(rectified_flow_tt=model, x0=x0, x1=x1_train, reg_coeff=reg_coeff,
                            iterations=n_itr, tol=tol)
    gen_sample = model.sample_ode(z0=initial_model.sample(torch.Size([n_samples])), N=1000)[-1]
    gen_sample_filtered = filter_tensor(x=gen_sample)
    gen_sinkhorn = samples_loss_(x1_test, gen_sample_filtered).item()
    print(f"with r = {(r1, r2)}gen_sinkhorn value = {gen_sinkhorn}")
    return {
        'loss': gen_sinkhorn,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
    }


def tt_recflow_hopt(init_model: Distribution, target_dataset_name: str):
    # https://github.com/hyperopt/hyperopt/issues/835
    space = {'r1': hp.randint('r1', 5, 10),
             'r2': hp.randint('r2', 5, 10),
             'init_model': init_model,
             'dataset_name': target_dataset_name,
             'N': 10000}
    trials = Trials()
    best = fmin(fn=hopt_objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
    print(f"Best parameters = {best}")
    print(f"Opt loss = {trials.best_trial['result']['loss']}")


# Main
if __name__ == '__main__':
    D = 10.
    M = D + 5
    VAR = 0.3
    DOT_SIZE = 4
    COMP = 3
    n_samples = 10000
    data_dim = 2
    model_type = "tt"  # can be nn or tt
    target_dataset_name = "swissroll"
    initial_model = MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
    samples_0 = initial_model.sample(torch.Size([n_samples]))
    samples_1 = get_target_samples(dataset_name=target_dataset_name, n_samples=n_samples)
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
    x_pairs = torch.stack([x_0, x_1], dim=1)

    recflow_model = None
    if model_type == "nn":
        print("Training nn-recflow")
        iterations = 10000
        batch_size = 2048
        input_dim = 2
        recflow_model = RectifiedFlowNN(model=MLP(input_dim, hidden_num=100), num_steps=100)
        optimizer = torch.optim.Adam(recflow_model.model.parameters(), lr=5e-3)

        recflow_model, loss_curve = train_rectified_flow_nn(recflow_model, optimizer, x_pairs, batch_size,
                                                            iterations)
        plt.plot(np.linspace(0, iterations, iterations + 1), loss_curve[:(iterations + 1)])
        plt.title('Training Loss Curve')
        plt.savefig("loss_curve_recflow_nn_1.png")

    elif model_type == "tt":
        tt_recflow_hopt(init_model=initial_model, target_dataset_name=target_dataset_name)
        print("tt recflow hopt finished")
        sys.exit(-1)
    else:
        raise ValueError(f"Unsupported recflow model type : {type(model_type)}")

    assert recflow_model is not None, "recflow_model is not initialized or trained"
    print("Drawing generated samples vs actual and trajectories")
    draw_plot(recflow_model, z0=initial_model.sample([2000]), z1=samples_1.detach().clone(), N=1000)
    print("Generating sinkhorn values")
    samples_loss = SamplesLoss(loss="sinkhorn")
    samples_11 = get_target_samples(dataset_name=target_dataset_name, n_samples=n_samples)
    samples_12 = get_target_samples(dataset_name=target_dataset_name, n_samples=n_samples)
    ref_sinkhorn = samples_loss(samples_11, samples_12)
    print(f"ref sinkhorn value = {ref_sinkhorn}")

    generated_sample = recflow_model.sample_ode(z0=initial_model.sample(torch.Size([n_samples])), N=1000)[-1]
    generated_sample_filtered = filter_tensor(x=generated_sample)
    max_ = torch.max(generated_sample_filtered, dim=0)
    min_ = torch.min(generated_sample_filtered, dim=0)
    print(f"max = {max_}")
    print(f"min = {min_}")
    gen_sinkhorn_1 = samples_loss(samples_11, generated_sample_filtered)
    gen_sinkhorn_2 = samples_loss(samples_12, generated_sample_filtered)
    gen_sinkhorn_avg = (gen_sinkhorn_1 + gen_sinkhorn_2) / 2.0
    print(f"generated sinkhorn 1 = {gen_sinkhorn_1}")
    print(f"generated sinkhorn 2 = {gen_sinkhorn_2}")
    print(f"generated sinkhorn avg = {gen_sinkhorn_avg}")
    print("Finished")

"""
Results Log
------------------
Best parameters = {'r1': 8, 'r2': 6}
Opt loss = 0.2500009078991041
tt recflow hopt finished
"""