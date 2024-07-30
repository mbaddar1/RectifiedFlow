import sys

import numpy as np
import torch
from bspline import splinelab, bspline
from scipy.interpolate import BSpline

from RectifiedFlow.tutorial.functional_tt_fabrique import orthpoly

# Links
# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
# https://www.ibiblio.org/e-notes/Splines/basis.html
# https://stackoverflow.com/questions/45927965/how-to-extract-the-bspline-basis-from-scipy-interpolate-bspline
# https://www.sciencedirect.com/topics/engineering/spline-basis-function
# https://github.com/johntfoster/bspline
import numpy as np
from matplotlib import pyplot as plt

"""
Memoization in Python 
    * https://wiingy.com/learn/python/memoization-using-decorators-in-python/ 
    * https://www.geeksforgeeks.org/memoization-using-decorators-in-python/ 
"""


def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0
    if t[i + k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i]) / (t[i + k] - t[i]) * B(x, k - 1, i, t)
    if t[i + k + 1] == t[i + 1]:
        c2 = 0.0
    else:
        c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B(x, k - 1, i + 1, t)
    return c1 + c2


class BSplinesBasis:

    def __init__(self, x_low, x_high, n_knots, degree):
        # TODO make list of knots arrays based on data_dim
        self.degree = degree
        self.n_knots = n_knots
        a1 = np.linspace(start=float(x_low), stop=x_high, num=n_knots)
        # step = (x_high - x_low) / n_knots
        slack = 0.01
        a2 = np.linspace(start=float(x_high + slack), stop=float(x_high + slack + 0.01 * degree), num=degree + 1)
        self.u_ = np.concatenate([a1, a2])
        self.memory = {}
        self.all_calls = 0
        self.num_memoized_calls = 0

    def calculate_N(self, x, i):
        x = np.round(x, 4)  # to make memoization more effective
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
        assert 0 <= i <= (self.n_knots - 1)
        return self.__N(x, i, self.degree)

    def calculate_basis_vector(self, x):
        basis_vectors = list(map(lambda x_d: [self.calculate_N(x_d, i) for i in range(self.n_knots)], x))
        return basis_vectors

    def __N(self, x, i, p):
        self.all_calls += 1
        # print(f"Calculating N({u},{i},{p})")
        if (x, i, p) in self.memory.keys():
            # print(f"Memoized Value : N({u},{i},{p})")
            self.num_memoized_calls += 1
            return self.memory[(x, i, p)]
        if p == 0:
            val = 1 if self.u_[i] <= x < self.u_[i + 1] else 0
        else:
            c1 = (x - self.u_[i]) / (self.u_[i + p] - self.u_[i])
            N1 = self.__N(x, i, p - 1)
            c2 = (self.u_[i + p + 1] - x) / (self.u_[i + p + 1] - self.u_[i + 1])
            N2 = self.__N(x, i + 1, p - 1)
            val = c1 * N1 + c2 * N2
        self.memory[(x, i, p)] = val
        return val


def bsplines_basis_v1():
    p = 3  # order of spline (as-is; 3 = cubic)
    nknots = 11  # number of knots to generate (here endpoints count only once)
    tau = [0.1, 0.33]  # collocation sites (i.e. where to evaluate)

    knots = np.linspace(0, 1, nknots)  # create a knot vector without endpoint repeats
    k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, p)  # create a spline basis of order p on knots k

    A0 = B.collmat(tau)  # collocation matrix for function value at sites tau
    x = k
    y = A0[0, :]
    # plt.plot(k, A0[0, :])
    # plt.savefig("splines.jpg")
    print("Finished")


def bsplines_unit_test():
    """
        This function is to test my Bsplines implementation vs a naive Bsplines function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

        However, the values are not the same with bsplines method here
        https://github.com/johntfoster/bspline
    """
    degree = 2
    nknots = 10
    u_low = 0.0
    u_high = 1.0
    bsp2 = BSplinesBasis(x_low=u_low, x_high=u_high, n_knots=nknots, degree=degree)
    for i in range(0, nknots):
        for x in np.arange(u_low, u_high, 0.01):
            bval = bsp2.calculate_N(x=x, i=i)
            bval2 = B(x=x, k=degree, i=i, t=bsp2.u_)  # pass the augmented knots
            assert np.abs(bval - bval2) < 1e-6, f"N(x={x},i={i},p={degree}) are not equal : {bval}!={bval2}"
    print("Finished Unit Test")


if __name__ == '__main__':
    # bsplines_basis_v1()
    bsplines_unit_test()

    # TODO - to delete
    ### Sandbox Code ###
    # sys.exit(-1)
    # n_knots = 100
    # u_low = 0.0
    # u_high = 1.0
    # bsp1 = ""
    # bsp2 = BSplinesBasis(u_low=u_low, u_high=u_high, n_knots=n_knots, degree=2)
    # # bsp.N(u=0.1, i=1, p=2)
    # # print(bsp.num_memoized_calls)
    # u_arr = np.linspace(start=0.0, stop=1.0, num=50 + 1)
    # for i in range(n_knots):
    #     bsp_arr = []
    #     for u in u_arr:
    #         b = bsp2.calculate_N(u, i)
    #         bsp_arr.append(b)
    #     plt.plot(u_arr, bsp_arr)
    # print(f"all_calls = {bsp2.all_calls}")
    # print(f"memoized_calls = {bsp2.num_memoized_calls}")
    # print(f"fraction = {float(bsp2.num_memoized_calls) / float(bsp2.all_calls)}")
    # plt.savefig("bsplines.png")
    # print("finished")
    # # bsp.N(u=0.1, i=0, p=1)
    # # d = 2
    # # degrees = [5] * d
    # # domain = [[-1.0, 1.0] for _ in range(d)]
    # # op = orthpoly(degrees, domain)
    # # x = torch.tensor([-2, 0]).view(1, -1)
    # # feat = op(x)
    # # print("Finished")
    # # # knots = np.arange(-1, 1.001, 0.1)
    # # # c = np.array([-1, 2, 0, -1])
    # # # k = 2
    # # # bspl = BSpline.basis_element(t=knots)
    # # # spl_ = bspl(0.1)
    # # #
    # # # print(spl_)
    # # # import numpy as np
    # # # from scipy.interpolate import BSpline
    # # #
    # # # b = BSpline.basis_element([0, 1, 2, 3, 4])
    # # # u = b(2)
    # # # print("finished")q
