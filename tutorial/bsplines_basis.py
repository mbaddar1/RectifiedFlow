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


class BSplinesBasis:

    def __init__(self, u_low, u_high, n_knots, degree):
        self.degree = degree
        self.n_knots = n_knots
        a1 = np.linspace(start=float(u_low), stop=u_high, num=n_knots + 1)
        step = (u_high - u_low) / n_knots
        a2 = np.linspace(start=float(u_high + step), stop=float(u_high + degree * step), num=degree + 1)
        self.u_ = np.concatenate([a1, a2])
        self.memory = {}
        self.all_calls = 0
        self.num_memoized_calls = 0

    def calculate_N(self, u, i):
        u = np.round(u, 2)  # to make memoization more effective
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
        assert 0 <= i <= (self.n_knots - 1)
        return self.__N(u, i, self.degree)

    def __N(self, u, i, p):
        self.all_calls += 1
        # print(f"Calculating N({u},{i},{p})")
        if (u, i, p) in self.memory.keys():
            # print(f"Memoized Value : N({u},{i},{p})")
            self.num_memoized_calls += 1
            return self.memory[(u, i, p)]
        if p == 0:
            val = 1 if self.u_[i] <= u < self.u_[i + 1] else 0
        else:
            c1 = (u - self.u_[i]) / (self.u_[i + p] - self.u_[i])
            N1 = self.__N(u, i, p - 1)
            c2 = (self.u_[i + p + 1] - u) / (self.u_[i + p + 1] - self.u_[i + 1])
            N2 = self.__N(u, i + 1, p - 1)
            val = c1 * N1 + c2 * N2
        self.memory[(u, i, p)] = val
        return val


def bsplines_basis_v1():
    p = 3  # order of spline (as-is; 3 = cubic)
    nknots = 11  # number of knots to generate (here endpoints count only once)
    tau = [0.1, 0.33]  # collocation sites (i.e. where to evaluate)

    knots = np.linspace(0, 1, nknots)  # create a knot vector without endpoint repeats
    k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, 2)  # create a spline basis of order p on knots k

    A0 = B.collmat(tau)  # collocation matrix for function value at sites tau
    x = k
    y = A0[0, :]
    # plt.plot(k, A0[0, :])
    # plt.savefig("splines.jpg")
    print("Finished")


if __name__ == '__main__':
    # bsplines_basis_v1()
    n_knots = 100
    u_low = 0.0
    u_high = 1.0
    bsp = BSplinesBasis(u_low=u_low, u_high=u_high, n_knots=n_knots, degree=2)
    # bsp.N(u=0.1, i=1, p=2)
    # print(bsp.num_memoized_calls)
    u_arr = np.linspace(start=0.0, stop=1.0, num=50 + 1)
    for i in range(n_knots):
        bsp_arr = []
        for u in u_arr:
            b = bsp.calculate_N(u, i)
            bsp_arr.append(b)
        plt.plot(u_arr, bsp_arr)
    print(f"all_calls = {bsp.all_calls}")
    print(f"memoized_calls = {bsp.num_memoized_calls}")
    print(f"fraction = {float(bsp.num_memoized_calls) / float(bsp.all_calls)}")
    plt.savefig("bsplines.png")
    print("finished")
    # bsp.N(u=0.1, i=0, p=1)
    # d = 2
    # degrees = [5] * d
    # domain = [[-1.0, 1.0] for _ in range(d)]
    # op = orthpoly(degrees, domain)
    # x = torch.tensor([-2, 0]).view(1, -1)
    # feat = op(x)
    # print("Finished")
    # # knots = np.arange(-1, 1.001, 0.1)
    # # c = np.array([-1, 2, 0, -1])
    # # k = 2
    # # bspl = BSpline.basis_element(t=knots)
    # # spl_ = bspl(0.1)
    # #
    # # print(spl_)
    # # import numpy as np
    # # from scipy.interpolate import BSpline
    # #
    # # b = BSpline.basis_element([0, 1, 2, 3, 4])
    # # u = b(2)
    # # print("finished")q
