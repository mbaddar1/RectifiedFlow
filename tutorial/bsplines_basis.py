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
def bsplines_basis_v1():
    p = 3  # order of spline (as-is; 3 = cubic)
    nknots = 11  # number of knots to generate (here endpoints count only once)
    tau = [0.1, 0.33]  # collocation sites (i.e. where to evaluate)

    knots = np.linspace(0, 1, nknots)  # create a knot vector without endpoint repeats
    k = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, p)  # create spline basis of order p on knots k

    A0 = B.collmat(tau)  # collocation matrix for function value at sites tau
    print("Finished")


if __name__ == '__main__':
    bsplines_basis_v1()
    pass
    # d = 2
    # degrees = [5] * d
    # domain = [[-1.0, 1.0] for _ in range(d)]
    # op = orthpoly(degrees, domain)
    # x = torch.tensor([-2, 0]).view(1, -1)
    # feat = op(x)
    # print("Finished")
    # knots = np.arange(-1, 1.001, 0.1)
    # c = np.array([-1, 2, 0, -1])
    # k = 2
    # bspl = BSpline.basis_element(t=knots)
    # spl_ = bspl(0.1)
    #
    # print(spl_)
    # import numpy as np
    # from scipy.interpolate import BSpline
    #
    # b = BSpline.basis_element([0, 1, 2, 3, 4])
    # u = b(2)
    # print("finished")q