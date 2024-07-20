import torch


def my_mvn_log_prob1(mio, Sigma, sample):
    d = mio.shape[0]
    SigmaInv = torch.linalg.inv(Sigma)
    SigmaDet = torch.linalg.det(Sigma)
    term1 = -0.5 * d * torch.log(2 * torch.tensor(torch.pi))
    term2 = -0.5 * torch.log(SigmaDet)
    term3 = -0.5 * torch.matmul(torch.matmul((sample - mio).T, SigmaInv), sample - mio)
    res = term1 + term2 + term3
    return res


if __name__ == '__main__':
    # test log_prob of mvn
    d = 4
    Sigma = 0.1 * torch.eye(d)
    SigmaInv = torch.linalg.inv(Sigma)
    mio = torch.zeros(d)
    sample1 = torch.tensor([0.1, -0.1, 0.8, 0.0])
    mvn = torch.distributions.MultivariateNormal(loc=mio, covariance_matrix=Sigma)
    log_prob_1 = mvn.log_prob(value=sample1)
    print(log_prob_1)
    log_prob_1 = my_mvn_log_prob1(mio=mio, Sigma=Sigma, sample=sample1)
    print(log_prob_1)
