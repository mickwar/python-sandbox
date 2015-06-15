# A Bayesian Gaussian process model fit via MCMC
import random
import numpy as np
import scipy as sc
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt

random.seed(1)

x = np.matrix([[-4.6], [-4.2], [-4.0], [-3.8], [-3.7],
               [-3.1], [-2.0], [-1.9], [-1.6], [-0.5],
               [-0.1], [ 0.1], [ 0.4], [ 0.5], [ 0.8],
               [ 1.2], [ 1.5], [ 1.7], [ 2.5], [ 3.1]])

y = np.matrix([[ 5.1], [ 4.8], [ 3.9], [ 3.5], [ 3.9],
               [ 2.0], [ 1.3], [ 1.7], [ 1.9], [-2.3],
               [-3.5], [-2.8], [-2.4], [-2.2], [-0.4],
               [ 1.0], [ 1.1], [ 1.3], [ 1.0], [ 2.1]])

# calculating the prior mean function (from rasmussen example)
# x is a matrix
# np.multiply for element-wise multiplication
def mean_f(x, theta_m):
    a = theta_m[0]
    b = theta_m[1]
    c = theta_m[2]
    return (a * np.multiply(x, x) + b * x + c)

# alternate way to compute mean
#def mean_f(x, theta_m):
#    a = theta_m[0]
#    b = theta_m[1]
#    c = theta_m[2]
#    out = np.zeros(x.shape)
#    for i in range(x.size):
#        out[i] = a * x[i]**2 + b * x[i] + c
#    return (out)

# calculate the covariances given the distances d and covariance
# parameters theta_k
# d is a square matrix
def cov_f(d, theta_k):
    sig_y = theta_k[0]
    sig_n = theta_k[1]
    l = theta_k[2]
    return (sig_y * np.exp(-d/(2*l**2)) + np.identity(d.shape[0])*sig_n)

# d not necessarily square (for cross covariances)
def cov_f_cross(d, theta_k):
    sig_y = theta_k[0]
    l = theta_k[2]
    return (sig_y * np.exp(-d/(2*l**2)))

# calculate (manhattan) distance matrix between x and y
# right now just assumes x and y are vectors
def dist(x, y):
    out = np.asmatrix(np.zeros([x.shape[0], y.shape[0]]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            out[i, j] = abs(x[i] - y[j])
    return (out)

# calculate log posterior
# x and y are the data
# d is the distance matrix
# params is the vector of parameters
def calc_post(x, y, d, params):
    mu = mean_f(x, params[range(0, 3)])
    sigma = cov_f(d, params[range(3, 6)])
    # likelihood
    out = -0.5 * np.log(np.linalg.det(sigma)) - 0.5 * (y-mu).T * (sigma.I * (y-mu))
    # priors
    out += scipy.stats.norm.logpdf(params[0], 0, 1)
    out += scipy.stats.norm.logpdf(params[1], 0, 1)
    out += scipy.stats.norm.logpdf(params[2], 0, 1)
    out += scipy.stats.gamma.logpdf(params[3], a=1, scale=1)
    out += scipy.stats.gamma.logpdf(params[4], a=1, scale=1)
    out += scipy.stats.gamma.logpdf(params[5], a=1, scale=1)
    return (out)


# distance matrix 
dx = dist(x, x)

### mcmc setting
def autotune(accept, target = 0.25, k = 2.5):
    return ((1+(np.cosh(accept-target)-1)*(k-1)/(np.cosh(target-
        np.ceil(accept-target))-1))**np.sign(accept-target))

nburn = 10000
nmcmc = 10000
nparams = 6
params = np.zeros([nburn+nmcmc, nparams])
accept = np.zeros([nburn+nmcmc, nparams])
window = 100

params[0,] = [0, 0, 0, 1, 1, 1]
sigs = np.ones(nparams)

# parameter bounds
lower = [-np.inf, -np.inf, -np.inf, 0, 0, 0]
upper = np.zeros(nparams) + np.inf

# initial stuff
post = calc_post(x, y, dx, params[0,])
cand_param = params[0,]

dx = np.multiply(dx, dx)

# mcmc loop (start at 1 since python begins index at 0)
for i in range(1, nburn+nmcmc):
    if (np.floor((i+1.0)/window) == (i+1.0)/window):
        print i
    params[i,] = params[i-1,]
    for j in range(0, nparams):
        cand = random.gauss(params[i,j], sigs[j])
        if (cand >= lower[j]) & (cand <= upper[j]):
            cand_param[j] = cand
            cand_post = calc_post(x, y, dx, cand_param)
            if np.log(random.uniform(0, 1)) < (cand_post - post):
                post = cand_post
                params[i,j] = cand
                accept[i,j] = 1 
            else:
                cand_param[j] = params[i,j]
        else:
            cand_param[j] = params[i,j]
        if (np.floor((i+1.0)/window) == (i+1.0)/window) & (i < nburn):
            sigs[j] *= autotune(np.mean(accept[range(i-window+1, i+1),j]), k = max(window / 50, 1.1))

params = params[nburn:,]
accept = accept[nburn:,]

[np.mean(params[:,0]), np.mean(params[:,1]), np.mean(params[:,2]), np.mean(params[:,3]), np.mean(params[:,4]), np.mean(params[:,5])]

[np.mean(accept[:,0]), np.mean(accept[:,1]), np.mean(accept[:,2]), np.mean(accept[:,3]), np.mean(accept[:,4]), np.mean(accept[:,5])]

sigs

plt.plot(params[:,0])
plt.plot(params[:,1])
plt.plot(params[:,2])
plt.plot(params[:,3])
plt.plot(params[:,4])
plt.plot(params[:,5])
plt.show()

plt.plot(x, y, '+')
plt.grid()
plt.show()

### posterior predictions


# Python's equivalent to R's seq
pred_x = np.linspace(-5, 5, 100)
pred_x = np.asmatrix(pred_x).T

# test set points (cross distance)
dxp = dist(x, pred_x)
dxp = np.multiply(dxp, dxp)

# distance
dpp = dist(pred_x, pred_x)
dpp = np.multiply(dpp, dpp)

pred_y = np.asmatrix(np.zeros([nmcmc, len(pred_x)]))
for i in range(nmcmc):
    if (np.floor((i+1.0)/window) == (i+1.0)/window):
        print i
    t_m = params[i, range(0, 3)]
    t_k = params[i, range(3, 6)]
    kstar = cov_f_cross(dxp, t_k)
    sig_inv = cov_f(dx, t_k).I
    sig_star = cov_f(dpp, t_k)
    post_mu = np.squeeze(np.asarray(mean_f(pred_x, t_m) + kstar.T * sig_inv * (y - mean_f(x, t_m))))
    post_cov = sig_star - kstar.T * sig_inv * kstar
    pred_y[i,] = np.random.multivariate_normal(post_mu, post_cov)

### plotting
pred_mean = pred_y.mean(axis = 0).T
pred_upper = np.percentile(pred_y, 97.5, axis = 0)
pred_lower = np.percentile(pred_y, 2.5, axis = 0)

#plt.plot(pred_x, pred_y[range(50),].T, color='#1e90ff')
plt.plot(pred_x, pred_mean, color='#1e90ff')
plt.plot(pred_x, pred_upper, color='#bbbbbb')
plt.plot(pred_x, pred_lower, color='#bbbbbb')
plt.plot(x, y, color="#000000", linewidth=0, marker='+', markersize=10, markeredgewidth=3)
plt.show()

