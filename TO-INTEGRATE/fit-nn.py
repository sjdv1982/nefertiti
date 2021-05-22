import numpy as np
import scipy.stats
import sys, os
from numpy.polynomial.polynomial import Polynomial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

greedy_rmsd_file = sys.argv[1]
randombest_rmsd_file = sys.argv[2]
nres = int(sys.argv[3]) # number of residues
nproto = int(sys.argv[4]) # size of prototype library
outfile = sys.argv[5]  # plot file

fragsize = 4  # tetraresidue fragments
nfrags = nres - 3

skip_nn = 1000 # don't use the first values to fit the ensemble size
skip_last_randombest = 10000 # skip last values of the randombest ensemble (seems to plateau off)

# Parameters that were used to generate the randombest ensemble
nnsize = 100000
inverse_factor = 1000  # keep the random RMSD that are in the top 0.1 percentile
chunksize = 100000
chunks = int(nnsize * inverse_factor / chunksize + 0.99999)
randombest_samples = chunks * chunksize
randombest_logfactor = np.log(nproto) * nfrags - np.log(chunks * chunksize)

def get_log_nnsize(nearnative_ensemble, logfactor):
    nnsize = len(nearnative_ensemble)
    nbins = int(nnsize / 50)
    h,bins = np.histogram(nearnative_ensemble,bins=nbins)
    h2 = h.cumsum()
    log_nnsize = np.log(h2) + logfactor
    return bins[:-1], log_nnsize

def fit(nearnative_ensemble, logfactor):
    nnsize = len(nearnative_ensemble)
    nbins = int(nnsize / 50)
    h,bins = np.histogram(nearnative_ensemble,bins=nbins)
    h2 = h.cumsum()
    mask = (h2 > skip_nn)
    log_nnsize = np.log(h2) + logfactor
    rmsds = bins[:-1]

    # full second-degree polynomial fit
    ### c,b,a = Polynomial.fit(x=log_nnsize[mask], y=rmsds[mask], deg=2).convert().coef
    # or: linear regression, with no first-order term
    b = 0
    a, c, _, _, _ = scipy.stats.linregress(log_nnsize[mask]**2, rmsds[mask])
    p = 1/a
    q = (b*b-4*a*c)/(4*a*a)
    r = -b/(2*a)
    return a,b,c, p,q,r

def fit_with_intercept(nearnative_ensemble, logfactor, intercept):
    nnsize = len(nearnative_ensemble)
    nbins = int(nnsize / 50)
    h,bins = np.histogram(nearnative_ensemble,bins=nbins)
    h2 = h.cumsum()
    mask = (h2 > skip_nn)
    log_nnsize = np.log(h2[mask]) + logfactor
    rmsds = 0.5 * (bins[:-1] + bins[1:])[mask]
    rmsds = np.concatenate((
        np.repeat(intercept, 10000),
        rmsds
    ))
    log_nnsize = np.concatenate((
        np.repeat(0, 10000),
        log_nnsize
    ))

    # full second-degree polynomial fit
    ### c,b,a = Polynomial.fit(x=log_nnsize, y=rmsds, deg=2).convert().coef
    # or: linear regression, with no first-order term
    b = 0
    a, c, _, _, _ = scipy.stats.linregress(log_nnsize**2, rmsds)

    p = 1/a
    q = (b*b-4*a*c)/(4*a*a)
    r = -b/(2*a)
    return a,b,c, p,q,r

randombest = np.load(randombest_rmsd_file)
randombest = randombest[:-skip_last_randombest]
bins_randombest, log_nnsize_randombest = get_log_nnsize(randombest, randombest_logfactor)
greedy = np.load(greedy_rmsd_file)
print("best greedy", greedy.min())
print("worst greedy", greedy.max())
greedy_intercept = fit(greedy, 0)[2]
print("greedy intercept", greedy_intercept)
a,b,c, p,q,r = fit_with_intercept(
    randombest, randombest_logfactor,
    intercept=greedy_intercept
)
print("p:", "%.6g" % p )
print("q:", "%.6g" % q )
log_nnsize_randombest_fitted3 = np.sqrt(p*bins_randombest+q) + r

def format_fn2(tick_val, tick_pos):
    if tick_val < 0:
        return ''
    else:
        size = np.exp(np.sqrt(tick_val))
        return "%.2e" % size

fig = plt.figure()
ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FuncFormatter(format_fn2))
ax.plot(bins_randombest, log_nnsize_randombest**2, color="blue",linewidth=5)
ax.plot(bins_randombest, log_nnsize_randombest_fitted3**2,color="red")
ax.plot([greedy_intercept], [0],color="red", marker="x")
plt.savefig(outfile)
