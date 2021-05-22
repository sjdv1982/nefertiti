import numpy as np
from grow.fit import fit
import scipy.stats
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial

dummy = False
if len(sys.argv) > 1:
    assert sys.argv[1:] == ["dummy"]
    dummy = True

from matplotlib.ticker import FuncFormatter, MaxNLocator
def format_fn(tick_val, tick_pos):
    if tick_val < 0:
        return ''
    else:
        size = np.exp(tick_val)
        return "%.2e" % size

def format_fn2(tick_val, tick_pos):
    if tick_val < 0:
        return ''
    else:
        size = np.exp(np.sqrt(tick_val))
        return "%.2e" % size

skip_nn = 1000 # don't use the first values to fit the ensemble size
skip_last_randombest = 10000 # skip last values of the randombest ensemble (seems to plateau off)

dodecacommon = np.load("dodecacommon-aligned.npy")

prototypes_file = "prototypes/prototypes.npy"
if dummy:
    prototypes_file = "prototypes-dummy/dummy-prototypes.npy"
bbsize = 4

prototypes = np.load(prototypes_file)
assert len(prototypes.shape) == 3, prototypes.shape
assert prototypes.shape[-1] == 3, prototypes.shape
nproto = len(prototypes)
fragsize = prototypes.shape[1]
protosize = fragsize//bbsize

dodecacommon = np.load("dodecacommon-aligned.npy")
nfrags = dodecacommon.shape[1] - protosize + 1

# Parameters that were used to generate the randombest ensemble
nnsize = 100000
inverse_factor = 1000  # keep the random RMSD that are in the top 0.1 percentile
chunksize = 100000
chunks = int(nnsize * inverse_factor / chunksize + 0.99999)
randombest_samples = chunks * chunksize
randombest_logfactor = np.log(nproto) * nfrags - np.log(chunks * chunksize)

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
    return a,b,c, p,q,r, rmsds, log_nnsize, mask


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

_ = """
logN = log(ensemble_size)
logN2 = (logN)**2
Eq1: RMSD = a * logN **2 + b * logN + c
Eq2: logN = sqrt(p*RMSD + q) + r    (inverse of Eq1)

Statistics to be printed, per column:

1.  motif
2.  ex_best_rmsd: ex_c from exhaustive search
3.  ex_100k_rmsd: worst RMSD from exhaustive search
4.  ex_a: parameter a for Eq1 for exhaustive search ensemble
5.  ex_b: parameter b for Eq1 for exhaustive search ensemble
6.  ex_c: parameter c for Eq1 for exhaustive search ensemble
7.  ex_p: parameter p for Eq2 for exhaustive search ensemble
8.  ex_q: parameter q for Eq2 for exhaustive search ensemble
9.  ex_r: parameter r for Eq2 for exhaustive search ensemble
10. ex_corr_fit: logN of exhaustive search ensemble:
      observed vs calculated via Eq2,
      Pearson correlation
11. ex_corr_logN2: logN2 vs RMSD, Pearson correlation

12. rnd_best_rmsd: ex_c from random search
13. rnd_100k_rmsd: worst RMSD from random search
14. rnd_a: parameter a for Eq1 for random search ensemble
15. rnd_b: parameter b for Eq1 for random search ensemble
16. rnd_c: parameter c for Eq1 for random search ensemble
17. rnd_p: parameter p for Eq2 for random search ensemble
18. rnd_q: parameter q for Eq2 for random search ensemble
19. rnd_r: parameter r for Eq2 for random search ensemble
20. rnd_corr_fit: logN of random search ensemble:
      observed vs calculated via Eq2,
      Pearson correlation
21. rnd_corr_logN2: logN2 vs RMSD, Pearson correlation

22. rndb_a: parameter a for Eq1 for random search ensemble, with ex_c
23. rndb_b: parameter b for Eq1 for random search ensemble, with ex_c
24. rndb_c: parameter c for Eq1 for random search ensemble, with ex_c
25. rndb_p: parameter p for Eq2 for random search ensemble, with ex_c
26. rndb_q: parameter q for Eq2 for random search ensemble, with ex_c
27. rndb_r: parameter r for Eq2 for random search ensemble, with ex_c
28. rndb_corr_fit: logN of random search ensemble, with ex_c:
      observed vs calculated via Eq2,
      Pearson correlation
29. rndb_size_100k_rmsd:
    Extrapolated size for the near-native ensemble at ex_100k_rmsd (true size: 100k)

30. greedy_c: parameter c for Eq1, estimated with greedy search

31. rndg_a: parameter a for Eq1 for random search ensemble, with greedy_c
32. rndg_b: parameter b for Eq1 for random search ensemble, with greedy_c
33. rndg_c: parameter c for Eq1 for random search ensemble, with greedy_c
34. rndg_p: parameter p for Eq2 for random search ensemble, with greedy_c
35. rndg_q: parameter q for Eq2 for random search ensemble, with greedy_c
36. rndg_r: parameter r for Eq2 for random search ensemble, with greedy_c
37. rndg_corr_fit: logN of random search ensemble, with greedy_c:
      observed vs calculated via Eq2,
      Pearson correlation
38. rndg_size_100k_rmsd:
    Extrapolated size for the near-native ensemble at ex_100k_rmsd (true size: 100k)

"""
header = [l.split()[1].rstrip(":") for l in _.splitlines() if len(l.strip()) and l.split()[0][-1] == "." and l.split()[0][:-1].isnumeric()]
print(" ".join(header))
for motif in range(1, 100+1):
    print(motif, end=" ")
    exhaustive_file = "prototype-superimpose-motif-%d-rmsd.npy" % motif
    if dummy:
        exhaustive_file = "dummy-superimpose-motif-%d-rmsd.npy" % motif
    exhaustive = np.load(exhaustive_file)
    print("%.4f" % exhaustive[0], "%.4f" % exhaustive[-1], end=" ")
    a,b,c, p,q,r, exhaustive_rmsds, log_nnsize_exhaustive, ex_mask = fit(exhaustive, 0)
    print("%.6g" % a, "%.6g" % b, "%.6g" % c, end=" ")
    print("%.6g" % p, "%.6g" % q, "%.6g" % r, end=" ")
    log_nnsize_exhaustive_fitted = np.sqrt(p*exhaustive_rmsds+q) + r
    exhaustive_coeff = a,b,c, p,q,r
    ex_corr_fit = scipy.stats.pearsonr(
        log_nnsize_exhaustive[ex_mask],
        log_nnsize_exhaustive_fitted[ex_mask]
    )[0]
    print("%.6f" % ex_corr_fit, end=" ")
    ex_corr_logN2 = scipy.stats.pearsonr(
        log_nnsize_exhaustive**2,
        exhaustive_rmsds
    )[0]
    print("%.6f" % ex_corr_logN2, end=" ")

    randombest_file = "randombest-prototype-%d.npy" % motif
    if dummy:
        randombest_file = "randombest-dummy-%d.npy" % motif
    randombest = np.load(randombest_file)
    randombest = randombest[:-skip_last_randombest]
    print("%.4f" % randombest[0], "%.4f" % randombest[-1], end=" ")
    a,b,c, p,q,r, randombest_rmsds, log_nnsize_randombest, rnd_mask = fit(randombest, randombest_logfactor)
    print("%.6g" % a, "%.6g" % b, "%.6g" % c, end=" ")
    print("%.6g" % p, "%.6g" % q, "%.6g" % r, end=" ")
    log_nnsize_randombest_fitted = np.sqrt(p*randombest_rmsds+q) + r
    randombest_coeff = a,b,c, p,q,r
    rnd_corr_fit = scipy.stats.pearsonr(
        log_nnsize_randombest[rnd_mask],
        log_nnsize_randombest_fitted[rnd_mask]
    )[0]
    print("%.6f" % rnd_corr_fit, end=" ")
    rnd_corr_logN2 = scipy.stats.pearsonr(
        log_nnsize_randombest[rnd_mask]**2,
        randombest_rmsds[rnd_mask]
    )[0]
    print("%.6f" % rnd_corr_logN2, end=" ")

    a,b,c, p,q,r = fit_with_intercept(
        randombest, randombest_logfactor,
        intercept=exhaustive_coeff[2]
    )
    print("%.6g" % a, "%.6g" % b, "%.6g" % c, end=" ")
    print("%.6g" % p, "%.6g" % q, "%.6g" % r, end=" ")

    log_nnsize_randombest_fitted2 = np.sqrt(p*randombest_rmsds+q) + r
    rndb_corr_fit = scipy.stats.pearsonr(
        log_nnsize_randombest[rnd_mask],
        log_nnsize_randombest_fitted2[rnd_mask]
    )[0]
    print("%.6f" % rndb_corr_fit, end=" ")
    rndb_size_100k_rmsd =  int(np.exp(np.sqrt(p*exhaustive[-1]+q) + r)+0.5)
    print("%d" % rndb_size_100k_rmsd, end=" ")

    log_extrapolate_bwd2 = np.sqrt(p*exhaustive_rmsds+q) + r
    p,q,r = exhaustive_coeff[3:]
    log_extrapolate_fwd = np.sqrt(p*randombest_rmsds+q) + r
    p,q,r = randombest_coeff[3:]
    log_extrapolate_bwd = np.sqrt(p*exhaustive_rmsds+q) + r

    greedy_file = "greedy-prototype-%d-rmsd.npy" % motif
    if dummy:
        greedy_file = "greedy-dummy-%d-rmsd.npy" % motif
    greedy = np.load(greedy_file)

    greedy_intercept = fit(greedy, 0)[2]
    print("%.6g" % greedy_intercept, end=" ")
    a,b,c, p,q,r = fit_with_intercept(
        randombest, randombest_logfactor,
        intercept=greedy_intercept
    )
    print("%.6g" % a, "%.6g" % b, "%.6g" % c, end=" ")
    print("%.6g" % p, "%.6g" % q, "%.6g" % r, end=" ")

    log_extrapolate_bwd3 = np.sqrt(p*exhaustive_rmsds+q) + r

    log_nnsize_randombest_fitted3 = np.sqrt(p*randombest_rmsds+q) + r
    rndg_corr_fit = scipy.stats.pearsonr(
        log_nnsize_randombest[rnd_mask],
        log_nnsize_randombest_fitted3[rnd_mask]
    )[0]
    print("%.6f" % rndg_corr_fit, end=" ")
    rndg_size_100k_rmsd =  int(np.exp(np.sqrt(p*exhaustive[-1]+q) + r)+0.5)
    print("%d" % rndg_size_100k_rmsd, end=" ")
    print()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(format_fn2))
    ax.plot(exhaustive_rmsds, log_nnsize_exhaustive**2, color="blue",linewidth=5)
    ax.plot(exhaustive_rmsds[ex_mask], log_nnsize_exhaustive_fitted[ex_mask]**2, color="lightgreen")
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd**2)
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd2**2,color="pink")
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd3**2,color="red")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest[rnd_mask]**2, color="blue",linewidth=5)
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted[rnd_mask]**2,color="lightgreen")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted2[rnd_mask]**2,color="pink")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted3[rnd_mask]**2,color="red")
    ax.plot(randombest_rmsds[rnd_mask], log_extrapolate_fwd[rnd_mask]**2)
    outfile = "analysis/rmsd-logN2-%d.png" % (motif)
    if dummy:
        outfile = "dummy-" + outfile
    plt.savefig(outfile)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(format_fn2))
    ax.plot(exhaustive_rmsds, log_nnsize_exhaustive**2, color="blue",linewidth=5)
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest[rnd_mask]**2, color="blue",linewidth=5)
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd3**2,color="red")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted3[rnd_mask]**2,color="red")
    outfile = "analysis/fit-nn-%d.png" % (motif)
    if dummy:
        outfile = "dummy-" + outfile
    plt.savefig(outfile)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.plot(exhaustive_rmsds, log_nnsize_exhaustive, color="blue", linewidth=5)
    ax.plot(exhaustive_rmsds[ex_mask], log_nnsize_exhaustive_fitted[ex_mask], color="lightgreen")
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd)
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd2,color="pink")
    ax.plot(exhaustive_rmsds, log_extrapolate_bwd3,color="red")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest[rnd_mask], color="blue", linewidth=5)
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted[rnd_mask],color="lightgreen")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted2[rnd_mask],color="pink")
    ax.plot(randombest_rmsds[rnd_mask], log_nnsize_randombest_fitted3[rnd_mask],color="red")
    ax.plot(randombest_rmsds[rnd_mask], log_extrapolate_fwd[rnd_mask])
    outfile = "analysis/rmsd-logN-%d.png" % (motif)
    if dummy:
        outfile = "dummy-" + outfile
    plt.savefig(outfile)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_y = log_nnsize_exhaustive[0]
    max_y = log_nnsize_randombest[-1]
    ax.plot( (min_y,max_y), (min_y,max_y), linewidth=5)
    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda tick_val, tick_pos: ""))
    ax.plot(log_nnsize_exhaustive, log_nnsize_exhaustive_fitted)
    ax.plot(log_nnsize_exhaustive, log_extrapolate_bwd,color="brown")
    ax.plot(log_nnsize_exhaustive, log_extrapolate_bwd2,color="pink")
    ax.plot(log_nnsize_exhaustive, log_extrapolate_bwd3,color="red")
    ax.plot(log_nnsize_randombest[rnd_mask], log_nnsize_randombest_fitted[rnd_mask],color="lightgreen")
    ax.plot(log_nnsize_randombest[rnd_mask], log_nnsize_randombest_fitted2[rnd_mask],color="red")
    ax.plot(log_nnsize_randombest[rnd_mask], log_extrapolate_fwd[rnd_mask])


    outfile = "analysis/logN-logN-%d.png" % (motif)
    if dummy:
        outfile = "dummy-" + outfile
    plt.savefig(outfile)

    plt.close('all')
