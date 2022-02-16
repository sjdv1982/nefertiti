"""
Answer the question:
if you sample around structure A, that has a certain RMSD X from structure B,
 how many of the samples are closer to B than X?

How many of them are closer than 0.99 * X, 0.95 * X, etc?

How does this scale with the sampling distance around structure A?
"""

import functools
import sys, os, numpy as np
from math import acos, pi
from nefertiti.functions.superimpose import superimpose
from nefertiti.functions.prepare_backbone import prepare_backbone
from calc_rmsd import calc_rmsd

pattern = sys.argv[1]  # e.g. ../nnfit-suppmat/data/kbest-octa7-k-100000
trajfile = pattern + "-traj.npy"
assert os.path.exists(trajfile), trajfile
rmsdfile = pattern + "-rmsd.npy"
assert os.path.exists(rmsdfile), rmsdfile
rmsd_original = np.load(rmsdfile)

refe_array = sys.argv[2]  # e.g. ../benchmarks/octacommon-aligned.npy
refe_index = int(sys.argv[3])  # structure A. position in refe_array, starting at 1. "7" for octa7
alt_refe_index = int(sys.argv[4])  # structure B

fraglib = sys.argv[5] # e.g. ../fraglib/dummy.npy
refe_array = np.load(refe_array)
refe_struc0 = refe_array[refe_index-1]  # (nfrags, 4, 3) backbone coordinates

alt_refe_struc0 = refe_array[alt_refe_index-1]  # (nfrags, 4, 3) backbone coordinates

traj = np.load(trajfile)
fraglib = np.load(fraglib)

fraglen = fraglib.shape[1]
_, refe_struc, _, _ = prepare_backbone(refe_struc0, fraglen) # (nfrags, fraglen, 4, 3)
refe_struc = refe_struc.reshape(-1, 4)[:, :3]
_, alt_refe_struc, _, _ = prepare_backbone(alt_refe_struc0, fraglen) # (nfrags, fraglen, 4, 3)
alt_refe_struc = alt_refe_struc.reshape(-1, 4)[:, :3]

_, rmsd_X = superimpose(refe_struc, alt_refe_struc)


print()
print("Original RMSD: {:.3f}".format(rmsd_X))

rmsd_alt = calc_rmsd(traj, alt_refe_struc0, fraglib)
print("Original RMSD: {:.3f}".format(rmsd_X))
print("RMSD of nearest-native ({:.3f} A): {:.3f} A".format(rmsd_original[0], rmsd_alt[0]))
print()

from scipy.special import betainc, gamma
from scipy.optimize import minimize

def calc_hypercap_area(angle, dimensionality):
    # From https://math.stackexchange.com/questions/2238156/what-is-the-surface-area-of-a-cap-on-a-hypersphere
    # Formula by S. Li
    half_n = dimensionality/2
    hypersphere_area = 2*pi**half_n / gamma(half_n) 
    a = (dimensionality - 1) / 2
    b = 0.5   
    x = np.sin(angle)**2
    return hypersphere_area * betainc(a,b,x)

def calc_hypersphere_area(dimensionality):
    half_n = dimensionality/2
    return 2*pi**half_n / gamma(half_n)

def error_function(dimensionality0, angle, target_frac):
    dimensionality = dimensionality0[0]
    if dimensionality <= 0: 
        return 999999
    #hypercap_area = calc_hypercap_area(angle, dimensionality)
    #hypersphere_area = calc_hypersphere_area(dimensionality)
    #frac = hypercap_area / hypersphere_area
    a = (dimensionality - 1) / 2
    b = 0.5   
    x = np.sin(angle)**2
    frac = betainc(a,b,x)
    
    return (frac - target_frac)**2 


for top in 100000, 10000:
    mean_sampling = rmsd_original[:top].mean()
    print("#" * 50)
    print("# top {:d}, mean_sampling {:.2f} A".format(top, mean_sampling))
    print("#" * 50)
    print()
    for improvement in 0, 0.02, 0.05, 0.1, 0.15:
        threshold = rmsd_X * (1-improvement)
        print("Improvement threshold: {:d} % ({:.2f} A)".format(int(100*improvement), improvement*rmsd_X))
        frac = (rmsd_alt[:top] <= threshold).sum() / len(rmsd_alt[:top])
        print("Percentage that passes: {:.2f}".format(frac*100))
        if frac > 0:
            cosang = (mean_sampling**2 + rmsd_X **2 - threshold**2)/(2 * mean_sampling * rmsd_X )
            ang = acos(cosang)
            print("Hypercap angle: {:.1f} degrees".format(ang/pi * 180))
            func = functools.partial(error_function, angle=ang, target_frac=frac)
            best_error = None
            dimensionality = None
            for ini_dim in np.arange(0.5, 100, 0.5):
                result = minimize(func,x0=np.array([ini_dim]))
                if not result.success:
                    continue
                if dimensionality is None or result.fun < best_error:
                    dimensionality = result.x[0]
                    best_error = result.fun
                    #print(best_error, dimensionality)
            print("Dimensionality: {:.2f}".format(dimensionality))
        print()
    print()
#np.save(resultfile, rmsd)