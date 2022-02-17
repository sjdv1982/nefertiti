"""
Answer the question:
if you sample around structure A, that has a certain RMSD R from structure B,
 how many of the samples are closer to B than R?

How many of them are closer than 0.99 * R, 0.95 * R, etc?

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
structure_A_index = int(sys.argv[3])  # structure A. position in refe_array, starting at 1. "7" for octa7
structure_B_index = int(sys.argv[4])  # structure B

fraglib = sys.argv[5] # e.g. ../fraglib/dummy.npy
refe_array = np.load(refe_array)
structure_A_0 = refe_array[structure_A_index-1]  # (nfrags, 4, 3) backbone coordinates

structure_B_0 = refe_array[structure_B_index-1]  # (nfrags, 4, 3) backbone coordinates

traj = np.load(trajfile)
fraglib = np.load(fraglib)

fraglen = fraglib.shape[1]
_, structure_A, _, _ = prepare_backbone(structure_A_0, fraglen) # (nfrags, fraglen, 4, 3)
structure_A = structure_A.reshape(-1, 4)[:, :3]
_, structure_B, _, _ = prepare_backbone(structure_B_0, fraglen) # (nfrags, fraglen, 4, 3)
structure_B = structure_B.reshape(-1, 4)[:, :3]

_, rmsd_R = superimpose(structure_A, structure_B)


print()
print("Original RMSD between structures A and B: {:.3f} Å".format(rmsd_R))

rmsd_alt = calc_rmsd(traj, structure_B_0, fraglib)
print("RMSD of nearest-native ({:.3f} Å to structure A) to structure B: {:.3f} Å".format(rmsd_original[0], rmsd_alt[0]))
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
    msd_original = rmsd_original[:top]**2
    msd_alt = rmsd_alt[:top]**2
    mean_sampling = np.sqrt(msd_original.mean())
    print("#" * 50)
    print("# top {:d}, mean sampling {:.2f} Å".format(top, mean_sampling))
    print("#" * 50)
    print()
    alt_mean_rmsd = np.sqrt(msd_alt.mean())
    print("Root mean distance to structure B: {:.2f} Å".format(alt_mean_rmsd))
    alt_mean_rmsd_expected = np.sqrt((msd_original + rmsd_R**2).mean())
    #dist_scale = np.sqrt(alt_mean_rmsd / alt_mean_rmsd_expected)
    print("(expected based on independent distances): {:.2f} Å".format(alt_mean_rmsd_expected))
    #print(dist_scale)
    print()
    for improvement in 0, 0.02, 0.05, 0.1, 0.15, 0.2:
        threshold = rmsd_R * (1-improvement) #* dist_scale
        print("Improvement threshold: {:d} % ({:.2f} Å)".format(int(100*improvement), improvement*rmsd_R))
        npass = (rmsd_alt[:top] <= threshold).sum()
        frac = npass / len(rmsd_alt[:top])
        print("Percentage that passes: {:.2f}".format(frac*100))
        if npass >= 10:
            cosang = (mean_sampling**2 + rmsd_R **2 - threshold**2)/(2 * mean_sampling * rmsd_R )
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