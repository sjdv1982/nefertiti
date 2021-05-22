CASP14 T0133 D1, 100 residues
=============================

Random search, generating 100 000 random trajectories
that are the best 0.1 % (i.e. 100 million were sampled)

PEP-FOLD library (120 prototypes)
=================================
0.1 % threshold: 10.6039 A 

together with a greedy search of 2000 trajectories, fitted with fit-nn.py => fit-nn.png
best greedy: 0.6648
worst greedy: 0.6728
greedy intercept: 0.664484580852933

p and q are NOT realistic (see fit-nn.png). Instead, infer p and q from first principles 
(NOT using RMSD of random trajectories, only their THRESHOLD)

100 residues = 97 fragments
0.1 percentile: log(120)*97 - log(1000) = 457.4789437708763
p: 0.1 percentile ** 2 / (threshold - greedy intercept) = 21056.266909877904
q: -greedy intercept * p = -13991.564691937703
log(NN) = sqrt(21056.266909877904 * RMSD - 13991.564691937703)
for RMSD=1.58 (AlphaFold2): log(NN) = 138.84285010640406  (10**60.2986836529316)
for RMSD=2.21 (Baker): log(NN) = 180.39618947996786 (10**78.34506964752349), 18 orders of magnitude
for RMSD=3.14 (best of the others): log(NN) = 228.3092494952382  (10*99.15), 39 orders of magnitude

Dummy library (111 prototypes)
==============================

0.1 % threshold: 10.2518 A

together with a greedy search of 2000 trajectories, fitted with fit-nn.py => fit-nn-dummy.png
best greedy: 0.904768
worst greedy: 0.91372
greedy intercept 0.9049755892645333

p and q are NOT realistic (see fit-nn.png). Instead, infer p and q from first principles 
(NOT using RMSD of random trajectories, only their THRESHOLD)

0.1 percentile: log(111)*97 - log(1000) = 449.9166742483143
p: 0.1 percentile ** 2 / (threshold - greedy intercept) = 21657.09931632724
q: -greedy intercept * p = -19599.146215553763
log(NN) = sqrt(21657.09931632724 * RMSD - 19599.146215553763)
for RMSD=1.58 (AlphaFold2): log(NN) = 120.90934911843368  (10**52.51026313264955)
for RMSD=2.21 (Baker): log(NN) = 168.11616006062425 (10**73.01192063309296), 20.5 orders of magnitude
for RMSD=3.14 (best of the others): log(NN) = 220.00942170214842 (10**95.54887781196858), 33 orders of magnitude

