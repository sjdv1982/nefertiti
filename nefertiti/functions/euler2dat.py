import numpy as np
from io import StringIO

def euler2dat(euler: np.ndarray):
    """writes out Euler angles in ATTRACT 2-body .dat format"""
    result = StringIO()
    print("""#pivot auto
#centered receptor: true
#centered ligands: true""", file=result)
    for n, (phi, ssi, rot) in enumerate(euler):
        print("#{}".format(n+1), file=result)
        print("0 0 0 0 0 0", file=result)
        print("{:.9f} {:.9f} {:.9f} 0 0 0".format(phi, ssi, rot), file=result)
    return result.getvalue()

if __name__ == "__main__":    
    import sys
    euler = np.load(sys.argv[1])
    dat_txt = euler2dat(euler)
    print(dat_txt, end="")

    