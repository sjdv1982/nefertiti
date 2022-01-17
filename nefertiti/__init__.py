import opt_einsum
import numpy as np
np.einsum = opt_einsum.contract