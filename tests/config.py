import pytest

import array_api_strict as nparr
import array_api_compat.numpy as np

import parallelproj_python as pp

# generate list of array_api modules / device combinations to test

xp_dev_list = []
xp_dev_list.append((np, "cpu"))
xp_dev_list.append((nparr, None))

if pp.PARALLELPROJ_CUDA == 1:
    try:
        import array_api_compat.cupy as cp

        xp_dev_list.append((cp, cp.cuda.Device(0)))
    except ImportError:
        pass

try:
    import array_api_compat.torch as torch

    xp_dev_list.append((torch, "cpu"))
except ImportError:
    pass

if pp.PARALLELPROJ_CUDA == 1:
    try:
        import array_api_compat.torch as torch

        xp_dev_list.append((torch, "cuda"))
    except ImportError:
        pass

pytestmark = pytest.mark.parametrize("xp,dev", xp_dev_list)
