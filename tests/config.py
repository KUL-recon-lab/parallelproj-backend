import pytest
import importlib

import array_api_compat.numpy as np

import parallelproj_backend as pp

xp_dev_list = []
xp_dev_list.append((np, "cpu"))

if importlib.util.find_spec("array_api_strict") is not None:
    import array_api_strict as nparr
    xp_dev_list.append((nparr, None))

if importlib.util.find_spec("array_api_compat.torch") is not None and importlib.util.find_spec("torch") is not None:
    torch_available = True
    import array_api_compat.torch as torch
    xp_dev_list.append((torch, "cpu"))

if pp.PARALLELPROJ_CUDA == 1:
    if torch_available:
        xp_dev_list.append((torch, "cuda"))
    if importlib.util.find_spec("array_api_compat.cupy") is not None and importlib.util.find_spec("cupy") is not None:
        import array_api_compat.cupy as cp
        xp_dev_list.append((cp, cp.cuda.Device(0)))

pytestmark = pytest.mark.parametrize("xp,dev", xp_dev_list)
