import numpy as np

def from_hom(v):
    """
    homogeneous to inhomogeneous vector
    """
    v = np.asarray(v)
    v_inhom = v[..., :-1] / v[..., -1:]
    return v_inhom

def to_hom(v):
    """
    inhomogeneous to homogeneous vector
    """
    v = np.asarray(v)
    v_hom = np.concatenate(
        [v, np.ones(v.shape[:-1]+(1,), dtype=v.dtype)],
        axis=-1
    )
    return v_hom