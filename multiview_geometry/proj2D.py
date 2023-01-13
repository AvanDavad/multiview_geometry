import numpy as np
from multiview_geometry import utils

def get_line_points_xy(line, eps=1e-6):
    p1, p2 = get_line_points(line, eps=eps)
    p1_ih = utils.from_hom(p1)
    p2_ih = utils.from_hom(p2)
    x = [p1_ih[0], p2_ih[0]]
    y = [p1_ih[1], p2_ih[1]]
    return x, y

def get_line_points(line, eps=1e-6):
    """
    line in projective 2-space.
    get 2 points on the line. These 2 points should be
    outside of the field of view.
    The line is represented as a homogeneous 3-vector.
    """
    A_line = get_line_repr(line)
    p1 = A_line[:,0]
    p2 = A_line[:,1]

    # finding points not in infinity
    Pts = np.column_stack([p1, p2, p1+p2]) # 2 of these must be finite
    indices = np.where(Pts[-1] != 0.0)[0]

    # p1, p2 are finite (homogeneous) points
    p1 = Pts[:, indices[0]]
    p2 = Pts[:, indices[1]]

    tinf = -p1[-1] / p2[-1] # p1 + tinf * p2 is an ideal

    p3 = p1 + (tinf - eps) * p2
    p4 = p1 + (tinf + eps) * p2

    return p3, p4

def line_conic_intersection(line, C):
    if not np.allclose(C, C.T):
        raise ValueError(f"conic matrix C must be symmetric: {C}")
    A_line = get_line_repr(line)
    M = A_line.T @ C @ A_line
    a, b, c = M[1,1], 2*M[0,1], M[0,0]
    t1, t2 = solve_quadratic(a, b, c)
    vt1 = np.array([1.0, t1])
    vt2 = np.array([1.0, t2])
    p1 = A_line.dot(vt1)
    p2 = A_line.dot(vt2)
    return p1, p2

def get_line_repr(line):
    assert line.shape == (3,)
    u, _, _ = np.linalg.svd(line[:, None], full_matrices=True)
    A_line = u[:,1:]
    return A_line

def solve_quadratic(a, b, c):
    """
    solve a*x^2 + b*x + c = 0
    """
    if a == 0.0:
        x = -c / b
        return (x, x)
    D = b*b - 4*a*c
    assert D >= 0, f"no real roots ({a}, {b}, {c})"
    sD = np.sqrt(D)
    x1 = (-b + sD) / (2*a)
    x2 = (-b - sD) / (2*a)
    return (x1, x2)

##########
# CONICS #
##########

def get_mask_for_plot_lines(arr, xmin, xmax, ymin, ymax):
    """
    helper function
    """
    assert arr.shape[1] == 2
    assert arr.ndim == 2
    mask_x = np.logical_and(arr[:,0] < xmax, arr[:,0] > xmin)
    mask_y = np.logical_and(arr[:,1] < ymax, arr[:,1] > ymin)
    mask = np.logical_and(mask_x, mask_y)
    return mask


def mask_dilation(mask):
    """
    helper function

    mask_dilation([0,0,0,1,0,0,0,0,0,1])
    ->            [0,0,1,1,1,0,0,0,1,1]
    """
    mask = np.asarray(mask).astype(int)
    diff = np.diff(mask)
    wh_s = np.where(diff==1)[0]
    wh_e = np.where(diff==-1)[0] + 1
    mask[wh_s] = 1
    mask[wh_e] = 1
    return mask

def iter_mask(mask):
    """
    helper function
    """
    mask = np.concatenate([[0], mask.astype(int), [0]])
    diff = np.diff(mask)
    wh_s = np.where(diff==1)[0]
    wh_e = np.where(diff==-1)[0]
    for s, e in zip(wh_s, wh_e):
        yield (s, e)

def get_points_for_conic_A(A, xmin, xmax, ymin, ymax, logmin, logmax, n):
    """
    helper function
    """
    t = np.sort(np.concatenate([-np.logspace(logmin, logmax, n), np.logspace(logmin, logmax, n)]))
    vt = np.column_stack([np.ones_like(t), t, t*t])
    xt = vt @ A.T
    xt_ih = xt[:, :2] / xt[:, 2:3]
    mask = get_mask_for_plot_lines(xt_ih, xmin, xmax, ymin, ymax)
    mask = mask_dilation(mask)
    points = []
    for s, e in iter_mask(mask):
        points.append(xt_ih[s:e])
    return points

def get_conic_points(C, xmin=-3, xmax=3, ymin=-3, ymax=3, logmin=-5, logmax=5, n=1000, M=[0, 0]):
    assert np.allclose(C, C.T)
    M = np.array([M[0], M[1], 1.0])
    assert M.dot(C.dot(M)) != 0.0, "M is on the conic"

    CM_line = C.dot(M)
    L, R = line_conic_intersection(CM_line, C)
    A = np.column_stack(
        [
            -0.5 * (M.dot(C.dot(M))) / (L.dot(C).dot(R)) * L,
            M,
            R,
        ]
    )

    points = get_points_for_conic_A(A, xmin, xmax, ymin, ymax, logmin, logmax, n)

    return points

def get_Jc(A):
    """
    helper function

    A is a 3x3 matrix to parametrize a conic: A * [1 t t^2]^T is a point (t is the parameter)
    C is the conic matrix:

    C = a    b/2   d/2
        b/2  c     e/2
        d/2  e/2   f

    the derivative (Jacobian) of "(A vt)^T C (A vt)" w.r.t [a b c d e f]^T is Jc
    where vt := [1 t t^2]^T

    This function returns Jc, a 5x6 matrix.
    """
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = A.flatten()
    Jc = np.array(
        [
            [a11**2, a11*a21, a21**2, a11*a31, a21*a31, a31**2],
            [2*a11*a12, a11*a22+a12*a21, 2*a21*a22, a11*a32+a12*a31, a21*a32+a22*a31, 2*a31*a32],
            [2*a11*a13+a12**2, a11*a23+a12*a22+a13*a21, 2*a21*a23+a22**2, a11*a33+a12*a32+a13*a31, a21*a33+a22*a32+a23*a31, 2*a31*a33+a32**2],
            [2*a12*a13, a12*a23+a13*a22, 2*a22*a23, a12*a33+a13*a32, a22*a33+a23*a32, 2*a32*a33],
            [a13**2, a13*a23, a23**2, a13*a33, a23*a33, a33**2]
        ]
    )
    return Jc

def cvec2C(cvec):
    """
    c is the vector form of C: [a, b, c, d, e, f]
    where C =
               a   b/2  d/2
              b/2   c   e/2
              d/2  e/2   f
    """
    assert cvec.shape == (6,)
    a, b, c, d, e, f = cvec
    C = np.array(
        [
            [a, b/2, d/2],
            [b/2, c, e/2],
            [d/2, e/2, f],
        ]
    )
    return C

def C2cvec(C):
    assert C.shape == (3, 3)
    return np.array(
        [
            C[0,0],
            C[0,1] + C[1,0],
            C[1,1],
            C[0,2] + C[2,0],
            C[1,2] + C[2,1],
            C[2,2]
        ]
    )

def get_C_from_A(A):
    """
    given a 3x3 matrix A that parametrizes a conic,
    get the conic matrix C.
    """
    Jc = get_Jc(A)
    null_vec = np.linalg.svd(Jc)[2][-1]
    C = cvec2C(null_vec)
    return C
