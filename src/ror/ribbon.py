"""
This module implement different ways of computing Twist and Writhe for a ribbon.
The functions used in the work on catenane are `local_twist`, `twist`, and `writhe`.
"""
import numpy as np
from numba import njit


def local_twist(n: np.array, tangents: np.array) -> np.array:
    """
    Computes twist for a circular ribbon

    Parameters
    ----------
    normals: np.array
        normals to the ribbon
    tangents: np.array
        tangents to the ribbon

    Returns
    -------
    np.array
        twist

    Notes
    -----
    See

    """
    normals = np.array(n)
    tangents = np.array(tangents)
    t_norms = np.linalg.norm(tangents, axis=1)
    tangents /= t_norms[:, np.newaxis]
    normals -= np.einsum('ij,ij->i', normals, tangents)[:, np.newaxis] * tangents
    n_norms = np.linalg.norm(normals, axis=1)
    normals /= n_norms[:, np.newaxis]
    # test another arrangement of normals.
    # normals *= ((np.arange(normals.shape[0]) % 2 - 0.5) * 2)[:, np.newaxis]
    # ----
    p = np.cross(tangents, np.roll(tangents, -1, axis=0))
    p_norms = np.linalg.norm(p, axis=1)
    p /= p_norms[:, np.newaxis]
    #
    n_dot_p = np.einsum('ij,ij->i', normals, p)
    n_cross_p = np.cross(normals, p)
    sgn_np = np.sign(np.einsum('ij,ij->i', tangents, n_cross_p))
    alfa = np.arccos(n_dot_p) * sgn_np
    #
    p_dot_n1 = np.einsum('ij,ij->i', np.roll(normals, -1, axis=0), p)
    p_cross_n1 = np.cross(p, np.roll(normals, -1, axis=0))
    sgn_pn1 = np.sign(np.einsum('ij,ij->i', np.roll(tangents, -1, axis=0), p_cross_n1))
    gamma = np.arccos(p_dot_n1) * sgn_pn1
    # one must ensure that the angles are defined betwen -pi and pi ...!
    tw = np.mod(alfa + gamma + np.pi, 2 * np.pi) - np.pi
    return tw / (2 * np.pi)


def local_twist_das(n: np.array, tangents: np.array):
    """
    Twist definition for normal vectors not orthogonal
    to tangents.
    Taken from: Chou FC, Lipfert J, Das R, PLOS Comp-BIO 10(8) 2014 e1003756.

    UNUSED IN THE MANUSCRIPT

    Parameters
    ----------
    n: np.array
        normals of the ribbon (not necessarily orthogonal to tangent vectors)
    tangents: np.array
        tangents to the ribbon

    Returns
    -------
    np.array
       Twist
    """
    n_norms = np.linalg.norm(n, axis=1)
    normals = n / n_norms[:, np.newaxis]
    # ----
    p = np.cross(tangents, np.roll(tangents, -1, axis=0))
    p_norms = np.linalg.norm(p, axis=1)
    p /= p_norms[:, np.newaxis]
    # beta
    p1 = np.roll(p, -1, axis=0)
    p_dot_p1 = np.einsum('ij,ij->i', p, p1)
    p_cross_p1 = np.cross(p, p1)
    sgn_np = np.sign(np.einsum('ij,ij->i', tangents, p_cross_p1))
    beta = np.arccos(p_dot_p1) * sgn_np
    # alfa
    n_dot_p = np.einsum('ij,ij->i', normals, p)
    n_cross_p = np.cross(normals, p)
    sgn_np = -np.sign(np.einsum('ij,ij->i', np.roll(tangents, -1, axis=0), n_cross_p))
    alfa = np.arccos(n_dot_p) * sgn_np
    # tw should take care of 2\pi factors
    tw = np.roll(alfa, -1) + beta - alfa
    tw += (tw < -np.pi) * 2 * np.pi - (tw > np.pi) * 2 * np.pi
    # one must ensure that the angles are defined betwen -pi and pi ...!
    return tw / (2 * np.pi)


def twist(normals: np.array, tangents: np.array) -> np.array:
    """
    Parameters
    ----------
    normals: np.array
        normals to the ribbon
    tangents: np.array
        tangents to the ribbon

    Returns
    -------
    float
        sum of local twist obtained from ribbon.local_twist(normals, tangents)
    """
    tw = local_twist(normals, tangents)
    return tw.sum()


def bending_angles(tangents: np.array) -> np.array:
    """
    Parameters
    ----------
    tangents: np.array
        contains bond vectors (tangents)

    Returns
    -------
    np.array
        bending angles between subsequent bonds
    """

    bonds = tangents
    b = np.linalg.norm(bonds, axis=1)
    bonds /= b[:, None]
    # scalar product row by row..
    cosines = np.einsum('ij,ij->i', bonds, np.roll(bonds, -1, axis=0))
    angles = np.arccos(cosines)
    return angles


def bond_lengths(tangents: np.array) -> np.array:
    """
    Compute the lengths of tangent vectors (bonds) within a ribbon.

    Parameters
    ----------
    tangents: np.array
        tangent vectors

    Returns
    -------
    np.array
    """
    return np.linalg.norm(tangents, axis=1)


def rg2(vertices: np.array) -> float:
    """
    Compute Rg2 of a catenane

    Parameters
    ----------
    vertices: np.array
        vertices of the ribbon

    Returns
    -------
    float
        value of Rg^2.
    """

    vertices -= vertices.mean()
    dists = np.linalg.norm(vertices, axis=1)
    dists2 = dists ** 2
    return dists2.mean()


def r2l(v, l, periodic=False):
    v1 = np.roll(v, -l, axis=0)
    if periodic:
        w = v1 - v
    else:
        n = v.shape[0]
        k = n - l
        w = v1[:k] - v[:k]
    d = np.linalg.norm(w, axis=1)
    d2 = d ** 2
    return d2.mean()


def writhe_discrete(vertices: np.array, tangents: np.array) -> float:
    """
    Compute the writhe of a closed curve using the discretized Gauss self-linking number

    Parameters
    ----------
    vertices: np.array
        vertices of the curve
    tangents: np.array
        tangents to the closed curve (bonds)
    Returns
    -------
    float: writhe of the curve


    Notes
    -----
    Computes a discretized version of the Gauss integral definition of Writhe.

    .. math::

        Wr = \\frac{1}{4\\pi} \\sum_{i=1}^n \\sum_{j=1}^n \mathbf{\\hat{t_i}\\times\\hat{t_j}\\cdot\\frac{r_i-r_j}{\|r_i-r_j\|^3}

    UNUSED IN THE MANUSCRIPT
    """
    # subtract r_i - r_j for all i and j
    dx = (vertices[:, np.newaxis] - vertices).reshape(-1, vertices.shape[1])
    denom = np.linalg.norm(dx, axis=1) ** 3
    # cross_product t_i X t_j for all i and j
    cross_prods = np.cross(tangents[:, np.newaxis], tangents).reshape((-1, tangents.shape[1]))
    numer = np.einsum('ij,ij->i', cross_prods, dx)
    # divide by |r_i - r_j|^3 only where this is > 0. Otherwise take 0.
    to_sum = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
    # sum all components and return
    return to_sum.sum() / (4 * np.pi)


@njit
def writhe(coords):
    """
    Compute the writhe as specified in Klenin, Langowski, Biopolymers 54, 2001
    Parameters
    ----------
    coords: x,y,z coords of the a closed loop (polymer, ribbon, or catenane)

    Returns
    -------
    writhe (float)
    """
    n = len(coords)
    wr = 0
    for i in range(2, n):
        # one has to be careful to include all segments
        for j in range((i + 1) // n, i - 1):
            w_ij = coords[j] - coords[i]
            w_ijp1 = coords[j + 1] - coords[i]
            w_ip1jp1 = coords[j + 1] - coords[(i + 1) % n]
            w_ip1j = coords[j] - coords[(i + 1) % n]
            n_1 = np.cross(w_ij, w_ijp1)
            n_2 = np.cross(w_ijp1, w_ip1jp1)
            n_3 = np.cross(w_ip1jp1, w_ip1j)
            n_4 = np.cross(w_ip1j, w_ij)
            #
            n_1 /= np.linalg.norm(n_1)
            n_2 /= np.linalg.norm(n_2)
            n_3 /= np.linalg.norm(n_3)
            n_4 /= np.linalg.norm(n_4)
            omega = np.arcsin(n_1.dot(n_2)) + np.arcsin(n_2.dot(n_3)) + \
                    np.arcsin(n_3.dot(n_4)) + np.arcsin(n_4.dot(n_1))
            v = np.cross(w_ijp1, coords[(i + 1) % n] - coords[i])
            wr += omega * np.sign(v.dot(w_ij))
    wr /= (2 * np.pi)  # there is a multiplying factor 2 due to the fact that I'm counting only half the chain
    return wr
