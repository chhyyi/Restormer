from aotools.functions.zernike import zernike_nm
import numpy as np
import torch
from torch.fft import fftn, fftshift, ifftshift, ifftn
def fftn_with_shift(x):
    return fftshift(fftn(ifftshift(x)))

def ifftn_with_shift(x):
    return fftshift(ifftn(ifftshift(x)))

def zern_polynomial(size, idx, pad=0, norm="max1", device = torch.device("cuda")):
    """
    Return idx-th zernike polynomial as a (size+2*pad, size+2*pad) shaped 2D torch tensor,
    (idx in OSA/ANSI index)

    norm:
        "max1"(default) : rescale max-value to be 1. following [zernike polynomial implementation on MATLAB by paul fricker](https://kr.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials), it says
            | "For the non-normalized polynomials, max(Znm(r=1,theta))=1 for all [n,m]."

        "p2v2" : normalize as peak-to-valley to become 2.
        "noll": 'noll' normalization, where integration over unit circle of (zern_pol)**2 becomes unity.
    """
    n = np.ceil((-3+np.sqrt(9+8*idx))/2).astype(int)
    m = 2*idx - n*(n+2)
    zern_pol = torch.zeros((size+2*pad, size+2*pad)).to(device=device)
    if pad != 0:
        zern_pol[pad:-pad, pad:-pad] = torch.from_numpy(zernike_nm(n, m, size)).to(device=device)
    elif pad == 0:
        zern_pol = torch.from_numpy(zernike_nm(n, m, size)).to(device=device)

    if norm=="max1":
        zern_pol = zern_pol/zern_pol.max()
    elif norm=="p2v2":
        zern_pol = 2.0*zern_pol/(zern_pol.max()-zern_pol.min())
    elif norm=="noll":
        pass
    else:
        raise NotImplementedError
    return zern_pol

def is_in_circle(size, ctf_size):
    """
    return circular mask for coherent transfer function (optics)
    """
    x, y = np.meshgrid(np.linspace(-1., 1., size), np.linspace(-1., 1., size))
    is_in_circle = torch.tensor([[1 if i<=ctf_size else 0 for i in j] for j in np.power(np.sum((x**2, y**2), axis=0), 0.5)])
    return is_in_circle