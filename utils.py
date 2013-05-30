def make_unitless(xspec_arr, radius_arr=None, ndim=None):
    """multiply by  surface area in ndim sphere / 2pi^ndim * |k|^D
    (e.g. k^3 / 2 pi^2 in 3D)
    """
    if radius_arr is None:
        radius_arr = binning.radius_array(xspec_arr)

    if ndim is None:
        ndim = xspec_arr.ndim

    factor = 2. * math.pi ** (ndim / 2.) / scipy.special.gamma(ndim / 2.)
    factor /= (2. * math.pi) ** ndim
    return xspec_arr * radius_arr ** ndim * factor
