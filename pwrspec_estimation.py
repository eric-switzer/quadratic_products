"""
    TODO:
        test that the array, weight shapes and axes are compatible
        calculate the full window function instead of just the diagonal
            i, j -> delta k -> nearest index k
        make plots of the real data's 3D power
        make uniform noise unit test
        make radius array unit test (anisotropic axes)
"""
import numpy as np
import scipy.special
import math
from core import algebra
import copy
import gc
from utils import fftutil
from utils import binning
import h5py


def cross_power_est(arr1, arr2, weight1, weight2,
                    window="blackman", nonorm=False):
    """Calculate the cross-power spectrum of a two nD fields.

    The arrays must be identical and have the same length (physically
    and in pixel number) along each axis.

    inputs are clobbered to save memory
    """

    # make the axes
    ndim = arr1.ndim
    k_axes = tuple(["k_" + axis_name for axis_name in arr1.axes])
    info = {'axes': k_axes, 'type': 'vect'}
    width = np.zeros(ndim)
    for axis_index in range(ndim):
        n_axis = arr1.shape[axis_index]
        axis_name = arr1.axes[axis_index]
        axis_vector = arr1.get_axis(axis_name)
        delta_axis = abs(axis_vector[1] - axis_vector[0])
        width[axis_index] = delta_axis

        k_axis = np.fft.fftshift(np.fft.fftfreq(n_axis, d=delta_axis))
        k_axis *= 2. * math.pi
        delta_k_axis = abs(k_axis[1] - k_axis[0])

        k_name = k_axes[axis_index]
        info[k_name + "_delta"] = delta_k_axis
        info[k_name + "_centre"] = 0.
        #print k_axis
        #print k_name, n_axis, delta_axis

    if window:
        # along all axes
        #window_function = fftutil.window_nd(arr1.shape, name=window)

        # apodize along frequency only
        window_func = getattr(np, window)
        window_function = window_func(arr1.shape[0])
        window_function = window_function[:, None, None]

        weight1 *= window_function
        weight2 *= window_function
        del window_function

    arr1 *= weight1
    arr2 *= weight2

    # correct for the weighting
    fisher_diagonal = np.sum(weight1 * weight2)

    fft_arr1 = np.fft.fftshift(np.fft.fftn(arr1))

    fft_arr2 = np.fft.fftshift(np.fft.fftn(arr2))

    fft_arr1 *= fft_arr2.conj()
    xspec = fft_arr1.real
    del fft_arr1, fft_arr2
    gc.collect()

    xspec /= fisher_diagonal

    # make the axes
    xspec_arr = algebra.make_vect(xspec, axis_names=k_axes)

    xspec_arr.info = info
    #print xspec_arr.get_axis("k_dec")

    if not nonorm:
        xspec_arr *= width.prod()

    return xspec_arr


def cross_power_est_highmem(arr1, arr2, weight1, weight2,
                    window="blackman", nonorm=False):
    """Calculate the cross-power spectrum of a two nD fields.

    The arrays must be identical and have the same length (physically
    and in pixel number) along each axis.

    Same goal as above without the emphasis on saving memory.
    This is the "tried and true" legacy function.
    """
    if window:
        window_function = fftutil.window_nd(arr1.shape, name=window)
        weight1 *= window_function
        weight2 *= window_function

    warr1 = arr1 * weight1
    warr2 = arr2 * weight2
    ndim = arr1.ndim

    fft_arr1 = np.fft.fftshift(np.fft.fftn(warr1))
    fft_arr2 = np.fft.fftshift(np.fft.fftn(warr2))
    xspec = fft_arr1 * fft_arr2.conj()
    xspec = xspec.real

    # correct for the weighting
    product_weight = weight1 * weight2
    xspec /= np.sum(product_weight)

    # make the axes
    k_axes = tuple(["k_" + axis_name for axis_name in arr1.axes])
    xspec_arr = algebra.make_vect(xspec, axis_names=k_axes)

    info = {'axes': k_axes, 'type': 'vect'}
    width = np.zeros(ndim)
    for axis_index in range(ndim):
        n_axis = arr1.shape[axis_index]
        axis_name = arr1.axes[axis_index]
        axis_vector = arr1.get_axis(axis_name)
        delta_axis = abs(axis_vector[1] - axis_vector[0])
        width[axis_index] = delta_axis

        k_axis = np.fft.fftshift(np.fft.fftfreq(n_axis, d=delta_axis))
        k_axis *= 2. * math.pi
        delta_k_axis = abs(k_axis[1] - k_axis[0])

        k_name = k_axes[axis_index]
        info[k_name + "_delta"] = delta_k_axis
        info[k_name + "_centre"] = 0.
        #print k_axis
        #print k_name, n_axis, delta_axis

    xspec_arr.info = info
    #print xspec_arr.get_axis("k_dec")

    if not nonorm:
        xspec_arr *= width.prod()

    return xspec_arr
