def suggest_bins(input_array, truncate=True, nbins=40, logbins=True,
                 radius_arr=None):
    """Bin the points in an array by radius

    Parameters
    ----------
    input_array: np.ndarray
        array over which to bin
    truncate: boolean
        maximum radius is the smallest dimension of the array
    nbins: scalar
        number of bins to use if not given some in advance
    logbins: boolean
        generate a log-spaced binning
    radius_arr: np.ndarray
        optional array of |k| to avoid recalculation
    TODO: throw out bins where the counts/bin are too low
    """
    if radius_arr is None:
        radius_arr = radius_array(input_array)

    radius_sorted = np.sort(radius_arr.flat)

    if truncate:
        axis_range = np.zeros((input_array.ndim))
        for axis_index in range(input_array.ndim):
            axis_name = input_array.axes[axis_index]
            axis = input_array.get_axis(axis_name)
            axis_range[axis_index] = axis.max()

        max_r = axis_range.min()
    else:
        max_r = radius_sorted[-1]

    # ignore the bin at k=0 (index 0 in sorted)
    if logbins:
        bins = np.logspace(math.log10(radius_sorted[1]),
                           math.log10(max_r),
                           num=(nbins + 1), endpoint=True)
    else:
        bins = np.linspace(radius_sorted[1], max_r,
                           num=(nbins + 1), endpoint=True)

    print "%d bins from %10.15g to %10.15g" % (nbins, radius_sorted[1], max_r)

    return bins


def bin_edges(bins, log=False):
    """report the bin edges and centers using the same convention as
    np.histogram. `log` reports the log-center
    """
    bin_left, bin_right = bins[:-1], bins[1:]
    if log:
        bin_center = 10 ** (0.5 * (np.log10(bin_left) +
                                   np.log10(bin_right)))
    else:
        bin_center = 0.5 * (bin_left + bin_right)

    return bin_left, bin_center, bin_right


def bin_an_array(input_array, bins, radius_arr=None):
    """Bin the points in an array by radius (n-dim)

    Parameters
    ----------
    input_array: np.ndarray
        array over which to bin
    bins: np.ndarray
        the bins
    radius_arr: np.ndarray
        optional array of |k| to avoid recalculation
    """
    if radius_arr is None:
        radius_arr = radius_array(input_array)

    radius_flat = radius_arr.flat
    arr_flat = input_array.flat

    counts_histo = np.histogram(radius_flat, bins)[0]
    binsum_histo = np.histogram(radius_flat, bins,
                                weights=arr_flat)[0]

    binavg = binsum_histo / counts_histo.astype(float)

    return counts_histo, binavg


def bin_an_array_2d(input_array, radius_array_x, radius_array_y,
                    bins_x, bins_y):
    """Bin the points in an array by radius (n-dim)

    Parameters
    ----------
    input_array: np.ndarray
        array over which to bin
    radius_array_x: np.ndarray
        array of the x-vectors at each point in the series
    radius_array_y: np.ndarray
        array of the y-vectors at each point in the series
    bins_x and bins_y: np.ndarray
        the bins in x and y
    """
    radius_flat_x = radius_array_x.flatten()
    radius_flat_y = radius_array_y.flatten()
    arr_flat = input_array.flatten()

    counts_histo = np.histogram2d(radius_flat_x, radius_flat_y,
                                  bins=(bins_x, bins_y))[0]
    binsum_histo = np.histogram2d(radius_flat_x, radius_flat_y,
                                  bins=(bins_x, bins_y),
                                  weights=arr_flat)[0]

    binavg = binsum_histo / counts_histo.astype(float)

    return counts_histo, binavg


def find_edges(axis, delta=None):
    """
    service function for bin_catalog_data which
    finds the bin edges for the histogram
    """
    if delta is None:
        delta = axis[1] - axis[0]

    edges = np.array(axis) - delta / 2.
    return np.append(edges, edges[-1] + delta)


def print_edges(sample, edges, name):
    """print bin edges for a catalog
    """
    print "Binning %s from range (%5.3g, %5.3g) into (%5.3g, %5.3g)" % (
          name, min(sample), max(sample), min(edges), max(edges))


def radius_array(input_array, zero_axes=[]):
    """Find the Euclidian distance of all the points in an array using their
    axis meta-data. e.g. x_axis[0]^2 + y_axis[0]^2. (n-dim)
    optionally give the list of axes to set to zero -- for example, in 3D:
    zero_axes = [0] leaves the x^2 out of x^2+y^2+z^2
    zero_axes = [1,2] leave the y^2 and z^2 out of x^2+y^2+z^2 (e.g. x^2)
    """
    index_array = np.indices(input_array.shape)
    scale_array = np.zeros(index_array.shape)

    for axis_index in range(input_array.ndim):
        axis_name = input_array.axes[axis_index]
        axis = input_array.get_axis(axis_name)
        if axis_index in zero_axes:
            axis = np.zeros_like(axis)

        scale_array[axis_index, ...] = axis[index_array[axis_index, ...]]

    scale_array = np.rollaxis(scale_array, 0, scale_array.ndim)

    return np.sum(scale_array ** 2., axis=-1) ** 0.5


def convert_2d_to_1d_pwrspec(pwr_2d, counts_2d, bin_kx, bin_ky, bin_1d,
                             weights_2d=None, nullval=np.nan,
                             null_zero_counts=True):
    """take a 2D power spectrum and the counts matrix (number of modex per k
    cell) and return the binned 1D power spectrum
    pwr_2d is the 2D power
    counts_2d is the counts matrix
    bin_kx is the x-axis
    bin_ky is the x-axis
    bin_1d is the k vector over which to return the result
    weights_2d is an optional weight matrix in 2d; otherwise use counts

    null_zero_counts sets elements of the weight where there are zero counts to
    zero.
    """
    # find |k| across the array
    index_array = np.indices(pwr_2d.shape)
    scale_array = np.zeros(index_array.shape)
    scale_array[0, ...] = bin_kx[index_array[0, ...]]
    scale_array[1, ...] = bin_ky[index_array[1, ...]]
    scale_array = np.rollaxis(scale_array, 0, scale_array.ndim)
    radius_array = np.sum(scale_array ** 2., axis=-1) ** 0.5

    radius_flat = radius_array.flatten()
    pwr_2d_flat = pwr_2d.flatten()
    counts_2d_flat = counts_2d.flatten()
    if weights_2d is not None:
        weights_2d_flat = weights_2d.flatten()
    else:
        weights_2d_flat = counts_2d_flat.astype(float)

    #print weights_2d_flat.shape, pwr_2d_flat.shape

    old_settings = np.seterr(invalid="ignore")
    weight_pwr_prod = weights_2d_flat * pwr_2d_flat
    weight_pwr_prod[np.isnan(weight_pwr_prod)] = 0.
    weight_pwr_prod[np.isinf(weight_pwr_prod)] = 0.
    weights_2d_flat[np.isnan(weight_pwr_prod)] = 0.
    weights_2d_flat[np.isinf(weight_pwr_prod)] = 0.

    if null_zero_counts:
        weight_pwr_prod[counts_2d_flat == 0] = 0.
        weights_2d_flat[counts_2d_flat == 0] = 0.

    counts_histo = np.histogram(radius_flat, bin_1d,
                                weights=counts_2d_flat)[0]

    weights_histo = np.histogram(radius_flat, bin_1d,
                                weights=weights_2d_flat)[0]

    binsum_histo = np.histogram(radius_flat, bin_1d,
                                weights=weight_pwr_prod)[0]

    # explicitly handle cases where the counts are zero
    #binavg = np.zeros_like(binsum_histo)
    #binavg[weights_histo > 0.] = binsum_histo[weights_histo > 0.] / \
    #                             weights_histo[weights_histo > 0.]
    #binavg[weights_histo <= 0.] = nullval
    #old_settings = np.seterr(invalid="ignore")
    binavg = binsum_histo / weights_histo
    binavg[np.isnan(binavg)] = nullval
    # note that if the weights are 1/sigma^2, then the variance of the weighted
    # sum is just 1/sum of weights; so return Gaussian errors based on that
    gaussian_errors = np.sqrt(1./weights_histo)
    gaussian_errors[np.isnan(binavg)] = nullval

    np.seterr(**old_settings)

    return counts_histo, gaussian_errors, binavg
