def calculate_xspec(cube1, cube2, weight1, weight2,
                    window="blackman", unitless=True, bins=None,
                    truncate=False, nbins=40, logbins=True, return_3d=False):

    print "finding the signal power spectrum"
    pwrspec3d_signal = cross_power_est(cube1, cube2, weight1, weight2,
                                       window=window)

    radius_arr = binning.radius_array(pwrspec3d_signal)

    if bins is None:
        bins = binning.suggest_bins(pwrspec3d_signal,
                                          truncate=truncate,
                                          logbins=logbins,
                                          nbins=nbins,
                                          radius_arr=radius_arr)

    if unitless:
        print "making the power spectrum unitless"
        pwrspec3d_signal = make_unitless(pwrspec3d_signal,
                                         radius_arr=radius_arr)

    print "calculating the 1D histogram"
    counts_histo, binavg = binning.bin_an_array(pwrspec3d_signal, bins,
                                                radius_arr=radius_arr)

    del radius_arr
    gc.collect()

    print "calculating the 2D histogram"
    # find the k_perp by not including k_nu in the distance
    radius_arr_perp = binning.radius_array(pwrspec3d_signal,
                                           zero_axes=[0])

    # find the k_perp by not including k_RA,Dec in the distance
    radius_arr_parallel = binning.radius_array(pwrspec3d_signal,
                                               zero_axes=[1, 2])

    # TODO: do better independent binning; for now:
    bins_x = copy.deepcopy(bins)
    bins_y = copy.deepcopy(bins)
    counts_histo_2d, binavg_2d = binning.bin_an_array_2d(pwrspec3d_signal,
                                                         radius_arr_perp,
                                                         radius_arr_parallel,
                                                         bins_x, bins_y)

    del radius_arr_perp
    del radius_arr_parallel
    gc.collect()

    bin_left_x, bin_center_x, bin_right_x = binning.bin_edges(bins_x,
                                                              log=logbins)

    bin_left_y, bin_center_y, bin_right_y = binning.bin_edges(bins_y,
                                                              log=logbins)

    bin_left, bin_center, bin_right = binning.bin_edges(bins, log=logbins)

    pwrspec2d_product = {}
    pwrspec2d_product['bin_x_left'] = bin_left_x
    pwrspec2d_product['bin_x_center'] = bin_center_x
    pwrspec2d_product['bin_x_right'] = bin_right_x
    pwrspec2d_product['bin_y_left'] = bin_left_y
    pwrspec2d_product['bin_y_center'] = bin_center_y
    pwrspec2d_product['bin_y_right'] = bin_right_y
    pwrspec2d_product['counts_histo'] = counts_histo_2d
    pwrspec2d_product['binavg'] = binavg_2d

    pwrspec1d_product = {}
    pwrspec1d_product['bin_left'] = bin_left
    pwrspec1d_product['bin_center'] = bin_center
    pwrspec1d_product['bin_right'] = bin_right
    pwrspec1d_product['counts_histo'] = counts_histo
    pwrspec1d_product['binavg'] = binavg

    if not return_3d:
        return pwrspec2d_product, pwrspec1d_product
    else:
        return pwrspec3d_signal, pwrspec2d_product, pwrspec1d_product


def calculate_xspec_file(cube1_file, cube2_file, bins,
                    weight1_file=None, weight2_file=None,
                    truncate=False, window="blackman",
                    return_3d=False, unitless=True):

    cube1 = algebra.make_vect(algebra.load(cube1_file))
    cube2 = algebra.make_vect(algebra.load(cube2_file))

    if weight1_file is None:
        weight1 = algebra.ones_like(cube1)
    else:
        weight1 = algebra.make_vect(algebra.load(weight1_file))

    if weight2_file is None:
        weight2 = algebra.ones_like(cube2)
    else:
        weight2 = algebra.make_vect(algebra.load(weight2_file))

    print cube1.shape, cube2.shape, weight1.shape, weight2.shape
    return calculate_xspec(cube1, cube2, weight1, weight2, bins=bins,
                           window=window, unitless=unitless,
                           truncate=truncate, return_3d=return_3d)


def test_with_random(unitless=True):
    """Test the power spectral estimator using a random noise cube"""

    delta = 1.33333
    cube1 = algebra.make_vect(np.random.normal(0, 1,
                                size=(257, 124, 68)))

    info = {'axes': ["freq", "ra", "dec"], 'type': 'vect',
            'freq_delta': delta / 3.78, 'freq_centre': 0.,
            'ra_delta': delta / 1.63, 'ra_centre': 0.,
            'dec_delta': delta, 'dec_centre': 0.}
    cube1.info = info
    cube2 = copy.deepcopy(cube1)

    weight1 = algebra.ones_like(cube1)
    weight2 = algebra.ones_like(cube2)

    bin_left, bin_center, bin_right, counts_histo, binavg = \
                    calculate_xspec(cube1, cube2, weight1, weight2,
                                    window="blackman",
                                    truncate=False,
                                    nbins=40,
                                    unitless=unitless,
                                    logbins=True)

    if unitless:
        pwrspec_input = bin_center ** 3. / 2. / math.pi / math.pi
    else:
        pwrspec_input = np.ones_like(bin_center)

    volume = 1.
    for axis_name in cube1.axes:
        axis_vector = cube1.get_axis(axis_name)
        volume *= abs(axis_vector[1] - axis_vector[0])

    pwrspec_input *= volume

    for specdata in zip(bin_left, bin_center,
                        bin_right, counts_histo, binavg,
                        pwrspec_input):
        print ("%10.15g " * 6) % specdata


if __name__ == '__main__':
    test_with_random(unitless=False)


