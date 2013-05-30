import h5py

def load_transferfunc(beamtransfer_file, modetransfer_file, treatment_list,
                      beam_treatment="0modes"):
    r"""Given two hd5 files containing a beam and mode loss transfer function,
    load them and make a composite transfer function"""
    # TODO: check that both files have the same treatment cases

    if modetransfer_file is not None:
        print "Applying 2d transfer from " + modetransfer_file
        modetransfer_2d = h5py.File(modetransfer_file, "r")

    if beamtransfer_file is not None:
        print "Applying 2d transfer from " + beamtransfer_file
        beamtransfer_2d = h5py.File(beamtransfer_file, "r")

    # make a list of treatments or cross-check if given one
    if modetransfer_file is not None:
        if treatment_list is None:
            treatment_list = modetransfer_2d.keys()
        else:
            assert modetransfer_2d.keys() == treatment_list, \
                    "mode transfer treatments do not match data"
    else:
        if treatment_list is None:
            treatment_list = [beam_treatment]

    # given both
    if (beamtransfer_file is not None) and (modetransfer_file is not None):
        print "using the product of beam and mode transfer functions"
        transfer_dict = {}

        for treatment in modetransfer_2d:
            transfer_dict[treatment] = modetransfer_2d[treatment].value
            transfer_dict[treatment] *= \
                                    beamtransfer_2d[beam_treatment].value

    # given mode only
    if (beamtransfer_file is None) and (modetransfer_file is not None):
        print "using just the mode transfer function"
        transfer_dict = {}

        for treatment in modetransfer_2d:
            transfer_dict[treatment] = modetransfer_2d[treatment].value

    # given beam only
    if (beamtransfer_file is not None) and (modetransfer_file is None):
        print "using just the beam transfer function"
        transfer_dict = {}
        for treatment in treatment_list:
            transfer_dict[treatment] = beamtransfer_2d[beam_treatment].value

    # no transfer function
    if (beamtransfer_file is None) and (modetransfer_file is None):
        print "not using transfer function"
        transfer_dict = None

    return transfer_dict



