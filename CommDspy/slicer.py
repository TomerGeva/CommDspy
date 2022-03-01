import numpy as np


def slicer(slicer_in_mat, levels=None):
    """
    :param slicer_in_mat:capture or data at the slicer input
    :param levels: levels of the constellation. If None then [-3,-1,1,3]. MUST be monotonically increasing
    :return: a matrix of the same dimension with the sliced data
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    try:
        y, x = slicer_in_mat.shape
    except AttributeError:
        slicer_im_mat = np.array(slicer_in_mat)
        try:
            y, x = slicer_im_mat.shape
        except ValueError:
            x = slicer_in_mat.shape[0]
            y = 0
    except ValueError:
        x = slicer_in_mat.shape[0]
        y = 0
    if levels is None:
        levels = np.array([-3, -1, 1, 3])
    # ==================================================================================================================
    # Creating the diff matrix --> col vec minus row vec --> matrix
    # ==================================================================================================================
    slicer_in_vec = np.reshape(slicer_in_mat, [-1, 1])
    diff_mat = abs(slicer_in_vec - levels)
    # ==================================================================================================================
    # Slicing
    # ==================================================================================================================
    constellation_points = np.argmin(diff_mat, axis=1)
    constellation_points = np.reshape(constellation_points, [y, x]) if y > 0 else np.reshape(constellation_points, [-1])
    slicer_out_mat = np.array(levels[constellation_points])
    return slicer_out_mat
