import numpy as np


def mueller_muller_step(signal_chunk, reference_signal, pre_weight=1, post_weight=1, tol=1e-3):
    """
    :param signal_chunk: The raw signal
    :param reference_signal: The symbol stream without errors
    :param pre_weight: the weight of precursor in the ratio between the precursor and the postcursor
    :param post_weight: the weight of postcursor in the ratio between the precursor and the
    :param tol: tolerance of the step direction.
        1. If the amplitudes are the same up to the tolerance, the result will be 0
        2. If the pre is higher than the post, the step will be towards the pre, i.e. -1
        3. If the post is higher than the post, the step will be towards the pre, i.e. 1
    :return: Function computes the estimation of the first precursor and the first postcursor and compares the
             amplitudes of the both. The goal of the mm CDR is to choose the phase which makes the precursor and
             postcursor equal in amplitude.
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    reference_pre  = reference_signal[1:]
    reference_post = reference_signal[:-1]
    # ==================================================================================================================
    # Computing the coefs
    # ==================================================================================================================
    pre_coef  = reference_pre.dot(signal_chunk[:-1])
    post_coef = reference_post.dot(signal_chunk[1:])
    step      =  post_coef * post_weight - pre_coef * pre_weight
    if abs(step) < tol:
        return 0
    else:
        return np.sign(step).astype(int)
