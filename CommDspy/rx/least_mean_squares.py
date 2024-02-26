import numpy as np
from CommDspy.rx.slicer import slicer

def ddlms_grad(input_vec, output_vec, levels, ffe_tap_idx=np.array([]), dfe_tap_idx=np.array([]), reference_vec=None):
    """
    :param input_vec: Numpy array of inputs to the FFE-DFEused to compute the MSE and tap gradients
    :param output_vec: Numpy array of outputs of the FFE-DFE (Slicer inputs), used to compute the MSE and tap gradients
    :param levels: Constellation levels, should be a numpy array of floats
    :param ffe_tap_idx: numpy array containing the indices of the FFE taps for which we want to compute the gradient
    :param dfe_tap_idx: numpy array containing the indices of the DFE taps for which we want to compute the gradient
    :param reference_vec: If None, assuming that there are no slicer errors and performing slicing to recover the
                          reference data. If not None, should have the same length as "input_vec" and correspond to the
                          reference level for each input
    :return: Decision Directed Least Mean Square gradient calculator.
    Function computes the MSE as well as the gradient w.r.t. each tap inputted.
        1. If ffe_tap_idx and dfe_tap_idx are empty, only the MSE
        2. If ffe_tap_idx is empty and dfe_tap_idx is not empty, returns the MSE and DFE grad
        3. If ffe_tap_idx is not empty and dfe_tap_idx is empty, returns the MSE and FFE grad
        4. If neither are empty, returns the MSE, FFE grad and DFE grad

    According to the FFE DFE model, the gradient is computed as follows:


                  -----               -----
                  \                   \
    \tilde{x_n} =  > b_i * y_{n-i} -   > a_j * \hat{x_{n-j}}
                  /                   /
                  -----               -----
                    i                   j

                N-1                                 ;
               -----                                ;
           1   \                                    ;
    MSE = ---   > (\tilde{x_n} - \hat{x_n})^2       ; (\tilde{x_n} - \hat{x_n}) === e_n
           N   /                                    ;
               -----                                ;
               n=0                                  ;

                 N-1                                ;                  N-1
                -----                               ;                 -----
    dMSE      1 \                                   ;     dMSE     -1 \
    ----- =  --- > e_n * y_{n-i}                    ;     ----- =  --- > e_n * hat{x_{n-j}}
    db_i     2N /                                   ;     da_j     2N /
                -----                               ;                 -----
                n=0                                 ;                 n=0

    """

    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    if reference_vec is None:
        reference_vec = slicer(output_vec, levels)
    elif len(reference_vec) != len(output_vec):
        raise ValueError(
            f'Length of the reference vector ({len(reference_vec):d} is different than the input vector ({len(output_vec)}')
    # ==================================================================================================================
    # Computing the MSE
    # ==================================================================================================================
    e_n = output_vec - reference_vec
    mse = np.mean(e_n**2)
    # ==================================================================================================================
    # Computing the FFE + DFE derivatives
    # ==================================================================================================================
    grad_ffe, grad_dfe = _compute_derivatives(input_vec, reference_vec, ffe_tap_idx, dfe_tap_idx, e_n)
    # ==================================================================================================================
    # Returning
    # ==================================================================================================================
    return _return_function(mse, grad_ffe, grad_dfe, ffe_tap_idx.size, dfe_tap_idx.size)

def soft_ddlms_grad(input_vec, output_vec, levels, sigma=1, ffe_tap_idx=np.array([]), dfe_tap_idx=np.array([]), reference_vec=None):
    """
        :param input_vec: Numpy array of inputs to the FFE-DFEused to compute the MSE and tap gradients
        :param output_vec: Numpy array of outputs of the FFE-DFE (Slicer inputs), used to compute the loss and tap gradients
        :param levels: Constellation levels, should be a numpy array of floats
        :param sigma: standatd deviation of each gaussians composing the gaussian mixture
        :param ffe_tap_idx: numpy array containing the indices of the FFE taps for which we want to compute the gradient
        :param dfe_tap_idx: numpy array containing the indices of the DFE taps for which we want to compute the gradient
        :param reference_vec: If None, assuming that there are no slicer errors and performing slicing to recover the
                              reference data. If not None, should have the same length as "input_vec" and correspond to the
                              reference level for each input
        :return: This function is based on the following article:
                 "A soft decision-directed LMS algorithm for blind equalization" (1993). DOI: 10.1109/26.216497

                 Soft Decision Directed Least Mean Square gradient calculator.

                 Function computes thee soft-decision directed LMS function and gradients. This function assumes that
                 all levels are transmitted in equi-probable probabilities and that the received signal should be a
                 gaussian mixture with equal standard deviation of sigma.
            Given the sample y_i from 'input_vec', we assume that:

                                FFE                 DFE
                            -----               -----                 
                            \                   \                     
              \tilde{x_n} =  > b_i * y_{n-i} -   > a_j * \hat{x_{n-j}}
                            /                   /                     
                            -----               -----                 
                              i                   j                   

        Gaussian Mixture Model
                                               -----
                                   p           \           (\tilde{x_n} - mu_j)^2
        f(\tilde{x_n}) =  ------------------ *  >    -1 * ------------------------
                         sqrt(2*pi*sigma^2)    /  e^            2 * sigma^2
                                               -----
                                                 j
            where
            - 'p' is 1/len(levels) ; the equiprobable probability of transmitting each symbol
            - mu_j is the jth level in the levels vector

            The function tries to minimize the following loss function:

            L(f(\tilde{x_n});w_vec) = min{E[-1*log(f(\tilde{x_n})]} ;
                          w_vec
            where:
                w_vec are that b_i and a_j, i.e. the FFE and DFE coefficients





                                                            -----                          -----                       -----
            d L(f(\tilde{x_n}) ; w_vec)          1          \     \tilde{x_n} - mu_j       |      (\tilde{x_n} - mu_j)^2   |
           ---------------------------- = -------------- *   >   -------------------- * e^ |-1 *  ------------------------ | = e_n
                d\tilde{x_n}              f(\tilde{x_n})    /          sigma^2             |           2 * sigma^2         |
                                                            -----                          -----                       -----




                        N-1                                ;                  N-1
                       -----                               ;                 -----
            dL       1 \                                   ;      dL      -1 \
           ----- =  --- > e_n * y_{n-i}                    ;     ----- =  --- > e_n * hat{x_{n-j}}
           db_i     2N /                                   ;     da_j     2N /
                       -----                               ;                 -----
                       n=0                                 ;                 n=0

        This is a generalization of the following article:
        A Soft Decision-Directed LMS Algorithm for Blind Equalization (1993)
                Steven J. Nowlan and Geoffrey E. Hinton
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    if reference_vec is None:
        reference_vec = slicer(output_vec, levels)
    p             = 1 / len(levels)
    # ==================================================================================================================
    # Computing the loss
    # ==================================================================================================================
    # temp = np.exp(-1 * (output_vec[None, :] - levels[:, None])**2 / (2 * sigma**2))
    # f    = p / np.sqrt(2 * np.pi * sigma**2) * np.sum(temp, axis=0)
    # e_n  = output_vec - np.sum(temp * levels[:, None], axis=0) / np.sum(temp, axis=0)
    temp1 = np.exp(-1 * (output_vec[None, :] - levels[:, None])**2 / (2 * sigma**2))
    temp2 = (output_vec[None, :] - levels[:, None]) / (sigma**2) * temp1
    f     = p / np.sqrt(2 * np.pi * sigma**2) * np.sum(temp1, axis=0)
    e_n  = 1 / f * np.sum(temp2, axis=0)

    loss = np.mean(-1 * np.mean(np.log(f)))
    # ==================================================================================================================
    # Computing the FFE + DFE derivatives
    # ==================================================================================================================
    grad_ffe, grad_dfe = _compute_derivatives(input_vec, reference_vec, ffe_tap_idx, dfe_tap_idx, e_n)
    # ==================================================================================================================
    # Returning
    # ==================================================================================================================
    return _return_function(loss, grad_ffe, grad_dfe, ffe_tap_idx.size, dfe_tap_idx.size)

def _compute_derivatives(input_vec, reference_vec, ffe_tap_idx, dfe_tap_idx, e_n):
    """
        :param input_vec: Numpy array of inputs to the FFE-DFE used to compute the FFE tap gradients (Slicer inputs)
        :param reference_vec: Numpy array of slicer outputs, used to compute the DFE tap gradients
        :param ffe_tap_idx: indices of FFE taps, for gradient computation
        :param dfe_tap_idx: indices of DFE taps, for gradient computation. Can be strictly positive larger than 0
        :param e_n: Numpy array of the errors, used to compute the gradients
        :return:
        Function computes the FFE and DFE gradients, according to the inputs
    """
    N             = len(input_vec)
    grad_ffe      = np.zeros_like(ffe_tap_idx, dtype=float)
    grad_dfe      = np.zeros_like(dfe_tap_idx, dtype=float)
    # ==================================================================================================================
    # Computing the FFE derivatives
    # ==================================================================================================================
    for idx, ii in enumerate(ffe_tap_idx):
        start_idx = max([0, ii])
        stop_idx = min([N, N + ii])  # stop is exclusive in python
        mask_vec = np.arange(start_idx, stop_idx)
        grad_ffe[idx] = 0.5 * np.mean(e_n[mask_vec] * input_vec[mask_vec - ii])
    # ==================================================================================================================
    # Computing the DFE derivatives
    # ==================================================================================================================
    for idx, jj in enumerate(dfe_tap_idx):
        mask_vec = np.arange(jj, N)  # stop is exclusive in python and DFE taps are strictly positive
        grad_dfe[idx] = -0.5 * np.mean(e_n[mask_vec] * reference_vec[mask_vec - jj])
    return grad_ffe, grad_dfe

def _return_function(loss, grad_ffe, grad_dfe, ffe_size, dfe_size):
    # ==================================================================================================================
    # Returning
    # ==================================================================================================================
    if ffe_size == 0 and dfe_size == 0:
        return loss
    elif ffe_size == 0:
        return loss, grad_dfe
    elif dfe_size == 0:
        return loss, grad_ffe
    else:
        return loss, grad_ffe, grad_dfe
