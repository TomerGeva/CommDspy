import numpy as np
from scipy import signal
import os
import CommDspy as cdsp
import random
from test.auxiliary import generate_pattern, generate_and_pass_channel
from CommDspy.rx.ffe_dfe_model import ffe_dfe


def equalization_prbs_test(prbs_type):
    """
    :param prbs_type:
    :return: Testing the equalization function. This is dine via inserting the PRBS through an IIR, which the inverse of
             is a FIR and can be thus be checked according to the reference IIR. Nevertheless, the IIR should have all
             poles inside the unit circle, or else the resulting signal will be unstable
    """
    channel_ref, channel_out, constellation, _, assert_str =  _get_test_parameters(prbs_type)
    # ==================================================================================================================
    # Running DUT
    # ==================================================================================================================
    precursors        = np.argmax(abs(channel_ref))
    pn_inv_postcoding = channel_ref[precursors] != np.max(channel_ref)
    postcursors       = len(channel_ref) - precursors - 1
    equ_dut = cdsp.equalization_estimation_prbs(prbs_type, channel_out, constellation,
                                                prbs_full_scale=False,
                                                ffe_postcursor=postcursors, ffe_precursor=precursors, dfe_taps=0,
                                                normalize=False,
                                                bit_order_inv=False,
                                                pn_inv_precoding=False,
                                                gray_coded=False,
                                                pn_inv_postmapping=pn_inv_postcoding)
    # ==================================================================================================================
    # Comparing results
    # ==================================================================================================================
    assert np.all(np.abs(channel_ref - equ_dut[0]) < 5e-3), assert_str
    # ==================================================================================================================
    # Testing the ffe_dfe function
    # ==================================================================================================================
    dfe_taps = np.array([0])
    levels = np.array([-1, 1])
    ffe_taps = equ_dut[0]
    slicer_in_dut     = ffe_dfe(channel_out, ffe_taps, dfe_taps, levels=levels)
    slicer_in_vec_dut = slicer_in_dut[0][len(ffe_taps):]
    slicer_in_ref     = signal.lfilter(ffe_taps, 1, channel_out)[len(ffe_taps):]
    assert np.allclose(slicer_in_ref, slicer_in_vec_dut), assert_str


def equalization_lms_test(prbs_type):
    channel_ref, channel_out, constellation, ref_pattern, assert_str =  _get_test_parameters(prbs_type)
    precursors        = np.argmax(abs(channel_ref))
    postcursors       = len(channel_ref) - precursors - 1
    pn_inv_postcoding = channel_ref[precursors] != np.max(channel_ref)
    tap_idx_vec       = np.arange(-1*precursors, postcursors+1)
    # ==================================================================================================================
    # Running LMS
    # ==================================================================================================================
    equ_dut             = np.zeros_like(channel_ref)
    equ_dut[precursors] = 1
    mse_vec             = [1e3]
    count               = 0
    prbs_len            = 2 ** prbs_type.value - 1
    while count < 100000 and mse_vec[-1] > 1e-5:
        count += 1
        # ----------------------------------------------------------------------------------------------------------
        # Passing through the FFE + DFE
        # ----------------------------------------------------------------------------------------------------------
        slicer_in_dut, _ = ffe_dfe(channel_out, equ_dut)
        slicer_in_dut    = slicer_in_dut[len(equ_dut):]
        # ----------------------------------------------------------------------------------------------------------
        # Aligning
        # ----------------------------------------------------------------------------------------------------------
        pattern_aligned, _  = cdsp.rx.lock_pattern_to_signal(ref_pattern[:prbs_len], slicer_in_dut)
        pattern_aligned_rep = np.tile(pattern_aligned, int(np.ceil(len(ref_pattern) / prbs_len)))[:len(slicer_in_dut)]
        mse, grad_ffe       = cdsp.rx.lms_grad(channel_out[len(equ_dut):], slicer_in_dut, cdsp.get_levels(constellation), tap_idx_vec, reference_vec=pattern_aligned_rep)
        equ_dut -= grad_ffe * 0.003
        if abs(mse - mse_vec[-1]) < 1e-7:
            break
        mse_vec.append(mse)
    # ==================================================================================================================
    # Comparing results
    # ==================================================================================================================
    assert np.all(np.abs(channel_ref - equ_dut) < 1e-2), assert_str


def _get_test_parameters(prbs_type):
    ref_pattern, constellation, gray_coding = generate_pattern(prbs_type)
    channel_ref, channel_out, constellation, assert_str = generate_and_pass_channel(ref_pattern, prbs_type, constellation, gray_coding)
    return channel_ref, channel_out, constellation, ref_pattern, assert_str
