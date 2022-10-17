import numpy as np
import CommDspy as cdsp

class PrbsGen:
    def __init__(self, prbs_data):
        self.prbs_type  = prbs_data.prbs_type
        self.gen_poly   = prbs_data.gen_poly
        self.chunk_size = prbs_data.chunk_size
        self.seed       = prbs_data.seed if prbs_data.seed is not None else np.ones_like(self.gen_poly)
        self.seed_prev  = None

    def __iter__(self):
        return self

    def __next__(self):
        chunk = np.zeros(self.chunk_size).astype(int)
        for ii in range(self.chunk_size):
            chunk[ii] = np.mod(self.gen_poly.dot(self.seed), 2)
            self.seed_prev = self.seed
            self.seed      = np.concatenate(([chunk[ii]], self.seed[:-1]))
        return chunk

    def get_seed(self):
        return self.seed

    def get_prev_seed(self):
        return self.seed_prev

class Encoder:
    def __init__(self, coding_data):
        self.constellation           = coding_data.constellation
        self.bits_per_symbol         = coding_data.bits_per_symbol
        self.bit_order_inv           = coding_data.bit_order_inv
        self.inv_msb                 = coding_data.inv_msb
        self.inv_lsb                 = coding_data.inv_lsb
        self.pn_inv                  = coding_data.pn_inv
        self.coding_gray             = coding_data.coding_gray
        self.coding_differential     = coding_data.coding_differential
        self.coding_bin_differential = coding_data.coding_bin_differential
        self.coding_manchester       = coding_data.coding_manchester
        self.coding_bipolar          = coding_data.coding_bipolar
        self.coding_mlt3             = coding_data.coding_mlt3
        self.coding_diff_manchester  = coding_data.coding_diff_manchester
        self.coding_linear           = coding_data.coding_linear
        self.coding_conv             = coding_data.coding_conv
        self.G                       = coding_data.G
        self.feedback                = coding_data.feedback
        self.use_feedback            = coding_data.use_feedback

    def __call__(self, pattern_block):
        # ==============================================================================================================
        # Starting with binary encoding options
        # ==============================================================================================================
        pattern_coded = pattern_block.copy()
        if self.coding_bin_differential:
            pattern_coded = cdsp.tx.coding_differential(pattern_coded, cdsp.ConstellationEnum.OOK)
        if self.coding_manchester:
            pattern_coded = cdsp.tx.coding_manchester(pattern_coded)
        if self.coding_bipolar:
            pattern_coded = cdsp.tx.coding_bipolar(pattern_coded)
        if self.coding_mlt3:
            pattern_coded = cdsp.tx.coding_mlt3(pattern_coded)
        if self.coding_diff_manchester:
            pattern_coded = cdsp.tx.coding_differential_manchester(pattern_coded)
        if self.coding_linear:
            pattern_coded = cdsp.tx.coding_linear(pattern_coded)
        if self.coding_conv:
            pattern_coded = cdsp.tx.coding_conv(pattern_coded, self.G, self.feedback, self.use_feedback)
        # ==============================================================================================================
        # Converting to symbols if needed
        # ==============================================================================================================
        if self.bits_per_symbol > 1:
            pattern_coded = cdsp.tx.bin2symbol(pattern_coded,
                                               num_of_symbols=2**self.bits_per_symbol,
                                               bit_order_inv=self.bit_order_inv,
                                               inv_msb=self.inv_msb,
                                               inv_lsb=self.inv_lsb,
                                               pn_inv=self.pn_inv)
        elif np.isclose(int(self.bits_per_symbol), self.bits_per_symbol):
        # meaning two levels, only pn_inv is relevant here
            pattern_coded = 1 - pattern_coded
        else:
        # meaning three levels, at this point it is {0, 1, 2}
            pattern_coded = 2 - pattern_coded
        # ==============================================================================================================
        # Symbol encoding
        # ==============================================================================================================
        if self.coding_differential:
            pattern_coded = cdsp.tx.coding_differential(pattern_coded, self.constellation)
        if self.coding_gray:
            pattern_coded = cdsp.tx.coding_gray(pattern_coded, self.constellation)

        return pattern_coded

class Transmitter:
    def __init__(self, prbs_data, coding_data, mapping_data):
        self.prbs_gen = iter(PrbsGen(prbs_data))
        self.encoder  = Encoder(coding_data)
        self.constellation = mapping_data.constellation
        self.levels        = mapping_data.levels

    def generate(self):
        prbs_chunk         = next(self.prbs_gen)
        encoded_prbs_chunk = self.encoder(prbs_chunk)  # need to sort memory for encoding with memory, e.g. differential
        mapped_prbs_chunk  = cdsp.tx.mapping(encoded_prbs_chunk,
                                             constellation=self.constellation,
                                             levels=self.levels)
        return mapped_prbs_chunk


