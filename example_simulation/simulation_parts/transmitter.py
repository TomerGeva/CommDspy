import numpy as np
import CommDspy as cdsp

class PrbsGen:
    def __init__(self, prbs_data):
        self.prbs_type  = prbs_data.prbs_type
        self.gen_poly   = prbs_data.gen_poly
        self.chunk_size = prbs_data.chunk_size
        self.seed       = prbs_data.seed if prbs_data.seed is not None else np.ones_like(self.gen_poly)

    def __iter__(self):
        return self

    def __next__(self):
        chunk, self.seed = cdsp.tx.prbs_gen(self.gen_poly, self.seed, self.chunk_size)
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
            pattern_coded = cdsp.tx.coding_linear(pattern_coded, self.G)
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
            if self.pn_inv:
                pattern_coded = 1 - pattern_coded
        else:
        # meaning three levels, at this point it is {0, 1, 2}
            if self.pn_inv:
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
        # ==============================================================================================================
        # Memory
        # ==============================================================================================================
        self.prbs_chunk         = np.zeros(prbs_data.chunk_size).astype(int)
        self.encoded_prbs_chunk = np.zeros(prbs_data.chunk_size).astype(int)
        self.mapped_prbs_chunk  = np.zeros(prbs_data.chunk_size).astype(int)

    def generate(self):
        self.prbs_chunk         = next(self.prbs_gen)
        self.encoded_prbs_chunk = self.encoder(self.prbs_chunk)  # need to sort memory for encoding with memory, e.g. differential
        self.mapped_prbs_chunk  = cdsp.tx.mapping(self.encoded_prbs_chunk,
                                                  constellation=self.constellation,
                                                  levels=self.levels)
        return self.mapped_prbs_chunk


