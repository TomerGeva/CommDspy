import numpy as np
import CommDspy as cdsp

class Checker:
    def __init__(self, prbs_data):
        self.prbs_data       = prbs_data
        self.prbs_vec, _     = cdsp.tx.prbs_gen(prbs_data.gen_poly, prbs_data.seed, (2**prbs_data.prbs_type.value)-1)
        self.check_sequences = []
        self.acc_errors      = 0
        self.acc_symbols     = 0

    def check_ber(self, bit_vec):
        _, _, error_bit = cdsp.rx.prbs_checker(self.prbs_data.prbs_type, bit_vec,
                                               init_lock=False,
                                               prbs_seq=self.prbs_vec)
        self.acc_errors  += sum(error_bit)
        self.acc_symbols += len(error_bit)
        self.check_sequences.append(list(bit_vec))
        return sum(error_bit) / len(error_bit)
