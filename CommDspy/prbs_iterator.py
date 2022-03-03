import numpy as np


class PrbsIterator:
    """
    This class hold the prbs iterator, whenever the next function is called, the function pops out the next bit in the
    PRBS series. the iterator hold the seed and the generating polynomial.

    NOTE - BOTH POLY_COEFF AND INIT_SEED SHOULD BE NUMPY ARRAYS!

    ploy_coeff should be equal to the degree of the generating polynomial g(x). Example - for g(x) = 1+x+x^2+x^12+x^13,
        use poly_coeff = [1 1 0 0 0 0 0 0 0 0 0 1 1]

    init_seed should have the same length as "poly_coeff"
    """
    def __init__(self, poly_coeff, init_seed):
        self.poly_coeff = poly_coeff
        self.seed = init_seed
        self.seed_prev = None

    def __iter__(self):
        return self

    def __next__(self):
        single_patt = np.mod(self.poly_coeff.dot(self.seed), 2)
        self.seed_prev = self.seed
        self.seed = np.concatenate(([single_patt], self.seed[:-1]))
        return single_patt

    def get_seed(self):
        return self.seed

    def get_prev_seed(self):
        return self.seed_prev
