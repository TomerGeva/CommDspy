import numpy as np
from collections import deque
from CommDspy.rx.slicer import slicer

class Filter:
    def __init__(self, b, a, z_fir=None, z_iir=None):
        """
        Init the filter object. The Filtering is done by the following difference equation:
                        N-1                  M
                       -----               -----
                       \                   \
        a[0] * y[n] =   > b[i] * x[n-i] -   > a[j] * y[n-j]
                       /                   /
                       -----               -----
                       i = 0               j = 1
        :param b: FIR coefficients of the filter
        :param a: IIR coefficients of the filter
        :param z_fir: FIR initial memory. If None assumes '0' memory. Must have the same length as 'b'
        :param z_iir: IIR initial memory. If None assumes '0' memory. Must have the same length as len('a') - 1
        """
        self.b = b
        self.a = a
        self._z_fir = None
        self._z_iir = None
        self.set_z_fir(z_fir)
        self.set_z_iir(z_iir)

    def set_z_fir(self, z_fir):
        if z_fir is None:
            self._z_fir = deque([0] * len(self.b), maxlen=len(self.b))
        elif len(z_fir) == len(self.b):
            self._z_fir = deque(z_fir, maxlen=len(self.b))
        else:
            raise ValueError(f'Length of z_fir is {len(z_fir):d} instead of {len(self.b):d}')

    def set_z_iir(self,z_iir):
        if z_iir is None:
            self._z_iir = deque([0] * (len(self.a) - 1), maxlen=len(self.a)-1)
        elif len(z_iir) == len(self.a)-1:
            self._z_iir = deque(z_iir, maxlen=len(z_iir))
        else:
            raise ValueError(f'Length of z_fir is {len(z_iir):d} instead of {len(self.a)-1:d}')

    def _process(self, x):
        self._z_fir.appendleft(x)
        a0_y = np.dot(self.b, self._z_fir) - np.dot(self.a[1:], self._z_iir)
        y    = a0_y / self.a[0]
        self._z_iir.appendleft(y)
        return y

    def __call__(self, data_chunk):
        y_vec = np.zeros_like(data_chunk)
        for ii, x in enumerate(data_chunk):
            y_vec[ii] = self._process(x)
        return y_vec

class FilterTII:
    def __init__(self, b, a, z=None):
        """
        Init the filter object. The Filtering is done by the following difference equation:
                        N-1                  M
                       -----               -----
                       \                   \
        a[0] * y[n] =   > b[i] * x[n-i] -   > a[j] * y[n-j]
                       /                   /
                       -----               -----
                       i = 0               j = 1
        This filter works according to the transposed II filter model, allowing to use half the memory that the normal
        IIR filter uses (and it is faster). The model is shown in the sketch below:

                            b[0]
        x[n] --------------> X -----------> + -------------------------> a[0] * y[n]
                  |                         ^                       |
                  |                         |                       |
                  |                   ------------                  |
                  |                   |    z^-1  |                  |
                  |                   ------------                  |
                  |                         ^                       |
                  |         b[1]            |          -a[1]        |
                  |--------> X -----------> + <--------- X ---------|
                  |                         ^                       |
                  |                         |                       |
                  |                   ------------                  |
                  |                   |    z^-1  |                  |
                  |                   ------------                  |
                  |                         ^                       |
                  |         b[2]            |          -a[2]        |
                  |--------> X -----------> + <--------- X ---------|
                  :                                                 :
                  :                                                 :
                  :                                                 :
                  |                         ^                       |
                  |                         |                       |
                  |                   ------------                  |
                  |                   |    z^-1  |                  |
                  |                   ------------                  |
                  |                         ^                       |
                  |         b[N]            |          -a[N]        |
                  |--------> X -----------> + <--------- X ---------|

        :param b: FIR coefficients of the filter
        :param a: IIR coefficients of the filter
        :param z: state initial memory. If None assumes '0' memory. Must have the same length as:
                  max{len('b'), len('a')} - 1

        """
        self.b  = b
        self.a  = a
        self._z = None
        self.set_z(z)
        self._b_vec = np.concatenate([self.b[1:], np.zeros(max(len(self._z) - len(self.b) + 1, 0))])
        self._a_vec = np.concatenate([self.a[1:], np.zeros(max(len(self._z) - len(self.a) + 1, 0))])

    def set_z(self, z):
        z_len = max(len(self.b), len(self.a)) - 1
        if z is None:
            self._z = [0] * z_len
        elif len(z) == z_len:
            self._z = z
        else:
            raise ValueError(f'Length of z is {len(z):d} instead of {z_len:d}')

    def _process(self, x):
        # Creating y
        a0_y = self.b[0] * x + self._z[0]
        y    = a0_y / self.a[0]
        # Advancing the memory
        z_temp = np.concatenate([self._z[1:], [0]])
        # Filling the new memory
        self._z = (x * self._b_vec) + z_temp - (y * self._a_vec)
        return y

    def __call__(self, data_chunk):
        y_vec = np.zeros_like(data_chunk)
        for ii, x in enumerate(data_chunk):
            y_vec[ii] = self._process(x)
        return y_vec

class FfeDfeFilterTII(FilterTII):
    def __init__(self, ffe, dfe, levels, z=None):
        """
        Init the filter object. The Filtering is done by the following difference equation:
                        N-1                  M
                       -----               -----
               ~       \                   \         ^
        d[0] * x[n] =   > f[i] * x[n-i] -   > d[j] * x[n-j]
                       /                   /
                       -----               -----
                       i = 0               j = 1
        This filter works according to the transposed II filter model, allowing to use half the memory that the normal
        IIR filter uses (and it is faster). The model is shown in the sketch below:

                          ffe[0]                             ------------
        x[n] --------------> X -----------> + -\tilde{x[n]}->|  slicer  | ------> a[0] * \hat{x[n]}
                  |                         ^                ------------   |
                  |                         |                               |
                  |                   ------------                          |
                  |                   |    z^-1  |                          |
                  |                   ------------                          |
                  |                         ^                               |
                  |       ffe[1]            |        -dfe[1]                |
                  |--------> X -----------> + <--------- X -----------------|
                  |                         ^                               |
                  |                         |                               |
                  |                   ------------                          |
                  |                   |    z^-1  |                          |
                  |                   ------------                          |
                  |                         ^                               |
                  |       ffe[2]            |        -dfe[2]                |
                  |--------> X -----------> + <--------- X -----------------|
                  :                                                         :
                  :                                                         :
                  :                                                         :
                  |                         ^                               |
                  |                         |                               |
                  |                   ------------                          |
                  |                   |    z^-1  |                          |
                  |                   ------------                          |
                  |                         ^                               |
                  |       ffe[N]            |        -dfe[N]                |
                  |--------> X -----------> + <--------- X -----------------|
        :param ffe: FFE coefficients of the filter
        :param dfe: DFE coefficients of the filter
        :param levels: the decision levels of the constellation
        :param z: state initial memory. If None assumes '0' memory. Must have the same length as:
                  max{len('b'), len('a')} - 1
        """
        super().__init__(ffe, dfe, z)
        self.levels = levels

    def _process(self, x):
        # Creating \tilde{x} and the sliced \hat{x[n]}
        a0_xtilde = self.b[0] * x + self._z[0]
        xtilde = a0_xtilde / self.a[0]
        xhat   = slicer(np.array([xtilde]), self.levels)
        # Advancing the memory
        z_temp = np.concatenate([self._z[1:], [0]])
        # Filling the new memory
        self._z = (x * self._b_vec) + z_temp - (xhat * self._a_vec)
        return xtilde

