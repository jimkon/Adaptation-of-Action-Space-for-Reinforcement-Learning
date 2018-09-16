import random
from numpy.random import randint


class sd_list:
    """docstring for sd_list."""

    def __init__(self, size, dims):
        super(sd_list, self).__init__()

        assert size > 0, "Size must be positive"
        assert dims >= 0, "dims must be positive or zero"

        self._size = int(size)
        self._dims = int(dims)

        self._buffer = []

    def _random_location(self, n=1):
        return randint(0, len(self._buffer)-1, size=n)

    def add(self, sample):
        if self._dims > 0:
            assert len(sample) == self._dims, "Samples must have dims={} dimensions".format(self._dims)

        if len(self._buffer) < self._size:
            self._buffer.append(sample)
        else:
            self._buffer[self._random_location()] = sample

    def elements(self):
        return self._buffer

    def get_samples(self, n):
        return self._buffer[self._random_location(n=n)]
