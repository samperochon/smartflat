
import numpy as np
from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints


class CostCustom(BaseCost):
    model = ""

    def __init__(self):
        """Initialize the object."""
        self.min_size = 2
        self.gram = None


    def fit(self, gram):
        """Sets parameters of the instance.
        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)
        Returns:
            self
        """
        self.signal = gram
        self.gram = gram
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].
        Args:
            start (int): start of the segment
            end (int): end of the segment
        Returns:
            segment cost
        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val