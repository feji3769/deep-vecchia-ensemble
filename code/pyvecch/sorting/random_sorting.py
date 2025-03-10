from .sorting_strategy import SortingStrategy
import numpy as np

class RandomSorting(SortingStrategy):
    def sort_data(self, X):
        n = X.shape[0]
        return np.arange(0, n)