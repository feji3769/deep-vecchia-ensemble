from .sorting_strategy import SortingStrategy
import _pyvecch as maxmin_cpp

class MaxMinSorting(SortingStrategy):
    def __init__(self, max_group_size):
        self.max_group_size = max_group_size

    def sort_data(self, X):
        if X.dim() > 2:
            raise Exception("exact_max_min does not support batch operations.")
        if X.dim() < 2:
            raise Exception("X must be a 2 dimensional tensor.")
        ind  = np.arange(0, X.shape[0])
        ind_og = np.arange(0, X.shape[0])
        np.random.shuffle(ind)
        return ind_og[ind[self.grouped_exact_max_min(X[ind]).squeeze()]]


    def grouped_exact_max_min(self, X):
        N = X.shape[0]
        if N > self.max_group_size:
            ord1 = _grouped_exact_max_min(X[0:int(N//2)]).view(-1, 1)
            ord2 = _grouped_exact_max_min(X[int(N//2):]).view(-1,1) + N // 2
            if N % 2 == 0:
                ord = torch.cat((ord1, ord2), axis = -1).view(-1, 1)
            else:
                ord = torch.cat((
                torch.cat((ord1,ord2[0:-1]), axis = -1).view(-1,1), 
                ord2[-1].view(-1,1)))
            return ord
        else:
            return exact_max_min(X)


    def exact_max_min(self, X):
        """PyTorch implementation of exact max min from GPvecchia:
        https://github.com/katzfuss-group/GPvecchia/blob/master/src/MaxMin.cpp.
        
        X : (Nxd) torch tensor. Does not support batches. 
        
        returns (N,) torch tensor of location in max min ordering. """

        return maxmin_cpp.MaxMincpp(X).type(torch.LongTensor) - 1
