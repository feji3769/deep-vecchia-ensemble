import torch
from ..sorting import RandomSorting

class NeighborOracle():
    def __init__(self, x, y, m, sorting_strategy=None, store_dist = False, sort_by_dist = False, **kwargs):
        if sorting_strategy is None:
            sorting_strategy = RandomSorting()
        self.sorting_strategy = sorting_strategy
        sorted_index = self.sorting_strategy.sort_data(x)
        self.x = x[sorted_index]
        self.y = y[sorted_index]
        self.m = m
        self.index = self.index_trainer(self.x, **kwargs)
        if store_dist:
            self.nn_array, self.nn_dist = self.construct_ordered_neighbors(\
                x, self.m, store_dist = store_dist, sort_by_dist = sort_by_dist)
        else:
            self.nn_array = self.construct_ordered_neighbors(x, self.m)

    def reorder_data(self, input_transform):
        transformed_x = input_transform.transform_query(self.x)
        sorted_index = self.sorting_strategy.sort_data(transformed_x)
        self.x = self.x[sorted_index]
        self.y = self.y[sorted_index]
        self.index = self.index_trainer(transformed_x)
        self.nn_array = self.construct_ordered_neighbors(transformed_x, self.m)

    def index_trainer(self, x, **kwargs):
        raise NotImplementedError

    def get_neighbors(self, subset):
        return self.nn_array[subset]

    def __getitem__(self, index):
        return self.x[index,...], self.y[index,...]

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def input_dim(self):
        return self.x.shape[-1]

    def query(self, x_query, k, **kwargs):
        if len(x_query.shape) > 2:
            D, I = self.batch_query(x_query, k, **kwargs)
        else:
            D, I = self.index.search(x_query, k = int(k))
        return D, I

    def construct_ordered_neighbors(self, x, m, store_dist = False, sort_by_dist = False):
        '''
        Find nearest neighbors subject to ordering. 
        Inputs
        x : nxp numpy array of ordered data. 
        m : int number of neighbors. 
        store_dist (bool): should the distances be saved?
        sort_by_dist(bool): sort the neighbor indices by distance.
        returns nearest neighbors according to ordered data. 
        '''
        n = x.shape[0]
        nbrs = (torch.arange(1, m+1) * -1).unsqueeze(0)
        nbrs = nbrs.tile((n, 1))
        if store_dist:
            dist = torch.ones((n, m)) * 100
            pair_dist = torch.nn.PairwiseDistance()
        for i in range(1, m+1):
            nbrs_set = torch.arange(0,i)

            if sort_by_dist or store_dist:
                dist_set = pair_dist(x[i:(i+1), :], x[0:(i), :])

            if sort_by_dist:
                sort_ind = torch.sort(nbrs_set)[1]
                nbrs_set = nbrs_set[sort_ind]
                dist_set = dist_set[sort_ind]

            if store_dist:
                dist[i, 0:i] = dist_set
            nbrs[i,0:i] = nbrs_set


        self.index.add(x[0:(m+1)])

        for i in range(m+1, n):
            dist_set, nbrs_set = self.index.search(x[i:(i+1), :], k = int(m))
            if sort_by_dist:
                sorted_ind = torch.arange(m)
            else:
                sorted_ind = torch.sort(nbrs_set[0], descending=True)[1]
            nbrs[i,:] = nbrs_set[0][sorted_ind]
            #nbrs[i,:] = torch.sort(nbrs_set[0], descending=True)[0]#old
            if store_dist:
                dist[i,:] = dist_set[0][sorted_ind]
            self.index.add(x[i:(i+1),:])
        if store_dist:
            return nbrs, dist
        else:
            return nbrs


    def batch_query(self, x_query, k, **kwargs):
        is_self = kwargs.get("is_self", False)
        # determine data dimension and number of locations.
        xq_shape = x_query.shape
        d = xq_shape[-1]
        test_shape = xq_shape[0:-1]
        num_locs = torch.prod(torch.tensor(test_shape))
        # perform query.
        if is_self: # if x_query in database exclude it from query.
            nn_dist, nn_ind = self.query(x_query.view((num_locs, d)).detach(), k+1, return_d = True)
            nn_ind = nn_ind[:, 1:]
            nn_dist = nn_dist[:, 1:]
        else: # x_query is not in database. 
            nn_dist, nn_ind = self.query(x_query.view((num_locs, d)).detach(), k, return_d = True)
        # reshape output for downstream. 
        output_shape = test_shape + (k,)
        nn_dist = nn_dist.reshape(output_shape)
        nn_ind = nn_ind.reshape(output_shape)
        # return NN indices and NN distances
        return nn_dist, nn_ind


