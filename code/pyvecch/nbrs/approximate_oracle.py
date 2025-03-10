import faiss
from .neighbor_oracle import NeighborOracle
import faiss.contrib.torch_utils

class ApproximateOracle(NeighborOracle):
    def __init__(self, x, y, m, n_list=100, n_probe=10, sorting_stategy = None, store_dist = False, sort_by_dist=False):
        super().__init__(x,y,m, sorting_stategy, n_list = n_list, n_probe = n_probe, \
        store_dist = store_dist, sort_by_dist=sort_by_dist)

        
    def index_trainer(self, x, **kwargs):
        n_list  = kwargs.get("n_list", 100)
        n_probe = kwargs.get("n_probe", 10)
        d = int(x.shape[-1])
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_list)
        index.nprobe = n_probe
        index.train(x)
        return index

if __name__ == '__main__':
    unittest.main()