import faiss
from .neighbor_oracle import NeighborOracle
import faiss.contrib.torch_utils

class ExactOracle(NeighborOracle):
    def __init__(self, x, y, m, prediction_stategy=None, store_dist = False, sort_by_dist=False):
        super().__init__(x,y,m, prediction_stategy, store_dist=store_dist, sort_by_dist=sort_by_dist)
        
    def index_trainer(self, x, **kwargs):
        d = int(x.shape[-1])
        index = faiss.IndexFlatL2(d)
        index.train(x)
        return index