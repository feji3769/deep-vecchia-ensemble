#from torch.utils.cpp_extension import load
#import os

#pc = False
#if pc:
#    os.environ["CXX"] = "/usr/bin/g++"
#else:
#    os.environ['CXX'] = "/home/felix.jimenez/anaconda3/envs/torch/bin/x86_64-conda-linux-gnu-c++"
#load(name="maxmin", sources=[os.path.dirname(os.path.abspath(__file__)) + "/maxMin.cpp"])
from .random_sorting import RandomSorting
#from .max_min_sorting import MaxMinSorting

if __name__ == "__main__":
    print("yes")
    import time
    start = time.time()
    x = torch.rand((4000, 200))
    #maxmin.MaxMincpp(x)
    print(time.time() - start)