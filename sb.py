import numpy as np
import cupy as cp


if __name__ == "__main__":
    arr = cp.ones((2, 256))
    ss = cp.sum(arr)
    print(ss)