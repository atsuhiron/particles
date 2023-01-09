import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt


def func(x, a, b, c):
    return x*x*a + b*x + c


if __name__ == "__main__":
    path_c = "C:/Users/atsur/dev/particles/particles/cupy.npy"
    path_n = "C:/Users/atsur/dev/particles/particles/numpy.npy"

    arr_n: np.ndarray = np.load(path_n)
    arr_c: np.ndarray = np.load(path_c)
    arr = np.array([arr_n[0], arr_n[1], arr_c[0], arr_c[1]])

    means = np.mean(arr, axis=2)
    stds = np.std(arr, axis=2)
    means[2, 0] = arr[2, 0, 1:].mean()
    stds[2, 0] = arr[2, 0, 1:].std()
    names = ["NumPy (cpu, Core i7-1260P) float 32bit", "Numpy (cpu, Core i7-1260P) float 64bit",
             "Cupy  (gpu, RTX T550)          float 32bit", "Cupy  (gpu, RTX T550)          float 64bit"]
    alphas = [1.0, 0.3, 1.0, 0.3]
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]

    xx = np.array([64, 128, 256, 512, 1024, 2048])
    apx_xx = np.logspace(1.7, 3.4, 64)
    for ii in range(4):
        para, _ = so.curve_fit(func, xx, means[ii], p0=[0.01, 0.01, 0.01], sigma=stds[ii])
        print(para)
        apx_yy = func(apx_xx, *para)
        plt.plot(xx, means[ii], "o", markersize=8, label=names[ii], ls="", alpha=alphas[ii], color=colors[ii])
        plt.plot(apx_xx, apx_yy, color=colors[ii], alpha=alphas[ii])
    plt.xlabel("Number of particles")
    plt.ylabel("Calculating time [s]")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1e-2, 1e2])
    plt.show()