from xp_array_factory import xparray
import xp_array_factory as xaf

np = xaf.import_numpy_or_cupy()


class LogArray:
    def __init__(self, arr: xparray):
        # [x, y, vx, vy, fx, fy]
        self.arr = arr

    def get(self, index: int) -> tuple[xparray, xparray, xparray, xparray]:
        return self.arr[index, 0], self.arr[index, 1], self.arr[index, 4], self.arr[index, 5]

    def get_xy_lim(self) -> tuple[float, float, float, float]:
        return float(self.arr[:, 0].min()), float(self.arr[:, 0].max()), \
               float(self.arr[:, 1].min()), float(self.arr[:, 0].max())

    def get_total_steps(self) -> int:
        return self.arr.shape[0]
