from xp_array_factory import xparray
import xp_array_factory as xaf

np = xaf.import_numpy_or_cupy()


class LogArray:
    def __init__(self, total_steps: int, p_num: int, is_double: bool = False):
        self.total_steps = total_steps
        self.p_num = p_num
        if is_double:
            dtype = np.float64
        else:
            dtype = np.float32

        size = (self.total_steps, self.p_num)
        self.x = np.zeros(size, dtype=dtype)
        self.y = np.zeros(size, dtype=dtype)
        self.vx = np.zeros(size, dtype=dtype)
        self.vy = np.zeros(size, dtype=dtype)
        self.fx = np.zeros(size, dtype=dtype)
        self.fy = np.zeros(size, dtype=dtype)

    def log(self, index: int, x: xparray, y: xparray, vx: xparray, vy: xparray, fx: xparray, fy: xparray):
        self.x[index] = x
        self.y[index] = y
        self.vx[index] = vx
        self.vy[index] = vy
        self.fx[index] = fx
        self.fy[index] = fy

    def get(self, index: int) -> tuple[xparray, xparray, xparray, xparray]:
        return self.x[index], self.y[index], self.fx[index], self.fy[index]

    def get_xy_lim(self) -> tuple[float, float, float, float]:
        return float(self.x.min()), float(self.x.max()), float(self.y.min()), float(self.y.max())