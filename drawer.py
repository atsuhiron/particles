from typing import Optional
from typing import Iterable
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import tqdm

from xp_array_factory import xparray
import xp_array_factory as xaf
from log_array import LogArray
import path_service as ps

np = xaf.import_numpy_or_cupy()
matplotlib.use('Agg')


@dataclass
class DrawQuiverParam:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


draw_quiver_args = tuple[xparray, xparray, xparray, xparray, str, DrawQuiverParam]


def draw_quiver(x: xparray, y: xparray, fx: xparray, fy: xparray,
                f_name: Optional[str] = None, draw_param: Optional[DrawQuiverParam] = None):
    f_norm = np.linalg.norm(np.c_[fx, fy], axis=1)
    if xaf.is_cupy_env():
        x, y, fx, fy, f_norm = xaf.to_numpy(x, y, fx, fy, f_norm)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o", mfc="#00000050", markersize=8)
    ax.quiver(x, y, fx / f_norm, fy / f_norm, f_norm)
    ax.set_xlim([draw_param.x_min, draw_param.x_max])
    ax.set_ylim([draw_param.y_min, draw_param.y_max])
    if f_name is None:
        plt.show()
    else:
        fig.savefig(f_name)
        plt.close()


def _show_quiver_wrap(args: draw_quiver_args):
    draw_quiver(*args)


def _make_args(log_arr: LogArray, path_service: ps.PathService) -> Iterable[draw_quiver_args]:
    dqp = DrawQuiverParam(*log_arr.get_xy_lim())
    for i in range(log_arr.total_steps):
        x, y, fx, fy = log_arr.get(i)
        frame_path = path_service.gen_frame_path(i)
        yield x, y, fx, fy, frame_path, dqp


def save_frames(log_arr: LogArray, path_service: ps.PathService, num: int):
    with tqdm.tqdm(total=log_arr.total_steps, desc="Drawing") as p_bar:
        with ThreadPool(num) as pool:
            for _ in pool.imap_unordered(_show_quiver_wrap, _make_args(log_arr, path_service), chunksize=2):
                p_bar.update(1)
        p_bar.close()
