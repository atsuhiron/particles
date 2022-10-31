import os
from typing import Tuple

import tqdm

from xp_array_factory import xparray
import xp_array_factory as xaf
import path_service as ps
import log_array
import drawer

np = xaf.import_numpy_or_cupy()


def calc_rel_vec(x_pos_arr: xparray, y_pos_arr: xparray) -> Tuple[xparray, xparray]:
    x_mat_h, x_mat_v = np.meshgrid(x_pos_arr, x_pos_arr)
    y_mat_h, y_mat_v = np.meshgrid(y_pos_arr, y_pos_arr)
    # TODO: なんかおかしくなったらここを反転させる
    return x_mat_h - x_mat_v, y_mat_h - y_mat_v


def calc_interactive_force(coef: float, x_pos_arr: xparray, y_pos_arr: xparray) -> Tuple[xparray, xparray]:
    rel_x, rel_y = calc_rel_vec(x_pos_arr, y_pos_arr)
    # shape = (N, N)
    dist_sq = np.square(rel_x) + np.square(rel_y)
    dist_sq[dist_sq == 0] = 1.0

    # shape = (N, N, 2)
    rel = np.c_[rel_x[:, :, np.newaxis], rel_y[:, :, np.newaxis]]
    # shape = (N, N, 1)
    norm = np.linalg.norm(rel, axis=2)[:, :, np.newaxis]
    norm[norm == 0] = 1.0

    # shape = (N, N, 2)
    inter_force = coef * rel / norm / dist_sq[:, :, np.newaxis]
    inter_force_sum = np.sum(inter_force, axis=0)

    return inter_force_sum[:, 0], inter_force_sum[:, 1]


def calc_delta_v(step: float, mass: float, fx: xparray, fy: xparray) -> Tuple[xparray, xparray]:
    return step * fx / mass, step * fy / mass


def calc_delta_x(step: float, vx: xparray, vy: xparray) -> Tuple[xparray, xparray]:
    return step * vx, step * vy


if __name__ == "__main__":
    frame_dir = "frames"
    p_num = 36
    step_size = 0.01
    if_coef = 1.0
    _mass = 1.0
    total_steps = 200
    use_double = False

    # initialize
    np.random.seed(125)
    if use_double:
        dtype = np.float64
    else:
        dtype = np.float32
    _x = np.random.random(p_num).astype(dtype) * 4
    _y = np.random.random(p_num).astype(dtype) * 4
    _vx, _vy = np.zeros_like(_x), np.zeros_like(_y)
    _log_arr = log_array.LogArray(total_steps, p_num)
    path = ps.PathService(frame_dir)
    path.reset_dir()

    _fx, _fy = calc_interactive_force(if_coef, _x, _y)
    _log_arr.log(0, _x, _y, _vx, _vy, _fx, _fy)
    for i in tqdm.tqdm(range(1, total_steps), desc="Calc"):
        _fx, _fy = calc_interactive_force(if_coef, _x, _y)
        _dvx, _dvy = calc_delta_v(step_size, _mass, _fx, _fy)
        # ここに抵抗の処理
        _vx += _dvx
        _vy += _dvy
        _dx, _dy = calc_delta_x(step_size, _vx, _vy)
        _x += _dx
        _y += _dy
        _log_arr.log(i, _x, _y, _vx, _vy, _fx, _fy)

    drawer.save_frames(_log_arr, path, 4)

    mov_path = "out.mp4"
    os.system(
        "ffmpeg -r 20 -i {} -vcodec libx264 -pix_fmt yuv420p -r 20 {}".format(path.gen_template_frame_path(), mov_path))
