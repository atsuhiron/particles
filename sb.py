from typing import Tuple

import matplotlib.pyplot as plt

from xp_array_factory import xparray
import xp_array_factory as xaf

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


def show_quiver(x: xparray, y: xparray, fx: xparray, fy: xparray):
    f_norm = np.linalg.norm(np.c_[fx, fy], axis=1)
    if xaf.is_cupy_env():
        x, y, fx, fy, f_norm = xaf.to_numpy(x, y, fx, fy, f_norm)
    vec_ratio = 1
    plt.plot(x, y, "ko", markersize=8)
    plt.quiver(x, y, fx/f_norm*vec_ratio, fy/f_norm*vec_ratio, f_norm)
    plt.show()


if __name__ == "__main__":
    p_num = 36
    step_size = 0.1
    if_coef = 1.0
    _mass = 1.0
    np.random.seed(125)

    _x = np.random.random(p_num).astype(np.float32) * 10
    _y = np.random.random(p_num).astype(np.float32) * 10
    _vx, _vy = np.zeros_like(_x), np.zeros_like(_y)

    _fx, _fy = calc_interactive_force(if_coef, _x, _y)
    show_quiver(_x, _y, _fx, _fy)
    _dvx, _dvy = calc_delta_v(step_size, _mass, _fx, _fy)
    # ここに抵抗の処理
    _vx += _dvx
    _vy += _dvy
    _dx, _dy = calc_delta_x(step_size, _vx, _vy)
    _x += _dx
    _y += _dy

    show_quiver(_x, _y, _fx, _fy)
