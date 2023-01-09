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


def calc_friction_resistance(coef: float, vx_arr: xparray, vy_arr: xparray) -> xparray:
    v_arr = np.array([vx_arr, vy_arr])
    norm = np.linalg.norm(v_arr, axis=0)
    resistance_norm = coef * norm
    norm[norm == 0] = 1.0
    unit_v_arr = v_arr / norm
    return resistance_norm * unit_v_arr


def calc_delta_v(step: float, mass: float, fx: xparray, fy: xparray) -> Tuple[xparray, xparray]:
    return step / mass * fx, step / mass * fy


def calc_delta_x(step: float, vx: xparray, vy: xparray) -> Tuple[xparray, xparray]:
    return step * vx, step * vy


def calc_step(x: xparray, y: xparray, fx: xparray, fy: xparray, vx: xparray, vy: xparray,
              interactive_force_coef: float, friction_resistance_coef: float, mass: float, step_size: float):
    # 相互作用による力
    _dfx, _dfy = calc_interactive_force(interactive_force_coef, x, y)
    fx += _dfx
    fy += _dfy

    # 粘性抵抗による力
    _dfx, _dfy = calc_friction_resistance(friction_resistance_coef, vx, vy)
    fx += _dfx
    fy += _dfy

    _dvx, _dvy = calc_delta_v(step_size, mass, fx, fy)
    vx += _dvx
    vy += _dvy
    _dx, _dy = calc_delta_x(step_size, vx, vy)
    x += _dx
    y += _dy
    return x, y, fx, fy, vx, vy


def calc(particle_num: int, total_steps: int, use_double: bool,
         interactive_force_coef: float, friction_resistance_coef: float, mass: float, step_size: float) -> xparray:
    if use_double:
        dtype = np.float64
    else:
        dtype = np.float32
    x = np.random.random(particle_num).astype(dtype) * 4
    y = np.random.random(particle_num).astype(dtype) * 4
    vx, vy = np.zeros_like(x).astype(dtype), np.zeros_like(y).astype(dtype)
    log_arr = np.zeros((total_steps, 6, particle_num), dtype=dtype)

    fx, fy = calc_interactive_force(interactive_force_coef, x, y)
    log_arr[0] = np.array([x, y, vx, vy, fx, fy])
    fx, fy = np.zeros_like(x), np.zeros_like(y)
    #for ii in tqdm.tqdm(range(1, total_steps), desc="Calc"):
    for ii in range(1, total_steps):
        x, y, fx, fy, vx, vy = calc_step(x, y, fx, fy, vx, vy,
                                         interactive_force_coef, friction_resistance_coef, mass, step_size)
        log_arr[ii] = np.array([x, y, vx, vy, fx, fy])
    return log_arr


if __name__ == "__main__":
    import time
    frame_dir = "frames"
    #p_num = 36
    _step_size = 0.01
    if_coef = 1.0
    fr_coef = -0.05
    _mass = 1.0
    _total_steps = 200
    #_use_double = True

    # initialize
    np.random.seed(125)
    path = ps.PathService(frame_dir)
    path.reset_dir()

    # run
    p_num_arr = [64, 128, 256, 512, 1028, 2048]
    use_double_arr = [False, True]
    iter_num = 20
    la = np.zeros((2, len(p_num_arr), iter_num), dtype=np.float64)
    for i in range(2):
        for j in tqdm.tqdm(range(len(p_num_arr))):
            for k in tqdm.tqdm(range(iter_num), leave=False):
                s = time.time()
                result_arr = calc(p_num_arr[j], _total_steps, use_double_arr[i], if_coef, fr_coef, _mass, _step_size)
                e = time.time()
                la[i, j, k] = e-s

    # result_arr = calc(p_num, _total_steps, _use_double, if_coef, fr_coef, _mass, _step_size)
    # drawer.save_frames(log_array.LogArray(result_arr), path, 6)
    #
    # mov_path = "out.mp4"
    # template = "ffmpeg -r 20 -i {} -vcodec libx264 -pix_fmt yuv420p -r 20 -loglevel error {}"
    # os.system(template.format(path.gen_template_frame_path(), mov_path))
