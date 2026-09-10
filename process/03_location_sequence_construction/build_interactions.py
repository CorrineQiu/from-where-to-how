#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build shared-F0 right-hand interaction sequences from existing annotation caches.
Retains the 2026-01-12 numerical pipeline; paths are explicit CLI arguments.

Source: My_Code_qwen/build_interaction_json_new_0112.py
Source SHA-256: 28df79b988edf13bc14da68191fb8024f2d7c670f663a7cf62cff565ff98a143
Only trusted local pickle/joblib inputs are supported. Model assets and datasets
are user-supplied and are not distributed by this repository.
"""

SOURCE_FILE = "My_Code_qwen/build_interaction_json_new_0112.py"
SOURCE_SHA256 = "28df79b988edf13bc14da68191fb8024f2d7c670f663a7cf62cff565ff98a143"


import os
import os.path as osp
import json
import pickle
import argparse
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# LVIS
# =========================
def load_lvis_name_to_id(lvis_cat_json: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}

    if not lvis_cat_json or (not os.path.exists(lvis_cat_json)):
        print(f"[WARN] LVIS category json not found: {lvis_cat_json}")
        return name_to_id, id_to_name

    with open(lvis_cat_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "categories" in data:
        cats = data["categories"]
    elif isinstance(data, list):
        cats = data
    else:
        print(f"[WARN] Unknown LVIS json format: {type(data)}")
        return name_to_id, id_to_name

    for idx, c in enumerate(cats):
        if not isinstance(c, dict):
            continue
        nm = c.get("name", None)
        if isinstance(nm, str) and nm.strip():
            nm = nm.strip()
            name_to_id[nm] = idx
            id_to_name[idx] = nm

    print(f"[INFO] Loaded LVIS categories: {len(name_to_id)}")
    return name_to_id, id_to_name


# =========================
# Utils: basic
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def same_file(a, b):
    """Resolve path aliases, including existing hard links."""
    if osp.realpath(a) == osp.realpath(b):
        return True
    try:
        return osp.samefile(a, b)
    except OSError:
        return False


def preflight_outputs(paths, overwrite=False, input_paths=()):
    """Check every planned artifact before writing the first one."""
    for path in paths:
        if any(same_file(path, source) for source in input_paths):
            raise ValueError(f"Output would replace an input file: {path}")
        if osp.exists(path) and not overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite explicitly")

def norm_ts(x):
    return float(np.round(float(x), 6))


def nearest_existing_frame_with_err(frames_sorted: List[int], target_frame: int) -> Tuple[int, float]:
    arr = np.asarray(frames_sorted, dtype=np.int64)
    if arr.size == 0:
        return int(target_frame), float("inf")
    pos = int(np.searchsorted(arr, target_frame))
    if pos <= 0:
        fr = int(arr[0])
        return fr, float(abs(fr - target_frame))
    if pos >= len(arr):
        fr = int(arr[-1])
        return fr, float(abs(fr - target_frame))
    left = int(arr[pos - 1])
    right = int(arr[pos])
    if abs(left - target_frame) <= abs(right - target_frame):
        return left, float(abs(left - target_frame))
    else:
        return right, float(abs(right - target_frame))


def get_hand_xyz_at_frame(h: dict, vert_id: int) -> Optional[np.ndarray]:
    if h is None or ('verts' not in h):
        return None
    v = np.asarray(h['verts'])
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] <= int(vert_id):
        return None
    xyz = v[int(vert_id)]
    xyz = np.asarray(xyz, dtype=np.float32)
    if not np.isfinite(xyz).all():
        return None
    return xyz


def get_hand_xyz_by_timestamp(
    ts: float,
    human_full: dict,
    frames_sorted: List[int],
    fps: float,
    vert_id: int,
    max_full_frame_err: float,
) -> Optional[Tuple[np.ndarray, int]]:
    nominal_frame = int(np.floor(float(ts) * float(fps)))
    fr_near, ferr = nearest_existing_frame_with_err(frames_sorted, nominal_frame)
    if ferr > float(max_full_frame_err) * float(fps):
        return None
    h = human_full.get(int(fr_near), None)
    xyz = get_hand_xyz_at_frame(h, vert_id)
    if xyz is None:
        return None
    return xyz, int(fr_near)


# =========================
# Hand parsing from narration
# =========================
def parse_hand_from_narration(narr: Any) -> str:
    """
    return: 'left' / 'right' / 'both' / 'unknown'
    """
    if not isinstance(narr, str):
        return "unknown"
    s = narr.strip().lower()

    has_left = ("left hand" in s) or ("with his left" in s) or ("with her left" in s) or ("left-handed" in s)
    has_right = ("right hand" in s) or ("with his right" in s) or ("with her right" in s) or ("right-handed" in s)
    has_both_phrase = ("both hands" in s) or ("two hands" in s) or ("with both" in s)

    if has_both_phrase or (has_left and has_right):
        return "both"
    if has_right:
        return "right"
    if has_left:
        return "left"
    return "unknown"


# =========================
# Object token helpers
# =========================
def split_object_tokens(obj_str: Any, split_token: str = "@@@@@") -> List[str]:
    if not isinstance(obj_str, str):
        return []
    parts = [p.strip() for p in obj_str.split(split_token)]
    return [p for p in parts if p]


def merge_object_tokens_keep_order(tokens: List[str], join_token: str = ",") -> str:
    seen = set()
    out = []
    for t in tokens:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        out.append(tt)
    return join_token.join(out)


# =========================
# Window selection (same as you)
# =========================
def select_anticipation_window(df, timestamp_col='timestamp', future_time=60.):
    subset_dict = {}
    subset_uncompressed_dict = {}
    current_time = {}

    for index, row in df.iterrows():
        timestamp = float(row[timestamp_col])
        window_start = timestamp
        window_end = window_start + float(future_time)

        window_df = df[(df[timestamp_col] >= window_start) & (df[timestamp_col] <= window_end)]

        if 'object' in window_df.columns:
            # Avoid hashing ndarray-valued annotation columns in a MultiIndex.
            # This is the same object-token expansion as the historical stack.
            window_df_expanded = window_df.assign(
                object=window_df["object"].str.split("@@@@@")
            ).explode("object").dropna(subset=["object"]).reset_index(drop=True)
            window_df_sorted = window_df_expanded.sort_values(by=['object', 'timestamp'])
            window_df_unique = window_df_sorted.drop_duplicates(subset='object', keep='first')
            window_df_unique.reset_index(drop=True, inplace=True)
        else:
            window_df_sorted = window_df.sort_values(by=['timestamp']).copy()
            window_df_unique = window_df_sorted.copy()
            window_df_unique.reset_index(drop=True, inplace=True)

        subset_dict[index] = window_df_unique
        subset_uncompressed_dict[index] = window_df_sorted
        current_time[index] = timestamp

    return subset_dict, subset_uncompressed_dict, current_time


# =========================
# Observation trajectory
# =========================
def build_observation_traj(
    human_full: dict,
    frames_sorted: List[int],
    obs_start_s: float,
    obs_end_s: float,
    fps: float,
    right_hand_vert_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if obs_end_s <= obs_start_s:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    fr0 = int(np.floor(obs_start_s * fps))
    fr1 = int(np.floor(obs_end_s * fps))

    times = []
    xyzs = []

    arr = np.asarray(frames_sorted, dtype=np.int64)
    l = int(np.searchsorted(arr, fr0, side="left"))
    r = int(np.searchsorted(arr, fr1, side="right"))
    for fr in arr[l:r]:
        h = human_full.get(int(fr), None)
        xyz = get_hand_xyz_at_frame(h, right_hand_vert_id)
        if xyz is None:
            continue
        times.append(float(fr) / float(fps))
        xyzs.append(xyz.astype(np.float32))

    if len(times) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.asarray(times, dtype=np.float32), np.asarray(xyzs, dtype=np.float32)


# =========================
# Check: same time multi-location
# =========================
def check_same_time_multi_location(interactions: List[Dict[str, Any]], dist_eps: float = 1e-4) -> List[Dict[str, Any]]:
    if not interactions:
        return []
    buckets: Dict[float, List[np.ndarray]] = {}
    for it in interactions:
        if "time" not in it or "location" not in it:
            continue
        t = float(it["time"])
        loc = np.asarray(it["location"], dtype=np.float32).reshape(3,)
        buckets.setdefault(t, []).append(loc)

    issues = []
    for t, locs in buckets.items():
        if len(locs) <= 1:
            continue
        arr = np.stack(locs, axis=0)
        d = np.linalg.norm(arr - arr[0:1], axis=1)
        maxd = float(np.max(d))
        if maxd > dist_eps:
            issues.append({
                "time": float(t),
                "count": int(len(locs)),
                "max_dist": maxd,
                "locations": arr.tolist(),
            })
    return issues


# =========================
# Pose averaging: last 1s -> pick most-stable 10 frames -> average
# =========================
def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return q


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def _average_quaternions(quats: np.ndarray) -> np.ndarray:
    if quats.shape[0] == 1:
        return quats[0]
    q0 = quats[0]
    aligned = []
    for q in quats:
        if np.dot(q0, q) < 0:
            aligned.append(-q)
        else:
            aligned.append(q)
    Q = np.stack(aligned, axis=0)
    A = np.zeros((4,4), dtype=np.float64)
    for q in Q:
        A += np.outer(q, q)
    A /= float(Q.shape[0])
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[0] < 0:
        q_avg = -q_avg
    q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-12)
    return q_avg


def _get_device_pose_from_human_frame(h: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if h is None:
        return None
    if "rotation_matrix" not in h or "device_trajectory" not in h:
        return None
    R = np.asarray(h["rotation_matrix"], dtype=np.float64)
    t = np.asarray(h["device_trajectory"], dtype=np.float64).reshape(-1)
    if R.shape != (3,3) or t.size < 3:
        return None
    t = t[:3]
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return None
    return R, t


def _rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """angle between rotations (deg) from R = R1^T R2"""
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def _pick_most_stable_window(
    poses_R: List[np.ndarray],
    poses_t: List[np.ndarray],
    win_frames: int,
) -> Tuple[int, int, float]:
    """
    在给定序列里选运动最小的窗口（平移增量 + 旋转增量）。
    返回 (s, e, cost) (闭区间)
    """
    N = len(poses_R)
    if N <= win_frames:
        return 0, N - 1, 0.0

    best_cost = 1e18
    best = (0, win_frames - 1)

    for s in range(0, N - win_frames + 1):
        e = s + win_frames - 1
        trans = 0.0
        rot = 0.0
        for k in range(s, e):
            trans += float(np.linalg.norm(poses_t[k + 1] - poses_t[k]))
            rot += _rot_angle_deg(poses_R[k], poses_R[k + 1])
        cost = trans + 0.01 * rot
        if cost < best_cost:
            best_cost = cost
            best = (s, e)

    return best[0], best[1], float(best_cost)


def compute_F0_from_last_second(
    human_full: dict,
    frames_sorted: List[int],
    obs_end_time_s: float,
    fps: float,
    last_sec: float = 1.0,
    min_pose_frames: int = 10,
    win_frames: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    用 observation_end_time 前 last_sec 的 device pose，
    先选“最稳定的 win_frames”窗口，再对窗口内 pose 做平均，得到 F0 (R0,t0)。

    若可用帧数 < min_pose_frames：fallback 到 obs_end 最近单帧。
    """
    fr_end = int(np.floor(float(obs_end_time_s) * float(fps)))
    n_frames = int(round(float(last_sec) * float(fps)))
    fr_start = fr_end - n_frames + 1

    poses_R = []
    poses_t = []
    used_frames = []

    for fr in range(fr_start, fr_end + 1):
        fr_near, _ = nearest_existing_frame_with_err(frames_sorted, fr)
        h = human_full.get(int(fr_near), None)
        out = _get_device_pose_from_human_frame(h)
        if out is None:
            continue
        R, t = out
        poses_R.append(R)
        poses_t.append(t)
        used_frames.append(int(fr_near))

    if len(used_frames) > 1:
        uniq = {}
        for i, fr in enumerate(used_frames):
            uniq.setdefault(fr, i)
        idxs = sorted(list(uniq.values()))
        poses_R = [poses_R[i] for i in idxs]
        poses_t = [poses_t[i] for i in idxs]
        used_frames = [used_frames[i] for i in idxs]

    if len(poses_R) < int(min_pose_frames):
        fr_near, _ = nearest_existing_frame_with_err(frames_sorted, fr_end)
        h = human_full.get(int(fr_near), None)
        out = _get_device_pose_from_human_frame(h)
        if out is None:
            return None
        R0, t0 = out
        dbg = {
            "method": "fallback_single_frame",
            "fr_end": int(fr_end),
            "used_frame": int(fr_near),
            "num_frames": 1,
        }
        return R0.astype(np.float64), t0.astype(np.float64), dbg

    win_frames_eff = int(min(int(win_frames), len(poses_R)))
    s, e, cost = _pick_most_stable_window(poses_R, poses_t, win_frames_eff)
    poses_R_win = poses_R[s:e+1]
    poses_t_win = poses_t[s:e+1]
    used_frames_win = used_frames[s:e+1]

    t_stack = np.stack(poses_t_win, axis=0)
    t0 = np.mean(t_stack, axis=0)

    quats = np.stack([_rotmat_to_quat(R) for R in poses_R_win], axis=0)
    q0 = _average_quaternions(quats)
    R0 = _quat_to_rotmat(q0)

    dbg = {
        "method": "stable_window_avg",
        "fr_end": int(fr_end),
        "fr_start": int(fr_start),
        "num_frames_available": int(len(poses_R)),
        "win_frames": int(win_frames_eff),
        "win_cost": float(cost),
        "used_frames_min": int(np.min(used_frames)),
        "used_frames_max": int(np.max(used_frames)),
        "used_frames_win_min": int(np.min(used_frames_win)),
        "used_frames_win_max": int(np.max(used_frames_win)),
    }
    return R0.astype(np.float64), t0.astype(np.float64), dbg


# =========================
# Canonical transform + normalization
# =========================
def world_to_F0(points_w: np.ndarray, R0: np.ndarray, t0: np.ndarray) -> np.ndarray:
    p = np.asarray(points_w, dtype=np.float64)
    orig_shape = p.shape
    p = p.reshape(-1, 3)
    out = (R0.T @ (p - t0.reshape(1,3)).T).T
    return out.reshape(orig_shape).astype(np.float32)


def normalize_xyz(p_f0: np.ndarray, pos_scale_m: float) -> np.ndarray:
    s = float(pos_scale_m)
    p = np.asarray(p_f0, dtype=np.float32) / max(s, 1e-6)
    p = np.clip(p, -1.0, 1.0)
    return p


def normalize_time_obs(obs_times_s: np.ndarray, obs_end_time_s: float, observation_time: float) -> np.ndarray:
    t = np.asarray(obs_times_s, dtype=np.float32)
    denom = max(float(observation_time), 1e-6)
    return (t - float(obs_end_time_s)) / denom


def normalize_time_future(inter_times_s: np.ndarray, current_time_s: float, future_time: float) -> np.ndarray:
    t = np.asarray(inter_times_s, dtype=np.float32)
    denom = max(float(future_time), 1e-6)
    return (t - float(current_time_s)) / denom


def _obj_set(obj_str: Any) -> Tuple[str, ...]:
    if not isinstance(obj_str, str) or (not obj_str.strip()):
        return tuple()
    toks = [t.strip() for t in obj_str.split(",") if t.strip()]
    toks = sorted(set(toks))
    return tuple(toks)


def compress_dense_repeated_interactions(
    interactions_norm: List[Dict[str, Any]],
    dense_dt: float = 3.0,
    pos_eps_f0: float = 0.12,
    use_pos_field: str = "location_f0_m",
    compress_mode: str = "first",
) -> List[Dict[str, Any]]:
    if not interactions_norm:
        return []

    dense_dt = float(dense_dt)
    pos_eps_f0 = float(pos_eps_f0)

    xs = sorted(interactions_norm, key=lambda it: float(it.get("time", 0.0)))

    def get_pos(it):
        p = it.get(use_pos_field, None)
        if isinstance(p, np.ndarray):
            p = p.tolist()
        if isinstance(p, (list, tuple)) and len(p) == 3:
            try:
                arr = np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
                if np.isfinite(arr).all():
                    return arr
            except Exception:
                return None
        return None

    def finalize_segment(seg: List[Dict[str, Any]]) -> Dict[str, Any]:
        if compress_mode == "first":
            rep = dict(seg[0])
        elif compress_mode == "middle":
            rep = dict(seg[len(seg)//2])
        elif compress_mode == "mean":
            rep = dict(seg[0])
            for fld in [use_pos_field, "location_norm", "location"]:
                ps = []
                for it in seg:
                    p = it.get(fld, None)
                    if isinstance(p, np.ndarray):
                        p = p.tolist()
                    if isinstance(p, (list, tuple)) and len(p) == 3:
                        try:
                            arr = np.asarray(p, dtype=np.float32)
                            if np.isfinite(arr).all():
                                ps.append(arr)
                        except Exception:
                            pass
                if ps:
                    m = np.mean(np.stack(ps, 0), axis=0)
                    rep[fld] = m.astype(np.float32).tolist()
        else:
            rep = dict(seg[0])

        rep["segment_len"] = int(len(seg))
        rep["segment_t_start"] = float(seg[0].get("time", 0.0))
        rep["segment_t_end"] = float(seg[-1].get("time", 0.0))
        rep["segment_duration"] = float(rep["segment_t_end"] - rep["segment_t_start"])
        return rep

    out: List[Dict[str, Any]] = []
    seg: List[Dict[str, Any]] = [xs[0]]

    prev = xs[0]
    prev_t = float(prev.get("time", 0.0))
    prev_pos = get_pos(prev)
    prev_obj = _obj_set(prev.get("object", ""))

    for cur in xs[1:]:
        cur_t = float(cur.get("time", 0.0))
        cur_pos = get_pos(cur)
        cur_obj = _obj_set(cur.get("object", ""))

        dt = cur_t - prev_t
        dpos = None
        if (prev_pos is not None) and (cur_pos is not None):
            dpos = float(np.linalg.norm(cur_pos - prev_pos))

        same_obj = (cur_obj == prev_obj)
        close_time = (dt <= dense_dt)
        close_pos = (dpos is not None and dpos <= pos_eps_f0) if (prev_pos is not None and cur_pos is not None) else True

        if same_obj and close_time and close_pos:
            seg.append(cur)
        else:
            out.append(finalize_segment(seg))
            seg = [cur]

        prev = cur
        prev_t = cur_t
        prev_pos = cur_pos
        prev_obj = cur_obj

    out.append(finalize_segment(seg))
    return out


# =========================
# Env extraction helpers (保持原样)
# =========================
def _try_parse_obj_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    parts = name.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def obb_corners_to_center_size_rot(corners_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = np.asarray(corners_w, dtype=np.float64).reshape(8,3)
    center = np.mean(C, axis=0)

    X = C - center.reshape(1,3)
    cov = (X.T @ X) / float(X.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]

    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    proj = X @ axes
    minv = np.min(proj, axis=0)
    maxv = np.max(proj, axis=0)
    size = (maxv - minv)

    R_box = axes
    return center.astype(np.float64), size.astype(np.float64), R_box.astype(np.float64)


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32).reshape(3,3)
    return np.concatenate([R[:,0], R[:,1]], axis=0).astype(np.float32)


def extract_env_objects_tokens(
    full_take: dict,
    R0: np.ndarray,
    t0: np.ndarray,
    pos_scale_m: float,
    max_objects: int = 30,
    name_to_cat_id: Optional[Dict[str, int]] = None,
    cat_id_to_name: Optional[Dict[int, str]] = None,
    log_count_norm_denom: float = 10.0,
) -> List[Dict[str, Any]]:
    objs = full_take.get("objects", None)
    if not isinstance(objs, dict):
        return []

    names = objs.get("object_names", None)
    counts = objs.get("object_counts", None)
    rrc = objs.get("rrc", None)
    if names is None or counts is None or rrc is None:
        return []

    rrc_arr = np.asarray(rrc)
    if rrc_arr.ndim == 3 and rrc_arr.shape[1:] == (3, 8):
        rrc_arr = rrc_arr.transpose(0, 2, 1)
    if not (rrc_arr.ndim == 3 and rrc_arr.shape[1:] == (8, 3)):
        return []

    name_to_cat_id = name_to_cat_id or {}
    cat_id_to_name = cat_id_to_name or {}

    unk_id = 0
    num_cats_with_unk = len(name_to_cat_id) + 1

    N = min(len(names), rrc_arr.shape[0], int(max_objects))
    out: List[Dict[str, Any]] = []

    s = max(float(pos_scale_m), 1e-6)
    denom = max(float(log_count_norm_denom), 1e-6)

    for i in range(N):
        raw_name = str(names[i])
        nm = _try_parse_obj_name(raw_name)

        cat_id_raw = name_to_cat_id.get(nm, None)
        cat_id = int(cat_id_raw) + 1 if cat_id_raw is not None else unk_id
        cat_name = cat_id_to_name.get(cat_id - 1, nm) if cat_id > 0 else "unknown"

        cnt = float(counts[i]) if i < len(counts) else 0.0
        log_cnt = float(np.log1p(max(cnt, 0.0)))
        log_cnt_norm = float(np.clip(log_cnt / denom, 0.0, 1.0))

        corners = np.asarray(rrc_arr[i], dtype=np.float64)
        center_w, size_w, R_box_w = obb_corners_to_center_size_rot(corners)

        center_f0 = world_to_F0(center_w.reshape(1, 3), R0, t0).reshape(3,)
        center_norm = np.clip(center_f0 / s, -1.0, 1.0).astype(np.float32)

        R_box_f0 = (R0.T @ R_box_w).astype(np.float32)
        rot6d = rotmat_to_6d(R_box_f0).astype(np.float32)

        size_f0 = size_w.astype(np.float32)
        size_norm = np.clip(size_f0 / s, 0.0, 1.0).astype(np.float32)

        out.append({
            "category_id": int(cat_id),
            "category_name": str(cat_name),
            "raw_name": str(raw_name),
            "num_categories_with_unk": int(num_cats_with_unk),

            "center": center_norm.tolist(),
            "size": size_norm.tolist(),
            "rot6d": rot6d.tolist(),

            "log_count": float(log_cnt),
            "log_count_norm": float(log_cnt_norm),
        })

    return out


def extract_env_voxel_tokens(
    full_take: dict,
    R0: np.ndarray,
    t0: np.ndarray,
    pos_scale_m: float,
    max_voxels: int = 512,
    sample_strategy: str = "random",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    if "env_voxels" not in full_take or "voxel_centers" not in full_take:
        return []
    env = np.asarray(full_take["env_voxels"])
    centers = np.asarray(full_take["voxel_centers"])
    if env.shape != (16,16,16) or centers.shape != (16,16,16,3):
        return []

    idxs = np.argwhere(env > 0)
    if idxs.shape[0] == 0:
        return []

    if idxs.shape[0] > int(max_voxels):
        rng = np.random.default_rng(int(seed))
        choose = rng.choice(idxs.shape[0], size=int(max_voxels), replace=False)
        idxs = idxs[choose]

    out = []
    pts_w = centers[idxs[:,0], idxs[:,1], idxs[:,2]]
    pts_f0 = world_to_F0(pts_w, R0, t0)
    pts_norm = normalize_xyz(pts_f0, pos_scale_m)

    cls_ids = env[idxs[:,0], idxs[:,1], idxs[:,2]].astype(np.int32)

    for j in range(idxs.shape[0]):
        out.append({
            "class_id": int(cls_ids[j]),
            "center_f0_m": pts_f0[j].tolist(),
            "center_f0_norm": pts_norm[j].tolist(),
            "ijk": [int(idxs[j,0]), int(idxs[j,1]), int(idxs[j,2])],
        })
    return out


# =========================
# Load full pkl
# =========================
def load_full_take_pkl(dataset_root: str, take_name: str) -> Optional[dict]:
    pkl_path = osp.join(dataset_root, f"{take_name}_humans_objects_interactions.pkl")
    if not osp.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# =========================
# Build right-hand-only interactions with min_gap
# =========================
def build_future_interactions_right_only(
    df_ev_raw: pd.DataFrame,
    human_full: dict,
    frames_sorted: List[int],
    current_time: float,
    future_time: float,
    fps: float,
    right_hand_vert_id: int,
    max_full_frame_err: float,
    min_gap: float,
    unknown_hand_mode: str = "right",  # <- 默认改成 right：unknown 当作 right
) -> List[Dict[str, Any]]:
    """
    只输出右手交互（both/two hands 也当作 right；unknown 也可按 unknown_hand_mode 处理）：
      - time
      - location (right hand world xyz)
      - frame_idx
      - object (合并去重，仅右手)
    min_gap: 对每个 object token 去抖；无 object 用 __noobj__
    """
    if df_ev_raw is None or len(df_ev_raw) == 0:
        return []

    df = df_ev_raw.copy()
    if "timestamp" not in df.columns:
        return []

    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp_norm"] = df["timestamp"].map(norm_ts)

    end_time = norm_ts(current_time + float(future_time))
    df = df[(df["timestamp_norm"] >= float(current_time)) & (df["timestamp_norm"] < float(end_time))].copy()
    if len(df) == 0:
        return []

    # parse hand
    if "narration" in df.columns:
        df["hand"] = df["narration"].apply(parse_hand_from_narration)
    else:
        df["hand"] = "unknown"

    # unknown handling
    if unknown_hand_mode not in ["drop", "right", "left"]:
        unknown_hand_mode = "right"  # <- 非法值也默认 right（避免误删）
    if unknown_hand_mode == "drop":
        df = df[df["hand"] != "unknown"].copy()
    else:
        df.loc[df["hand"] == "unknown", "hand"] = unknown_hand_mode

    if len(df) == 0:
        return []

    # keep ONLY right-hand events; treat 'both' as 'right'
    df = df[df["hand"].isin(["right", "both"])].copy()
    if len(df) == 0:
        return []

    df["hand"] = "right"  # unify

    # ---- build candidate (t, obj_token) list for debounce ----
    candidates: List[Tuple[float, Optional[str]]] = []
    for _, row in df.iterrows():
        t = float(row["timestamp_norm"])
        obj_tokens = []
        if "object" in df.columns and isinstance(row.get("object", None), str):
            obj_tokens = split_object_tokens(row["object"], split_token="@@@@@")
        if len(obj_tokens) == 0:
            candidates.append((t, None))
        else:
            for ot in obj_tokens:
                candidates.append((t, ot))

    if len(candidates) == 0:
        return []

    candidates.sort(key=lambda x: (x[0], "" if x[1] is None else x[1]))

    # ---- debounce by object token ----
    mg = float(min_gap)
    last_keep: Dict[str, float] = {}
    kept: List[Tuple[float, Optional[str]]] = []

    for (t, obj) in candidates:
        if mg <= 0.0:
            kept.append((t, obj))
            continue
        key = obj if obj is not None else "__noobj__"
        lt = last_keep.get(key, None)
        if lt is None or (t - lt) >= mg:
            kept.append((t, obj))
            last_keep[key] = t

    if len(kept) == 0:
        return []

    # ---- group by timestamp -> one interaction per ts ----
    ts2objs: Dict[float, List[str]] = {}
    for t, obj in kept:
        if obj is None:
            ts2objs.setdefault(float(t), [])
        else:
            ts2objs.setdefault(float(t), []).append(str(obj))

    ts_list = sorted(ts2objs.keys())

    # get right-hand xyz for each ts
    interactions: List[Dict[str, Any]] = []
    for ts in ts_list:
        out = get_hand_xyz_by_timestamp(
            ts=ts,
            human_full=human_full,
            frames_sorted=frames_sorted,
            fps=fps,
            vert_id=right_hand_vert_id,
            max_full_frame_err=max_full_frame_err,
        )
        if out is None:
            continue
        xyz, fr = out

        obj_str = merge_object_tokens_keep_order(ts2objs.get(ts, []), join_token=",")

        it = {
            "time": float(ts),
            "hand": "right",
            "location": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "frame_idx": int(fr),
        }
        if obj_str:
            it["object"] = obj_str

        interactions.append(it)

    interactions.sort(key=lambda x: float(x["time"]))
    return interactions


# =========================
# Build cache with canonical transform + normalization
# =========================
def build_cache_points(
    split: str,
    scenarios: str,
    dataset_root: str,
    annotations_root: str,
    takes_txt_dir: str,
    takes_json: str,
    out_cache: str,
    lvis_cat_json: str,

    observation_time: float,
    anticipation_time: float,
    future_time: float,

    right_hand_vert_id: int,
    full_fps: float,
    max_full_frame_err: float,

    last_sec_for_pose_avg: float,
    pose_avg_win_frames: int,
    pos_scale_m: float,
    log_count_norm_denom: float,

    include_env_objects: bool,
    max_env_objects: int,
    include_env_voxels: bool,
    max_env_voxels: int,

    export_first_n_json: int,
    viz_first_n: int,
    viz_out_dir: str,

    same_time_dist_eps: float,
    obs_len_eps: float,

    min_gap: float,
    unknown_hand_mode: str,
    print_interaction_stats: bool,

    compress_dense: bool,
    dense_dt: float,
    pos_eps_f0: float,
    compress_mode: str,
    take_name_filter: str = "",
    max_samples: int = 0,
    overwrite: bool = False,
):
    if osp.exists(out_cache) and not overwrite:
        raise FileExistsError(f"Output exists: {out_cache}; use --overwrite explicitly")
    os.makedirs(os.path.dirname(out_cache) or ".", exist_ok=True)
    if viz_first_n:
        raise ValueError("The historical visualization option has no implementation; use --viz_first_n 0")

    name_to_cat_id, cat_id_to_name = load_lvis_name_to_id(lvis_cat_json)

    split_for_desc = 'val' if split == 'test' else split
    with open(os.path.join(annotations_root, f"atomic_descriptions_{split_for_desc}.json")) as f:
        descriptions = json.load(f)['annotations']
    uids_in_desc = set(descriptions.keys()) if isinstance(descriptions, dict) else set(descriptions)

    take2uid = {}
    take2scenario = {}
    with open(takes_json, encoding="utf-8") as f:
        takes_meta = json.load(f)
    for t in takes_meta:
        take2uid[t['take_name']] = t['take_uid']
        take2scenario[t['take_name']] = t.get('parent_task_name', 'Unknown')

    with open(os.path.join(takes_txt_dir, f"{split}_takes.txt")) as f:
        split_takes = [x.strip() for x in f.readlines() if x.strip()]

    all_takes = [
        x.split("_humans_objects_interactions.pkl")[0]
        for x in os.listdir(dataset_root)
        if x.endswith("_humans_objects_interactions.pkl")
    ]

    norm_target = scenarios.strip().lower()
    takes = [
        t for t in all_takes
        if (t in split_takes)
        and (take2uid.get(t) in uids_in_desc)
        and (norm_target == "all" or str(take2scenario.get(t, "")).strip().lower() == norm_target)
    ]
    if take_name_filter:
        takes = [t for t in takes if t == take_name_filter]
    if not takes:
        raise ValueError("No takes matched the supplied inputs, split, and scenario")
    print(f"[INFO] takes after filters: {len(takes)} (split={split}, scenario={scenarios})")

    dataset: List[Dict[str, Any]] = []
    scenario_mapping: Dict[str, List[int]] = {}

    report = {
        "split": split,
        "scenarios": scenarios,
        "out_cache": out_cache,
        "takes_total": len(takes),
        "samples_total": 0,
        "samples_kept": 0,
        "samples_with_interactions": 0,
        "filtered_obs_short": 0,
        "pose_failed": 0,
        "same_time_multi_location_samples": [],
    }

    per_sample_counts: List[int] = []
    per_sample_meta: List[Tuple[str, int, int]] = []

    for take_name in tqdm(takes, desc="Build cache (canonical+norm)"):
        if max_samples > 0 and len(dataset) >= max_samples:
            break
        full = load_full_take_pkl(dataset_root, take_name)
        if full is None:
            continue
        if "human" not in full or "orig_interaction_dataset" not in full:
            continue

        human_full = full["human"]
        frames_sorted = sorted([int(k) for k in human_full.keys()])
        if len(frames_sorted) == 0:
            continue

        df_full = full["orig_interaction_dataset"]
        if not isinstance(df_full, pd.DataFrame) or ("timestamp" not in df_full.columns):
            continue

        _, interaction_data_uncompressed, current_time_dict = select_anticipation_window(
            df_full, future_time=future_time
        )

        for datum_idx in interaction_data_uncompressed:
            if max_samples > 0 and len(dataset) >= max_samples:
                break
            curr_df_uncompressed = interaction_data_uncompressed[datum_idx]

            current_time = float(current_time_dict[datum_idx])
            observation_end_time = max(0.0, current_time - float(anticipation_time))
            observation_start_time = max(0.0, observation_end_time - float(observation_time))

            obs_len = float(observation_end_time - observation_start_time)
            if obs_len + float(obs_len_eps) < float(observation_time):
                report["filtered_obs_short"] += 1
                continue

            pose_out = compute_F0_from_last_second(
                human_full=human_full,
                frames_sorted=frames_sorted,
                obs_end_time_s=observation_end_time,
                fps=full_fps,
                last_sec=last_sec_for_pose_avg,
                min_pose_frames=10,
                win_frames=int(pose_avg_win_frames),
            )
            if pose_out is None:
                report["pose_failed"] += 1
                continue
            R0, t0, pose_dbg = pose_out

            obs_t_w, obs_xyz_w = build_observation_traj(
                human_full=human_full,
                frames_sorted=frames_sorted,
                obs_start_s=observation_start_time,
                obs_end_s=observation_end_time,
                fps=full_fps,
                right_hand_vert_id=right_hand_vert_id,
            )
            if obs_xyz_w.shape[0] == 0:
                report["filtered_obs_short"] += 1
                continue

            # ====== right-hand-only interactions ======
            current_time_normed = norm_ts(current_time_dict[datum_idx])
            interactions = build_future_interactions_right_only(
                df_ev_raw=curr_df_uncompressed,
                human_full=human_full,
                frames_sorted=frames_sorted,
                current_time=current_time_normed,
                future_time=float(future_time),
                fps=float(full_fps),
                right_hand_vert_id=int(right_hand_vert_id),
                max_full_frame_err=float(max_full_frame_err),
                min_gap=float(min_gap),
                unknown_hand_mode=str(unknown_hand_mode),
            )

            if len(interactions) > 0:
                issues = check_same_time_multi_location(interactions, dist_eps=same_time_dist_eps)
                if len(issues) > 0:
                    report["same_time_multi_location_samples"].append({
                        "uid": f"{take_name}_{int(datum_idx)}",
                        "take_name": take_name,
                        "datum_idx": int(datum_idx),
                        "issues": issues,
                    })

            obs_xyz_f0 = world_to_F0(obs_xyz_w, R0, t0)
            obs_xyz_norm = normalize_xyz(obs_xyz_f0, pos_scale_m)
            obs_t_norm = normalize_time_obs(obs_t_w, observation_end_time, observation_time)

            inter_t_w = np.asarray([it["time"] for it in interactions], dtype=np.float32) if interactions else np.zeros((0,), dtype=np.float32)
            inter_xyz_w = np.asarray([it["location"] for it in interactions], dtype=np.float32) if interactions else np.zeros((0,3), dtype=np.float32)

            inter_xyz_f0 = world_to_F0(inter_xyz_w, R0, t0) if inter_xyz_w.shape[0] > 0 else inter_xyz_w
            inter_xyz_norm = normalize_xyz(inter_xyz_f0, pos_scale_m) if inter_xyz_f0.shape[0] > 0 else inter_xyz_f0
            inter_t_norm = normalize_time_future(inter_t_w, current_time_normed, future_time) if inter_t_w.shape[0] > 0 else inter_t_w

            interactions_norm: List[Dict[str, Any]] = []
            for i, it in enumerate(interactions):
                it2 = dict(it)
                it2["time_rel_s"] = float(it["time"] - current_time_normed)
                it2["time_norm"] = float(inter_t_norm[i])
                it2["location_f0_m"] = inter_xyz_f0[i].tolist()
                it2["location_norm"] = inter_xyz_norm[i].tolist()
                interactions_norm.append(it2)

            if compress_dense:
                interactions_norm = compress_dense_repeated_interactions(
                    interactions_norm,
                    dense_dt=dense_dt,
                    pos_eps_f0=pos_eps_f0,
                    use_pos_field="location_f0_m",
                    compress_mode=compress_mode,
                )

            env_objects = []
            if include_env_objects:
                env_objects = extract_env_objects_tokens(
                    full_take=full,
                    R0=R0,
                    t0=t0,
                    pos_scale_m=pos_scale_m,
                    max_objects=max_env_objects,
                    name_to_cat_id=name_to_cat_id,
                    cat_id_to_name=cat_id_to_name,
                    log_count_norm_denom=log_count_norm_denom,
                )

            env_voxels = []
            if include_env_voxels:
                env_voxels = extract_env_voxel_tokens(
                    full_take=full,
                    R0=R0,
                    t0=t0,
                    pos_scale_m=pos_scale_m,
                    max_voxels=max_env_voxels,
                    sample_strategy="random",
                    seed=(hash(take_name) ^ int(datum_idx)) & 0xFFFFFFFF,
                )

            sample = {
                "take_name": take_name,
                "datum_idx": int(datum_idx),

                "observation_start_time_s": float(observation_start_time),
                "observation_end_time_s": float(observation_end_time),
                "current_time_s": float(current_time_normed),
                "anticipation_time_s": float(anticipation_time),
                "observation_time_s": float(observation_time),
                "future_time_s": float(future_time),

                "anticipation_time_norm_obs": float(anticipation_time) / max(float(observation_time), 1e-6),
                "anticipation_time_norm_future": float(anticipation_time) / max(float(future_time), 1e-6),

                "F0_pose_debug": pose_dbg,
                "F0_pos_scale_m": float(pos_scale_m),

                "trajectory": {
                    "time_s_world": obs_t_w.astype(np.float32),
                    "time_norm": obs_t_norm.astype(np.float32),
                    "xyz_world_m": obs_xyz_w.astype(np.float32),
                    "xyz_f0_m": obs_xyz_f0.astype(np.float32),
                    "xyz_norm": obs_xyz_norm.astype(np.float32),
                },

                "interactions": interactions_norm,

                "env_objects": env_objects,
                "env_voxel_tokens": env_voxels,
            }

            dataset.append(sample)

            c = int(len(interactions_norm))
            per_sample_counts.append(c)
            per_sample_meta.append((take_name, int(datum_idx), c))

            DEBUG_PRINT_N = 5
            if len(dataset) <= DEBUG_PRINT_N:
                sid = len(dataset) - 1
                print("\n================ DEBUG SAMPLE ================")
                print(f"[sid={sid}] take={take_name} datum_idx={datum_idx}")
                print("current_time_s:", current_time_normed, "future_time_s:", future_time)
                print("min_gap:", float(min_gap), "unknown_hand_mode:", unknown_hand_mode)
                print("pose_dbg:", pose_dbg)
                print("[interactions] num_interactions:", len(sample["interactions"]))
                for it in sample["interactions"][:5]:
                    print("  - time:", it["time"],
                          "obj:", it.get("object", ""),
                          "time_rel_s:", it.get("time_rel_s"),
                          "loc_norm:", it.get("location_norm"))
                print("==============================================\n")

            scn = take2scenario.get(take_name, "Unknown")
            scenario_mapping.setdefault(scn, []).append(len(dataset) - 1)

            report["samples_total"] += 1
            report["samples_kept"] += 1
            if len(interactions_norm) > 0:
                report["samples_with_interactions"] += 1

        del full, human_full, frames_sorted
        gc.collect()

    cache_obj = {
        "dataset": dataset,
        "scenario_mapping": scenario_mapping,
        "config": {
            "split": split,
            "scenarios": scenarios,

            "observation_time": float(observation_time),
            "anticipation_time": float(anticipation_time),
            "future_time": float(future_time),

            "right_hand_vert_id": int(right_hand_vert_id),
            "full_fps": float(full_fps),
            "max_full_frame_err": float(max_full_frame_err),

            "min_gap": float(min_gap),
            "unknown_hand_mode": str(unknown_hand_mode),

            "canonical_frame": {
                "method": "F0 = avg device pose over most-stable window within last_sec before observation_end_time",
                "last_sec_for_pose_avg": float(last_sec_for_pose_avg),
                "pose_avg_win_frames": int(pose_avg_win_frames),
                "pos_scale_m": float(pos_scale_m),
                "time_norm_obs": "t_norm = (t - obs_end)/observation_time in [-1,0]",
                "time_norm_future": "t_norm = (t - current_time)/future_time in [0,1]",
            },

            "env": {
                "include_env_objects": bool(include_env_objects),
                "max_env_objects": int(max_env_objects),
                "include_env_voxels": bool(include_env_voxels),
                "max_env_voxels": int(max_env_voxels),
            },

            "lvis": {
                "lvis_cat_json": lvis_cat_json,
                "unknown_id": 0,
                "num_categories_with_unknown": int(len(name_to_cat_id) + 1),
            },
            "log_count_norm": {
                "method": "log_count_norm = clip(log1p(count) / denom, 0, 1)",
                "denom": float(log_count_norm_denom),
            },

            "filters": {
                "obs_len_required_s": float(observation_time),
                "obs_len_eps": float(obs_len_eps),
            },

            "compress_dense": {
                "enabled": bool(compress_dense),
                "dense_dt": float(dense_dt),
                "pos_eps_f0": float(pos_eps_f0),
                "compress_mode": str(compress_mode),
            },
        },
        "report": report,
    }

    cache_obj["config"]["release_source"] = {"file": SOURCE_FILE, "sha256": SOURCE_SHA256}
    if not dataset:
        raise ValueError("No source samples were constructed")
    planned_outputs = [out_cache, osp.splitext(out_cache)[0] + "_report.json"]
    if export_first_n_json > 0:
        n = min(export_first_n_json, len(dataset))
        planned_outputs.append(osp.splitext(out_cache)[0] + f"_first{n}_samples.json")
    input_paths = [takes_json, lvis_cat_json,
                   osp.join(takes_txt_dir, f"{split}_takes.txt"),
                   osp.join(annotations_root, f"atomic_descriptions_{split_for_desc}.json")]
    input_paths.extend(osp.join(dataset_root, f"{take}_humans_objects_interactions.pkl")
                       for take in takes)
    preflight_outputs(planned_outputs, overwrite=overwrite, input_paths=input_paths)
    with open(out_cache, "wb" if overwrite else "xb") as f:
        pickle.dump(cache_obj, f)

    print("========================================")
    print(f"[SAVE CACHE] {out_cache}")
    print(f"  samples_kept = {len(dataset)}")

    report_path = osp.splitext(out_cache)[0] + "_report.json"
    with open(report_path, "w" if overwrite else "x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[SAVE REPORT] {report_path}")

    if export_first_n_json > 0:
        n = min(export_first_n_json, len(dataset))
        out_samples = []
        for i in range(n):
            s = dataset[i]
            ss = dict(s)
            ss["trajectory"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in s["trajectory"].items()
            }
            out_samples.append(ss)

        samples_json_path = osp.splitext(out_cache)[0] + f"_first{n}_samples.json"
        with open(samples_json_path, "w" if overwrite else "x", encoding="utf-8") as f:
            json.dump(out_samples, f, ensure_ascii=False, indent=2)
        print(f"[SAVE SAMPLES JSON] {samples_json_path}")

    if print_interaction_stats and len(per_sample_counts) > 0:
        arr = np.asarray(per_sample_counts, dtype=np.int64)
        mean_v = float(arr.mean())
        min_v = int(arr.min())
        max_v = int(arr.max())
        p90 = float(np.quantile(arr, 0.90))
        p99 = float(np.quantile(arr, 0.99))
        max_i = int(np.argmax(arr))
        max_take, max_datum, max_cnt = per_sample_meta[max_i]

        print("================================================================================")
        print("[INTERACTIONS STATS] (RIGHT HAND ONLY + min_gap per object)")
        print("================================================================================")
        print(f"- min_gap: {float(min_gap)} sec")
        print(f"- unknown_hand_mode: {unknown_hand_mode}")
        print(f"- total_samples: {len(arr)}")
        print(f"- mean_interactions: {mean_v:.4f}")
        print(f"- min_interactions:  {min_v}")
        print(f"- p90_interactions:  {p90:.4f}")
        print(f"- p99_interactions:  {p99:.4f}")
        print(f"- max_interactions:  {max_v}  (take={max_take}, datum_idx={max_datum})")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--scenarios", type=str, default="Cooking")
    ap.add_argument("--out_cache", type=str, required=True)

    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--annotations_root", type=str, required=True)
    ap.add_argument("--takes_txt_dir", type=str, required=True)

    ap.add_argument("--observation_time", type=float, default=30.0)
    ap.add_argument("--anticipation_time", type=float, default=5.0)
    ap.add_argument("--future_time", type=float, default=60.0)

    ap.add_argument("--right_hand_vert_id", type=int, default=5777)
    ap.add_argument("--full_fps", type=float, default=30.0)
    ap.add_argument("--max_full_frame_err", type=float, default=0.49)

    ap.add_argument("--last_sec_for_pose_avg", type=float, default=1.0,
                    help="use last 1s (~30 frames) before observation_end_time to compute pose candidates")
    ap.add_argument("--pose_avg_win_frames", type=int, default=10,
                    help="pick the most stable window of this many frames inside last_sec_for_pose_avg (0108 logic)")

    ap.add_argument("--pos_scale_m", type=float, default=5.0)

    ap.add_argument("--include_env_objects", type=int, default=1, choices=[0,1])
    ap.add_argument("--max_env_objects", type=int, default=30)
    ap.add_argument("--include_env_voxels", type=int, default=0, choices=[0,1])
    ap.add_argument("--max_env_voxels", type=int, default=512)

    ap.add_argument("--export_first_n_json", type=int, default=5)
    ap.add_argument("--viz_first_n", type=int, default=0)
    ap.add_argument("--viz_out_dir", type=str, default="./viz_out")

    ap.add_argument("--same_time_dist_eps", type=float, default=1e-3)
    ap.add_argument("--obs_len_eps", type=float, default=1e-3)

    ap.add_argument("--compress_dense", type=int, default=1, choices=[0,1],
                    help="compress dense repeated interactions with same objects and small movement")
    ap.add_argument("--dense_dt", type=float, default=3.0,
                    help="max time gap (sec) to consider interactions in same dense segment")
    ap.add_argument("--pos_eps_f0", type=float, default=0.12,
                    help="max F0 position delta (meters) to consider as same segment")
    ap.add_argument("--compress_mode", type=str, default="first", choices=["first","middle","mean"],
                    help="representative point for a segment")

    ap.add_argument(
        "--lvis_cat_json",
        type=str,
        required=True,
    )
    ap.add_argument(
        "--log_count_norm_denom",
        type=float,
        default=10.0,
    )

    # New (0107 right-hand only + debounce)
    ap.add_argument("--min_gap", type=float, default=0.0,
                    help="debounce threshold (sec) per right-hand object token; 0 disables")

    # ✅ 默认改为 right：未标明左右手 => 当作 right
    ap.add_argument("--unknown_hand_mode", type=str, default="right", choices=["drop", "right", "left"],
                    help="how to handle narrations without explicit hand; default=right (treat unknown as right)")

    ap.add_argument("--print_interaction_stats", type=int, default=1, choices=[0,1])

    ap.add_argument("--takes_json", required=True, help="Ego-Exo take metadata JSON")
    ap.add_argument("--take_name", default="", help="Optional single-take filter")
    ap.add_argument("--max_samples", type=int, default=0, help="Limit kept source samples; 0 means all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for path in [args.dataset_root, args.annotations_root, args.takes_txt_dir, args.takes_json, args.lvis_cat_json]:
        if not osp.exists(path):
            ap.error(f"Input does not exist: {path}")
    if args.pos_scale_m <= 0 or args.full_fps <= 0 or args.pose_avg_win_frames <= 0:
        ap.error("Scale, FPS, and averaging window must be positive")

    build_cache_points(
        split=args.split,
        scenarios=args.scenarios,
        dataset_root=args.dataset_root,
        annotations_root=args.annotations_root,
        takes_txt_dir=args.takes_txt_dir,
        takes_json=args.takes_json,
        take_name_filter=args.take_name,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        out_cache=args.out_cache,
        lvis_cat_json=args.lvis_cat_json,

        observation_time=args.observation_time,
        anticipation_time=args.anticipation_time,
        future_time=args.future_time,

        right_hand_vert_id=args.right_hand_vert_id,
        full_fps=args.full_fps,
        max_full_frame_err=args.max_full_frame_err,

        last_sec_for_pose_avg=args.last_sec_for_pose_avg,
        pose_avg_win_frames=int(args.pose_avg_win_frames),
        pos_scale_m=args.pos_scale_m,
        log_count_norm_denom=args.log_count_norm_denom,

        include_env_objects=bool(args.include_env_objects),
        max_env_objects=args.max_env_objects,
        include_env_voxels=bool(args.include_env_voxels),
        max_env_voxels=args.max_env_voxels,

        export_first_n_json=args.export_first_n_json,
        viz_first_n=args.viz_first_n,
        viz_out_dir=args.viz_out_dir,

        same_time_dist_eps=args.same_time_dist_eps,
        obs_len_eps=args.obs_len_eps,

        min_gap=float(args.min_gap),
        unknown_hand_mode=str(args.unknown_hand_mode),
        print_interaction_stats=bool(args.print_interaction_stats),

        compress_dense=bool(args.compress_dense),
        dense_dt=float(args.dense_dt),
        pos_eps_f0=float(args.pos_eps_f0),
        compress_mode=str(args.compress_mode),
    )


if __name__ == "__main__":
    main()
