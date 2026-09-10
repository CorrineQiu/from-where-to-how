
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Attach Route-6 SMPL states to retained event timestamps in shared F0.
Native pelvis translations and parent-relative body rotations are preserved.
Camera-chain meshes are rigidly mapped to existing scene meshes, never fitted
to future loc targets. Missing events retain their historical validity masks.
This stage requires licensed SMPL assets and existing WHAM/full-take caches.

Source: My_Code_qwen/data_pose_chunk/build_pose_chunks_route6_anchor_fixed_0316.py
Source SHA-256: ad7471ed4ae29bd133273ec7076e69f41c432c26bb27484b9ad3e8e398e10a35
Only trusted local pickle/joblib inputs are supported. Model assets and datasets
are user-supplied and are not distributed by this repository.
"""

SOURCE_FILE = "My_Code_qwen/data_pose_chunk/build_pose_chunks_route6_anchor_fixed_0316.py"
SOURCE_SHA256 = "ad7471ed4ae29bd133273ec7076e69f41c432c26bb27484b9ad3e8e398e10a35"


import os
import os.path as osp
import argparse
import pickle
from glob import glob
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch


# =========================
# IO
# =========================
def load_pkl(path: str):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def same_file(a, b):
    """Resolve path aliases, including existing hard links."""
    if osp.realpath(a) == osp.realpath(b):
        return True
    try:
        return osp.samefile(a, b)
    except OSError:
        return False


def save_pkl(obj: Any, path: str, overwrite: bool = False):
    out_dir = osp.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb" if overwrite else "xb") as f:
        pickle.dump(obj, f)


# =========================
# Generic helpers
# =========================
def _as_float_array(x: Any, size: Optional[int] = None) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if size is not None and arr.size < size:
        return None
    return arr


def _decode_xyz_bytes(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, np.ndarray):
        arr = np.asarray(x).reshape(-1)
        if arr.size >= 3:
            return arr[:3].astype(np.float32)
        return None
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3].astype(np.float32)
        return None
    if isinstance(x, (bytes, bytearray, memoryview)):
        buf = bytes(x)
        for dtype in (np.float32, np.float64):
            try:
                arr = np.frombuffer(buf, dtype=dtype)
                if arr.size >= 3:
                    return arr[:3].astype(np.float32)
            except Exception:
                pass
    return None


def nearest_existing_frame_with_err(frames_sorted: np.ndarray, target_frame: int) -> Tuple[int, float]:
    arr = np.asarray(frames_sorted, dtype=np.int64)
    if arr.size == 0:
        return int(target_frame), float("inf")
    pos = int(np.searchsorted(arr, target_frame))
    if pos <= 0:
        fr = int(arr[0])
        return fr, float(abs(fr - target_frame))
    if pos >= arr.size:
        fr = int(arr[-1])
        return fr, float(abs(fr - target_frame))
    left = int(arr[pos - 1])
    right = int(arr[pos])
    if abs(left - target_frame) <= abs(right - target_frame):
        return left, float(abs(left - target_frame))
    return right, float(abs(right - target_frame))


# =========================
# Rotation utils
# =========================
def axis_angle_to_rotmat_batch(rotvec: np.ndarray) -> np.ndarray:
    v = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    N = v.shape[0]
    theta = np.linalg.norm(v, axis=1)

    eps = 1e-8
    A = np.empty((N,), dtype=np.float64)
    B = np.empty((N,), dtype=np.float64)

    small = theta < eps
    th = theta.copy()

    th_ns = th[~small]
    A[~small] = np.sin(th_ns) / th_ns
    B[~small] = (1.0 - np.cos(th_ns)) / (th_ns * th_ns)

    th_s = th[small]
    th2 = th_s * th_s
    A[small] = 1.0 - th2 / 6.0 + (th2 * th2) / 120.0
    B[small] = 0.5 - th2 / 24.0 + (th2 * th2) / 720.0

    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    K = np.zeros((N, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -vz
    K[:, 0, 2] = vy
    K[:, 1, 0] = vz
    K[:, 1, 2] = -vx
    K[:, 2, 0] = -vy
    K[:, 2, 1] = vx

    K2 = np.matmul(K, K)
    I = np.eye(3, dtype=np.float64)[None, :, :]
    R = I + A[:, None, None] * K + B[:, None, None] * K2
    return R.astype(np.float32)


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    return np.clip(R[..., :, :2].reshape(*R.shape[:-2], 6), -1.0, 1.0).astype(np.float32)


def rot6d_to_rotmat_np(rot6d: np.ndarray) -> np.ndarray:
    x = np.asarray(rot6d, dtype=np.float32).reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]

    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-12)
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=1)
    R = np.stack([b1, b2, b3], axis=-1)
    return R.astype(np.float32)


# =========================
# Rigid / F0 transforms
# =========================
def apply_rigid_points(x: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3,)
    y = (R @ x.T).T + t[None, :]
    return y.astype(np.float32)


def rigid_fit_kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    Y = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    assert X.shape == Y.shape and X.shape[1] == 3

    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    Xc = X - muX
    Yc = Y - muY

    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = muY.reshape(3,) - R @ muX.reshape(3,)
    fit = apply_rigid_points(X, R, t)
    mpjpe_mm = float(np.linalg.norm(fit - Y, axis=1).mean() * 1000.0)
    return R.astype(np.float32), t.astype(np.float32), mpjpe_mm


def world_to_F0_points(pw: np.ndarray, R0: np.ndarray, t0: np.ndarray) -> np.ndarray:
    pw = np.asarray(pw, dtype=np.float64).reshape(-1, 3)
    R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    t0 = np.asarray(t0, dtype=np.float64).reshape(3,)
    pf0 = (R0.T @ (pw - t0[None, :]).T).T
    return pf0.astype(np.float32)


def world_to_F0_rot(Rw: np.ndarray, R0: np.ndarray) -> np.ndarray:
    Rw = np.asarray(Rw, dtype=np.float64).reshape(3, 3)
    R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    return (R0.T @ Rw).astype(np.float32)


def normalize_xyz_clip(p: np.ndarray, pos_scale_m: float) -> np.ndarray:
    s = max(float(pos_scale_m), 1e-6)
    return np.clip(np.asarray(p, dtype=np.float32) / s, -1.0, 1.0).astype(np.float32)


# =========================
# Quaternion averaging for F0
# =========================
def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
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
    return q / (np.linalg.norm(q) + 1e-12)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def average_quats(quats: np.ndarray) -> np.ndarray:
    if quats.shape[0] == 1:
        return quats[0]
    q0 = quats[0]
    aligned = [(-q if np.dot(q0, q) < 0 else q) for q in quats]
    Q = np.stack(aligned, axis=0)
    A = np.zeros((4, 4), dtype=np.float64)
    for q in Q:
        A += np.outer(q, q)
    A /= float(Q.shape[0])
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[0] < 0:
        q_avg = -q_avg
    return q_avg / (np.linalg.norm(q_avg) + 1e-12)


def rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def pick_stable_window(Rs: List[np.ndarray], ts: List[np.ndarray], win: int) -> Tuple[int, int, float]:
    N = len(Rs)
    if N <= win:
        return 0, N - 1, 0.0
    best_cost = 1e18
    best = (0, win - 1)
    for s in range(0, N - win + 1):
        e = s + win - 1
        trans = 0.0
        rot = 0.0
        for k in range(s, e):
            trans += float(np.linalg.norm(ts[k + 1] - ts[k]))
            rot += rot_angle_deg(Rs[k], Rs[k + 1])
        cost = trans + 0.01 * rot
        if cost < best_cost:
            best_cost = cost
            best = (s, e)
    return best[0], best[1], float(best_cost)


# =========================
# SMPL helper
# =========================
class SMPLForwardHelper:
    def __init__(self, model_path: str, gender: str = "neutral", num_body_joints: int = 23, device: str = "cpu"):
        import smplx
        self.device = torch.device(device)
        self.num_body_joints = int(num_body_joints)
        self.smpl = smplx.SMPLLayer(
            model_path=model_path,
            gender=gender,
            num_body_joints=self.num_body_joints,
        ).to(self.device)
        self.smpl.eval()
        for p in self.smpl.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        root_aa: np.ndarray,
        body_aa: np.ndarray,
        betas: np.ndarray,
        transl: np.ndarray,
        batch_size: int = 256,
    ):
        root_aa = np.asarray(root_aa, dtype=np.float32).reshape(-1, 3)
        body_aa = np.asarray(body_aa, dtype=np.float32).reshape(-1, self.num_body_joints * 3)
        betas = np.asarray(betas, dtype=np.float32).reshape(root_aa.shape[0], -1)
        transl = np.asarray(transl, dtype=np.float32).reshape(-1, 3)

        N = root_aa.shape[0]
        root_R = axis_angle_to_rotmat_batch(root_aa).reshape(N, 1, 3, 3).astype(np.float32)
        body_R = axis_angle_to_rotmat_batch(body_aa.reshape(-1, 3)).reshape(
            N, self.num_body_joints, 3, 3
        ).astype(np.float32)

        verts_all = []
        native_joints_all = []
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            out = self.smpl(
                global_orient=torch.from_numpy(root_R[s:e]).to(self.device),
                body_pose=torch.from_numpy(body_R[s:e]).to(self.device),
                betas=torch.from_numpy(betas[s:e]).to(self.device),
                transl=torch.from_numpy(transl[s:e]).to(self.device),
                pose2rot=False,
            )
            verts_all.append(out.vertices.detach().cpu().numpy())
            native_joints_all.append(out.joints.detach().cpu().numpy())
        verts = np.concatenate(verts_all, axis=0).astype(np.float32)
        native_joints = np.concatenate(native_joints_all, axis=0).astype(np.float32)
        return verts, native_joints


# =========================
# Joint regressor
# =========================
def load_joint_regressor(path: str) -> np.ndarray:
    ext = osp.splitext(path)[1].lower()
    if ext == ".npy":
        W = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        W = z[list(z.keys())[0]]
    else:
        obj = load_pkl(path)
        if isinstance(obj, dict):
            for k in ["joint_regressor", "J_regressor", "regressor", "weights"]:
                if k in obj:
                    W = obj[k]
                    break
            else:
                if len(obj) == 1:
                    W = list(obj.values())[0]
                else:
                    raise ValueError(f"Cannot infer joint regressor from keys={list(obj.keys())}")
        else:
            W = obj
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"joint regressor must be 2D, got {W.shape}")
    return W


def regress_joints_from_verts(verts: np.ndarray, joint_regressor: np.ndarray) -> np.ndarray:
    V = np.asarray(verts, dtype=np.float32)
    W = np.asarray(joint_regressor, dtype=np.float32)
    return np.einsum("jv,...vc->...jc", W, V).astype(np.float32)


# =========================
# Paths
# =========================
def resolve_full_take_path(root: str, take: str) -> Optional[str]:
    p = osp.join(root, f"{take}_humans_objects_interactions.pkl")
    if osp.exists(p):
        return p
    matches = glob(osp.join(root, "**", f"{take}_humans_objects_interactions.pkl"), recursive=True)
    return sorted(matches)[0] if matches else None


def resolve_largest_area_path(root: str, take: str) -> Optional[str]:
    cands = [osp.join(root, f"{take}_largest_area.pkl"), osp.join(root, take, "largest_area.pkl")]
    for p in cands:
        if osp.exists(p):
            return p
    matches = glob(osp.join(root, "**", f"{take}_largest_area.pkl"), recursive=True)
    return sorted(matches)[0] if matches else None


def unpack_largest_frame_entry(frame_entry: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if frame_entry is None:
        return None, None
    if isinstance(frame_entry, dict):
        if any(k in frame_entry for k in ["pose", "pose_world", "trans", "verts", "betas"]):
            return "unknown", frame_entry
        if len(frame_entry) == 1:
            cam_name = list(frame_entry.keys())[0]
            return cam_name, list(frame_entry.values())[0]
    return None, None


# =========================
# loc-step hand extraction
# =========================
LOC_DECODE_NONE = 0
LOC_DECODE_F0_M = 1
LOC_DECODE_F0_NORM = 2
LOC_DECODE_WORLD = 3

LOC_HAND_M_KEYS = [
    "location_f0_m",
    "right_hand_loc_f0_m",
    "hand_loc_f0_m",
]
LOC_HAND_NORM_KEYS = [
    "location_norm",              # chunk builder 当前最常见
    "location_f0_norm",
    "right_hand_loc_f0_norm",
    "right_hand_f0_norm",
    "hand_loc_f0_norm",
    "hand_pos_f0_norm",
    "loc_f0_norm",
]
LOC_HAND_WORLD_KEYS = [
    "right_hand_world_loc",
    "right_hand_loc_world",
    "right_hand_world",
]


def extract_loc_right_hand_fields(
    interaction: Optional[Dict[str, Any]],
    R0: np.ndarray,
    t0: np.ndarray,
    pos_scale_m: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, bool]:
    """
    returns:
      loc_f0_m_exact_or_none,
      loc_f0_norm_or_none,
      decode_mode_id,
      norm_saturated
    """
    if not isinstance(interaction, dict):
        return None, None, LOC_DECODE_NONE, False

    # exact F0 meters
    for k in LOC_HAND_M_KEYS:
        if k in interaction and interaction[k] is not None:
            arr = _decode_xyz_bytes(interaction[k])
            if arr is not None:
                norm = normalize_xyz_clip(arr, pos_scale_m)
                sat = bool(np.max(np.abs(norm)) >= 0.999)
                return arr.astype(np.float32), norm.astype(np.float32), LOC_DECODE_F0_M, sat

    # F0 normalized only
    for k in LOC_HAND_NORM_KEYS:
        if k in interaction and interaction[k] is not None:
            arr = _decode_xyz_bytes(interaction[k])
            if arr is not None:
                arr = np.clip(arr.astype(np.float32), -1.0, 1.0)
                sat = bool(np.max(np.abs(arr)) >= 0.999)
                return None, arr, LOC_DECODE_F0_NORM, sat

    # world bytes / arrays
    for k in LOC_HAND_WORLD_KEYS:
        if k in interaction and interaction[k] is not None:
            pw = _decode_xyz_bytes(interaction[k])
            if pw is not None:
                loc_f0_m = world_to_F0_points(pw.reshape(1, 3), R0, t0).reshape(3,)
                loc_f0_norm = normalize_xyz_clip(loc_f0_m, pos_scale_m)
                sat = bool(np.max(np.abs(loc_f0_norm)) >= 0.999)
                return loc_f0_m.astype(np.float32), loc_f0_norm.astype(np.float32), LOC_DECODE_WORLD, sat

    return None, None, LOC_DECODE_NONE, False


# =========================
# time parsing
# =========================
def get_pose_f0_anchor_time(sample: Dict[str, Any]) -> Optional[float]:
    src_meta = sample.get("source_meta", None)
    if isinstance(src_meta, dict):
        v = src_meta.get("source_observation_end_time_s", None)
        if v is not None:
            return float(v)

    v = sample.get("observation_end_time_s", None)
    if v is not None:
        return float(v)

    v = sample.get("observation_end_time", None)
    if v is not None:
        return float(v)
    return None


def get_chunk_observation_end_time(sample: Dict[str, Any]) -> Optional[float]:
    v = sample.get("observation_end_time_s", None)
    if v is not None:
        return float(v)
    v = sample.get("observation_end_time", None)
    if v is not None:
        return float(v)
    return None


def get_ts_from_interaction(it: Any) -> Optional[float]:
    if not isinstance(it, dict):
        return None
    for k in ["time_s", "timestamp_s", "ts_s", "t_s", "time", "timestamp", "ts", "t"]:
        if k in it and it[k] is not None:
            try:
                return float(it[k])
            except Exception:
                pass
    for k in ["timestamp_us", "ts_us", "time_us", "tracking_timestamp_us"]:
        if k in it and it[k] is not None:
            try:
                return float(it[k]) / 1e6
            except Exception:
                pass
    return None


def get_time_rel_s_from_interaction(it: Any, anchor_time: float) -> Optional[float]:
    if not isinstance(it, dict):
        return None
    for k in ["time_rel_s", "time_rel", "dt_s", "delta_t_s"]:
        if k in it and it[k] is not None:
            try:
                return float(it[k])
            except Exception:
                pass
    ts = get_ts_from_interaction(it)
    if ts is not None:
        return float(ts) - float(anchor_time)
    return None


# =========================
# Fail codes
# =========================
FAIL_SUCCESS = 0
FAIL_SKIP_USER_MASK0 = 1
FAIL_NO_TS = 2
FAIL_EXTRACT_NONE = 3
FAIL_MISSING_ROUTE6 = 4


# =========================
# Build per-take cache
# =========================
def build_take_cache_route6(
    full_take: dict,
    largest_area: dict,
    smpl_helper: SMPLForwardHelper,
    joint_regressor: np.ndarray,
    right_hand_vert_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    human = full_take["human"]
    frames = np.array(sorted([int(k) for k in human.keys()]), dtype=np.int32)
    F = frames.shape[0]

    R_dev_world = np.zeros((F, 3, 3), dtype=np.float32)
    t_dev_world = np.zeros((F, 3), dtype=np.float32)
    has_dev = np.zeros((F,), dtype=np.uint8)

    beta_dim = None
    source_cam_name = [""] * F
    has_source = np.zeros((F,), dtype=np.uint8)

    source_root_orient6d_cam = np.zeros((F, 6), dtype=np.float32)
    source_root_trans_cam = np.zeros((F, 3), dtype=np.float32)
    source_body_pose_local6d = np.zeros((F, 23, 6), dtype=np.float32)
    source_body_pose_local_aa = np.zeros((F, 23, 3), dtype=np.float32)
    source_right_hand_cam = np.zeros((F, 3), dtype=np.float32)
    has_betas = np.zeros((F,), dtype=np.uint8)

    cam_to_bench_R = np.zeros((F, 3, 3), dtype=np.float32)
    cam_to_bench_t = np.zeros((F, 3), dtype=np.float32)
    cam_to_bench_vertex_fit_mm = np.full((F,), np.nan, dtype=np.float32)
    source_param_to_stored_verts_mm = np.full((F,), np.nan, dtype=np.float32)

    root_orient6d_bench_world = np.zeros((F, 6), dtype=np.float32)
    root_trans_bench_world = np.zeros((F, 3), dtype=np.float32)
    right_hand_bench_world = np.zeros((F, 3), dtype=np.float32)

    valid_rows = []
    for i, fr in enumerate(frames):
        h = human.get(int(fr), None)
        if h is None:
            continue

        try:
            Rw = np.asarray(h.get("rotation_matrix", None), dtype=np.float32).reshape(3, 3)
            tw = np.asarray(h.get("device_trajectory", None), dtype=np.float32).reshape(-1)[:3]
            if np.isfinite(Rw).all() and np.isfinite(tw).all():
                R_dev_world[i] = Rw
                t_dev_world[i] = tw
                has_dev[i] = 1
        except Exception:
            pass

        cam_name, e = unpack_largest_frame_entry(largest_area.get(int(fr), None))
        if e is None:
            continue

        pose = _as_float_array(e.get("pose", None), size=72)
        trans = _as_float_array(e.get("trans", None), size=3)
        betas = _as_float_array(e.get("betas", None), size=1)
        verts_saved = e.get("verts", None)

        if pose is None or trans is None or betas is None or verts_saved is None:
            continue

        verts_saved = np.asarray(verts_saved, dtype=np.float32)
        if verts_saved.ndim != 2 or verts_saved.shape[1] != 3 or not np.isfinite(verts_saved).all():
            continue

        bench_verts = np.asarray(h.get("verts", None), dtype=np.float32)
        if bench_verts.ndim != 2 or bench_verts.shape[1] != 3 or not np.isfinite(bench_verts).all():
            continue

        if beta_dim is None:
            beta_dim = int(betas.shape[0])
        if int(betas.shape[0]) != int(beta_dim):
            continue

        valid_rows.append({
            "i": i,
            "frame": int(fr),
            "cam_name": str(cam_name),
            "root_aa": pose[:3].astype(np.float32),
            "body_aa": pose[3:72].astype(np.float32),
            "trans": trans[:3].astype(np.float32),
            "betas": betas.astype(np.float32),
            "verts_saved": verts_saved.astype(np.float32),
            "bench_verts": bench_verts.astype(np.float32),
        })

    if len(valid_rows) == 0:
        J = int(joint_regressor.shape[0])
        source_joints_cam = np.zeros((F, J, 3), dtype=np.float32)
        joints_bench_world = np.zeros((F, J, 3), dtype=np.float32)
        source_betas = np.zeros((F, 0), dtype=np.float32)
        return {
            "frames": frames,
            "R_dev_world": R_dev_world,
            "t_dev_world": t_dev_world,
            "has_dev": has_dev,

            "source_cam_name": source_cam_name,
            "has_source": has_source,

            "source_root_orient6d_cam": source_root_orient6d_cam,
            "source_root_trans_cam": source_root_trans_cam,
            "source_body_pose_local6d": source_body_pose_local6d,
            "source_body_pose_local_aa": source_body_pose_local_aa,
            "source_joints_cam": source_joints_cam,
            "source_right_hand_cam": source_right_hand_cam,
            "source_betas": source_betas,
            "has_betas": has_betas,

            "cam_to_bench_R": cam_to_bench_R,
            "cam_to_bench_t": cam_to_bench_t,
            "cam_to_bench_vertex_fit_mm": cam_to_bench_vertex_fit_mm,
            "source_param_to_stored_verts_mm": source_param_to_stored_verts_mm,

            "root_orient6d_bench_world": root_orient6d_bench_world,
            "root_trans_bench_world": root_trans_bench_world,
            "joints_bench_world": joints_bench_world,
            "right_hand_bench_world": right_hand_bench_world,
        }, {}

    root_aa = np.stack([r["root_aa"] for r in valid_rows], axis=0)
    body_aa = np.stack([r["body_aa"] for r in valid_rows], axis=0)
    trans = np.stack([r["trans"] for r in valid_rows], axis=0)
    betas = np.stack([r["betas"] for r in valid_rows], axis=0)

    verts_recon, native_joints_cam = smpl_helper.forward(root_aa, body_aa, betas, trans, batch_size=256)
    eval_joints_cam = regress_joints_from_verts(verts_recon, joint_regressor)
    J = int(eval_joints_cam.shape[1])

    source_joints_cam = np.zeros((F, J, 3), dtype=np.float32)
    joints_bench_world = np.zeros((F, J, 3), dtype=np.float32)
    source_betas = np.zeros((F, beta_dim), dtype=np.float32)

    cam_vocab: Dict[str, int] = {}
    for idx, row in enumerate(valid_rows):
        i = int(row["i"])
        cam_name = row["cam_name"]
        if cam_name not in cam_vocab:
            cam_vocab[cam_name] = len(cam_vocab)

        has_source[i] = 1
        source_cam_name[i] = cam_name
        source_betas[i] = row["betas"]
        has_betas[i] = 1

        R_root_cam = axis_angle_to_rotmat_batch(row["root_aa"].reshape(1, 3))[0]
        source_root_orient6d_cam[i] = rotmat_to_6d(R_root_cam.reshape(1, 3, 3)).reshape(6,)
        source_body_pose_local_aa[i] = row["body_aa"].reshape(23, 3)
        source_body_pose_local6d[i] = rotmat_to_6d(axis_angle_to_rotmat_batch(row["body_aa"].reshape(23, 3)))
        source_joints_cam[i] = eval_joints_cam[idx]
        source_right_hand_cam[i] = row["verts_saved"][right_hand_vert_idx]
        source_root_trans_cam[i] = native_joints_cam[idx, 0, :3]

        source_param_to_stored_verts_mm[i] = float(np.linalg.norm(verts_recon[idx] - row["verts_saved"], axis=1).mean() * 1000.0)

        Rcb, tcb, fit_mm = rigid_fit_kabsch(row["verts_saved"], row["bench_verts"])
        cam_to_bench_R[i] = Rcb
        cam_to_bench_t[i] = tcb
        cam_to_bench_vertex_fit_mm[i] = fit_mm

        root_bench_R = Rcb @ R_root_cam
        root_orient6d_bench_world[i] = rotmat_to_6d(root_bench_R.reshape(1, 3, 3)).reshape(6,)
        root_trans_bench_world[i] = apply_rigid_points(native_joints_cam[idx, 0:1, :3], Rcb, tcb).reshape(3,)
        joints_bench_world[i] = apply_rigid_points(eval_joints_cam[idx], Rcb, tcb).reshape(J, 3)
        right_hand_bench_world[i] = apply_rigid_points(row["verts_saved"][right_hand_vert_idx:right_hand_vert_idx+1], Rcb, tcb).reshape(3,)

    return {
        "frames": frames,
        "R_dev_world": R_dev_world,
        "t_dev_world": t_dev_world,
        "has_dev": has_dev,

        "source_cam_name": source_cam_name,
        "has_source": has_source,

        "source_root_orient6d_cam": source_root_orient6d_cam,
        "source_root_trans_cam": source_root_trans_cam,
        "source_body_pose_local6d": source_body_pose_local6d,
        "source_body_pose_local_aa": source_body_pose_local_aa,
        "source_joints_cam": source_joints_cam,
        "source_right_hand_cam": source_right_hand_cam,
        "source_betas": source_betas,
        "has_betas": has_betas,

        "cam_to_bench_R": cam_to_bench_R,
        "cam_to_bench_t": cam_to_bench_t,
        "cam_to_bench_vertex_fit_mm": cam_to_bench_vertex_fit_mm,
        "source_param_to_stored_verts_mm": source_param_to_stored_verts_mm,

        "root_orient6d_bench_world": root_orient6d_bench_world,
        "root_trans_bench_world": root_trans_bench_world,
        "joints_bench_world": joints_bench_world,
        "right_hand_bench_world": right_hand_bench_world,
    }, cam_vocab


# =========================
# F0 from device pose
# =========================
def compute_F0_Rt_from_take_cache(
    take_cache: Dict[str, Any],
    obs_end_time_s: float,
    fps: float,
    last_sec: float,
    win_frames: int,
    min_pose_frames: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    frames = take_cache["frames"]
    R_all = take_cache["R_dev_world"].astype(np.float64)
    t_all = take_cache["t_dev_world"].astype(np.float64)
    has = take_cache["has_dev"]

    fr_end = int(np.floor(float(obs_end_time_s) * float(fps)))
    n_frames = int(round(float(last_sec) * float(fps)))
    fr_start = fr_end - n_frames + 1

    Rs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    used_frames: List[int] = []

    for fr in range(fr_start, fr_end + 1):
        fr_near, _ = nearest_existing_frame_with_err(frames, fr)
        idx = int(np.searchsorted(frames, fr_near))
        if idx >= frames.size or int(frames[idx]) != int(fr_near):
            continue
        if int(has[idx]) == 0:
            continue
        Rs.append(R_all[idx])
        ts.append(t_all[idx])
        used_frames.append(int(fr_near))

    if len(used_frames) > 1:
        uniq = {}
        for i, fr in enumerate(used_frames):
            uniq.setdefault(fr, i)
        idxs = sorted(list(uniq.values()))
        Rs = [Rs[i] for i in idxs]
        ts = [ts[i] for i in idxs]
        used_frames = [used_frames[i] for i in idxs]

    if len(Rs) == 0:
        return None

    if len(Rs) < int(min_pose_frames):
        fr_near, _ = nearest_existing_frame_with_err(frames, fr_end)
        idx = int(np.searchsorted(frames, fr_near))
        if idx >= frames.size or int(frames[idx]) != int(fr_near) or int(has[idx]) == 0:
            return None
        R0 = R_all[idx]
        t0 = t_all[idx]
        dbg = {
            "method": "fallback_single_frame_device_pose",
            "fr_end": int(fr_end),
            "used_frame": int(fr_near),
            "num_frames": 1,
        }
        return R0.astype(np.float64), t0.astype(np.float64), dbg

    win_eff = int(min(int(win_frames), len(Rs)))
    s, e, cost = pick_stable_window(Rs, ts, win=win_eff)
    Rs_win = Rs[s:e + 1]
    ts_win = ts[s:e + 1]
    used_win = used_frames[s:e + 1]

    t0 = np.mean(np.stack(ts_win, axis=0), axis=0)
    quats = np.stack([rotmat_to_quat(R) for R in Rs_win], axis=0)
    q0 = average_quats(quats)
    R0 = quat_to_rotmat(q0)

    dbg = {
        "method": "stable_window_avg_device_pose",
        "fr_end": int(fr_end),
        "fr_start": int(fr_start),
        "num_frames_available": int(len(Rs)),
        "win_frames": int(win_eff),
        "win_cost": float(cost),
        "used_frames_min": int(np.min(used_frames)),
        "used_frames_max": int(np.max(used_frames)),
        "used_frames_win_min": int(np.min(used_win)),
        "used_frames_win_max": int(np.max(used_win)),
    }
    return R0.astype(np.float64), t0.astype(np.float64), dbg


# =========================
# Extract aligned token
# =========================
def extract_route6_at_time(
    take_cache: Dict[str, Any],
    ts: float,
    fps: float,
    max_err_s: float,
    R0: np.ndarray,
    t0: np.ndarray,
    pos_scale_m: float,
    interaction: Optional[Dict[str, Any]],
    quality_source_tau_mm: float,
    quality_rigid_tau_mm: float,
    quality_hand_tau_mm: float,
    min_quality_weight: float,
    cam_vocab: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    frames = take_cache["frames"]
    target_frame = int(np.floor(float(ts) * float(fps)))
    fr, ferr = nearest_existing_frame_with_err(frames, target_frame)
    ferr_s = float(ferr) / float(fps)
    if ferr_s > float(max_err_s):
        return None

    idx = int(np.searchsorted(frames, fr))
    if idx >= frames.size or int(frames[idx]) != int(fr):
        return None
    if int(take_cache["has_source"][idx]) == 0:
        return None

    out = {
        "frame": int(fr),
        "frame_time_s": float(fr) / float(fps),
        "frame_err_s": float(ferr_s),
    }

    # camera chain
    root6_cam = take_cache["source_root_orient6d_cam"][idx].astype(np.float32)
    root_cam_R = rot6d_to_rotmat_np(root6_cam.reshape(1, 6))[0]

    out["root_orient6d_cam"] = root6_cam
    out["root_trans_cam_m"] = take_cache["source_root_trans_cam"][idx].astype(np.float32)
    out["body_pose_local6d"] = take_cache["source_body_pose_local6d"][idx].astype(np.float32)
    out["body_pose_local_aa"] = take_cache["source_body_pose_local_aa"][idx].astype(np.float32)
    out["joints_cam_m"] = take_cache["source_joints_cam"][idx].astype(np.float32)
    out["right_hand_cam_m"] = take_cache["source_right_hand_cam"][idx].astype(np.float32)
    out["betas"] = take_cache["source_betas"][idx].astype(np.float32)
    out["betas_ok"] = int(take_cache["has_betas"][idx])
    cam_name = take_cache["source_cam_name"][idx]
    out["source_cam_name"] = cam_name
    out["source_cam_idx"] = int(cam_vocab.get(cam_name, -1))

    q_src_mm = float(take_cache["source_param_to_stored_verts_mm"][idx])
    q_rigid_mm = float(take_cache["cam_to_bench_vertex_fit_mm"][idx])
    Rcb = take_cache["cam_to_bench_R"][idx].astype(np.float32)
    tcb = take_cache["cam_to_bench_t"][idx].astype(np.float32)

    out["source_param_to_stored_verts_mm"] = q_src_mm
    out["cam_to_bench_vertex_fit_mm"] = q_rigid_mm
    out["cam_to_bench_R"] = Rcb
    out["cam_to_bench_t"] = tcb

    # benchmark world
    root_bench_R = Rcb @ root_cam_R
    root_bench_t = take_cache["root_trans_bench_world"][idx].astype(np.float32)
    joints_bench = take_cache["joints_bench_world"][idx].astype(np.float32)
    hand_bench = take_cache["right_hand_bench_world"][idx].astype(np.float32)

    out["root_orient6d_bench_world"] = rotmat_to_6d(root_bench_R.reshape(1, 3, 3)).reshape(6,)
    out["root_trans_bench_world"] = root_bench_t
    out["joints_world"] = joints_bench
    out["right_hand_bench_world"] = hand_bench

    # F0
    root_f0_R = world_to_F0_rot(root_bench_R, R0)
    root_trans_f0_m = world_to_F0_points(root_bench_t.reshape(1, 3), R0, t0).reshape(3,)
    joints_f0_m = world_to_F0_points(joints_bench.reshape(-1, 3), R0, t0).reshape(joints_bench.shape[0], 3)
    hand_f0_m = world_to_F0_points(hand_bench.reshape(1, 3), R0, t0).reshape(3,)
    hand_f0_norm = normalize_xyz_clip(hand_f0_m, pos_scale_m)

    out["root_orient6d_f0"] = rotmat_to_6d(root_f0_R.reshape(1, 3, 3)).reshape(6,)
    out["root_trans_f0_m"] = root_trans_f0_m.astype(np.float32)
    out["root_trans_f0_norm"] = normalize_xyz_clip(root_trans_f0_m, pos_scale_m)
    out["joints_f0_m"] = joints_f0_m.astype(np.float32)
    out["joints_f0_norm"] = normalize_xyz_clip(joints_f0_m, pos_scale_m)
    out["right_hand_f0_m"] = hand_f0_m.astype(np.float32)
    out["right_hand_f0_norm"] = hand_f0_norm.astype(np.float32)

    # loc-pose coupling
    loc_f0_m_exact, loc_f0_norm, decode_mode, norm_saturated = extract_loc_right_hand_fields(
        interaction=interaction,
        R0=R0,
        t0=t0,
        pos_scale_m=pos_scale_m,
    )
    out["loc_hand_decode_mode"] = int(decode_mode)
    out["loc_hand_norm_saturated"] = int(norm_saturated)

    if loc_f0_norm is not None:
        out["pose_loc_right_hand_err_norm"] = float(np.linalg.norm(hand_f0_norm - loc_f0_norm))
    else:
        out["pose_loc_right_hand_err_norm"] = np.nan

    if loc_f0_m_exact is not None:
        hand_err_mm = float(np.linalg.norm(hand_f0_m - loc_f0_m_exact.reshape(3,)) * 1000.0)
        out["pose_loc_right_hand_err_mm"] = hand_err_mm
    else:
        hand_err_mm = np.nan
        out["pose_loc_right_hand_err_mm"] = np.nan

    # quality: 只在有 exact meter coupling 时才把 hand term 加进去
    q = np.exp(-q_src_mm / max(quality_source_tau_mm, 1e-6)) * np.exp(-q_rigid_mm / max(quality_rigid_tau_mm, 1e-6))
    if np.isfinite(hand_err_mm):
        q *= np.exp(-hand_err_mm / max(quality_hand_tau_mm, 1e-6))
    q = float(np.clip(q, min_quality_weight, 1.0))
    out["quality_weight"] = q
    return out


# =========================
# Alloc helper
# =========================
def alloc_side(T: int, J: int, beta_dim: int):
    return {
        "mask": np.zeros((T,), dtype=np.uint8),
        "frame": np.full((T,), -1, dtype=np.int32),
        "dt": np.full((T,), np.nan, dtype=np.float32),
        "time_s": np.full((T,), np.nan, dtype=np.float32),
        "time_rel_s": np.full((T,), np.nan, dtype=np.float32),
        "fail_code": np.full((T,), FAIL_SUCCESS, dtype=np.int32),

        "root_orient6d_cam": np.zeros((T, 6), dtype=np.float32),
        "root_trans_cam_m": np.zeros((T, 3), dtype=np.float32),
        "source_native_root_cam_m": np.zeros((T, 3), dtype=np.float32),
        "body_pose_local6d": np.zeros((T, 23, 6), dtype=np.float32),
        "body_pose_local_aa": np.zeros((T, 23, 3), dtype=np.float32),
        "joints_cam_m": np.zeros((T, J, 3), dtype=np.float32),
        "right_hand_cam_m": np.zeros((T, 3), dtype=np.float32),

        "root_orient6d_bench_world": np.zeros((T, 6), dtype=np.float32),
        "root_trans_bench_world": np.zeros((T, 3), dtype=np.float32),
        "joints_world": np.zeros((T, J, 3), dtype=np.float32),
        "right_hand_bench_world": np.zeros((T, 3), dtype=np.float32),

        "root_orient6d_f0": np.zeros((T, 6), dtype=np.float32),
        "root_trans_f0_m": np.zeros((T, 3), dtype=np.float32),
        "root_trans_f0_norm": np.zeros((T, 3), dtype=np.float32),
        "joints_f0_m": np.zeros((T, J, 3), dtype=np.float32),
        "joints_f0_norm": np.zeros((T, J, 3), dtype=np.float32),
        "right_hand_f0_m": np.zeros((T, 3), dtype=np.float32),
        "right_hand_f0_norm": np.zeros((T, 3), dtype=np.float32),

        "betas": np.zeros((T, beta_dim), dtype=np.float32),
        "betas_mask": np.zeros((T,), dtype=np.uint8),

        "source_cam_idx": np.full((T,), -1, dtype=np.int32),
        "loc_hand_decode_mode": np.full((T,), LOC_DECODE_NONE, dtype=np.int32),
        "loc_hand_norm_saturated": np.zeros((T,), dtype=np.uint8),

        "source_param_to_stored_verts_mm": np.full((T,), np.nan, dtype=np.float32),
        "cam_to_bench_vertex_fit_mm": np.full((T,), np.nan, dtype=np.float32),
        "pose_loc_right_hand_err_mm": np.full((T,), np.nan, dtype=np.float32),
        "pose_loc_right_hand_err_norm": np.full((T,), np.nan, dtype=np.float32),
        "quality_weight": np.ones((T,), dtype=np.float32),

        "cam_to_bench_R": np.zeros((T, 3, 3), dtype=np.float32),
        "cam_to_bench_t": np.zeros((T, 3), dtype=np.float32),
    }


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_chunk", required=True)
    ap.add_argument("--out_chunk", required=True)
    ap.add_argument("--full_take_root", required=True)
    ap.add_argument("--largest_area_root", required=True)
    ap.add_argument("--smpl_model_path", required=True)
    ap.add_argument("--joint_regressor_path", required=True)
    ap.add_argument("--gender", type=str, default="neutral")
    ap.add_argument("--num_body_joints", type=int, default=23)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max_err_s", type=float, default=0.49)
    ap.add_argument("--last_sec", type=float, default=1.0)
    ap.add_argument("--win_frames", type=int, default=10)
    ap.add_argument("--min_pose_frames", type=int, default=10)
    ap.add_argument("--pos_scale_m", type=float, default=5.0)
    ap.add_argument("--take_name", type=str, default="")
    ap.add_argument("--max_chunks", type=int, default=0)
    ap.add_argument("--debug_first_n", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--quality_source_tau_mm", type=float, default=10.0)
    ap.add_argument("--quality_rigid_tau_mm", type=float, default=5.0)
    ap.add_argument("--quality_hand_tau_mm", type=float, default=120.0)
    ap.add_argument("--min_quality_weight", type=float, default=0.05)
    ap.add_argument("--right_hand_vert_idx", type=int, default=5777)
    ap.add_argument("--cpu_threads", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for path in [args.in_chunk, args.full_take_root, args.largest_area_root,
                 args.smpl_model_path, args.joint_regressor_path]:
        if not osp.exists(path):
            ap.error(f"Input does not exist: {path}")
    output_paths = [args.out_chunk, osp.splitext(args.out_chunk)[0] + "_pose_fail_report.pkl"]
    for path in output_paths:
        if any(same_file(path, source) for source in [args.in_chunk, args.joint_regressor_path,
                                                     args.smpl_model_path]):
            ap.error("All output artifacts must differ from input files")
        if osp.exists(path) and not args.overwrite:
            ap.error(f"Output exists: {path}; use --overwrite explicitly")
    if args.cpu_threads <= 0 or args.fps <= 0 or args.pos_scale_m <= 0:
        ap.error("CPU threads, FPS, and position scale must be positive")
    torch.set_num_threads(args.cpu_threads)

    joint_regressor = load_joint_regressor(args.joint_regressor_path)
    smpl_helper = SMPLForwardHelper(
        model_path=args.smpl_model_path,
        gender=args.gender,
        num_body_joints=args.num_body_joints,
        device=args.device,
    )

    obj = load_pkl(args.in_chunk)
    ds = obj.get("dataset", [])
    if not isinstance(ds, list):
        raise ValueError("in_chunk['dataset'] must be a list")

    if args.take_name:
        ds = [s for s in ds if s.get("take_name", "") == args.take_name]
        print(f"[FILTER] take_name={args.take_name} -> chunks={len(ds)}")
    if args.max_chunks and args.max_chunks > 0:
        ds = ds[:int(args.max_chunks)]
        print(f"[LIMIT] max_chunks={args.max_chunks} -> chunks={len(ds)}")

    take2idxs: Dict[str, List[int]] = {}
    for i, s in enumerate(ds):
        take = s.get("take_name", None)
        if isinstance(take, str) and take:
            take2idxs.setdefault(take, []).append(i)

    print(f"[LOAD] chunks={len(ds)} takes={len(take2idxs)}")
    print(f"[ROOT] full_take_root={args.full_take_root}")
    print(f"[ROOT] largest_area_root={args.largest_area_root}")

    f0_cache: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    fail_ctr = Counter()
    bad_samples: List[Dict[str, Any]] = []
    processed = 0

    for take, idxs in take2idxs.items():
        full_take_path = resolve_full_take_path(args.full_take_root, take)
        largest_path = resolve_largest_area_path(args.largest_area_root, take)
        if full_take_path is None:
            print(f"[WARN] missing full_take for take={take}")
            fail_ctr["missing_full_take"] += 1
            continue
        if largest_path is None:
            print(f"[WARN] missing largest_area for take={take}")
            fail_ctr["missing_largest_area"] += 1
            continue

        for path in output_paths:
            if any(same_file(path, source) for source in [full_take_path, largest_path]):
                ap.error("Output would replace an upstream input cache")

        full = load_pkl(full_take_path)
        largest = load_pkl(largest_path)
        if "human" not in full:
            print(f"[WARN] invalid full_take (no human): {full_take_path}")
            continue

        take_cache, cam_vocab = build_take_cache_route6(
            full_take=full,
            largest_area=largest,
            smpl_helper=smpl_helper,
            joint_regressor=joint_regressor,
            right_hand_vert_idx=int(args.right_hand_vert_idx),
        )

        J = int(take_cache["joints_bench_world"].shape[1]) if take_cache["joints_bench_world"].size > 0 else int(joint_regressor.shape[0])
        BETA_DIM = int(take_cache["source_betas"].shape[1]) if take_cache["source_betas"].ndim == 2 else 0
        print(f"[TAKE] {take} frames={take_cache['frames'].shape[0]} chunks={len(idxs)} joints={J} betas={BETA_DIM} cams={cam_vocab}")

        for ii in idxs:
            s = ds[ii]

            pose_anchor_time = get_pose_f0_anchor_time(s)
            chunk_obs_end_time = get_chunk_observation_end_time(s)
            if pose_anchor_time is None:
                fail_ctr["sample_no_anchor_time"] += 1
                continue

            key = (take, float(np.round(pose_anchor_time, 6)))
            if key in f0_cache:
                R0, t0, dbg = f0_cache[key]
            else:
                Rt = compute_F0_Rt_from_take_cache(
                    take_cache=take_cache,
                    obs_end_time_s=float(pose_anchor_time),
                    fps=float(args.fps),
                    last_sec=float(args.last_sec),
                    win_frames=int(args.win_frames),
                    min_pose_frames=int(args.min_pose_frames),
                )
                if Rt is None:
                    fail_ctr["sample_f0_compute_failed"] += 1
                    continue
                R0, t0, dbg = Rt
                f0_cache[key] = (R0, t0, dbg)

            hist = s.get("history_interactions", [])
            tgt = s.get("target_interactions", [])
            H = len(hist) if isinstance(hist, list) else 0
            K = len(tgt) if isinstance(tgt, list) else 0
            user_tgt_mask = s.get("target_mask", None)

            Hx = alloc_side(H, J, BETA_DIM)
            Tx = alloc_side(K, J, BETA_DIM)

            sample_fail = {
                "take_name": s.get("take_name", None),
                "id": s.get("id", None),
                "datum_idx": s.get("datum_idx", None),
                "chunk_idx": s.get("chunk_idx", None),
                "H": int(H),
                "K": int(K),
                "hist_valid": 0,
                "tgt_valid": 0,
                "pose_anchor_time_s": float(pose_anchor_time),
                "chunk_obs_end_time_s": (float(chunk_obs_end_time) if chunk_obs_end_time is not None else np.nan),
                "hist_reasons": Counter(),
                "tgt_reasons": Counter(),
            }

            def process_token(side: str, j: int, interaction: Dict[str, Any], store: Dict[str, Any], check_user_mask: bool):
                if check_user_mask and isinstance(user_tgt_mask, list) and j < len(user_tgt_mask) and int(user_tgt_mask[j]) == 0:
                    fail_ctr[f"{side}_skip_user_mask0"] += 1
                    sample_fail[f"{side}_reasons"]["skip_user_mask0"] += 1
                    store["fail_code"][j] = FAIL_SKIP_USER_MASK0
                    return

                ts = get_ts_from_interaction(interaction)
                tr = get_time_rel_s_from_interaction(interaction, pose_anchor_time)
                if ts is not None:
                    store["time_s"][j] = float(ts)
                if tr is not None:
                    store["time_rel_s"][j] = float(tr)
                if ts is None:
                    fail_ctr[f"{side}_no_ts"] += 1
                    sample_fail[f"{side}_reasons"]["no_ts"] += 1
                    store["fail_code"][j] = FAIL_NO_TS
                    return

                out = extract_route6_at_time(
                    take_cache=take_cache,
                    ts=float(ts),
                    fps=float(args.fps),
                    max_err_s=float(args.max_err_s),
                    R0=R0,
                    t0=t0,
                    pos_scale_m=float(args.pos_scale_m),
                    interaction=interaction,
                    quality_source_tau_mm=float(args.quality_source_tau_mm),
                    quality_rigid_tau_mm=float(args.quality_rigid_tau_mm),
                    quality_hand_tau_mm=float(args.quality_hand_tau_mm),
                    min_quality_weight=float(args.min_quality_weight),
                    cam_vocab=cam_vocab,
                )
                if out is None:
                    fail_ctr[f"{side}_extract_failed"] += 1
                    sample_fail[f"{side}_reasons"]["extract_failed"] += 1
                    store["fail_code"][j] = FAIL_EXTRACT_NONE
                    return

                store["root_orient6d_cam"][j] = out["root_orient6d_cam"]
                store["root_trans_cam_m"][j] = out["root_trans_cam_m"]
                store["source_native_root_cam_m"][j] = out["root_trans_cam_m"]
                store["body_pose_local6d"][j] = out["body_pose_local6d"]
                store["body_pose_local_aa"][j] = out["body_pose_local_aa"]
                store["joints_cam_m"][j] = out["joints_cam_m"]
                store["right_hand_cam_m"][j] = out["right_hand_cam_m"]

                store["root_orient6d_bench_world"][j] = out["root_orient6d_bench_world"]
                store["root_trans_bench_world"][j] = out["root_trans_bench_world"]
                store["joints_world"][j] = out["joints_world"]
                store["right_hand_bench_world"][j] = out["right_hand_bench_world"]

                store["root_orient6d_f0"][j] = out["root_orient6d_f0"]
                store["root_trans_f0_m"][j] = out["root_trans_f0_m"]
                store["root_trans_f0_norm"][j] = out["root_trans_f0_norm"]
                store["joints_f0_m"][j] = out["joints_f0_m"]
                store["joints_f0_norm"][j] = out["joints_f0_norm"]
                store["right_hand_f0_m"][j] = out["right_hand_f0_m"]
                store["right_hand_f0_norm"][j] = out["right_hand_f0_norm"]

                if BETA_DIM > 0 and int(out["betas_ok"]) == 1:
                    store["betas"][j] = out["betas"]
                    store["betas_mask"][j] = 1
                store["source_cam_idx"][j] = int(out["source_cam_idx"])
                store["loc_hand_decode_mode"][j] = int(out["loc_hand_decode_mode"])
                store["loc_hand_norm_saturated"][j] = int(out["loc_hand_norm_saturated"])
                store["source_param_to_stored_verts_mm"][j] = float(out["source_param_to_stored_verts_mm"])
                store["cam_to_bench_vertex_fit_mm"][j] = float(out["cam_to_bench_vertex_fit_mm"])
                store["pose_loc_right_hand_err_mm"][j] = float(out["pose_loc_right_hand_err_mm"])
                store["pose_loc_right_hand_err_norm"][j] = float(out["pose_loc_right_hand_err_norm"])
                store["quality_weight"][j] = float(out["quality_weight"])
                store["cam_to_bench_R"][j] = out["cam_to_bench_R"]
                store["cam_to_bench_t"][j] = out["cam_to_bench_t"]

                store["mask"][j] = 1
                store["frame"][j] = out["frame"]
                store["dt"][j] = float(out["frame_time_s"] - float(ts))
                store["fail_code"][j] = FAIL_SUCCESS
                sample_fail[f"{side}_valid"] += 1

            for j in range(H):
                process_token("hist", j, hist[j], Hx, check_user_mask=False)
            for j in range(K):
                process_token("tgt", j, tgt[j], Tx, check_user_mask=True)

            hist_pose6d_world = np.concatenate([Hx["root_orient6d_bench_world"][:, None, :], Hx["body_pose_local6d"]], axis=1)
            hist_pose6d_f0 = np.concatenate([Hx["root_orient6d_f0"][:, None, :], Hx["body_pose_local6d"]], axis=1)
            tgt_pose6d_world = np.concatenate([Tx["root_orient6d_bench_world"][:, None, :], Tx["body_pose_local6d"]], axis=1)
            tgt_pose6d_f0 = np.concatenate([Tx["root_orient6d_f0"][:, None, :], Tx["body_pose_local6d"]], axis=1)

            s["pose_f0_anchor_time_s"] = float(pose_anchor_time)
            s["chunk_observation_end_time_s"] = (float(chunk_obs_end_time) if chunk_obs_end_time is not None else np.nan)
            s["pose_f0_from_source_loc"] = True

            s["f0_anchor_time_s_used_for_pose"] = float(pose_anchor_time)
            s["F0_pose_debug_for_pose"] = dbg
            s["F0_R_world"] = np.asarray(R0, dtype=np.float32)
            s["F0_t_world"] = np.asarray(t0, dtype=np.float32)
            s["pose_source_type"] = "route6_largest_area_camera_chain_plus_full_take_plus_device_f0"
            s["root_is_native_pelvis"] = True

            s["history_time_s"] = Hx["time_s"]
            s["history_time_rel_s"] = Hx["time_rel_s"]
            s["target_time_s"] = Tx["time_s"]
            s["target_time_rel_s"] = Tx["time_rel_s"]

            s["history_pose_mask"] = Hx["mask"]
            s["history_pose_frame"] = Hx["frame"]
            s["history_pose_time_delta_s"] = Hx["dt"]
            s["history_pose_fail_code"] = Hx["fail_code"]
            s["history_root_orient6d_cam"] = Hx["root_orient6d_cam"]
            s["history_root_trans_cam_m"] = Hx["root_trans_cam_m"]
            s["history_source_native_root_cam_m"] = Hx["source_native_root_cam_m"]
            s["history_body_pose_local6d"] = Hx["body_pose_local6d"]
            s["history_body_pose_local_aa"] = Hx["body_pose_local_aa"]
            s["history_joints_cam_m"] = Hx["joints_cam_m"]
            s["history_right_hand_cam_m"] = Hx["right_hand_cam_m"]
            s["history_root_orient6d_bench_world"] = Hx["root_orient6d_bench_world"]
            s["history_root_trans_bench_world"] = Hx["root_trans_bench_world"]
            s["history_joints_world"] = Hx["joints_world"]
            s["history_right_hand_bench_world"] = Hx["right_hand_bench_world"]
            s["history_root_orient6d_f0"] = Hx["root_orient6d_f0"]
            s["history_root_trans_f0_m"] = Hx["root_trans_f0_m"]
            s["history_root_trans_f0_norm"] = Hx["root_trans_f0_norm"]
            s["history_joints_f0_m"] = Hx["joints_f0_m"]
            s["history_joints_f0_norm"] = Hx["joints_f0_norm"]
            s["history_right_hand_f0_m"] = Hx["right_hand_f0_m"]
            s["history_right_hand_f0_norm"] = Hx["right_hand_f0_norm"]
            s["history_betas"] = Hx["betas"]
            s["history_betas_mask"] = Hx["betas_mask"]
            s["history_pose6d_world"] = hist_pose6d_world
            s["history_pose6d_f0"] = hist_pose6d_f0

            s["target_pose_mask"] = Tx["mask"]
            s["target_pose_frame"] = Tx["frame"]
            s["target_pose_time_delta_s"] = Tx["dt"]
            s["target_pose_fail_code"] = Tx["fail_code"]
            s["target_root_orient6d_cam"] = Tx["root_orient6d_cam"]
            s["target_root_trans_cam_m"] = Tx["root_trans_cam_m"]
            s["target_source_native_root_cam_m"] = Tx["source_native_root_cam_m"]
            s["target_body_pose_local6d"] = Tx["body_pose_local6d"]
            s["target_body_pose_local_aa"] = Tx["body_pose_local_aa"]
            s["target_joints_cam_m"] = Tx["joints_cam_m"]
            s["target_right_hand_cam_m"] = Tx["right_hand_cam_m"]
            s["target_root_orient6d_bench_world"] = Tx["root_orient6d_bench_world"]
            s["target_root_trans_bench_world"] = Tx["root_trans_bench_world"]
            s["target_joints_world"] = Tx["joints_world"]
            s["target_right_hand_bench_world"] = Tx["right_hand_bench_world"]
            s["target_root_orient6d_f0"] = Tx["root_orient6d_f0"]
            s["target_root_trans_f0_m"] = Tx["root_trans_f0_m"]
            s["target_root_trans_f0_norm"] = Tx["root_trans_f0_norm"]
            s["target_joints_f0_m"] = Tx["joints_f0_m"]
            s["target_joints_f0_norm"] = Tx["joints_f0_norm"]
            s["target_right_hand_f0_m"] = Tx["right_hand_f0_m"]
            s["target_right_hand_f0_norm"] = Tx["right_hand_f0_norm"]
            s["target_betas"] = Tx["betas"]
            s["target_betas_mask"] = Tx["betas_mask"]
            s["target_pose6d_world"] = tgt_pose6d_world
            s["target_pose6d_f0"] = tgt_pose6d_f0

            s["history_source_cam_idx"] = Hx["source_cam_idx"]
            s["target_source_cam_idx"] = Tx["source_cam_idx"]
            s["history_loc_hand_decode_mode"] = Hx["loc_hand_decode_mode"]
            s["target_loc_hand_decode_mode"] = Tx["loc_hand_decode_mode"]
            s["history_loc_hand_norm_saturated"] = Hx["loc_hand_norm_saturated"]
            s["target_loc_hand_norm_saturated"] = Tx["loc_hand_norm_saturated"]

            s["history_source_param_to_stored_verts_mm"] = Hx["source_param_to_stored_verts_mm"]
            s["target_source_param_to_stored_verts_mm"] = Tx["source_param_to_stored_verts_mm"]
            s["history_cam_to_bench_vertex_fit_mm"] = Hx["cam_to_bench_vertex_fit_mm"]
            s["target_cam_to_bench_vertex_fit_mm"] = Tx["cam_to_bench_vertex_fit_mm"]
            s["history_pose_loc_right_hand_err_mm"] = Hx["pose_loc_right_hand_err_mm"]
            s["target_pose_loc_right_hand_err_mm"] = Tx["pose_loc_right_hand_err_mm"]
            s["history_pose_loc_right_hand_err_norm"] = Hx["pose_loc_right_hand_err_norm"]
            s["target_pose_loc_right_hand_err_norm"] = Tx["pose_loc_right_hand_err_norm"]
            s["history_quality_weight"] = Hx["quality_weight"]
            s["target_quality_weight"] = Tx["quality_weight"]
            s["history_cam_to_bench_R"] = Hx["cam_to_bench_R"]
            s["target_cam_to_bench_R"] = Tx["cam_to_bench_R"]
            s["history_cam_to_bench_t"] = Hx["cam_to_bench_t"]
            s["target_cam_to_bench_t"] = Tx["cam_to_bench_t"]

            hist_valid = Hx["mask"].astype(bool)
            tgt_valid = Tx["mask"].astype(bool)
            s["history_quality_weight_mean"] = float(np.nanmean(Hx["quality_weight"][hist_valid])) if np.any(hist_valid) else np.nan
            s["target_quality_weight_mean"] = float(np.nanmean(Tx["quality_weight"][tgt_valid])) if np.any(tgt_valid) else np.nan
            s["target_pose_loc_right_hand_err_mm_mean"] = float(np.nanmean(Tx["pose_loc_right_hand_err_mm"][tgt_valid])) if np.any(tgt_valid) else np.nan
            s["target_pose_loc_right_hand_err_norm_mean"] = float(np.nanmean(Tx["pose_loc_right_hand_err_norm"][tgt_valid])) if np.any(tgt_valid) else np.nan

            h_cov = (sample_fail["hist_valid"] / sample_fail["H"]) if sample_fail["H"] > 0 else 1.0
            t_cov = (sample_fail["tgt_valid"] / sample_fail["K"]) if sample_fail["K"] > 0 else 1.0
            sample_fail["hist_cov"] = float(h_cov)
            sample_fail["tgt_cov"] = float(t_cov)
            effective_K = sample_fail["K"]
            if isinstance(user_tgt_mask, list):
                effective_K = int(sum(int(x) != 0 for x in user_tgt_mask[:sample_fail["K"]]))
            sample_fail["effective_K"] = int(effective_K)
            sample_fail["tgt_cov_effective"] = float(sample_fail["tgt_valid"] / effective_K) if effective_K > 0 else 1.0
            is_bad = (sample_fail["H"] > 0 and sample_fail["hist_valid"] < sample_fail["H"]) or (effective_K > 0 and sample_fail["tgt_valid"] < effective_K)
            if is_bad:
                fail_ctr["bad_samples"] += 1
                bad_samples.append({
                    **sample_fail,
                    "hist_reasons": dict(sample_fail["hist_reasons"]),
                    "tgt_reasons": dict(sample_fail["tgt_reasons"]),
                })

            if processed < int(args.debug_first_n):
                processed += 1
                print("\n================ DEBUG CHUNK ================")
                print(f"[{processed}] take={take} H={H} valid={int(Hx['mask'].sum())} K={K} valid={int(Tx['mask'].sum())}")
                print(f"pose_anchor_time_s={pose_anchor_time:.6f} chunk_obs_end_time_s={chunk_obs_end_time}")
                print("F0 dbg:", dbg)
                if np.any(Tx["mask"]):
                    qv = Tx["quality_weight"][Tx["mask"].astype(bool)]
                    print(f"target_quality mean={float(np.mean(qv)):.4f} min={float(np.min(qv)):.4f} max={float(np.max(qv)):.4f}")
                print("============================================\n")

    if not any(np.any(s.get("target_pose_mask", [])) or
               np.any(s.get("history_pose_mask", [])) for s in ds):
        raise RuntimeError("No valid pose events were attached; check input caches and timestamps")
    out_obj = dict(obj)
    out_obj["dataset"] = ds
    out_obj.setdefault("config", {})
    out_obj["config"]["pose_align_note"] = (
        "Route-6 solid builder (anchor-fixed). "
        "Pose source is current largest_area.pkl camera-chain pose. "
        "A per-frame rigid transform maps source camera-chain geometry to full_take benchmark-world mesh. "
        "Then F0 is computed with the SAME anchor priority as loc: source_meta.source_observation_end_time_s first. "
        "Core outputs are root_orient6d_f0, root_trans_f0_[m/norm], joints_f0_[m/norm], body_pose_local6d. "
        "loc-coupling supports location_norm / location_f0_m / bytes world hand fields."
    )
    out_obj["config"]["pose_params"] = {
        "fps": float(args.fps),
        "max_err_s": float(args.max_err_s),
        "last_sec": float(args.last_sec),
        "win_frames": int(args.win_frames),
        "min_pose_frames": int(args.min_pose_frames),
        "pos_scale_m": float(args.pos_scale_m),
        "full_take_root": str(args.full_take_root),
        "largest_area_root": str(args.largest_area_root),
        "smpl_model_path": str(args.smpl_model_path),
        "joint_regressor_path": str(args.joint_regressor_path),
        "right_hand_vert_idx": int(args.right_hand_vert_idx),
        "quality_source_tau_mm": float(args.quality_source_tau_mm),
        "quality_rigid_tau_mm": float(args.quality_rigid_tau_mm),
        "quality_hand_tau_mm": float(args.quality_hand_tau_mm),
        "min_quality_weight": float(args.min_quality_weight),
        "root_is_native_pelvis": True,
        "pose_f0_anchor_priority": [
            "source_meta.source_observation_end_time_s",
            "observation_end_time_s",
            "observation_end_time",
        ],
        "loc_hand_decode_mode_vocab": {
            str(LOC_DECODE_NONE): "none",
            str(LOC_DECODE_F0_M): "direct_f0_m",
            str(LOC_DECODE_F0_NORM): "direct_f0_norm",
            str(LOC_DECODE_WORLD): "world_bytes_or_world_array",
        },
    }

    out_obj["config"]["release_source"] = {"file": SOURCE_FILE, "sha256": SOURCE_SHA256}
    save_pkl(out_obj, args.out_chunk, overwrite=args.overwrite)
    print(f"[SAVE] {args.out_chunk}")

    report_path = osp.splitext(args.out_chunk)[0] + "_pose_fail_report.pkl"

    def _summ_stats(arr: np.ndarray) -> Dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"count": 0}
        return {
            "count": float(a.size),
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    src_fit = []
    rigid_fit = []
    hand_err_mm = []
    hand_err_norm = []
    q_weight = []
    for s in ds:
        for key in ["history_source_param_to_stored_verts_mm", "target_source_param_to_stored_verts_mm"]:
            if key in s:
                src_fit.append(np.asarray(s[key], dtype=np.float32))
        for key in ["history_cam_to_bench_vertex_fit_mm", "target_cam_to_bench_vertex_fit_mm"]:
            if key in s:
                rigid_fit.append(np.asarray(s[key], dtype=np.float32))
        for key in ["history_pose_loc_right_hand_err_mm", "target_pose_loc_right_hand_err_mm"]:
            if key in s:
                hand_err_mm.append(np.asarray(s[key], dtype=np.float32))
        for key in ["history_pose_loc_right_hand_err_norm", "target_pose_loc_right_hand_err_norm"]:
            if key in s:
                hand_err_norm.append(np.asarray(s[key], dtype=np.float32))
        for key in ["history_quality_weight", "target_quality_weight"]:
            if key in s:
                q_weight.append(np.asarray(s[key], dtype=np.float32))

    src_fit = np.concatenate(src_fit, axis=0) if len(src_fit) > 0 else np.array([], dtype=np.float32)
    rigid_fit = np.concatenate(rigid_fit, axis=0) if len(rigid_fit) > 0 else np.array([], dtype=np.float32)
    hand_err_mm = np.concatenate(hand_err_mm, axis=0) if len(hand_err_mm) > 0 else np.array([], dtype=np.float32)
    hand_err_norm = np.concatenate(hand_err_norm, axis=0) if len(hand_err_norm) > 0 else np.array([], dtype=np.float32)
    q_weight = np.concatenate(q_weight, axis=0) if len(q_weight) > 0 else np.array([], dtype=np.float32)

    report = {
        "fail_counter": dict(fail_ctr),
        "bad_samples": bad_samples,
        "n_samples_processed": int(len(ds)),
        "bad_samples_ratio": float(fail_ctr.get("bad_samples", 0) / max(len(ds), 1)),
        "source_param_to_stored_verts_mm_stats": _summ_stats(src_fit),
        "cam_to_bench_vertex_fit_mm_stats": _summ_stats(rigid_fit),
        "pose_loc_right_hand_err_mm_stats": _summ_stats(hand_err_mm),
        "pose_loc_right_hand_err_norm_stats": _summ_stats(hand_err_norm),
        "quality_weight_stats": _summ_stats(q_weight),
        "root_is_native_pelvis": True,
        "loc_hand_decode_mode_vocab": {
            str(LOC_DECODE_NONE): "none",
            str(LOC_DECODE_F0_M): "direct_f0_m",
            str(LOC_DECODE_F0_NORM): "direct_f0_norm",
            str(LOC_DECODE_WORLD): "world_bytes_or_world_array",
        },
        "fail_code_meaning": {
            "0": "success",
            "1": "skip_user_mask0",
            "2": "no_ts",
            "3": "extract_failed",
            "4": "missing_route6_core",
        },
    }
    save_pkl(report, report_path, overwrite=args.overwrite)
    print(f"[REPORT] {report_path}")
    print(f"[REPORT] bad_samples_ratio={report['bad_samples_ratio']:.6f}")
    print(f"[REPORT] source_param_to_stored_verts_mm_stats={report['source_param_to_stored_verts_mm_stats']}")
    print(f"[REPORT] cam_to_bench_vertex_fit_mm_stats={report['cam_to_bench_vertex_fit_mm_stats']}")
    print(f"[REPORT] pose_loc_right_hand_err_mm_stats={report['pose_loc_right_hand_err_mm_stats']}")
    print(f"[REPORT] pose_loc_right_hand_err_norm_stats={report['pose_loc_right_hand_err_norm_stats']}")
    print(f"[REPORT] quality_weight_stats={report['quality_weight_stats']}")
    print("[REPORT] top fail reasons:", fail_ctr.most_common(20))


if __name__ == "__main__":
    main()
