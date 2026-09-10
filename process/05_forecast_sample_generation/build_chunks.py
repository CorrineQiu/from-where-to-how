#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build event-count chunks from shared-F0 location sequences.
Cooking/Health/Bike use 10/5/4 events. History uses its own explicit compression
parameters. Empty future sequences yield no chunks, matching nozero filtering.
Single-process output is written to --out_cache; sharded output retains the
historical shard order when merged. Geometry and time arithmetic are unchanged.

Source: My_Code_qwen/dataset_continuous/build_cooking_chunk_10_new.py
Source SHA-256: e845b54d18f562f7b98bc414b8a26a04b318daa22fae2a0c45ffdaa9cc9b2027
Only trusted local pickle/joblib inputs are supported. Model assets and datasets
are user-supplied and are not distributed by this repository.
"""

SOURCE_FILE = "My_Code_qwen/dataset_continuous/build_cooking_chunk_10_new.py"
SOURCE_SHA256 = "e845b54d18f562f7b98bc414b8a26a04b318daa22fae2a0c45ffdaa9cc9b2027"


import os
import os.path as osp
import pickle
import argparse
import math
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# IO
# -------------------------
def load_cache(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "dataset" not in obj:
        raise ValueError(f"Bad cache format: {pkl_path}")
    return obj


def same_file(a, b):
    """Resolve path aliases, including existing hard links."""
    if osp.realpath(a) == osp.realpath(b):
        return True
    try:
        return osp.samefile(a, b)
    except OSError:
        return False


def save_cache(obj: Dict[str, Any], out_path: str, overwrite: bool = False) -> None:
    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb" if overwrite else "xb") as f:
        pickle.dump(obj, f)


# -------------------------
# Utils
# -------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_list(x: Any) -> List[Any]:
    """np.ndarray / list / tuple -> list（不引入 numpy 依赖）"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    # numpy ndarray 有 tolist
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return []
    return []


def _sort_interactions_by_time(interactions: Any) -> List[Dict[str, Any]]:
    xs: List[Dict[str, Any]] = []
    if isinstance(interactions, list):
        for it in interactions:
            if isinstance(it, dict):
                xs.append(it)
    xs.sort(key=lambda d: _safe_float(d.get("time", 0.0), 0.0))
    return xs


def _l2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _clone_interaction_for_chunk(it: Dict[str, Any], obs_end_time_s: float) -> Dict[str, Any]:
    """
    包含归一化后的 location、object、以及相对时间信息。
    - time_s: 绝对时间（秒）
    - time_rel_s: 相对 chunk 观测末端的时间（秒）= time_s - obs_end_time_s
    - location_norm: 归一化坐标（来自 build_interaction_json_new_0112 或 trajectory->压缩）
    - object: 字符串（可空）
    """
    out: Dict[str, Any] = {}

    # 注意：future interactions 的 key 是 "time"
    t_abs = _safe_float(it.get("time", it.get("time_s", 0.0)), 0.0)
    out["time_s"] = float(t_abs)
    out["time_rel_s"] = float(t_abs - float(obs_end_time_s))

    if "location_norm" in it:
        out["location_norm"] = it["location_norm"]
    else:
        if "location_f0_m" in it:
            out["location_f0_m"] = it["location_f0_m"]
        if "location" in it:
            out["location"] = it["location"]

    if "object" in it:
        out["object"] = it["object"]

    # debug / trace fields（可选保留）
    for k in ["frame_idx", "segment_len", "segment_t_start", "segment_t_end", "segment_duration"]:
        if k in it:
            out[k] = it[k]

    return out


# -------------------------
# Dense compress (对齐你未来 compress_dense_repeated_interactions 的逻辑)
# -------------------------
def _obj_set(obj_str: Any) -> Tuple[str, ...]:
    if not isinstance(obj_str, str) or (not obj_str.strip()):
        return tuple()
    toks = [t.strip() for t in obj_str.split(",") if t.strip()]
    toks = sorted(set(toks))
    return tuple(toks)


def compress_dense_repeated_interactions_like_0112(
    interactions_norm: List[Dict[str, Any]],
    dense_dt: float = 3.0,
    pos_eps_f0: float = 0.12,
    use_pos_field: str = "location_f0_m",
    compress_mode: str = "first",
) -> List[Dict[str, Any]]:
    """
    基本复刻你 build_interaction_json_new_0112.py 里的 compress_dense_repeated_interactions 思路：
    - 排序
    - 连续相同 object 且 dt<=dense_dt 且 dpos<=pos_eps_f0 的合并成一段 segment
    - compress_mode 决定用 first/middle/mean 表示该段
    - 增加 segment_len, segment_t_start, segment_t_end, segment_duration
    """
    if not interactions_norm:
        return []

    dense_dt = float(dense_dt)
    pos_eps_f0 = float(pos_eps_f0)

    xs = sorted(interactions_norm, key=lambda it: float(it.get("time", 0.0)))

    def get_pos(it: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        p = it.get(use_pos_field, None)
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            try:
                return (float(p[0]), float(p[1]), float(p[2]))
            except Exception:
                return None
        return None

    def finalize_segment(seg: List[Dict[str, Any]]) -> Dict[str, Any]:
        if compress_mode == "first":
            rep = dict(seg[0])
        elif compress_mode == "middle":
            rep = dict(seg[len(seg) // 2])
        elif compress_mode == "mean":
            rep = dict(seg[0])
            # 只对位置字段做均值，对齐 0112 的行为（time/object 等保持 rep 的）
            ps = []
            for it in seg:
                p = it.get(use_pos_field, None)
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    try:
                        ps.append((float(p[0]), float(p[1]), float(p[2])))
                    except Exception:
                        pass
            if ps:
                mx = sum([p[0] for p in ps]) / len(ps)
                my = sum([p[1] for p in ps]) / len(ps)
                mz = sum([p[2] for p in ps]) / len(ps)
                rep[use_pos_field] = [mx, my, mz]

            # 同时也把 location_norm 平均一下（若存在）
            psn = []
            for it in seg:
                p = it.get("location_norm", None)
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    try:
                        psn.append((float(p[0]), float(p[1]), float(p[2])))
                    except Exception:
                        pass
            if psn:
                mx = sum([p[0] for p in psn]) / len(psn)
                my = sum([p[1] for p in psn]) / len(psn)
                mz = sum([p[2] for p in psn]) / len(psn)
                rep["location_norm"] = [mx, my, mz]
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
            dpos = float(_l2(cur_pos, prev_pos))

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


# -------------------------
# 观测阶段：从 trajectory 连续轨迹“抽点并压缩成事件”
# -------------------------
def build_observed_events_from_trajectory(
    sample: Dict[str, Any],
    t_start: float,
    t_end: float,
    obs_object_name: str = "__obs__",
) -> List[Dict[str, Any]]:
    """
    从 sample["trajectory"] 里取出落在 [t_start, t_end) 的轨迹点，构造成与 interactions 类似的 dict：
      - time
      - location_f0_m   (用于与未来压缩一致的米制距离判断)
      - location_norm
      - object="__obs__" （让 same_obj 恒成立，复用 dense compress 逻辑）
    """
    traj = sample.get("trajectory", {})
    ts = _as_list(traj.get("time_s_world", []))  # 绝对时间(s)
    xyz_norm = _as_list(traj.get("xyz_norm", []))
    xyz_f0_m = _as_list(traj.get("xyz_f0_m", []))  # F0米制坐标（你的 0112 里有）

    if not ts or not xyz_norm:
        return []

    # 对齐长度
    L = min(len(ts), len(xyz_norm), len(xyz_f0_m) if xyz_f0_m else len(ts))
    if L <= 0:
        return []

    out: List[Dict[str, Any]] = []
    for i in range(L):
        t = ts[i]
        if t is None:
            continue
        tt = _safe_float(t, None)
        if tt is None:
            continue
        if not (float(t_start) <= float(tt) < float(t_end)):
            continue

        ln = xyz_norm[i] if i < len(xyz_norm) else None
        lf = xyz_f0_m[i] if (xyz_f0_m and i < len(xyz_f0_m)) else None

        if not (isinstance(ln, (list, tuple)) and len(ln) >= 3):
            continue
        if not (isinstance(lf, (list, tuple)) and len(lf) >= 3):
            continue

        try:
            ln3 = [float(ln[0]), float(ln[1]), float(ln[2])]
            lf3 = [float(lf[0]), float(lf[1]), float(lf[2])]
        except Exception:
            continue

        out.append({
            "time": float(tt),
            "location_f0_m": lf3,
            "location_norm": ln3,
            "object": str(obs_object_name),
        })

    # trajectory 本来按时间递增，但这里保险排序一下
    out.sort(key=lambda d: float(d.get("time", 0.0)))
    return out


# -------------------------
# Core: build chunks from one source sample
# -------------------------
def build_chunk_samples_from_one_source_sample(
    sample: Dict[str, Any],
    chunk_size: int,
    observation_time_s: float,
    include_history: bool,
    pad_to_chunk: bool,
    obs_dense_dt: float,
    obs_pos_eps_f0: float,
    obs_compress_mode: str,
) -> List[Dict[str, Any]]:
    take_name = str(sample.get("take_name", "NA"))
    datum_idx = _safe_int(sample.get("datum_idx", -1), -1)

    interactions_sorted = _sort_interactions_by_time(sample.get("interactions", []))
    n = len(interactions_sorted)
    if n == 0:
        return []

    # 保留 env_objects / env_voxel_tokens
    env_objects = sample.get("env_objects", None)
    env_voxel_tokens = sample.get("env_voxel_tokens", None)

    out_chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        tgt = interactions_sorted[start:end]
        if not tgt:
            continue

        # chunk 第一个 target 的绝对时间 t0
        t0 = _safe_float(tgt[0].get("time", 0.0), 0.0)
        obs_end = float(t0)
        obs_start = max(0.0, obs_end - float(observation_time_s))

        chunk_id = f"{take_name}_{datum_idx}_{chunk_idx}"

        # -------------------------
        # ✅ history_interactions = [t0-30, t0) 内发生的事件
        #    1) 观测轨迹(trajectory) -> 抽点 -> dense compress（与未来一致）
        #    2) 未来事件(interactions) -> 筛窗口
        # -------------------------
        history: List[Dict[str, Any]] = []
        if include_history:
            # 1) 从 trajectory 取出观测窗口点（注意：trajectory 覆盖 observation_start_time_s~observation_end_time_s）
            obs_raw = build_observed_events_from_trajectory(
                sample=sample,
                t_start=obs_start,
                t_end=obs_end,
                obs_object_name="__obs__",
            )

            # 2) 用与你未来一致的 dense compress 逻辑压缩观测点（dt + dpos + compress_mode）
            #    距离用 location_f0_m，与未来一致；阈值用 obs_pos_eps_f0（米）
            obs_comp = compress_dense_repeated_interactions_like_0112(
                interactions_norm=obs_raw,
                dense_dt=float(obs_dense_dt),
                pos_eps_f0=float(obs_pos_eps_f0),
                use_pos_field="location_f0_m",
                compress_mode=str(obs_compress_mode),
            )

            # 3) future interactions 中落在 [obs_start, obs_end) 的事件也属于 history
            fut_hist = []
            for it in interactions_sorted:
                tt = _safe_float(it.get("time", 0.0), 0.0)
                if (tt >= obs_start) and (tt < obs_end):
                    fut_hist.append(it)

            # 4) 合并并 clone 成 chunk 格式
            history_all = obs_comp + fut_hist
            history_all.sort(key=lambda d: _safe_float(d.get("time", 0.0), 0.0))
            history = [_clone_interaction_for_chunk(it, obs_end) for it in history_all]

        # targets（chunk 内）
        targets = [_clone_interaction_for_chunk(it, obs_end) for it in tgt]

        mask = [1] * len(targets)
        if pad_to_chunk and len(targets) < int(chunk_size):
            pad_k = int(chunk_size) - len(targets)
            for _ in range(pad_k):
                targets.append({
                    "time_s": None,
                    "time_rel_s": None,
                    "location_norm": None,
                    "object": None,
                })
            mask += [0] * pad_k

        chunk_sample: Dict[str, Any] = {
            "id": chunk_id,
            "take_name": take_name,
            "datum_idx": int(datum_idx),
            "chunk_idx": int(chunk_idx),

            "chunk_start_inter_idx": int(start),
            "chunk_end_inter_idx": int(end),

            # 抽帧用：以 t0 为观测末端
            "observation_start_time_s": float(obs_start),
            "observation_end_time_s": float(obs_end),

            "history_interactions": history,
            "target_interactions": targets,
            "target_mask": mask,

            "source_meta": {
                "source_current_time_s": sample.get("current_time_s", None),
                "source_future_time_s": sample.get("future_time_s", None),
                "source_observation_end_time_s": sample.get("observation_end_time_s", None),
            },

            # 记录观测压缩参数，方便追溯
            "obs_compress": {
                "dense_dt_s": float(obs_dense_dt),
                "pos_eps_f0_m": float(obs_pos_eps_f0),
                "compress_mode": str(obs_compress_mode),
                "note": "Observation events are obtained by compressing trajectory points in [t0-obs_time, t0) using the SAME dense compress logic as future interactions (dt + dpos + compress_mode), with dpos computed in F0 meters.",
            }
        }

        if env_objects is not None:
            chunk_sample["env_objects"] = env_objects
        if env_voxel_tokens is not None:
            chunk_sample["env_voxel_tokens"] = env_voxel_tokens

        out_chunks.append(chunk_sample)
        chunk_idx += 1

    return out_chunks


# -------------------------
# Sharding helpers
# -------------------------
def shard_out_path(out_cache: str, shard_id: int) -> str:
    base, ext = osp.splitext(out_cache)
    if ext.lower() != ".pkl":
        return f"{out_cache}.shard{shard_id}.pkl"
    return f"{base}.shard{shard_id}.pkl"


def merge_shards(out_cache: str, num_shards: int, overwrite: bool = False) -> None:
    merged: List[Dict[str, Any]] = []
    reports = []

    for k in range(num_shards):
        p = shard_out_path(out_cache, k)
        if same_file(p, out_cache):
            raise ValueError("Merged output must not replace a source shard")
        if not osp.exists(p):
            raise FileNotFoundError(f"Missing shard file: {p}")
        obj = load_cache(p)
        ds = obj.get("dataset", [])
        if not isinstance(ds, list):
            raise ValueError(f"Bad shard dataset list: {p}")
        merged.extend(ds)
        reports.append(obj.get("report", {}))

    out_obj = {
        "dataset": merged,
        "config": {
            "merged_from": [shard_out_path(out_cache, k) for k in range(num_shards)],
            "num_shards": int(num_shards),
        },
        "report": {
            "num_chunk_samples": len(merged),
            "shard_reports": reports,
        },
    }
    out_obj["config"]["release_source"] = {"file": SOURCE_FILE, "sha256": SOURCE_SHA256}
    save_cache(out_obj, out_cache, overwrite=overwrite)

    print("========================================")
    print(f"[MERGE DONE] -> {out_cache}")
    print(f"- num_shards      : {num_shards}")
    print(f"- merged samples  : {len(merged)}")
    print("========================================")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_cache", type=str, default="", help="input *_F0norm.pkl (build_interaction_json_new_0112 output)")
    ap.add_argument("--out_cache", type=str, required=True, help="output chunk cache pkl (final merged target name)")

    ap.add_argument("--chunk_size", type=int, default=10)
    ap.add_argument("--observation_time", type=float, default=30.0)

    ap.add_argument("--include_history", type=int, default=1, choices=[0, 1])
    ap.add_argument("--pad_to_chunk", type=int, default=1, choices=[0, 1])

    # ✅ 观测压缩参数（默认对齐 0112 的 compress_dense 默认值）
    ap.add_argument("--obs_dense_dt", type=float, default=3.0, help="dense_dt for compressing observation trajectory points into events (seconds)")
    ap.add_argument("--obs_pos_eps_f0", type=float, default=0.12, help="pos_eps_f0 for compressing observation events (meters in F0)")
    ap.add_argument("--obs_compress_mode", type=str, default="first", choices=["first", "middle", "mean"],
                    help="representative point for a dense segment in observation compression")

    # ✅ shard 并行参数
    ap.add_argument("--num_shards", type=int, default=1, help="total shards")
    ap.add_argument("--shard_id", type=int, default=0, help="which shard to run: 0..num_shards-1")

    # ✅ merge 模式
    ap.add_argument("--merge_only", type=int, default=0, choices=[0, 1], help="only merge shard outputs into out_cache")

    ap.add_argument("--max_source_samples", type=int, default=0, help="Limit source records before sharding; 0 means all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.chunk_size <= 0 or args.observation_time <= 0:
        ap.error("--chunk_size and --observation_time must be positive")
    if args.in_cache and same_file(args.in_cache, args.out_cache):
        ap.error("Input and output must be different files")

    if int(args.merge_only) == 1:
        if int(args.num_shards) <= 0:
            raise ValueError("--num_shards must be positive for merge")
        merge_shards(args.out_cache, int(args.num_shards), overwrite=args.overwrite)
        return

    if not args.in_cache:
        raise ValueError("--in_cache is required unless --merge_only 1")

    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards-1]")

    out_shard = (args.out_cache if args.num_shards == 1 else
                 shard_out_path(args.out_cache, int(args.shard_id)))
    if same_file(args.in_cache, out_shard):
        ap.error("The actual shard output and input must be different files")
    if osp.exists(out_shard) and not args.overwrite:
        ap.error(f"Output exists: {out_shard}; use --overwrite explicitly")

    src = load_cache(args.in_cache)
    dataset = src.get("dataset", [])
    if not isinstance(dataset, list):
        raise ValueError("in_cache['dataset'] is not a list")

    if args.max_source_samples > 0:
        dataset = dataset[:args.max_source_samples]

    chunk_dataset: List[Dict[str, Any]] = []

    kept_source = 0
    for i, s in enumerate(dataset):
        if (i % int(args.num_shards)) != int(args.shard_id):
            continue
        if not isinstance(s, dict):
            continue
        kept_source += 1
        chunks = build_chunk_samples_from_one_source_sample(
            sample=s,
            chunk_size=int(args.chunk_size),
            observation_time_s=float(args.observation_time),
            include_history=bool(args.include_history),
            pad_to_chunk=bool(args.pad_to_chunk),
            obs_dense_dt=float(args.obs_dense_dt),
            obs_pos_eps_f0=float(args.obs_pos_eps_f0),
            obs_compress_mode=str(args.obs_compress_mode),
        )
        chunk_dataset.extend(chunks)

    out_obj: Dict[str, Any] = {
        "dataset": chunk_dataset,
        "config": {
            "source_cache": args.in_cache,
            "chunk_size": int(args.chunk_size),
            "observation_time_s": float(args.observation_time),
            "include_history": bool(args.include_history),
            "pad_to_chunk": bool(args.pad_to_chunk),

            "obs_compress": {
                "dense_dt_s": float(args.obs_dense_dt),
                "pos_eps_f0_m": float(args.obs_pos_eps_f0),
                "compress_mode": str(args.obs_compress_mode),
                "distance_space": "F0 meters (trajectory.xyz_f0_m) to match future compression logic",
            },

            "num_shards": int(args.num_shards),
            "shard_id": int(args.shard_id),

            "id_format": "{take_name}_{datum_idx}_{chunk_idx}",
            "time_note": "No time normalization. Each interaction keeps time_rel_s relative to observation_end_time_s; absolute times kept in observation_start/end and time(time_s).",
            "env_note": "env_objects (and env_voxel_tokens if present) are copied from source sample into each chunk sample without modification.",
            "history_note": "history_interactions are events in [t0-obs_time, t0). Observation part is obtained by compressing trajectory points with SAME dense compress logic as future events.",
        },
        "report": {
            "num_source_samples_total": len(dataset),
            "num_source_samples_this_shard": kept_source,
            "num_chunk_samples_this_shard": len(chunk_dataset),
            "avg_chunks_per_source_sample_this_shard": (len(chunk_dataset) / max(1, kept_source)),
        },
    }

    out_obj["config"]["release_source"] = {"file": SOURCE_FILE, "sha256": SOURCE_SHA256}
    save_cache(out_obj, out_shard, overwrite=args.overwrite)

    print("========================================")
    print(f"[BUILD SHARD DONE]")
    print(f"[IN ] {args.in_cache}")
    print(f"[OUT] {out_shard}")
    print(f"- shard_id / num_shards : {int(args.shard_id)} / {int(args.num_shards)}")
    print(f"- source total samples  : {len(dataset)}")
    print(f"- source shard samples  : {kept_source}")
    print(f"- chunk shard samples   : {len(chunk_dataset)}")
    print(f"- chunk_size            : {int(args.chunk_size)}")
    print(f"- observation_time(s)   : {float(args.observation_time)}")
    print(f"- obs_compress dense_dt : {float(args.obs_dense_dt)}")
    print(f"- obs_compress pos_eps  : {float(args.obs_pos_eps_f0)} (F0 meters)")
    print(f"- obs_compress mode     : {str(args.obs_compress_mode)}")
    print("========================================")


if __name__ == "__main__":
    main()
