#!/usr/bin/env python3
"""Validate released numerical helpers against trusted local historical caches.

This does not re-run Detic, narrations, WHAM inference, or full-take attachment.
It rebuilds chunks twice and reconstructs cached SMPL states twice on CPU.
Only aggregate checks, hashes, and sample identifiers are written to JSON.
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROCESS = Path(__file__).resolve().parents[1]

def module(relative):
    path = PROCESS / relative
    spec = importlib.util.spec_from_file_location(path.stem,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def digest(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda: f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def same(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
    return a==b

def check_chunks(args):
    source = load(args.loc_cache)
    original = {s["id"]:s for s in load(args.chunk_cache)["dataset"]}
    limit = min(args.source_samples,len(source["dataset"]))
    outputs=[]
    with tempfile.TemporaryDirectory(prefix="process-chunk-validation-") as tmp:
        for repeat in (1,2):
            out = Path(tmp)/("repeat%d.pkl"%repeat)
            cmd = [sys.executable,str(PROCESS/"05_forecast_sample_generation/build_chunks.py"),
                   "--in_cache",args.loc_cache,"--out_cache",str(out),
                   "--chunk_size",str(args.chunk_size),"--max_source_samples",str(limit),
                   "--obs_dense_dt","3","--obs_pos_eps_f0","0.12","--obs_compress_mode","first"]
            subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120)
            outputs.append(load(out)["dataset"])
    assert same(outputs[0],outputs[1]), "Chunk re-run was not exact"
    assert outputs[0], "No chunks rebuilt"
    for row in outputs[0]:
        assert row["id"] in original, row["id"]
        assert same(row,original[row["id"]]), "Historical chunk mismatch: "+row["id"]
    return dict(status="PASS",source_samples=limit,rebuilt_chunks=len(outputs[0]),
                historical_comparison="exact by sample ID (historical shard ordering ignored)",
                rerun_comparison="exact",ids=[x["id"] for x in outputs[0]])

def reconstruct(pose,helper,W,s,side):
    valid=np.asarray(s[side+"_pose_mask"],bool)
    if not np.any(valid):
        return None
    root6=np.asarray(s[side+"_root_orient6d_cam"],np.float32)[valid]
    rootR=pose.rot6d_to_rotmat_np(root6)
    aa=Rotation.from_matrix(rootR).as_rotvec().astype(np.float32)
    body=np.asarray(s[side+"_body_pose_local_aa"],np.float32)[valid].reshape(-1,69)
    beta=np.asarray(s[side+"_betas"],np.float32)[valid]
    root=np.asarray(s[side+"_root_trans_cam_m"],np.float32)[valid]
    verts0,j0=helper.forward(aa,body,beta,np.zeros_like(root))
    delta=root-j0[:,0,:3]
    verts=verts0+delta[:,None,:]
    native_pelvis=j0[:,0,:3]+delta
    joints=pose.regress_joints_from_verts(verts,W)
    R=np.asarray(s[side+"_cam_to_bench_R"],np.float32)[valid]
    t=np.asarray(s[side+"_cam_to_bench_t"],np.float32)[valid]
    world=np.einsum("nij,nvj->nvi",R,joints)+t[:,None,:]
    f0=np.stack([pose.world_to_F0_points(x,s["F0_R_world"],s["F0_t_world"]) for x in world])
    rootworld=np.einsum("nij,nj->ni",R,native_pelvis)+t
    rootf0=pose.world_to_F0_points(rootworld,s["F0_R_world"],s["F0_t_world"])
    # Historical hand loc used stored source vertices; do not force this new
    # SMPL reconstruction to equal those measurements or an object-box point.
    checks={
       "native_pelvis_m":float(np.max(np.abs(native_pelvis-root))),
       "joints_cam_m":float(np.max(np.abs(joints-np.asarray(s[side+"_joints_cam_m"])[valid]))),
       "joints_f0_m":float(np.max(np.abs(f0-np.asarray(s[side+"_joints_f0_m"])[valid]))),
       "root_f0_m":float(np.max(np.abs(rootf0-np.asarray(s[side+"_root_trans_f0_m"])[valid]))),
    }
    for key,value in checks.items():
        assert value <= 5e-5,(s["id"],side,key,value)
    return checks,np.concatenate([native_pelvis.reshape(-1),joints.reshape(-1),f0.reshape(-1)])

def check_pose(args):
    pose=module("04_smpl_state_attachment/attach_smpl.py")
    loc=module("03_location_sequence_construction/build_interactions.py")
    cached=load(args.pose_cache)
    samples=cached["dataset"][:args.pose_samples]
    assert samples
    torch.set_num_threads(args.cpu_threads)
    helper=pose.SMPLForwardHelper(args.smpl_model_path,device="cpu")
    W=pose.load_joint_regressor(args.joint_regressor_path)
    maxima={}
    n=0
    digests=[]
    for repeat in (1,2):
        h=hashlib.sha256()
        for sample in samples:
            for side in ("history","target"):
                out=reconstruct(pose,helper,W,sample,side)
                if out is None:
                    continue
                checks,arr=out
                h.update(arr.tobytes())
                if repeat==1:
                    n+=int(np.sum(sample[side+"_pose_mask"]))
                    for k,v in checks.items():
                        maxima[k]=max(maxima.get(k,0.),v)
        digests.append(h.hexdigest())
    assert n>0
    assert digests[0]==digests[1], "CPU SMPL reconstruction re-run not exact"
    source={(s["take_name"],s["datum_idx"]):s for s in load(args.loc_cache)["dataset"]}
    f0_max=0.
    for sample in samples:
        src=source[(sample["take_name"],sample["datum_idx"])]
        tr=src["trajectory"]
        recomputed=loc.world_to_F0(np.asarray(tr["xyz_world_m"]),sample["F0_R_world"],sample["F0_t_world"])
        err=float(np.max(np.abs(recomputed-np.asarray(tr["xyz_f0_m"]))))
        f0_max=max(f0_max,err)
        assert err<=5e-5,("shared F0 mismatch",sample["id"],err)
        assert sample["pose_f0_anchor_time_s"]==sample["source_meta"]["source_observation_end_time_s"]
    return dict(status="PASS",samples=len(samples),valid_events=n,device="cpu",
                max_absolute_errors=maxima,shared_f0_max_abs_m=f0_max,
                tolerance_m=5e-5,rerun_comparison="exact array SHA-256",
                rerun_sha256=digests[0],
                scope="Reconstruction from saved real Route-6 states; not full upstream attachment",
                historical_gpu_cpu_caveat="Reconstruction comparison allows 0.05 mm numerical tolerance")

def check_attachment_cli(args):
    """Exercise the full attachment CLI with synthetic inputs and real SMPL assets."""
    pose=module("04_smpl_state_attachment/attach_smpl.py")
    helper=pose.SMPLForwardHelper(args.smpl_model_path,device="cpu")
    root_aa=np.array([[.1,.2,.3]],np.float32)
    body_aa=np.zeros((1,69),np.float32)
    beta=np.zeros((1,10),np.float32)
    transl=np.array([[1.,2.,3.]],np.float32)
    vertices,native=helper.forward(root_aa,body_aa,beta,transl)
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],np.float32)
    shift=np.array([.4,.5,.6],np.float32)
    world=pose.apply_rigid_points(vertices[0],rot,shift)
    anchor_shift=np.array([.1,.2,.3],np.float32)
    loc_m=world[5777]-anchor_shift+np.array([.1,0.,0.],np.float32)
    frames=list(range(1021,1051))+[1200]
    human={fr:{"verts":world,"rotation_matrix":np.eye(3,dtype=np.float32),
               "device_trajectory":anchor_shift} for fr in frames}
    source_entry={"pose":np.concatenate([root_aa[0],body_aa[0]]),"trans":transl[0],
                  "betas":beta[0],"verts":vertices[0]}
    largest={fr:{"camera00":source_entry} for fr in frames}
    event=lambda t: dict(time_s=t,time_rel_s=t-40.,location_f0_m=loc_m.tolist(),
                         location_norm=(loc_m/5.).tolist(),object="synthetic reference")
    sample=dict(id="synthetic_cli_0_0",take_name="synthetic_cli",datum_idx=0,chunk_idx=0,
                observation_start_time_s=10.,observation_end_time_s=40.,
                source_meta={"source_observation_end_time_s":35.},
                history_interactions=[event(34.5)],
                target_interactions=[event(40.),event(100.),
                                     dict(time_s=None,time_rel_s=None,location_norm=None,object=None)],
                target_mask=[1,1,0])
    results=[]
    with tempfile.TemporaryDirectory(prefix="process-attach-validation-") as tmp:
        root=Path(tmp)
        for filename,obj in [
            ("synthetic_cli_humans_objects_interactions.pkl",{"human":human}),
            ("synthetic_cli_largest_area.pkl",largest),("chunks.pkl",{"dataset":[sample],"config":{}})]:
            with (root/filename).open("wb") as f:
                pickle.dump(obj,f)
        for repeat in (1,2):
            out=root/("attached%d.pkl"%repeat)
            cmd=[sys.executable,str(PROCESS/"04_smpl_state_attachment/attach_smpl.py"),
                 "--in_chunk",str(root/"chunks.pkl"),"--out_chunk",str(out),
                 "--full_take_root",tmp,"--largest_area_root",tmp,
                 "--smpl_model_path",args.smpl_model_path,"--joint_regressor_path",args.joint_regressor_path,
                 "--device","cpu","--cpu_threads",str(args.cpu_threads),"--debug_first_n","0"]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
            if result.returncode:
                raise RuntimeError("Attachment CLI failed:\n"+result.stdout+"\n"+result.stderr)
            assert out.is_file()
            assert out.with_name(out.stem+"_pose_fail_report.pkl").is_file()
            row=load(out)["dataset"][0]
            np.testing.assert_array_equal(row["target_pose_mask"],[1,0,0])
            np.testing.assert_array_equal(row["target_pose_fail_code"],[0,3,1])
            np.testing.assert_array_equal(row["history_pose_mask"],[1])
            np.testing.assert_allclose(row["F0_t_world"],anchor_shift,atol=1e-7)
            np.testing.assert_allclose(row["target_root_trans_cam_m"][0],native[0,0,:3],atol=1e-6)
            assert row["pose_f0_anchor_time_s"]==35.
            assert abs(float(row["target_pose_loc_right_hand_err_mm"][0])-100.)<.01
            results.append(row)
    assert same(results[0],results[1]), "Full synthetic attachment re-run not exact"
    return dict(status="PASS",input_kind="Synthetic full-take and camera-chain records with user-supplied SMPL assets",
                cli_runs=2,rerun_comparison="exact dataset including NaN masks",
                history_pose_mask=[1],target_pose_mask=[1,0,0],target_pose_fail_code=[0,3,1],
                deliberately_offset_loc_mm=100.,
                checked=["output cache and failure report","native pelvis","source-F0 anchor",
                         "failed-event and padding masks","no pose fitting to loc"],
                scope="CLI integration only; not reconstruction of original WHAM/full-take input pairs")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("loc_cache","chunk_cache","pose_cache","smpl_model_path","joint_regressor_path","output"):
        ap.add_argument("--"+key,required=True)
    ap.add_argument("--chunk_size",type=int,default=4)
    ap.add_argument("--source_samples",type=int,default=3)
    ap.add_argument("--pose_samples",type=int,default=3)
    ap.add_argument("--cpu_threads",type=int,default=4)
    args=ap.parse_args()
    if args.source_samples<=0 or args.pose_samples<=0:
        ap.error("Sample limits must be positive")
    if Path(args.output).exists():
        ap.error("Output already exists; choose a new validation report path")
    checks={"chunks":check_chunks(args),"cached_pose_reconstruction":check_pose(args),
            "synthetic_attachment_cli":check_attachment_cli(args)}
    report={"material_passport":{"origin_skill":"academic-research-suite/experiment-agent",
              "origin_mode":"validate","origin_date":datetime.now(timezone.utc).isoformat(),
              "verification_status":"VERIFIED","version_label":"process_core_subset_v1",
              "scope":"Only the successful subset checks below; not end-to-end construction"},
            "checks":checks,"environment":{"python":platform.python_version(),
              "numpy":np.__version__,"torch":torch.__version__,"device":"cpu","threads":args.cpu_threads},
            "input_hashes":{k:{"filename":Path(getattr(args,k)).name,"sha256":digest(getattr(args,k))}
              for k in ("loc_cache","chunk_cache","pose_cache","joint_regressor_path")},
            "model_asset_hashes":{p.name:digest(p) for p in Path(args.smpl_model_path).glob("SMPL_NEUTRAL.pkl")},
            "code_hashes":{str(p.relative_to(PROCESS)):digest(p) for p in [
                PROCESS/"03_location_sequence_construction/build_interactions.py",
                PROCESS/"04_smpl_state_attachment/attach_smpl.py",
                PROCESS/"05_forecast_sample_generation/build_chunks.py"]},
            "not_run":["Detic inference","Llama narration grounding","WHAM inference",
                       "Full source-event construction from raw upstream inputs",
                       "Full-take Route-6 attachment from original WHAM/full-take pairs",
                       "Video frame extraction","Full dataset regeneration"]}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(checks,indent=2))
    print("Report:",output)

if __name__=="__main__":
    main()
