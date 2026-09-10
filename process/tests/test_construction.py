"""Synthetic regression tests; no datasets, weights, or network required."""
import importlib.util
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

PROCESS = Path(__file__).resolve().parents[1]

def module(relative):
    path = PROCESS / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

LOC = module("03_location_sequence_construction/build_interactions.py")
CHUNK = module("05_forecast_sample_generation/build_chunks.py")
POSE = module("04_smpl_state_attachment/attach_smpl.py")

def chunk_sample():
    return dict(take_name="synthetic", datum_idx=2, current_time_s=40.,
                observation_end_time_s=35., future_time_s=60.,
                trajectory=dict(time_s_world=[20., 24.], xyz_f0_m=[[0.,0.,0.],[1.,0.,0.]],
                                xyz_norm=[[0.,0.,0.],[.2,0.,0.]]),
                interactions=[dict(time=t, object="cup", location_norm=[.1,.2,.3],
                                   location_f0_m=[.5,1.,1.5]) for t in (40.,45.,50.)],
                env_objects=[dict(raw_name="cup-1", center=[0.,0.,0.])])

class ConstructionTests(unittest.TestCase):
    def test_hand_policy_and_vertex(self):
        verts = np.zeros((5778,3), np.float32)
        verts[5777] = [1.,2.,3.]
        human = {f: {"verts": verts} for f in (30,60,90,120)}
        df = pd.DataFrame({"timestamp":[1.,2.,3.,4.],
                           "narration":["right hand","left hand","both hands","stirs soup"],
                           "object":["cup"]*4})
        out = LOC.build_future_interactions_right_only(
            df,human,sorted(human),0.,5.,30.,5777,.49,0.,"right")
        self.assertEqual([x["time"] for x in out],[1.,3.,4.])
        self.assertTrue(all(x["location"] == [1.,2.,3.] for x in out))

    def test_array_valued_annotation_columns(self):
        df = pd.DataFrame({"timestamp":[40.,41.],"object":["cup@@@@@pan","cup"],
                           "narration":["right hand"]*2,
                           "geometry":[np.zeros((2,3)),np.ones((2,3))]})
        _, expanded, _ = LOC.select_anticipation_window(df)
        self.assertEqual(sorted(expanded[0]["object"].tolist()),["cup","cup","pan"])

    def test_dense_compression_is_consecutive(self):
        xs = [dict(time=float(i),object="cup",location_f0_m=[i*.1,0,0]) for i in range(3)]
        out = LOC.compress_dense_repeated_interactions(xs,dense_dt=1.,pos_eps_f0=.11)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]["segment_len"],3)
        self.assertEqual(out[0]["time"],0.)

    def test_shared_f0_and_metric_normalization(self):
        rot = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        tr = np.array([2.,3.,4.])
        xyz = np.array([[3.,3.,4.],[20.,3.,4.]])
        local = LOC.world_to_F0(xyz,rot,tr)
        np.testing.assert_allclose(local, POSE.world_to_F0_points(xyz,rot,tr))
        np.testing.assert_allclose(LOC.normalize_xyz(local,5.),np.clip(local/5.,-1.,1.))
        self.assertEqual(POSE.get_pose_f0_anchor_time(
            {"source_meta":{"source_observation_end_time_s":35.},"observation_end_time_s":40.}),35.)

    def test_chunk_anchor_padding_and_geometry(self):
        sample = chunk_sample()
        out = CHUNK.build_chunk_samples_from_one_source_sample(sample,2,30.,True,True,3.,.12,"first")
        self.assertEqual(len(out),2)
        self.assertEqual(out[0]["source_meta"]["source_observation_end_time_s"],35.)
        self.assertEqual(out[0]["observation_end_time_s"],40.)
        self.assertEqual(out[0]["target_interactions"][0]["time_rel_s"],0.)
        self.assertEqual(out[1]["target_mask"],[1,0])
        self.assertEqual(out[0]["env_objects"],sample["env_objects"])
        self.assertEqual([e["object"] for e in out[0]["history_interactions"]],["__obs__","__obs__"])
        sample["interactions"]=[]
        self.assertEqual(CHUNK.build_chunk_samples_from_one_source_sample(
            sample,2,30.,True,True,3.,.12,"first"),[])

    def test_loc_decode_does_not_invert_clipped_values(self):
        metric,norm,mode,saturated = POSE.extract_loc_right_hand_fields(
            {"location_norm":[1.,.2,-.2]},np.eye(3),np.zeros(3),5.)
        self.assertIsNone(metric)
        self.assertTrue(saturated)
        self.assertEqual(mode,POSE.LOC_DECODE_F0_NORM)

    def test_rigid_alignment(self):
        xyz = np.random.default_rng(42).normal(size=(20,3)).astype(np.float32)
        rot = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],np.float32)
        dst = xyz @ rot.T + [1.,2.,3.]
        r,t,error = POSE.rigid_fit_kabsch(xyz,dst)
        np.testing.assert_allclose(POSE.apply_rigid_points(xyz,r,t),dst,atol=1e-6)
        self.assertLess(error,.001)

    def test_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"cache.pkl")
            CHUNK.save_cache({"dataset":[]},path)
            with self.assertRaises(FileExistsError):
                CHUNK.save_cache({"dataset":[]},path)
            CHUNK.save_cache({"dataset":[1]},path,overwrite=True)

    def test_missing_inputs_fail_before_smpl_loading(self):
        argv = ["attach_smpl.py"]
        for flag in ("in_chunk","out_chunk","full_take_root","largest_area_root",
                     "smpl_model_path","joint_regressor_path"):
            argv += ["--"+flag,"/nonexistent/process-test/"+flag]
        with mock.patch.object(sys,"argv",argv), mock.patch.object(POSE,"SMPLForwardHelper") as smpl:
            with self.assertRaises(SystemExit) as ctx:
                POSE.main()
            self.assertEqual(ctx.exception.code,2)
            smpl.assert_not_called()

    def test_pose_report_cannot_replace_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"out_pose_fail_report.pkl"
            source.write_bytes(b"input must remain unchanged")
            argv=["attach_smpl.py","--in_chunk",str(source),"--out_chunk",str(Path(tmp)/"out.pkl"),
                  "--overwrite"]
            for flag in ("full_take_root","largest_area_root","smpl_model_path","joint_regressor_path"):
                argv += ["--"+flag,tmp]
            with mock.patch.object(sys,"argv",argv), mock.patch.object(POSE,"SMPLForwardHelper") as smpl:
                with self.assertRaises(SystemExit) as ctx:
                    POSE.main()
                self.assertEqual(ctx.exception.code,2)
                smpl.assert_not_called()
            self.assertEqual(source.read_bytes(),b"input must remain unchanged")

    def test_loc_input_alias_guard_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source.json"
            source.write_text("input")
            alias=Path(tmp)/"output_report.json"
            alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError,"replace an input"):
                LOC.preflight_outputs([str(alias)],overwrite=True,input_paths=[str(source)])
            self.assertEqual(source.read_text(),"input")

    def test_zero_attached_events_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["attach_smpl.py"]
            for flag in ("in_chunk","full_take_root","largest_area_root",
                         "smpl_model_path","joint_regressor_path"):
                argv += ["--"+flag,tmp]
            output = str(Path(tmp)/"empty_pose.pkl")
            argv += ["--out_chunk",output]
            with mock.patch.object(sys,"argv",argv), \
                 mock.patch.object(POSE,"SMPLForwardHelper"), \
                 mock.patch.object(POSE,"load_joint_regressor",return_value=np.zeros((19,6890))), \
                 mock.patch.object(POSE,"load_pkl",return_value={"dataset":[chunk_sample()]}), \
                 mock.patch.object(POSE,"resolve_full_take_path",return_value=None), \
                 mock.patch.object(POSE,"resolve_largest_area_path",return_value=None):
                with self.assertRaisesRegex(RuntimeError,"No valid pose events"):
                    POSE.main()
            self.assertFalse(Path(output).exists())

    def test_actual_shard_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"chunks.shard0.pkl"
            with source.open("wb") as stream:
                pickle.dump({"dataset":[chunk_sample()]},stream)
            before = source.read_bytes()
            cmd = [sys.executable,str(PROCESS/"05_forecast_sample_generation/build_chunks.py"),
                   "--in_cache",str(source),"--out_cache",str(Path(tmp)/"chunks.pkl"),
                   "--num_shards","2","--shard_id","0","--overwrite"]
            result = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("actual shard output",result.stderr)
            self.assertEqual(source.read_bytes(),before)

    def test_loc_cli_and_sidecar_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verts = np.zeros((5778,3),np.float32)
            verts[5777] = [1.,2.,3.]
            human = {fr:{"verts":verts,"rotation_matrix":np.eye(3),
                         "device_trajectory":np.zeros(3)} for fr in list(range(1020,1051))+[1200]}
            df = pd.DataFrame({"timestamp":[40.],"narration":["right hand"],
                               "object":["cup"],"geometry":[np.zeros((2,3))]})
            with (root/"synthetic_humans_objects_interactions.pkl").open("wb") as f:
                pickle.dump({"human":human,"orig_interaction_dataset":df},f)
            import json
            (root/"takes.json").write_text(json.dumps([
                {"take_name":"synthetic","take_uid":"uid","parent_task_name":"Cooking"}]))
            (root/"atomic_descriptions_train.json").write_text(json.dumps({"annotations":{"uid":[]}}))
            (root/"train_takes.txt").write_text("synthetic\n")
            (root/"categories.json").write_text(json.dumps([{"name":"cup"}]))
            target = root/"built.pkl"
            report = root/"built_report.json"
            cmd = [sys.executable,str(PROCESS/"03_location_sequence_construction/build_interactions.py"),
                   "--dataset_root",tmp,"--annotations_root",tmp,"--takes_txt_dir",tmp,
                   "--takes_json",str(root/"takes.json"),"--lvis_cat_json",str(root/"categories.json"),
                   "--out_cache",str(target),"--export_first_n_json","1"]
            report.write_text("existing report")
            blocked = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(blocked.returncode,0)
            self.assertIn("Output exists",blocked.stderr)
            self.assertFalse(target.exists())
            self.assertEqual(report.read_text(),"existing report")
            report.unlink()
            first_json=root/"built_first1_samples.json"
            first_json.write_text("existing first sample")
            blocked = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(blocked.returncode,0)
            self.assertFalse(target.exists())
            self.assertFalse(report.exists())
            self.assertEqual(first_json.read_text(),"existing first sample")
            first_json.unlink()
            result = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            with target.open("rb") as stream:
                built = pickle.load(stream)
            self.assertEqual(len(built["dataset"]),1)
            self.assertEqual(built["dataset"][0]["interactions"][0]["location"],[1.,2.,3.])
            self.assertTrue((root/"built_first1_samples.json").exists())

    def test_chunk_cli_and_invalid_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source.pkl"
            with source.open("wb") as stream:
                pickle.dump({"dataset":[chunk_sample()]},stream)
            target = Path(tmp)/"chunk.pkl"
            cmd = [sys.executable,str(PROCESS/"05_forecast_sample_generation/build_chunks.py"),
                   "--in_cache",str(source),"--out_cache",str(target),"--chunk_size","2"]
            result = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_suffix(".shard0.pkl").exists())
            invalid = subprocess.run(cmd[:-1]+["0"],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(invalid.returncode,0)

if __name__ == "__main__":
    unittest.main()
