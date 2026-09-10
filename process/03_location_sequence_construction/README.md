# Location sequence construction

[Pipeline overview](../README.md) · [Previous: shared coordinates](../02_shared_coordinate_construction/) · [Next: SMPL attachment](../04_smpl_state_attachment/)

## Purpose

Convert sparse narration-associated interactions into ordered continuous hand-location targets while suppressing repeated events. The output retains both semantic object context and continuous geometry.

## Inputs and outputs

The upstream narration stage uses [Llama 3](https://github.com/meta-llama/llama3) to associate narrated objects with the semantic vocabulary, following [FIction's interaction preparation](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md). This script consumes the resulting annotations; it does not run an LLM.

| Input | Expected layout |
|---|---|
| `--dataset_root` | `<take>_humans_objects_interactions.pkl` files containing scene-aligned `human` frames and an `orig_interaction_dataset` DataFrame |
| `--annotations_root` | `atomic_descriptions_train.json` and `atomic_descriptions_val.json` with an `annotations` mapping |
| `--takes_json` | Ego-Exo4D take metadata, including `take_name`, `take_uid`, and `parent_task_name` |
| `--takes_txt_dir` | `train_takes.txt`, `val_takes.txt`, and `test_takes.txt`, one take name per line |
| `--lvis_cat_json` | LVIS category metadata used to assign semantic IDs |

As in the source script, the test split looks up atomic descriptions in the validation narration file. The take list, not this lookup, defines the dataset split.

**Output:** an F0-normalized pickle dictionary containing `dataset`, configuration, and construction diagnostics. Each source sample retains the observation trajectory, future interaction sequence, environment objects, and reference-frame metadata.

## Run

Example for Bike Repair with the inspected source-cache merging settings:

```bash
python process/03_location_sequence_construction/build_interactions.py \
  --split test --scenarios "Bike Repair" \
  --dataset_root inputs/scene_aligned_takes \
  --annotations_root inputs/narrations \
  --takes_json inputs/takes.json \
  --takes_txt_dir inputs/splits \
  --lvis_cat_json inputs/lvis_v1_train_cat_info.json \
  --out_cache work/test_bike_F0norm.pkl \
  --observation_time 30 --anticipation_time 5 --future_time 60 \
  --right_hand_vert_id 5777 --unknown_hand_mode right \
  --min_gap 1.0 --compress_dense 1 \
  --dense_dt 4.0 --pos_eps_f0 0.15 --compress_mode first \
  --pose_avg_win_frames 10 --pos_scale_m 5.0 \
  --include_env_objects 1 --include_env_voxels 0 \
  --export_first_n_json 0 --viz_first_n 0
```

Replace the example `inputs/` and `work/` paths with your local paths. These directories are ignored by Git and their contents should not be committed. This example records an inspected Bike configuration; it is not a claim that every historical source export used the same settings.

## Hand proxy and event merging

The hand proxy is **right-hand SMPL mesh vertex 5777**. Right-hand and both-hand events use this point; unspecified-hand narrations are treated as right-hand events with `--unknown_hand_mode right`. Explicit left-only events are not added to this right-hand stream.

The builder retrieves the nearest available hand mesh frame for each timestamp and applies the shared reference and positional normalization. It operates on previously assembled interaction annotations; it does not implement a new hand-to-object contact-surface test.

Two chronological neighboring events can be merged when their object sets match and both of the following hold:

```text
current_time - previous_time <= dense_dt
EuclideanDistance(current_position_f0_m, previous_position_f0_m) <= pos_eps_f0
```

The checks compare **consecutive candidates**, so a chain's total span may exceed `dense_dt`. The distance is computed in **meters before normalization**, not directly in normalized coordinates. A 0.15-meter bound corresponds to 0.03 in coordinates scaled by 5 meters.

`compress_mode=first` retains the first event's location and timestamp and records the segment's extent. `min_gap` separately debounces repeated right-hand object tokens. The historic merger treats an unavailable position as passing the spatial test; upstream validity filtering and downstream masks therefore remain important.

## Temporal versus spatial reference

The source builder uses a five-second anticipation interval by default when forming its initial candidate windows. [Chunk generation](../05_forecast_sample_generation/) subsequently resets the observation endpoint to the first retained target timestamp and keeps the preceding 30 seconds. This resets the **time origin**, but preserves the source spatial F0 needed by pose attachment.

Takes must satisfy the supplied split list, narration UID availability, and scenario filter. The source builder skips samples with an observation interval shorter than 30 seconds, an unavailable reference frame, or unusable observed trajectories. A target hand frame outside `--max_full_frame_err` (default 0.49 seconds) is rejected.

Source samples with no retained events generate no forecast chunks. If reproducing a historical prefiltered `nozero` cache's exact sample IDs and source indices, apply the same source filtering and ordering before chunking.

Run [step 5's chunk command](../05_forecast_sample_generation/) next, then attach SMPL states with [step 4](../04_smpl_state_attachment/).
