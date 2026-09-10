# Location sequence construction

Convert sparse narration-associated interactions into ordered continuous hand-location targets, merging repeated events while preserving object context.

## Inputs and outputs

| Input | Expected layout |
|---|---|
| `--dataset_root` | `<take>_humans_objects_interactions.pkl` files containing scene-aligned `human` frames and an `orig_interaction_dataset` DataFrame |
| `--annotations_root` | `atomic_descriptions_train.json` and `atomic_descriptions_val.json` with an `annotations` mapping |
| `--takes_json` | Ego-Exo4D take metadata, including `take_name`, `take_uid`, and `parent_task_name` |
| `--takes_txt_dir` | `train_takes.txt`, `val_takes.txt`, and `test_takes.txt`, one take name per line |
| `--lvis_cat_json` | LVIS category metadata used to assign semantic IDs |

Take lists define the splits. Test narrations are read from the validation narration file.

**Output:** an F0-normalized pickle containing observation trajectories, future interaction sequences, environment objects, and reference metadata.

## Run

Example for Bike Repair:

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

Replace the example `inputs/` and `work/` paths with your local paths.

## Hand proxy and event merging

Use **right-hand SMPL mesh vertex 5777** as the hand-location proxy for right-hand, both-hand, and unspecified-hand events (`--unknown_hand_mode right`). Left-only events are excluded. Retrieve the nearest mesh frame within `--max_full_frame_err` (default 0.49 seconds), then apply F0 and divide by 5.

Two chronological neighboring events can be merged when their object sets match and both of the following hold:

```text
current_time - previous_time <= dense_dt
EuclideanDistance(current_position_f0_m, previous_position_f0_m) <= pos_eps_f0
```

The checks compare **consecutive candidates**, so a merged chain may span more than `dense_dt`. Distances use meters before normalization: 0.15 meters corresponds to 0.03 after division by 5. Missing positions pass the spatial merge check; retain validity masks for downstream filtering.

`compress_mode=first` keeps the first event's location and timestamp. `min_gap` separately suppresses repeated right-hand object tokens within one second.

Run [chunk generation](../05_forecast_sample_generation/) next, then [SMPL attachment](../04_smpl_state_attachment/). Chunking resets the time origin to the first target while preserving the source spatial F0.
