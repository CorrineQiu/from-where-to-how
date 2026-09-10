# SMPL state attachment

Pair each retained history and target timestamp with a full-body state in the same spatial frame as its interaction location.

## Inputs and outputs

Run [chunk generation](../05_forecast_sample_generation/) before this script.

| Input | Content |
|---|---|
| `--in_chunk` | Location chunks with timestamps, masks, and source-reference metadata |
| `--full_take_root` | Scene-aligned take records with human mesh frames and device poses |
| `--largest_area_root` | Selected-actor WHAM reconstructions, `<take>_largest_area.pkl` |
| `--smpl_model_path` | SMPL body-model directory |
| `--joint_regressor_path` | The SMPL-to-19-joint regressor |

**Output:** pose-augmented chunks with root rotation (6D), root translation (3, meters), body rotations (23 × 6D), joints (19 × 3, meters), and pose masks. Use `target_mask & target_pose_mask` for paired targets.

## Run

```bash
python process/04_smpl_state_attachment/attach_smpl.py \
  --in_chunk work/test_bike_chunk4.pkl \
  --out_chunk work/pose_bike_chunk4_test.pkl \
  --full_take_root inputs/scene_aligned_takes \
  --largest_area_root inputs/selected_actor_wham \
  --smpl_model_path inputs/body_models/smpl \
  --joint_regressor_path inputs/SMPL_to_J19.pkl \
  --fps 30 --max_err_s 0.49 \
  --last_sec 1.0 --win_frames 10 --min_pose_frames 10 \
  --pos_scale_m 5.0 --right_hand_vert_idx 5777 \
  --device cpu
```

Change input/output names for Health and Cooking. Add `--take_name TAKE_NAME` for one take, or select a CUDA device with `--device`.

## Alignment and attachment

1. Recover local body rotations and shape from the selected actor's [WHAM](https://github.com/yohanshin/WHAM) reconstruction using [SMPL](https://smpl.is.tue.mpg.de/).
2. Fit the rigid transform from the reconstructed SMPL surface to the corresponding scene-aligned human mesh.
3. Recover F0 at the inherited source reference timestamp. Transform the root orientation, root translation, and joints together.
4. Match each timestamp to its nearest scene frame. Reject the attachment if its time error exceeds 0.49 seconds or that frame has no usable SMPL state.
5. Save pose masks and alignment quality weights. Exclude chunks without pose fields from paired supervision.

`root_trans_f0_m` denotes the SMPL native pelvis position in F0 meters, rather than the body's raw translation offset. `joints_f0_m` stores positions from the supplied 19-joint regressor. Auxiliary normalized fields are saved separately.

## Quality fields

Reconstruction, rigid-fit, and hand-location agreement determine quality weights, using these defaults:

| Argument | Value |
|---|---:|
| `quality_source_tau_mm` | 10.0 |
| `quality_rigid_tau_mm` | 5.0 |
| `quality_hand_tau_mm` | 120.0 |
| `min_quality_weight` | 0.05 |
