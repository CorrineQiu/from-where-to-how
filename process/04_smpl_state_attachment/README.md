# SMPL state attachment

[Pipeline overview](../README.md) · [Previous: location sequences](../03_location_sequence_construction/) · [Next: forecast samples](../05_forecast_sample_generation/)

## Purpose

Pair every retained history/target timestamp with a full-body state in the same spatial frame as its interaction location. A hand location alone does not define the complete body configuration.

## Inputs and outputs

Run [chunk generation](../05_forecast_sample_generation/) before this script.

| Input | Content |
|---|---|
| `--in_chunk` | Location chunks with timestamps, masks, and source-reference metadata |
| `--full_take_root` | Scene-aligned take records with human mesh frames and device poses |
| `--largest_area_root` | Selected-actor WHAM reconstructions, `<take>_largest_area.pkl` |
| `--smpl_model_path` | SMPL body-model directory |
| `--joint_regressor_path` | The SMPL-to-19-joint regressor |

The upstream motion reconstruction uses [WHAM](https://github.com/yohanshin/WHAM) and [SMPL](https://smpl.is.tue.mpg.de/). See the [FIction preparation guide](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md) for actor selection and take assembly.

**Output:** pose-augmented location chunks, with root orientations, metric root translations, 23 parent-relative body rotations, metric joints, pose masks, frame-matching diagnostics, and alignment quality fields.

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

Add `--take_name TAKE_NAME` to process a single take. Change the input/output names to process Health and Cooking. CPU execution is supported; a CUDA device can be selected explicitly if desired.

## Alignment and attachment

1. Recover each source SMPL state from the selected actor, preserving its local body rotations and shape.
2. Recover the rigid relationship between the source SMPL surface and the corresponding scene-aligned mesh. A WHAM-native world trajectory is not automatically an Ego-Exo4D scene trajectory.
3. Use the location sample's inherited reference timestamp to recover the shared F0. Transform the root orientation, root translation, and joints together.
4. Find the nearest scene frame to each retained history/target timestamp, then check its time error and availability of a usable source SMPL state. An invalid nearest frame is rejected; the current implementation does not search onward for a different valid frame.
5. Preserve validity masks and alignment diagnostics. Missing frames or failed attachments are not valid pose supervision.

The primary `root_trans_f0_m` and `joints_f0_m` fields are metric; auxiliary normalized fields are stored separately. Root translation refers to the SMPL native pelvis in metric F0 coordinates. It is not interchangeable with a normalized hand position or the SMPL library's raw translation offset without accounting for the model's native pelvis.

The saved benchmark joint positions use the supplied 19-joint regressor. This does not change the 23 body-joint rotations stored for each state.

The original `target_mask` describes location validity; `target_pose_mask` separately describes successful pose attachment. Use their intersection for paired targets. The historical processing behavior can retain a chunk without pose fields when a take input or F0 is unavailable; such a chunk must be rejected by a paired-data consumer. Inspect the failure report rather than assuming that every record in the output has usable pose supervision.

## Quality fields

The source builder records reconstruction, rigid-fit, and hand-location agreement diagnostics and derives data-quality weights from them. Its defaults are:

| Argument | Value |
|---|---:|
| `quality_source_tau_mm` | 10.0 |
| `quality_rigid_tau_mm` | 5.0 |
| `quality_hand_tau_mm` | 120.0 |
| `min_quality_weight` | 0.05 |

These are annotation-quality settings, not a guarantee of exact hand–object contact. Keep the emitted configuration and diagnostics with generated annotations.
