# Shared coordinate construction

Express interaction locations, scene objects, and full-body motion with the same origin and axes to obtain geometrically aligned annotations.

## Inputs and outputs

**Inputs:** hand and object geometry aligned to the Ego-Exo4D scene frame, device poses, and the source sample's observation endpoint.

**Outputs:** a reference rotation `R0`, translation `t0` in meters, and local geometry with reference metadata inherited by each chunk.

## Transformation

```text
p_local = R0.T @ (p_world - t0)
R_local = R0.T @ R_world
p_normalized = clip(p_local / 5.0, -1, 1)
```

Apply the point transform to hand locations, object centers, root translations, and joints. Apply the orientation transform to global root and object orientations. Object dimensions and parent-relative body rotations are unchanged by this rigid transform.

Hand locations use `p_normalized`. Environment fields store `center = clip(center_local / 5, -1, 1)`, `size = clip(size_m / 5, 0, 1)`, and the F0 box orientation as `rot6d`. The primary pose fields `root_trans_f0_m` and `joints_f0_m` remain in **meters**; auxiliary normalized fields are stored separately.

## Implementation

[`build_interactions.py`](../03_location_sequence_construction/build_interactions.py) estimates F0 from the most stable ten-frame window in the final second before the source observation endpoint, averaging translation and quaternion orientation. It uses a nearby single pose when too few poses are available.

[`attach_smpl.py`](../04_smpl_state_attachment/attach_smpl.py) aligns the reconstructed SMPL surface to scene-aligned human vertices and recovers F0 using `source_meta.source_observation_end_time_s`. Chunking changes the time origin while retaining this spatial reference.

Use matching settings in the two scripts: `full_fps` / `fps`, `last_sec_for_pose_avg` / `last_sec`, `pose_avg_win_frames` / `win_frames`, and `pos_scale_m`.
