# Shared coordinate construction

[Pipeline overview](../README.md) · [Previous: object grounding](../01_scene_object_grounding/) · [Next: location sequences](../03_location_sequence_construction/)

## Purpose

Interaction locations, scene objects, and full-body motion must use the same origin and axes to provide geometrically coupled annotations. Normalizing an array is not a substitute for aligning its coordinate frame.

## Inputs and outputs

**Inputs:** world-space hand/scene geometry, device poses on the take timeline, and the original sample observation endpoint. Reconstructed human geometry must already be related to the Ego-Exo4D scene frame.

**Outputs:** a reference rotation `R0`, a reference translation `t0` in meters, local point/orientation representations, and source-reference provenance carried into each chunk.

## Transformation

For column-vector notation:

```text
p_local = R0.T @ (p_world - t0)
R_local = R0.T @ R_world
p_normalized = clip(p_local / 5.0, -1, 1)
```

Apply the point transform to hand locations, object centers, root translations, and joints. Apply the orientation transform to global root and object orientations. Object dimensions and parent-relative body rotations are unchanged by this rigid transform.

Hand locations are stored in normalized coordinates. The exported environment fields use `center = clip(center_local / 5, -1, 1)` and `size = clip(size_m / 5, 0, 1)`; `rot6d` encodes the box orientation in F0. Physical dimensions are unchanged by the rigid transform, but the serialized `size` field is normalized, not measured in meters.

The primary pose fields `root_trans_f0_m` and `joints_f0_m` remain in **meters**, in the same local frame. The builder also stores auxiliary normalized fields. Clipping is not invertible outside the original range.

## Implementation

The coordinate code is part of the construction scripts rather than a separate pass that would estimate a second reference:

- [`build_interactions.py`](../03_location_sequence_construction/build_interactions.py): `compute_F0_from_last_second`, `world_to_F0`, `normalize_xyz`, and object-box conversion.
- [`attach_smpl.py`](../04_smpl_state_attachment/attach_smpl.py): reference recovery, global orientation/root/joint transformation, and pose-to-location alignment checks.

The inspected interaction builder collects device poses in the final second before the **source sample's** observation endpoint, chooses the most stable ten-frame window when enough poses exist, and averages its translation and quaternion orientation. If insufficient poses are available, it falls back to a nearby single frame. The stored diagnostic metadata distinguishes these cases.

After chunking, pose attachment uses `source_meta.source_observation_end_time_s` as its spatial-reference timestamp. The chunk's own observation endpoint is only a fallback when that provenance is absent. Do not independently recenter pose at the chunk endpoint while retaining locations from the source frame.

Pose attachment re-estimates F0 from the source pose stream at this inherited timestamp. Keep the interaction builder's `full_fps`, `last_sec_for_pose_avg`, and `pose_avg_win_frames` equal to the pose builder's `fps`, `last_sec`, and `win_frames`, respectively, and use the same `pos_scale_m` in both. Matching timestamps alone is insufficient if these settings or source geometry differ.

WHAM's native world coordinates are not assumed to equal the scene frame. Pose attachment uses the available scene-aligned human vertices and the reconstructed SMPL surface to recover their rigid relationship before expressing the state in F0. Its alignment errors and quality fields should be inspected alongside the pose masks.
