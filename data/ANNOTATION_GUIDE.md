# Annotation field guide

[Data overview](README.md) · [Construction pipeline](../process/README.md)

This guide documents the main fields found in the inspected preprocessing and pose-loading interfaces. It is **not a frozen release schema**, and no per-sample annotation file is included with this guide. Internal variants may contain additional fields; a future release must specify its exact schema and provenance.

## Sample identity and context

| Field or group | Meaning |
|---|---|
| `id` | Forecasting sample identifier |
| `take_name` | Source take identifier; splitting must keep a take within one split |
| `datum_idx`, `chunk_idx` | Indices linking a chunk to its source sample |
| `observation_start_time_s`, `observation_end_time_s` | Observation bounds on the source take timeline, in seconds |
| `env_objects` | Structured object context; category and continuous box geometry, not learned tokens |
| `history_interactions` | Time-ordered observed hand-location history and available earlier interaction events |
| `target_interactions` | Ordered future interaction targets, padded when required |
| `target_mask` | Validity of future interaction steps; padded steps are zero |
| `source_meta` | Provenance information, including the original source observation endpoint when carried through chunk construction |

Observed trajectory points can carry the placeholder `__obs__`. Such points are hand-location observations and should not automatically be treated as confirmed object-contact events.

## Interaction entries

| Field | Meaning and units |
|---|---|
| `time_s` | Absolute timestamp within the source take, in seconds; not a calendar or Unix time |
| `time_rel_s` | Seconds relative to the first future interaction timestamp of this chunk |
| `location_norm` | Continuous 3D hand-location proxy in the sample-local frame, divided by 5 m and clipped per coordinate to `[-1, 1]` |
| `object` | Associated semantic object label(s), when available |

Earlier preprocessing records can also retain metric positions such as `location_f0_m`. A padded target may have null time/location/object entries and must be ignored according to its mask.

The inspected preprocessing uses fixed SMPL mesh vertex **5777** on the right hand as the location proxy. This is not the SMPL wrist joint, the hand centroid, a complete hand contour, or an explicitly recovered surface-contact point. Right-hand, both-hand, and unspecified-hand annotations are mapped to the unified right-hand proxy stream by the inspected parser.

## Attached pose fields

The prefix `history_` refers to observed history, and `target_` refers to future steps. For example, `target_root_trans_f0_m` stores future root translations.

| Field suffix | Per-state representation | Convention |
|---|---|---|
| `root_orient6d_f0` | 6 values | Global root orientation represented in the sample-local frame |
| `root_trans_f0_m` | 3 values | Root translation in meters |
| `body_pose_local6d` | 23 × 6 values | Non-root body-joint rotations relative to their parents |
| `joints_f0_m` | J × 3 values | 3D joint positions in meters; the serialized joint layout must be specified by the release |
| `pose_mask` | Validity flag | Whether the corresponding pose attachment is valid |

The model packs root orientation, root translation, and the 23 local joint rotations into a 147-dimensional SMPL state. Its separate 19-joint evaluation subset should not be confused with either the 23 local joint rotations or the complete serialized joint array.

## Coordinate and time conventions

- **Spatial reference:** interaction locations, object geometry, SMPL root motion, and 3D joints refer to the same sample-local origin and axes after alignment.
- **Numeric scale:** hand-location conditions are normalized; root translations and 3D joint positions used by the pose branch remain in meters. Pose translation residuals are also computed and applied in meters.
- **Object geometry:** rigid frame transformation changes centers and global orientations but not physical box dimensions.
- **Temporal reference:** the first future interaction has relative time zero. Observed history is earlier than zero, and later future events have positive offsets. Inter-event spacing is not assumed uniform.
- **Spatial versus temporal anchors:** resetting a chunk's relative-time origin does not by itself recompute its inherited spatial reference pose. Retain the source reference metadata needed to interpret its coordinates.
- **Masks:** all padded or invalid targets must be excluded from supervision and evaluation. A joint location/pose sample requires consistent validity and temporal correspondence across both targets.

Multiplying a normalized coordinate by 5 m returns its metric-scale value, but cannot recover an original coordinate that was lost through clipping. A release should retain metric annotations and document out-of-range handling when exact reversibility is required.
