# Scene object grounding

[Pipeline overview](../README.md) · [Next: shared coordinates](../02_shared_coordinate_construction/)

## Purpose

Associate scene objects with semantic labels and continuous metric geometry. These annotations describe object identity, location, and extent; they are not learned feature tokens.

## Inputs and outputs

**Inputs:** calibrated Aria views, the [LVIS vocabulary](https://www.lvisdataset.org/), and Ego-Exo4D SLAM resources including `semidense_points.csv.gz`, `semidense_observations.csv.gz`, and `online_calibration.jsonl`.

**Output:** one `<take_name>_object_bbs_obb.pkl` file per take, containing:

| Field | Content |
|---|---|
| `rrc` | Object box corners, shape N × 3 × 8, in scene/world meters |
| `object_names` | Semantic names associated with boxes |
| `object_counts` | Supporting observation counts used for object selection |

The subsequent take assembler combines these boxes with aligned human geometry and interaction annotations.

## Upstream processing

Follow the [FIction preparation guide](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md) with its [FIction-Detic integration](https://github.com/thechargedneutron/FIction-Detic), based on [Detic](https://github.com/facebookresearch/Detic).

1. Detect LVIS objects in calibrated egocentric images. The inspected local adaptation uses Aria left/right SLAM-camera streams and subsamples every six frames.
2. Use `map_3d_to_detic_objects.py` to associate observed SLAM point IDs with detected image regions. This produces per-take mapping tables.
3. Group associated points by semantic category, discard noise, and construct object boxes. The inspected Coherent4D adaptation uses DBSCAN with `eps=0.5` meters and `min_samples=100`, percentile trimming, and an XY principal-axis fit with vertical Z. These are yaw-oriented boxes, not unrestricted 3D principal-axis boxes.
4. Combine the object boxes with human and narration intermediates using the upstream `PuttingEverythingIn3D` preparation stage. The inspected assembly retains up to 30 objects ranked by supporting count.
5. Pass the resulting scene-aligned take records to [location construction](../03_location_sequence_construction/). Box centers and orientations are then expressed in the common F0 frame.

Run detection and SLAM association with the linked upstream tools, configuring their checkpoints, environments, and input paths. This folder describes the resulting data interface consumed by the subsequent construction scripts.

Continue with [shared coordinate construction](../02_shared_coordinate_construction/).
