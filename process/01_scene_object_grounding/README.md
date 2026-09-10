# Scene object grounding

Associate scene objects with semantic labels and continuous 3D geometry to provide spatial context for interaction annotations.

## Inputs and outputs

**Inputs:** calibrated Aria views, the [LVIS vocabulary](https://www.lvisdataset.org/), and Ego-Exo4D SLAM resources including `semidense_points.csv.gz`, `semidense_observations.csv.gz`, and `online_calibration.jsonl`.

**Output:** one `<take_name>_object_bbs_obb.pkl` file per take, containing:

| Field | Content |
|---|---|
| `rrc` | Object box corners, shape N × 3 × 8, in scene/world meters |
| `object_names` | Semantic names associated with boxes |
| `object_counts` | Supporting observation counts used for object selection |

## Processing

Follow the [FIction preparation guide](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md) with its [FIction-Detic integration](https://github.com/thechargedneutron/FIction-Detic), based on [Detic](https://github.com/facebookresearch/Detic).

1. Detect LVIS objects in Aria left/right SLAM-camera images, sampling every six frames.
2. Use `map_3d_to_detic_objects.py` to associate observed SLAM point IDs with detected image regions. This produces per-take mapping tables.
3. Cluster points by semantic category using DBSCAN with `eps=0.5` meters and `min_samples=100`. Apply percentile trimming and an XY principal-axis fit with vertical Z to obtain yaw-oriented boxes.
4. Combine boxes with aligned human geometry and interaction annotations using the upstream `PuttingEverythingIn3D` stage. Retain up to 30 objects ranked by supporting count.
5. Pass the scene-aligned take records to [location construction](../03_location_sequence_construction/), where box centers and orientations are transformed into the shared F0 frame.

Configure the linked upstream tools with your local input paths and model checkpoints before running detection and SLAM association.
