# Coherent4D data construction

[Repository](../README.md) · [Annotation metadata](../data/README.md) · [Annotation field guide](../data/ANNOTATION_GUIDE.md)

Coherent4D pairs continuous 3D interaction locations with temporally aligned full-body poses in a shared sample-local coordinate frame. This directory follows the five annotation stages in Section III-A of the [paper](https://arxiv.org/abs/2609.08636). Each stage has its own folder with inputs, outputs, implementation details, and the relevant tools.

## Annotation pipeline

![Coherent4D annotation pipeline](annotation_pipeline.png)

| Step | Guide | Input | Output |
|---|---|---|---|
| 1 | [Scene object grounding](01_scene_object_grounding/) | Egocentric frames, calibration, SLAM points | Semantic labels and metric 3D object boxes |
| 2 | [Shared coordinate construction](02_shared_coordinate_construction/) | Scene-aligned geometry and source reference poses | A common sample-local frame for locations, objects, and body motion |
| 3 | [Location sequence construction](03_location_sequence_construction/) | Narration matches and scene-aligned hand geometry | Timestamped continuous interaction sequences |
| 4 | [SMPL state attachment](04_smpl_state_attachment/) | Actor reconstructions, scene geometry, retained timestamps | Aligned root motion, body rotations, joints, and pose masks |
| 5 | [Forecast sample generation](05_forecast_sample_generation/) | Ordered locations and histories | Fixed-step samples with absolute/relative times and validity masks |

## Source preparation and execution order

Obtain the source takes and annotations through [Ego-Exo4D](https://docs.ego-exo4d-data.org/getting-started/). Required inputs include Aria video, synchronized exocentric views, camera calibration and trajectories, SLAM scene geometry, narrations, and take metadata. Keep all timestamps on the same take timeline.

The chapter order describes the annotations; it is not a shell execution order. Object grounding and WHAM reconstruction can run independently. The interaction builder needs scene-aligned human geometry, and the pose attachment script operates on already formed location chunks:

```text
Detic + SLAM ---------------------> object boxes -------+
                                                       |
Narrations + Llama object matching ---------------------+--> scene-aligned
                                                       |    take records
WHAM + actor selection + scene alignment --> body mesh -+
                                                            |
                               shared F0 + hand locations ---+
                                                            |
                                      build_interactions.py
                                                            |
                                         build_chunks.py
                                                            |
                         WHAM states + scene records --> attach_smpl.py
                                                            |
                              paired location / pose chunks with masks
```

The three construction scripts are provided in steps 3, 5, and 4, respectively. Step 2 documents the coordinate routines used by both builders. Step 1 documents the upstream grounding tools and the intermediate files consumed by these scripts. Third-party detector and motion reconstruction code is linked at its source.

## Running the construction scripts

Run commands from the repository root using Python 3.9. The dependency file includes NumPy, pandas, tqdm, joblib, PyTorch, and SMPL-X:

```bash
python -m pip install -r process/requirements.txt
python process/03_location_sequence_construction/build_interactions.py --help
python process/05_forecast_sample_generation/build_chunks.py --help
python process/04_smpl_state_attachment/attach_smpl.py --help
```

Each executable step's README gives its full command and expected input layout. SMPL model files and the 19-joint regressor are supplied through explicit paths for pose attachment. Only load pickle intermediates from trusted sources.

## Coordinate and timestamp conventions

| Annotation | Convention |
|---|---|
| Interaction location | `clip(p_local / 5.0, -1, 1)`, starting from a point in meters |
| Object geometry | Metric source boxes; exported centers and sizes are divided by 5 and clipped, with orientations in F0 |
| SMPL root translation and joints | Meters in the same local frame |
| SMPL root orientation | Rotated into the local frame |
| 23 body-joint rotations | Parent-relative rotations, unchanged by the global frame transform |
| Future time | Absolute take seconds and seconds relative to the first retained target |
| Observation history | The preceding 30 seconds, strictly before the first target |
| Padding | Explicit invalid slots, excluded from supervision and evaluation |

The source spatial reference is inherited by its chunks. Resetting a chunk's time origin does not recompute its spatial frame. Hand locations and pose root translations share an origin and axes, but not the same numeric scale.

The hand-location proxy is fixed right-hand SMPL mesh vertex **5777**, not a wrist joint or a measured surface-contact point. See the [annotation field guide](../data/ANNOTATION_GUIDE.md) for field definitions.
