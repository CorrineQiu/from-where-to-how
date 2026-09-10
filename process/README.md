# Data preparation

Construct Coherent4D from [Ego-Exo4D](https://docs.ego-exo4d-data.org/getting-started/) using the upstream [FIction preparation pipeline](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md) and the steps below.

![Coherent4D annotation pipeline](annotation_pipeline.png)

## Extract source poses, objects, and interactions

- Use [WHAM](https://github.com/yohanshin/WHAM) to reconstruct SMPL poses, then select the actor with `find_actor_from_outputs_largest_area.py` to obtain `<take_name>_largest_area.pkl`.
- Use [Detic](https://github.com/facebookresearch/Detic) and SLAM geometry to obtain `<take_name>_object_bbs_obb.pkl`. See [scene object grounding](01_scene_object_grounding/).
- Use [Llama 3](https://github.com/meta-llama/llama3) to match narrated interactions. Run FIction's `InteractionFromNarration/llama3.py` and `parse_outputs.py`, then combine the outputs with `PuttingEverythingIn3D/main.py` to obtain scene-aligned take records.

## Construct shared coordinates and location sequences

Run [`build_interactions.py`](03_location_sequence_construction/) on the assembled take records to transform hand locations and object geometry into the [shared local frame](02_shared_coordinate_construction/), merge repeated interactions, and save `*_F0norm.pkl`.

## Generate forecasting samples

Run [`build_chunks.py`](05_forecast_sample_generation/) to form sequences of 10 events for Cooking, 5 for Health, or 4 for Bike Repair. Each chunk retains its observation history, absolute/relative timestamps, and padding mask.

## Attach full-body poses

Run [`attach_smpl.py`](04_smpl_state_attachment/) on the location chunks and selected WHAM reconstructions. It pairs each timestamp with SMPL rotations, metric root translations, 19 joint positions, and a pose-validity mask in the shared local frame.

## Run

Use Python 3.9 and install the dependencies from the repository root:

```bash
python -m pip install -r process/requirements.txt
```

The linked step folders provide the commands and input paths. Execution order: `build_interactions.py` → `build_chunks.py` → `attach_smpl.py`.
