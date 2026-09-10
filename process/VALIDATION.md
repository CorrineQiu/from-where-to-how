# Construction validation

[Pipeline overview](README.md)

Validation date: 2026-09-10. These checks concern annotation processing, not forecasting-model performance.

## Verified scope

| Check | Result |
|---|---|
| Construction unit/CLI tests | 15 passed: hand proxy selection, array-valued annotation columns, event compression, shared F0, time origin, padding, masks, rigid transforms, and output/input protection |
| Real Bike Repair chunk construction | Three source records produced ten chunks, exactly matching the historical chunks by sample ID |
| Repeatability | The chunk construction was repeated twice with identical records |
| Real cached SMPL reconstruction | Three pose samples, totaling 22 valid history/target states, reconstructed on CPU |
| Shared loc/pose frame | World hand trajectories transformed using the cached pose F0 matched the saved local trajectories |
| Repeated SMPL reconstruction | Identical output-array SHA-256 across two runs |
| Full attachment CLI with synthetic intermediates | Two successful runs using synthetic scene/actor records and a real SMPL model asset; output, source F0, native pelvis, invalid/padded masks, and repeatability checked |
| Markdown | GitHub's Markdown rendering accepted the overview and its code-form coordinate expressions without the reported macro error |

The machine-readable [validation report](validation/bike_core_checks_20260910.json) records input and code hashes, environment, tested sample IDs, numerical errors, and the separate synthetic attachment integration check. It contains no source video, body-model assets, or per-frame geometry.

Maximum absolute errors in the real-cache checks:

| Quantity | Error in meters |
|---|---:|
| Reconstructed source-camera joints | 9.54e-7 |
| Reconstructed F0 joints | 6.56e-7 |
| Reconstructed F0 root translations | 2.98e-7 |
| Shared hand/pose F0 transform | 5.96e-8 |

The reconstruction comparison allows 5e-5 meters (0.05 mm) for numerical differences between the historical cache and CPU reconstruction. The repeatability comparison itself is exact. This small validation subset is **not** a full dataset audit.

## Reproduce the checks

Run the synthetic construction tests without source annotations:

```bash
python -m unittest discover -s process/tests -v
```

To repeat the real-intermediate checks with your local trusted caches and model assets:

```bash
python process/tests/validate_real_caches.py \
  --loc_cache inputs/test_bike_F0norm.pkl \
  --chunk_cache inputs/test_bike_chunk4.pkl \
  --pose_cache inputs/pose_bike_chunk4_test.pkl \
  --smpl_model_path inputs/body_models/smpl \
  --joint_regressor_path inputs/SMPL_to_J19.pkl \
  --chunk_size 4 --source_samples 3 --pose_samples 3 \
  --cpu_threads 4 \
  --output work/bike_validation.json
```

The checked environment used Python 3.9.7, NumPy 1.23.5, pandas 2.2.3, SciPy 1.10.1, joblib 1.4.2, tqdm 4.67.1, SMPL-X 0.1.28, and PyTorch 2.6.0+cu118. The reported construction checks used CPU only.

## Source provenance and packaging changes

| Published file | Historical source |
|---|---|
| [build_interactions.py](03_location_sequence_construction/build_interactions.py) | `My_Code_qwen/build_interaction_json_new_0112.py` |
| [build_chunks.py](05_forecast_sample_generation/build_chunks.py) | `My_Code_qwen/dataset_continuous/build_cooking_chunk_10_new.py` |
| [attach_smpl.py](04_smpl_state_attachment/attach_smpl.py) | `My_Code_qwen/data_pose_chunk/build_pose_chunks_route6_anchor_fixed_0316.py` |

Each script includes its original source SHA-256 and writes that provenance into the generated cache configuration. Publication changes make paths explicit, add bounded validation options and input/output guards, and remove machine-specific launch examples. A pandas compatibility fix expands array-valued annotation columns without requiring them to be hashable. Single-process chunk construction now writes directly to `--out_cache`; multi-shard merging retains the historical shard order.

The underlying hand proxy, metric spatial thresholds, source F0 selection, chunk timestamp convention, parent-relative body rotations, and pose alignment logic are retained. Exact reproduction of historical IDs also depends on source filtering and record order, not only on numerical parameters.

The construction follows the upstream interfaces of [FIction](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md), [Detic](https://github.com/facebookresearch/Detic), [Llama 3](https://github.com/meta-llama/llama3), and [WHAM](https://github.com/yohanshin/WHAM). Third-party model source trees and model assets are not copied into these scripts.

## Limits and implementation details

- **No raw-video rerun:** Detic inference, SLAM association, narration-to-object inference, WHAM inference, scene registration, and complete dataset regeneration were not rerun. Step 1 describes the upstream interface, not a newly tested detector implementation.
- **Pose attachment versus reconstruction:** the real-cache test reconstructs already attached SMPL states and checks their shared frame. It does not reselect actors or rerun full-take attachment from the original WHAM/scene-file pairs. Some historical selected-actor and scene-cache paths were unavailable in the audited workspace.
- **Event geometry:** the published converter consumes assembled interaction annotations. The audited conversion path does not add a separate hand-to-object OBB or contact-distance filter. The paper's geometric-consistency description should not be read as a verified additional filter in this converter.
- **Nearest frame:** pose attachment checks the closest scene frame and its validity; it does not search for a further valid SMPL state after a failed nearest-frame attachment.
- **Shared reference:** the two builders must use matching frame/scale settings and source geometry. A common inherited timestamp alone does not ensure identical F0.
- **Invalid records:** the location and pose masks serve different purposes. Missing full-take inputs can leave records without pose fields; a paired-annotation consumer must reject those records and intersect valid masks.
- **Source configuration:** historical future-event and observed-trajectory compression thresholds differ. Llama model variants also exist in the upstream working scripts; no single raw-annotation model version is claimed to have been revalidated here.

Full per-sample export requirements remain documented in the [annotation validation notes](../data/RELEASE_REVIEW.md).
