# Forecast sample generation

[Pipeline overview](../README.md) · [Previous: SMPL attachment](../04_smpl_state_attachment/)

## Purpose

Convert irregular interaction sequences into a fixed-step interface without discarding actual timestamps or treating padded targets as observations.

## Inputs and outputs

**Input:** a trusted `*_F0norm.pkl` dictionary containing a `dataset` list, source metadata, continuous hand trajectories, and ordered future interactions, produced by [step 3](../03_location_sequence_construction/).

**Output:** a chunk-cache dictionary with `dataset` and configuration metadata. Each sample contains its observation bounds, history interactions, padded targets, target mask, and inherited spatial-reference provenance. This location chunk is then passed to [step 4](../04_smpl_state_attachment/) for pose attachment.

| Domain | Target steps |
|---|---:|
| Cooking | 10 |
| Health | 5 |
| Bike Repair | 4 |

These horizons count interaction events, not seconds.

## Run

For example, to build Bike Repair chunks:

```bash
python process/05_forecast_sample_generation/build_chunks.py \
  --in_cache work/test_bike_F0norm.pkl \
  --out_cache work/test_bike_chunk4.pkl \
  --chunk_size 4 \
  --observation_time 30 \
  --include_history 1 \
  --pad_to_chunk 1 \
  --obs_dense_dt 3.0 \
  --obs_pos_eps_f0 0.12 \
  --obs_compress_mode first
```

Set `--chunk_size 5` for Health or `--chunk_size 10` for Cooking. Process each source split independently. The builder does not invent a new take split; take-level separation must already be defined by the inputs.

## Time, history, and padding

The first target's absolute take timestamp becomes `observation_end_time_s`, the temporal origin of the chunk. Future entries preserve `time_s` and `time_rel_s = time_s - observation_end_time_s`; the first relative timestamp is zero.

History is drawn from the preceding 30 seconds and excludes the target timestamp. It combines earlier interaction events with compressed samples of the observed hand trajectory. Trajectory-only history carries the placeholder `__obs__` and should not be interpreted as confirmed object contact.

The history compression parameters above are **3 seconds and 0.12 meters**. They are separate from the future-event parameters in the source interaction cache; do not silently equate the two configurations.

Incomplete tail chunks are padded, and their `target_mask` entries are zero. The source observation endpoint is preserved in `source_meta` so that pose attachment can recover the same spatial F0 as the original location annotations.

## Optional sharding

Use `--num_shards N --shard_id i` to construct a shard, with `i` from zero to `N-1`. After all shards complete, run the same output path with `--merge_only 1 --num_shards N`. Omit these options to use the single-process command above.
