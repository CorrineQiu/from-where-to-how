# Forecast sample generation

Group irregular interaction sequences into fixed-step samples while preserving timestamps and target validity.

## Inputs and outputs

**Input:** the `*_F0norm.pkl` cache from [location construction](../03_location_sequence_construction/), containing hand trajectories, future interactions, and source metadata.

**Output:** location chunks containing observation bounds, history interactions, padded targets, masks, and inherited spatial-reference metadata. Pass these chunks to [SMPL attachment](../04_smpl_state_attachment/) next.

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

Set `--chunk_size 5` for Health or `--chunk_size 10` for Cooking. Process each split independently using the existing take assignments.

## Time, history, and padding

The first target's absolute take timestamp becomes `observation_end_time_s`, the temporal origin of the chunk. Future entries preserve `time_s` and `time_rel_s = time_s - observation_end_time_s`; the first relative timestamp is zero.

History covers the preceding 30 seconds, excluding the first target timestamp. It combines earlier interactions with hand-trajectory samples compressed using **3 seconds and 0.12 meters**. Trajectory-only entries carry the placeholder `__obs__` rather than an interaction-object label.

Incomplete tail chunks are padded, and their `target_mask` entries are zero. The source observation endpoint is preserved in `source_meta` so that pose attachment can recover the same spatial F0 as the original location annotations.

## Optional sharding

Use `--num_shards N --shard_id i` to construct a shard, with `i` from zero to `N-1`. After all shards complete, run the same output path with `--merge_only 1 --num_shards N`. Omit these options to use the single-process command above.
