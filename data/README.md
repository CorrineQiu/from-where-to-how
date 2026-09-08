# Coherent4D data and annotations

[Repository](../README.md) · [Construction pipeline](../process/README.md) · [Annotation field guide](ANNOTATION_GUIDE.md)

This directory contains **aggregate metadata and annotation documentation**, not the full training dataset. Per-sample interaction and pose annotations, final take-level split manifests, model weights, and feature tensors are not included.

## Included files

| File | Contents | Provenance |
|---|---|---|
| [dataset_statistics.csv](dataset_statistics.csv) | Domain horizons, sample split counts, takes, valid targets, and object-label counts | Table I of the supplied manuscript; counts also appear on the project page |
| [dataset_task_sunburst_stats.csv](dataset_task_sunburst_stats.csv) | Counts for all 21 procedural tasks | Existing source CSV for the dataset distribution figure |
| [dataset_narration_verb_stats.csv](dataset_narration_verb_stats.csv) | 619 domain–verb rows describing future-target verb frequencies | Existing source CSV for the narration verb distribution figure |
| [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) | Meaning of the main annotation fields, coordinate conventions, timestamps, and masks | Manuscript and inspected local preprocessing interfaces |
| [RELEASE_REVIEW.md](RELEASE_REVIEW.md) | Checks performed and decisions about candidate annotation files | Documentation release review dated 2026-09-08 |

The two figure-source CSVs are copied without changing their contents. They contain aggregate category counts, not complete narrations, take identifiers, machine-specific paths, or per-person motion records. Original category spellings are retained for consistency with the figures.

## Reading the metadata

### Dataset statistics

`dataset_statistics.csv` reports forecasting samples, not video frames. `valid_future_targets` excludes padded future steps. `future_steps` is the number of target events per sample, not a duration in seconds. The `Total` row has no single forecasting horizon.

`interaction_object_labels` counts distinct labels within each domain. The overall 535 labels are a union across domains, not the sum of the three domain counts.

### Task distribution

- `domain`, `task_name`: the procedural domain and task label.
- `chunk_sequences`: the number of forecasting samples for this task.
- `percent_of_sequences`: the percentage of all 233,828 samples, not a within-domain percentage.
- `valid_future_targets`: the number of valid future targets associated with those samples.
- `unique_takes`: the number of source takes for that task.

### Verb distribution

- `domain`, `verb`: the domain and normalized first narration verb.
- `future_targets`: the number of valid future targets associated with that verb.
- `percent_of_future_targets`: the percentage of all 1,594,186 valid future targets, not a within-domain percentage.

Percentages are rounded. The 619 rows count domain–verb combinations and should not be interpreted as 619 distinct verbs across the whole dataset.

## Access to source data

Coherent4D is constructed from [Ego-Exo4D](https://ego-exo4d-data.org/). Obtain source videos, calibration, trajectories, point clouds, and annotations through the [official access procedure](https://docs.ego-exo4d-data.org/getting-started/), subject to the applicable agreement. This repository does not provide credentials or bypass that access process.

WHAM and SMPL resources should likewise be obtained from their [upstream sources](../process/README.md#models-and-resources). We do not bundle their pretrained weights or SMPL model files.

## Planned annotation release

Following the lightweight organization of [FIction's data directory](https://github.com/thechargedneutron/FIction/tree/main/data), the most useful additions would be final Coherent4D take-level split manifests and portable sample indices. They must be generated from the exact filtered dataset snapshot used for the reported experiments. FIction's original take lists and absolute feature paths are not substitutes for these files.

Before any per-sample release, we need to verify provenance, split disjointness, coordinate units, timestamp alignment, masks, portable paths, and redistribution terms. The full annotation package should be separately versioned with checksums and a documented schema. No download link or release date is promised here before those checks are complete.
