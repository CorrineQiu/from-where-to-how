# Annotation validation notes

Updated: 2026-09-08.

This document summarizes the included metadata, its validation, and requirements for per-sample annotation exports.

## Annotation file status

| Material | Availability | Validation notes |
|---|---|---|
| Dataset statistics from Table I | Included | Already reported in the manuscript and project page; aggregate counts only |
| Task distribution source CSV | Included | 21 task rows; totals match the reported samples, targets, and takes |
| Narration verb distribution source CSV | Included | 619 domain–verb rows; totals match the reported targets |
| Final Coherent4D take split manifests | Not included | Must be extracted and validated against the exact final filtered dataset |
| Legacy FIction take lists | Not included | The inspected lists contain 851 takes, whereas the reported Coherent4D snapshot has 787; they cannot serve as Coherent4D splits |
| Legacy `rgb_feature_paths*.json` | Not included | Environment-specific feature paths are not portable dataset annotations |
| Nine pose-augmented train/validation/test PKLs | Not included | About 13.8 GB in total; require per-sample provenance, alignment, split, and schema validation |
| Raw narrations, source media, and scene/pose intermediates | Not included | Inputs and intermediate outputs of the annotation pipeline, rather than portable final sample annotations |
| SMPL model files, pretrained model weights, and extracted feature tensors | Not included | Model assets and features, rather than dataset annotations |

## Checks on the included metadata

| Domain | Task-table samples | Valid targets in both aggregate sources | Takes in task metadata |
|---|---:|---:|---:|
| Cooking | 166,041 | 1,337,589 | 331 |
| Health | 27,598 | 114,964 | 205 |
| Bike Repair | 40,189 | 141,633 | 251 |
| Total | 233,828 | 1,594,186 | 787 |

The two figure-source CSVs contain aggregate task and verb counts. Per-sample PKL validation remains pending.

The figure-source files are preserved byte-for-byte. SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `dataset_task_sunburst_stats.csv` | `2cde46d08a829f8cd63a739aaf7b4dab10de453bd373b8ca97940c155e3f2775` |
| `dataset_narration_verb_stats.csv` | `6731c9571bf3fb1a08738ab2e89b00ee836aab2ffe701c19ecc6b9aa676bda39` |

`dataset_statistics.csv` is transcribed from Table I of the manuscript.

## Per-sample annotation validation

- [ ] Identify the exact dataset snapshot and export configurations used for the reported experiments.
- [ ] Generate portable split and sample manifests from that snapshot; verify take-level disjointness and counts.
- [ ] Verify the WHAM-to-scene alignment path and shared spatial-reference provenance for every included pose record.
- [ ] Verify hand-location normalization, metric root/joint fields, and all coordinate-conversion conventions.
- [ ] Check time origin, history boundaries, pose matching error, masks, and paired-target validity.
- [ ] Freeze event merging, debouncing, actor selection, and pose matching settings; do not substitute historical script defaults.
- [ ] Replace machine-specific paths with portable take and sample references.
- [ ] Document the annotation schema and provide archive checksums.
