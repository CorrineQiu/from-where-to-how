# Annotation release review

Review date: 2026-09-08.

This is a scoped repository-content review, not a legal clearance or a full validation of every training sample. It records why this update includes aggregate metadata but holds back per-sample annotations.

## Release decisions

| Candidate material | Decision for this update | Reason |
|---|---|---|
| Dataset statistics from Table I | Include | Already reported in the manuscript and project page; aggregate counts only |
| Task distribution source CSV | Include | 21 task rows; totals match the reported samples, targets, and takes; no record-level identifiers or machine paths |
| Narration verb distribution source CSV | Include | 619 domain–verb rows; totals match the reported targets; no complete narration text |
| Final Coherent4D take split manifests | Hold | Must be extracted and validated against the exact final filtered dataset |
| Legacy FIction take lists | Do not copy as Coherent4D splits | The inspected lists contain 851 takes, whereas the reported Coherent4D snapshot has 787 |
| Legacy `rgb_feature_paths*.json` | Do not copy | Environment-specific feature paths are not portable dataset annotations |
| Nine pose-augmented train/validation/test PKLs | Hold | About 13.8 GB in total; require record-level provenance, alignment, split, schema, and access review |
| Raw narrations, source media, and scene/pose intermediates | Hold | Source-data terms and record-level disclosure must be checked before redistribution |
| SMPL model files, pretrained model weights, and extracted feature tensors | Exclude | Outside this documentation update; obtain third-party assets through their official channels |
| Training, inference, and preprocessing code | Exclude | Code release is not part of this update |

The existing website demonstration media are unchanged by this review. Their presence does not establish permission to redistribute an entire source dataset or annotation archive.

## Checks on the included metadata

| Domain | Task-table samples | Valid targets in both aggregate sources | Takes in task metadata |
|---|---:|---:|---:|
| Cooking | 166,041 | 1,337,589 | 331 |
| Health | 27,598 | 114,964 | 205 |
| Bike Repair | 40,189 | 141,633 | 251 |
| Total | 233,828 | 1,594,186 | 787 |

The CSV headers and fields were inspected for machine-specific paths, take/person identifiers, full narration text, and media references. No such fields are included in these two aggregate sources. This does not imply that the underlying per-sample PKLs passed those checks.

The figure-source files are preserved byte-for-byte. SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `dataset_task_sunburst_stats.csv` | `2cde46d08a829f8cd63a739aaf7b4dab10de453bd373b8ca97940c155e3f2775` |
| `dataset_narration_verb_stats.csv` | `6731c9571bf3fb1a08738ab2e89b00ee836aab2ffe701c19ecc6b9aa676bda39` |

`dataset_statistics.csv` is transcribed from Table I rather than claimed to be freshly recomputed from every training record.

## Before releasing per-sample annotations

- [ ] Identify the exact dataset snapshot and export configurations used for the reported experiments.
- [ ] Generate portable split and sample manifests from that snapshot; verify take-level disjointness and counts.
- [ ] Verify the WHAM-to-scene alignment path and shared spatial-reference provenance for every included pose record.
- [ ] Verify hand-location normalization, metric root/joint fields, and all coordinate-conversion conventions.
- [ ] Check time origin, history boundaries, pose matching error, masks, and paired-target validity.
- [ ] Freeze event merging, debouncing, actor selection, and pose matching settings; do not substitute historical script defaults.
- [ ] Remove credentials, local absolute paths, unnecessary source text, and nonessential personal metadata.
- [ ] Review applicable Ego-Exo4D and third-party terms and choose an authorized distribution mechanism.
- [ ] Publish a versioned schema, archive checksums, and a clearly documented access procedure.

Ego-Exo4D requires its [official access agreement](https://docs.ego-exo4d-data.org/getting-started/), which includes redistribution restrictions. This review does not infer that all derived annotations can be made public simply because a related repository publishes split lists. WHAM also directs users to obtain [SMPL assets through registration](https://github.com/yohanshin/WHAM#registration).

Documentation was prepared with AI assistance and checked against the supplied manuscript, inspected preprocessing interfaces, and linked upstream documentation. Final annotation-release approval remains with the dataset authors.
