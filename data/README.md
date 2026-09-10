# Coherent4D statistics

Aggregate statistics used in the paper and dataset visualizations.

| File | Contents |
|---|---|
| [dataset_statistics.csv](dataset_statistics.csv) | Sample splits, takes, future steps, valid targets, and object-label counts |
| [dataset_task_sunburst_stats.csv](dataset_task_sunburst_stats.csv) | Sample and target counts for 21 procedural tasks |
| [dataset_narration_verb_stats.csv](dataset_narration_verb_stats.csv) | Future-target frequencies by domain and narration verb |

Future steps count interaction events; valid-target counts exclude padding. The 535 object labels are unique across domains. Percentage columns use dataset-wide totals.

To construct the annotations, follow [Data preparation](../process/README.md).
