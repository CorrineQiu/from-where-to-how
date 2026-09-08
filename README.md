# From Where to How

**From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video**

Coherent4D pairs continuous 3D interaction locations with temporally aligned full-body poses. HIGFlow models the corresponding forecasting problem as a cascaded *where-to-how* process.

[Project page](https://corrineqiu.github.io/from-where-to-how/) · [Dataset metadata](data/README.md) · [Data construction](process/README.md) · [Annotation guide](data/ANNOTATION_GUIDE.md)

## Release status

This repository currently provides the project website, qualitative media, data construction documentation, and aggregate dataset metadata. **Training/inference code, model weights, and per-sample training annotations are not included in this release.**

The `data/` directory contains the task and verb statistics used for the paper figures, the reported split counts, and an annotation field guide. See the [annotation release notes](data/RELEASE_REVIEW.md) for metadata validation and planned annotation exports.

## Coherent4D dataset

| Domain | Future steps | Takes | Train | Validation | Test | Total samples |
|---|---:|---:|---:|---:|---:|---:|
| Cooking | 10 | 331 | 136,079 | 15,435 | 14,527 | 166,041 |
| Health | 5 | 205 | 21,532 | 1,899 | 4,167 | 27,598 |
| Bike Repair | 4 | 251 | 35,987 | 3,150 | 1,052 | 40,189 |
| Total | — | 787 | 193,598 | 20,484 | 19,746 | 233,828 |

The dataset reported in the manuscript covers 21 procedural tasks and 535 interaction-object labels, with 1,594,186 valid future targets.

## Data construction

Open [`process/`](process/README.md) for the annotation pipeline figure and a step-by-step explanation:

1. Scene Object Grounding
2. Shared Coordinate Construction
3. Location Sequence Construction
4. SMPL State Attachment
5. Forecast Sample Generation

The guide links the relevant upstream resources, including Ego-Exo4D, FIction, Detic, LVIS, Llama 3, WHAM, and SMPL. It distinguishes annotation construction from the Qwen3-VL and V-JEPA representations used for forecasting.

## Local preview

The page uses only local static assets. Either open `index.html` directly or run a local server:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Publish with GitHub Pages

The site is published directly from the root of the `main` branch. In
**Settings → Pages**, use **Deploy from a branch**, select `main`, and select
`/(root)` as the folder.

The current project-page URL is:

```text
https://corrineqiu.github.io/from-where-to-how/
```

## Repository structure

```text
.
├── assets/
├── data/
│   ├── README.md
│   ├── ANNOTATION_GUIDE.md
│   ├── RELEASE_REVIEW.md
│   ├── dataset_statistics.csv
│   ├── dataset_task_sunburst_stats.csv
│   └── dataset_narration_verb_stats.csv
├── process/
│   ├── README.md
│   └── annotation_pipeline.png
├── index.html
├── styles.css
├── script.js
├── iiith_cooking_58_2_idx6839_original_switchcam.mp4
└── iiith_cooking_58_2_idx6839_rendered_continuous_pose.mp4
```

Paper and code links in the webpage are placeholders until public URLs are available.

## Acknowledgments

We thank the [Ego-Exo4D](https://ego-exo4d-data.org/) team and participants for making this research possible, and acknowledge [FIction](https://github.com/thechargedneutron/FIction) and the upstream annotation tools described in the construction guide.
