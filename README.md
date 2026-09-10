# From Where to How

**From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video**

Coherent4D pairs continuous 3D interaction locations with temporally aligned full-body poses. HIGFlow models the corresponding forecasting problem as a cascaded *where-to-how* process.

<p align="center">
  <a href="https://arxiv.org/abs/2609.08636"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="Paper on arXiv" height="28"></a>
  <a href="https://corrineqiu.github.io/from-where-to-how/"><img src="https://img.shields.io/badge/Project-From_Where_to_How-007EC6?style=flat" alt="From Where to How project page" height="28"></a>
</p>

## Coherent4D dataset

| Domain | Future steps | Takes | Train | Validation | Test | Total samples |
|---|---:|---:|---:|---:|---:|---:|
| Cooking | 10 | 331 | 136,079 | 15,435 | 14,527 | 166,041 |
| Health | 5 | 205 | 21,532 | 1,899 | 4,167 | 27,598 |
| Bike Repair | 4 | 251 | 35,987 | 3,150 | 1,052 | 40,189 |
| Total | — | 787 | 193,598 | 20,484 | 19,746 | 233,828 |

The dataset reported in the manuscript covers 21 procedural tasks and 535 interaction-object labels, with 1,594,186 valid future targets.

The [`data/`](data/README.md) directory contains the task and verb statistics used for the paper figures, the reported split counts, and an annotation field guide. See the [metadata validation notes](data/RELEASE_REVIEW.md) for validation details and annotation export requirements.

## Data construction

Open [`process/`](process/README.md) for the annotation pipeline figure and a step-by-step explanation:

1. [Scene object grounding](process/01_scene_object_grounding/)
2. [Shared coordinate construction](process/02_shared_coordinate_construction/)
3. [Location sequence construction](process/03_location_sequence_construction/)
4. [SMPL state attachment](process/04_smpl_state_attachment/)
5. [Forecast sample generation](process/05_forecast_sample_generation/)

Each step has its own guide. The construction scripts build continuous interaction sequences, form fixed-step chunks, and attach aligned SMPL states. The [pipeline overview](process/README.md) explains their execution order and links upstream tools. See the [construction validation](process/VALIDATION.md) for tested behavior and source provenance.

## Data and annotation files

```text
.
├── data/
│   ├── README.md
│   ├── ANNOTATION_GUIDE.md
│   ├── RELEASE_REVIEW.md
│   ├── dataset_statistics.csv
│   ├── dataset_task_sunburst_stats.csv
│   └── dataset_narration_verb_stats.csv
└── process/
    ├── README.md
    ├── annotation_pipeline.png
    ├── 01_scene_object_grounding/
    ├── 02_shared_coordinate_construction/
    ├── 03_location_sequence_construction/
    │   └── build_interactions.py
    ├── 04_smpl_state_attachment/
    │   └── attach_smpl.py
    ├── 05_forecast_sample_generation/
    │   └── build_chunks.py
    ├── tests/
    ├── requirements.txt
    └── VALIDATION.md
```

## Acknowledgments

We thank the [Ego-Exo4D](https://ego-exo4d-data.org/) team and participants for making this research possible, and acknowledge [FIction](https://github.com/thechargedneutron/FIction) and the upstream annotation tools described in the construction guide.
