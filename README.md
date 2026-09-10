<h1 align="center">From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video</h1>

<p align="center">
  <a href="mailto:qiaohuichu8599@gmail.com">Qiaohui Chu</a><sup>1,2</sup>,
  Haoyu Zhang<sup>1,2</sup>,
  Meng Liu<sup>3,*</sup>,
  Haoxiang Shi<sup>1,2</sup>,
  Dongmei Jiang<sup>2</sup>,
  Liqiang Nie<sup>1,*</sup>
</p>

<p align="center">
  <sup>1</sup> Harbin Institute of Technology (Shenzhen) &nbsp;&nbsp;
  <sup>2</sup> Pengcheng Laboratory<br>
  <sup>3</sup> Shandong University
</p>

<p align="center"><sup>*</sup> Corresponding authors</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.08636"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="Paper on arXiv" height="28"></a>
  <a href="https://corrineqiu.github.io/from-where-to-how/"><img src="https://img.shields.io/badge/Project-From_Where_to_How-007EC6?style=flat" alt="From Where to How project page" height="28"></a>
</p>

Coherent4D pairs continuous 3D interaction locations with temporally aligned full-body poses. HIGFlow models the corresponding forecasting problem as a cascaded *where-to-how* process.

## Coherent4D dataset

| Domain | Future steps | Takes | Train | Validation | Test | Total samples |
|---|---:|---:|---:|---:|---:|---:|
| Cooking | 10 | 331 | 136,079 | 15,435 | 14,527 | 166,041 |
| Health | 5 | 205 | 21,532 | 1,899 | 4,167 | 27,598 |
| Bike Repair | 4 | 251 | 35,987 | 3,150 | 1,052 | 40,189 |
| Total | — | 787 | 193,598 | 20,484 | 19,746 | 233,828 |

The dataset reported in the manuscript covers 21 procedural tasks and 535 interaction-object labels, with 1,594,186 valid future targets.

The [`data/`](data/README.md) directory contains aggregate split, task, and verb statistics.

## Data construction

Follow [Data preparation](process/README.md) to extract source annotations, construct shared coordinates and interaction sequences, generate forecasting samples, and attach SMPL poses. Each step includes the required tools, input/output files, and commands.

## Acknowledgments

We thank the [Ego-Exo4D](https://ego-exo4d-data.org/) team and participants for making this research possible, and acknowledge [FIction](https://github.com/thechargedneutron/FIction) and the upstream annotation tools described in the construction guide.
