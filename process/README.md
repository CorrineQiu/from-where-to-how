# Coherent4D data construction

[Repository](../README.md) · [Annotation metadata](../data/README.md) · [Annotation field guide](../data/ANNOTATION_GUIDE.md)

Coherent4D organizes continuous 3D interaction locations and full-body poses into time-aligned forecasting samples. The five stages below describe the annotation process in Section III-A of the manuscript.

## Annotation pipeline

![Five-stage Coherent4D annotation pipeline](annotation_pipeline.png)

The figure shows object grounding, a shared spatial reference, ordered hand locations, aligned body states, and the observation/forecast split. The visualization is an illustration of the annotations, not a model-generated prediction.

| Stage | Main input | Output passed to subsequent stages |
|---|---|---|
| 1. Scene Object Grounding | Egocentric frames, vocabulary, calibration, SLAM geometry | Semantic object labels and continuous 3D boxes |
| 2. Shared Coordinate Construction | World-space geometry and a sample reference pose | A common sample-local spatial reference |
| 3. Location Sequence Construction | Narrations, object matches, hand geometry | Ordered 3D interaction targets with timestamps |
| 4. SMPL State Attachment | Reconstructed actor motion and interaction timestamps | Paired location and full-body state sequences |
| 5. Forecast Sample Generation | Aligned histories and future sequences | Fixed-step samples with timestamps and validity masks |

These are logical stages rather than an executable scheduling dependency: motion reconstruction can run in parallel with object detection, and its hand geometry can be used when constructing interaction locations. Pose attachment then selects and aligns body states at the retained timestamps.

## Source preparation

Prepare the required [Ego-Exo4D resources](https://docs.ego-exo4d-data.org/getting-started/) for Cooking, Health, and Bike Repair. Relevant inputs include:

- The Aria egocentric video and synchronized exocentric views.
- Camera calibration, pose trajectories, and SLAM scene geometry.
- Timestamped narrations and source take metadata.

Keep video, narration, and pose timestamps on a consistent take timeline. The Aria stream provides the forecasting video input. Synchronized exocentric views support offline annotation construction and refinement; they are not future visual inputs to the forecasting model.

## 1. Scene Object Grounding

**Why:** interaction forecasting requires both the identity of an object and its location and extent in the scene. A category label alone does not provide the geometry needed to relate hand and body motion to that object.

**How:**

1. Apply [Detic](https://github.com/facebookresearch/Detic) with the [LVIS vocabulary](https://www.lvisdataset.org/) to identify candidate object regions in the egocentric stream.
2. Associate detected regions with Ego-Exo4D SLAM points using the available camera geometry and point observations. This establishes correspondence between image detections and scene-space geometry.
3. Cluster the associated 3D points and fit oriented bounding boxes to obtain spatial object instances.
4. Retain semantic categories together with continuous box geometry. Center, size, and orientation describe the resulting objects; an intermediate box file may encode this geometry using its corners instead.

**Output:** scene-grounded semantic objects and oriented 3D boxes. Their centers and orientations are later expressed in the shared local frame, while physical box dimensions remain unchanged by the rigid transform.

This is an annotation representation, **not a learned scene-token embedding**. Tokenization and feature encoding belong to the forecasting model rather than dataset construction.

## 2. Shared Coordinate Construction

**Why:** a hand location and a body pose can serve as coupled supervision only if they refer to the same origin and axes. Independently reconstructed camera or human coordinates cannot be compared directly without alignment.

**How:**

1. Bring camera-dependent observations and reconstructed human motion into the Ego-Exo4D scene/world reference using the appropriate calibration and motion-alignment procedure. A WHAM reconstruction's own world frame should not simply be assumed identical to the Ego-Exo4D scene frame.
2. Define a stable reference pose for each source sample, with rotation $R_{\mathrm{ref}}$ and translation $t_{\mathrm{ref}}$.
3. Map world-space points into the common sample-local frame:

$$
p^{\mathrm{loc}} = R_{\mathrm{ref}}^{\top}(p^{\mathrm w}-t_{\mathrm{ref}}).
$$

4. Apply this point transformation to interaction locations, object centers, SMPL root translations, and 3D joints. Transform global orientations with the same reference rotation. Keep the 23 non-root SMPL joint rotations relative to their parents, and keep object sizes unchanged.
5. Distinguish the shared spatial reference from the numerical scaling used by different model inputs:

| Quantity | Representation used downstream |
|---|---|
| Hand interaction locations | $\operatorname{clip}(p^{\mathrm{loc}}/s,-1,1)$, with $s=5\,\mathrm m$ |
| Pose root translations and 3D joints | Meters in the same sample-local frame |
| Root translation residuals | Meters during residual computation, clipping, and composition |
| Body joint rotations | Parent-relative rotations, not scaled by $s$ |

**Output:** geometrically aligned location, object, and body annotations with explicit units. Shared coordinates do not require every model tensor to have the same numerical scale.

A chunk can inherit the spatial reference of its source sample while resetting its temporal origin. Preserve this provenance rather than estimating a different spatial frame independently for the hand and pose streams. The exact reference and WHAM-to-scene alignment configuration must be fixed with the eventual annotation release.

## 3. Location Sequence Construction

**Why:** narration and interaction annotations are sparse events. Forecasting needs an ordered sequence of continuous 3D targets, without counting near-identical descriptions of the same interaction as separate locations.

**How:**

1. Use narration timestamps to identify candidate interaction events.
2. Use [Llama 3](https://github.com/meta-llama/llama3) to match the described object to the available semantic object vocabulary, following the interaction-mining approach of [FIction](https://github.com/thechargedneutron/FIction). Preserve missing-object cases when permitted by the construction rules rather than inventing an object identity.
3. Check consistency between the hand geometry and the candidate scene object. For each valid event, retain its timestamp, associated object when available, and continuous hand-location proxy.
4. The inspected preprocessing uses fixed right-hand SMPL mesh vertex **5777** as that proxy. It is not the SMPL wrist joint or an explicitly measured hand–object surface-contact point. Right-hand, both-hand, and unspecified-hand events are mapped to the unified proxy stream by the inspected parser.
5. Sort events chronologically. Apply the configured repeated-event and temporal/spatial merging checks. In the inspected merger, neighboring candidates associated with the same object set are compared by elapsed time and Euclidean separation between their metric hand locations.

For candidate times $t_i,t_j$ and metric local positions $p_i,p_j$, the temporal and spatial tests have the form

$$
|t_j-t_i|\leq\delta_{\mathrm{time}},\qquad
\lVert p_j-p_i\rVert_2\leq\delta_{\mathrm{position}}.
$$

Here the temporal threshold is in seconds and the spatial threshold is in meters. A threshold applied to normalized positions would need the corresponding scale conversion. Historical script defaults and example commands use different thresholds. The final export configuration must specify these values, any debounce interval, and the representative-event rule.

**Output:** ordered interaction events with continuous 3D locations and original timestamps. Event spacing can be nonuniform. Observed hand-location history can additionally contain compressed trajectory samples that are not individually labeled as object interactions.

## 4. SMPL State Attachment

**Why:** interaction locations describe *where* an action is directed but do not specify the corresponding whole-body configuration. Coupled forecasting requires a body state paired with each retained history or future timestamp.

**How:**

1. Reconstruct human motion with [WHAM](https://github.com/yohanshin/WHAM), using the [SMPL body model](https://smpl.is.tue.mpg.de/).
2. Select the primary actor and usable camera observations. Align the reconstructed trajectory to the Ego-Exo4D scene coordinate system before applying the shared sample-local transform.
3. Match each retained interaction/history timestamp to an available pose frame. Check temporal matching error and pose validity; unmatched or invalid attachments must not be treated as valid supervision.
4. Store root translation, root orientation, local body-joint rotations, and 3D joint positions. Express global components in the sample-local frame, while keeping local joint rotations parent-relative.
5. Preserve the correspondence between the location and pose validity masks, and retain alignment provenance for auditing.

**Output:** location and full-body pose sequences paired at the same target timestamps. The pose model uses 6D root rotation, 3D root translation, and 23 local 6D joint rotations, totaling 147 state dimensions. Evaluation can select a subset of joints without changing this state representation.

Per-record WHAM-to-scene alignment validation is tracked in the [annotation validation notes](../data/RELEASE_REVIEW.md).

## 5. Forecast Sample Generation

**Why:** models need a consistent observation/target interface, while the dataset must preserve the actual ordering and irregular timing of the interactions.

**How:**

1. Partition the ordered future events into domain-specific chunks: 10 steps for Cooking, 5 for Health, and 4 for Bike Repair. These are event counts, not seconds.
2. Let the first future interaction timestamp define the chunk time origin $t_0$. Use the preceding 30 seconds for the observation window, clipped at the beginning of a take when needed. Sample 30 egocentric frames and retain the available hand-location and pose history in that interval.
3. For each future event, retain its absolute take timestamp $t_k$ and relative offset $\Delta t_k=t_k-t_0$, together with its location and aligned SMPL state. The first target has offset zero; observed history precedes zero. No additional fixed anticipation gap is inserted into this chunk-level definition.
4. Pad incomplete tail sequences to the domain horizon and mark padded targets invalid. Exclude padding and invalid attachments from losses, metrics, and target counts.
5. Filter samples that fail the required grounding, coordinate, timestamp, or pose checks. Keep all samples from the same take in a single training, validation, or test split, then audit the final manifests against the reported counts.

**Output:** an observation context and a fixed number of future location/pose slots, accompanied by timestamps and validity masks. The dataset contains 233,828 forecasting samples across 787 takes, as reported in [the aggregate metadata](../data/README.md).

## Models and resources

| Resource | Role | Official link |
|---|---|---|
| Ego-Exo4D | Source video, scene geometry, calibration, and temporal annotations | [Website](https://ego-exo4d-data.org/) · [Documentation](https://docs.ego-exo4d-data.org/getting-started/) |
| FIction | Prior interaction annotation and preparation pipeline | [Repository](https://github.com/thechargedneutron/FIction) · [Preparation guide](https://github.com/thechargedneutron/FIction/blob/main/preprocess/README.md) |
| Detic | Object detection for scene grounding | [Repository](https://github.com/facebookresearch/Detic) |
| LVIS | Semantic object vocabulary; a dataset/vocabulary, not a separate predictor | [Website](https://www.lvisdataset.org/) |
| Llama 3 | Narration-to-object matching | [Official repository](https://github.com/meta-llama/llama3) |
| WHAM | Human motion reconstruction | [Repository](https://github.com/yohanshin/WHAM) |
| SMPL | Parametric human body representation | [Website](https://smpl.is.tue.mpg.de/) |

### Forecasting models are separate from annotation construction

[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and [V-JEPA 2](https://github.com/facebookresearch/vjepa2) provide semantic and dynamic representations in the HIGFlow forecasting stage, separate from the object grounding, narration matching, and body annotation steps above.

Exact model versions, checkpoints, and preprocessing parameters need to accompany the annotation export for reproducibility.
