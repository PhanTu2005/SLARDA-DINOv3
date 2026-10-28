# DINOv3 Integration Notes for SLARDA

This document is the implementation companion to the [main project README](README.md). It records the motivation, insertion point, tensor interfaces, training behavior, setup, and current validation status of the course-project integration.

## 1. Scope and design decision

SLARDA in this repository operates on multivariate sensor windows. DINOv3 ViT-S/16 is a pretrained vision Transformer that expects three-channel images. The integration therefore needs an adapter at the model boundary; directly passing the sensor tensor to DINOv3 would not satisfy its input contract.

The new `DINOv3_SLARDA` class is an optional model in `Autorgressive_Adaptation/models/models.py`. The training script selects it through its existing dynamic import:

```powershell
--base_model DINOv3_SLARDA
```

This design leaves `trainer/SLARDA.py`, its domain discriminator, its teacher update, dataset-specific configurations, and the original CNN model classes in place. It preserves the expected model return form:

```python
logits, (latent_features, sequence_features)
```

## 2. Architecture

![DINOv3 feature extraction in SLARDA: sensor matrix adapter, frozen ViT-S/16 and trainable projections.](figures/dinov3-slarda-architecture.svg)

The end-to-end feature path is:

```text
Input window [B, C, T]
    → sensor-to-image conversion [B, 3, 224, 224]
    → frozen DINOv3 ViT-S/16
    → patch tokens [B, 196, 384] + class token [B, 384]
    → trainable projections
    → SLARDA feature discriminator and class head
```

For ViT-S/16 at 224 × 224 input resolution, the 16-pixel patch size produces a 14 × 14 grid, or 196 patch tokens. DINOv3's ViT-S/16 embedding width is 384. The patch tokens are transposed to channel-first form and passed to a trainable `Conv1d(384, out_channels, kernel_size=1)`. The class token goes through a trainable `Linear(384, feat_dim)` layer and the task classifier.

### 2.1 Tensor interface by stage

| Stage | Tensor shape | Notes |
| --- | --- | --- |
| Data-loader batch | `[B, C, T]` | `C` must match the selected dataset configuration. For Opp_HAR, the model checks for 113 channels. |
| Grayscale matrix | `[B, 1, C, T]` | `tanh` bounds sensor values and maps them into grayscale intensity. |
| Resized image | `[B, 3, 224, 224]` | Bilinear resize, three repeated grayscale channels, then ImageNet normalization. |
| ViT patch output | `[B, 196, 384]` | Last-layer patch tokens for ViT-S/16 at 224 pixels. |
| Patch feature map | `[B, out_channels, 196]` | Patch tokens transposed and projected with a trainable 1×1 convolution. |
| Latent class vector | `[B, feat_dim]` | CLS token projected for classification and confident teacher pseudo-labeling. |
| Classifier output | `[B, num_classes]` | Logits consumed by the existing cross-entropy objective. |

The module validates rank and channel count at the input boundary. These checks make common data-layout mistakes fail with an informative error rather than later in a convolution or matrix multiplication.

### 2.2 Sensor-to-image conversion

`DINOv3_SLARDA.forward` expects a three-dimensional sensor batch `[B, C, T]`. For each window, it:

1. Applies `tanh` to bound sensor values and maps the result to grayscale intensity in `[0,1]`.
2. Adds one image channel and bilinearly resizes the `C × T` matrix to 224 × 224.
3. Repeats the grayscale channel three times to create an RGB-shaped tensor.
4. Applies the standard ImageNet channel means and standard deviations registered as model buffers.

The transformation allows an image backbone to accept sensor data, but it is a chosen representation rather than a physically meaningful photograph. The adapter does not introduce new sensor measurements. It also does not establish that vision pretraining is beneficial for this time-series task; that requires an experiment.

### 2.3 Frozen backbone and trainable layers

The DINOv3 backbone is loaded from the local Meta implementation with `torch.hub.load(..., source='local', weights=...)`. Its parameters have `requires_grad=False`; feature extraction runs under `torch.no_grad()`. The model's `train()` override keeps the backbone in evaluation mode even when the outer SLARDA model enters training mode.

| Component | Training behavior | Role |
| --- | --- | --- |
| DINOv3 ViT-S/16 | Frozen; always evaluation mode | Extract image-patch and class-token representations |
| Patch projection | Trainable | Map patch-token width 384 to configured `out_channels` |
| Class-token projection | Trainable | Map class-token width 384 to configured `feat_dim` |
| Classification head | Trainable | Produce the dataset's class logits |
| SLARDA discriminator and teacher logic | Existing training code | Align source/target feature distributions and update the teacher |

The source model, target model, and teacher are constructed by the existing SLARDA trainer using the selected model class. The DINOv3 backbone remains frozen in each instance; the trainable projections and classifier follow the existing source-only and target adaptation optimizers.

The current trainer constructs three model instances for a cross-domain scenario (source, target, teacher). Because each instance owns a frozen ViT-S/16, this is more memory-intensive than a single shared encoder. Freezing prevents gradient/optimizer updates to DINOv3 but does not share the weights between the three model objects. The trainer's behavior has not been refactored as part of this assignment.

## 3. What remains unchanged in SLARDA

The feature extractor is selected by `args.base_model`; the adaptation routine still follows the repository's existing control flow:

1. Construct source, target, and teacher models from the same selected model class.
2. If a source-only checkpoint for the chosen dataset/model/domain exists, load it. Otherwise, train the source model with source labels and save its checkpoint.
3. Initialize target and teacher weights from the source checkpoint. Freeze the source and teacher models.
4. Train the domain discriminator on detached source/target sequence features.
5. Train the target model to fool the discriminator and classify target samples that exceed the configured teacher confidence threshold.
6. Update teacher parameters from the target model using the configured momentum.

The DINOv3 extension changes the model instance used in these steps; it does not add a new SLARDA loss or alter the dataset split policy. The default Opp_HAR configuration selects an attention-based feature discriminator (`AR = 'ATT'`). Other dataset configurations can select different discriminator settings.

## 4. Interpreting the temporal features

The adapter rasterizes the entire sensor window as a `channels × time` matrix and runs it through a spatial vision Transformer. The resulting 196 patch tokens are flattened in the ViT's patch-grid order. The existing SLARDA discriminator receives this projected patch sequence.

This provides a compatible tensor sequence for the current trainer, but it is not a direct per-time-step encoding: a token corresponds to a 16 × 16 region of the resized sensor matrix, which can span both sensor and time coordinates. Therefore the alignment sequence has a different interpretation from the temporal feature sequence produced by the original 1D CNN. This is a methodological limitation to consider when evaluating the approach; the code does not claim that the token sequence preserves the original time axis without distortion.

## 5. Code map

| File | Integration point |
| --- | --- |
| [`Autorgressive_Adaptation/models/models.py`](Autorgressive_Adaptation/models/models.py) | `DINOv3_SLARDA`: local backbone load, shape checks, image adapter, token extraction, projection, and classifier |
| [`Autorgressive_Adaptation/args.py`](Autorgressive_Adaptation/args.py) | `--base_model`, `--dinov3_repo`, and `--dinov3_weights` options |
| [`Autorgressive_Adaptation/train_CD.py`](Autorgressive_Adaptation/train_CD.py) | Reads the model name and dynamically imports the selected model before domain adaptation |
| [`Autorgressive_Adaptation/trainer/SLARDA.py`](Autorgressive_Adaptation/trainer/SLARDA.py) | Constructs source, target and teacher models using the selected class |
| [`Autorgressive_Adaptation/config_files/Opp_HAR_Configs.py`](Autorgressive_Adaptation/config_files/Opp_HAR_Configs.py) | Default Opp_HAR channel, class, and feature dimensions |
| [`Autorgressive_Adaptation/dataloader/dataloader.py`](Autorgressive_Adaptation/dataloader/dataloader.py) | Loads prepared `.pt` splits and yields sensor batches and labels |

The model option is intentionally additive. Existing choices such as `CNN_Opp_HAR_SL`, `CNN_SL_bn`, and `EEG_M_SL` have not been removed.

## 6. Checkpoint and environment

The DINOv3 source repository does not contain the pretrained ViT-S/16 weights. Request access and download the matching backbone checkpoint from [Meta's DINOv3 access page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). The [official DINOv3 README](https://github.com/facebookresearch/dinov3#pretrained-models) describes the local PyTorch Hub loading interface used here. Choose ViT-S/16 so the checkpoint architecture matches the `dinov3_vits16` hub entry point used by this integration.

Required local inputs:

| Argument | Expected value |
| --- | --- |
| `--dinov3_repo` | Path to the local DINOv3 source checkout containing `hubconf.py` |
| `--dinov3_weights` | Path to the ViT-S/16 checkpoint file accepted by that DINOv3 checkout |

The checkpoint is an external model asset and should not be committed to GitHub. The integration uses the local checkout and explicit checkpoint path so model loading does not rely on downloading pretrained weights at runtime. Install PyTorch and the dependencies required by the DINOv3 checkout in the environment used for SLARDA; check the official DINOv3 repository for its current installation guidance.

There is no project-level, version-pinned `requirements.txt` for the SLARDA adaptation folder. Use an environment with the SLARDA dependencies and a PyTorch/torchvision combination supported by the selected DINOv3 source checkout. Record the environment used for any reported experiment; do not infer package compatibility from the syntax check alone.

## 7. Dataset and run instructions

The default dataset argument is `Opp_HAR`. The loader expects preprocessed PyTorch files under `Autorgressive_Adaptation/data/Opp_HAR/`:

```text
data/Opp_HAR/
├── train_a.pt
├── val_a.pt
├── test_a.pt
├── train_b.pt
├── val_b.pt
├── test_b.pt
└── ...
```

Each split file must load as a dictionary with `samples` and `labels` entries. The raw Opportunity files from [UCI](https://archive.ics.uci.edu/dataset/226/opportunity%2Bactivity%2Brecognition) are not directly consumed by this loader. See the [operational README](Autorgressive_Adaptation/README.md#data-layout) for preparation notes.

For Opp_HAR, `config_files/Opp_HAR_Configs.py` specifies 113 input channels, 16 output feature channels, a 16-dimensional classification latent vector, and four classes (`sit`, `stand`, `lie`, `walk`). Ensure the prepared labels are encoded as the integer class indices expected by `CrossEntropyLoss`. Since this legacy loader uses shape heuristics to arrange sensor axes, inspect one loaded batch and confirm it is `[B,113,T]` before a long run.

### Opp_HAR configuration values

| Setting | Current value | Used by |
| --- | ---: | --- |
| Input channels | 113 | Sensor input validation in `DINOv3_SLARDA` |
| DINO input resolution | 224 × 224 | ViT-S/16 image adapter |
| DINO patch grid | 14 × 14 = 196 tokens | Patch feature sequence |
| Patch embedding width | 384 | ViT-S/16 output before projection |
| Projected sequence width (`out_channels`) | 16 | Existing feature discriminator |
| Classification latent width (`feat_dim`) | 16 | Classifier and teacher path |
| Number of classes | 4 | Opp_HAR class logits |
| Source-only pretraining epochs | 30 | `configs.Supervised.num_epoch` |
| Epochs per SLARDA training stage | 15 | `configs.num_epoch` |
| Training batch size | 256 | Opp_HAR data loader config |
| Target-model learning rate | `0.5e-4` | SLARDA config |
| Discriminator learning rate | `0.5e-4` | SLARDA config |
| Teacher loss weight | 0.1 | Confident target pseudo-label loss |
| Teacher confidence threshold | 0.9 | Select target predictions for pseudo-labeling |
| Teacher momentum | 0.996 | Exponential moving-average update |
| Feature discriminator | Attention (`AR = 'ATT'`) | Opp_HAR SLARDA config |

These are the repository's current Opp_HAR configuration values, not hyperparameters selected by a new tuning study. For another dataset, use its own configuration rather than copying this table.

The training entrypoint discovers domains by selecting files whose names contain `train`, then reads the domain ID from the filename. Use the repository's one-character domain names (`a`, `b`, `c`, …) to match this code path. It creates ordered source-target pairs and skips pairs where source and target are the same domain.

### Runtime sequence

```text
train_CD.py
  → parse CLI options and import the selected model class
  → load Opp_HAR configuration and domain split files
  → create source/target model instances plus teacher
  → load or train source-only checkpoint
  → train discriminator and target model for each cross-domain pair
  → evaluate source and adapted target models
  → write logs, summaries and source-only checkpoints
```

The script also builds result summaries across runs. The actual output directory depends on `--save_dir`, `--experiment_description`, `--run_description`, dataset, model, and the experiment folders created by the runner. Inspect the generated directory and log file for the exact command outcome; this README does not assume that the run succeeded.

From the `Autorgressive_Adaptation` directory, run:

```powershell
python train_CD.py `
  --selected_dataset Opp_HAR `
  --base_model DINOv3_SLARDA `
  --dinov3_repo "C:\path\to\dinov3-main" `
  --dinov3_weights "C:\path\to\dinov3_vits16_checkpoint.pth" `
  --device cuda `
  --num_runs 1
```

`--num_runs 1` requests one random-seed run per source-target pair. The default is three. The data loader discovers source domains from the `train_<domain>.pt` files and the training script evaluates each ordered source-target pair with different domain IDs. Set `--selected_dataset` to another supported dataset only when its data files and configuration match that dataset's sensor and label format.

### Useful CLI options

| Option | Default / example | Purpose |
| --- | --- | --- |
| `--da_method` | `SLARDA` | Selects the trainer module imported by `train_CD.py` |
| `--selected_dataset` | `Opp_HAR` | Selects both the dataset directory and dataset configuration |
| `--base_model` | `DINOv3_SLARDA` | Selects the DINOv3 model class instead of an original CNN |
| `--dinov3_repo` | Local checkout path | Provides the local `hubconf.py` and model implementation |
| `--dinov3_weights` | Local `.pth` path | Provides the ViT-S/16 pretrained weights |
| `--device` | `cuda` | Selects the PyTorch execution device |
| `--num_runs` | `3` | Number of runs for each cross-domain scenario |
| `--save_dir` | `experiments_logs` | Root for experiment logs and summaries |

Keep the original CNN configuration and DINOv3 configuration separate when comparing models. Use the same data splits, run count, random-seed policy, and evaluation path for a fair baseline comparison, and record those choices with the results.

### Expected artifacts

The existing code writes source-only model checkpoints beneath `src_only_saved_models/<dataset>/` and experiment logs/results beneath `experiments_logs/` unless `--save_dir` is changed. Checkpoints are named with the dataset, selected model, and source-domain ID, so DINOv3 and CNN runs have separate paths. These outputs and the pretrained DINOv3 file are experiment artifacts, not source files to commit.

## 8. Validation record

| Check | Status |
| --- | --- |
| Python syntax compilation of the changed model and argument files | Passed |
| Load an actual DINOv3 checkpoint | Not run: checkpoint is not present in the project workspace |
| DINOv3 forward-pass smoke check | Not run |
| SLARDA training on prepared data | Not run: prepared dataset files are not present |
| Accuracy / baseline comparison | Not reported |

The first two missing assets needed to continue are the local checkpoint path and the prepared dataset files. Until the forward pass and training are run, this repository makes no empirical claim about accuracy, speed, or improvement over the original CNN.

## 9. Troubleshooting

| Message or symptom | Likely cause | Action |
| --- | --- | --- |
| `requires --dinov3_repo` | The local DINOv3 checkout path was omitted or points to the wrong folder. | Point to the directory containing `hubconf.py`. |
| `requires --dinov3_weights` | The checkpoint file path was omitted or does not exist. | Pass the full path to the downloaded ViT-S/16 checkpoint. |
| DINOv3 state-dict key/shape error | The checkpoint does not match the selected hub architecture. | Use the ViT-S/16 checkpoint with `dinov3_vits16`. |
| `Expected sensor input [batch, channels, time]` | The loader returned an unexpected number of dimensions. | Inspect `samples` and the first dataloader batch. |
| `Expected 113 sensor channels` | Sensor channels are in the wrong axis or the selected config does not match the data. | Confirm the final batch is `[B,113,T]` for Opp_HAR or choose the correct dataset/model configuration. |
| CUDA out-of-memory | Three model instances are created per source-target run. | Reduce batch size, use a smaller compatible setup, or use CPU for a slow smoke attempt; do not change the model width without matching the checkpoint. |

## 10. References

- Ragab et al., [Self-supervised Autoregressive Domain Adaptation for Time Series Data](https://arxiv.org/abs/2111.14834), IEEE TNNLS, DOI: [10.1109/TNNLS.2022.3183252](https://doi.org/10.1109/TNNLS.2022.3183252).
- Meta AI, [DINOv3 reference implementation and pretrained backbones](https://github.com/facebookresearch/dinov3).
- UCI Machine Learning Repository, [Opportunity Activity Recognition](https://archive.ics.uci.edu/dataset/226/opportunity%2Bactivity%2Brecognition).
