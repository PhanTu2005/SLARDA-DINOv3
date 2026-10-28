# Autoregressive Domain Adaptation

This directory contains the SLARDA source/target training implementation. The course-project DINOv3 encoder is added as an optional model choice for this training path. For the complete architecture explanation, see the repository's [main README](../README.md) and the [DINOv3 implementation notes](../README_DINOV3.md).

## Contents

| Path | Purpose |
| --- | --- |
| `train_CD.py` | Entry point for cross-domain training and evaluation |
| `train_supervised.py` | Source-only supervised baseline entry point |
| `models/models.py` | Original CNN models, DINOv3 model, and domain discriminator classes |
| `trainer/SLARDA.py` | Source model initialization, target adaptation, and teacher update |
| `trainer/source_only.py` | Source-domain supervised training used to initialize adaptation |
| `trainer/training_evaluation.py` | Training and evaluation helpers |
| `dataloader/dataloader.py` | Loads each domain's train, validation, and test `.pt` files |
| `config_files/` | Dataset-specific model and training settings |
| `args.py` | Dataset, model, device, and experiment CLI options |

## Data layout

The default dataset is `Opp_HAR`. Run commands from this directory; the loader resolves the dataset relative to the current working directory. Its expected directory is:

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

For every domain ID, provide all three splits. Each file must be loadable with `torch.load` and contain a dictionary with:

```python
{
    "samples": ...,  # windowed sensor data
    "labels": ...,   # integer class labels
}
```

The loader discovers domain IDs from `train_<domain>.pt` filenames. The `Opp_HAR` model configuration expects 113 input channels and four classes; after the loader's layout handling, each batch passed to the model must have shape `[batch, 113, time]`. The labels are expected to be class indices accepted by `CrossEntropyLoss`. Raw Opportunity `.dat` files must be converted to the prepared windowed `.pt` structure before running this code. The `data/` directory is ignored by Git, so dataset files stay out of the course submission.

To inspect one prepared split before training:

```powershell
python -c "import torch; p='data/Opp_HAR/train_a.pt'; d=torch.load(p, map_location='cpu'); print(d.keys()); print(d['samples'].shape, d['samples'].dtype); print(d['labels'].shape, d['labels'].dtype)"
```

## Select the DINOv3 model

The original encoders remain supported. To choose DINOv3, pass:

```text
--base_model DINOv3_SLARDA
```

The DINOv3 model also requires a local Meta DINOv3 checkout and a separately downloaded ViT-S/16 checkpoint. The checkpoint is not bundled in this repository. See [`README_DINOV3.md`](../README_DINOV3.md#checkpoint-and-environment) for access and model details.

## Run cross-domain adaptation

From this directory, configure the paths for your machine:

```powershell
python train_CD.py `
  --selected_dataset Opp_HAR `
  --base_model DINOv3_SLARDA `
  --dinov3_repo "C:\path\to\dinov3-main" `
  --dinov3_weights "C:\path\to\dinov3_vits16_checkpoint.pth" `
  --device cuda `
  --num_runs 1
```

The script enumerates source and target domains from the prepared split files and skips source-to-same-source pairs. `--num_runs 1` requests one run per cross-domain scenario; the default is three runs. For the original CNN baseline, select the corresponding model name from `args.py` and omit the DINOv3 path options.

### Training stages used by the existing SLARDA entry point

For each cross-domain pair, `trainer/SLARDA.py`:

1. Constructs source, target, and teacher instances of the selected model.
2. Loads an existing source-only checkpoint or trains the source model from labeled source data.
3. Freezes the source model and teacher; optimizes the target model and domain discriminator.
4. Uses target pseudo-labels above the configured confidence threshold and updates the teacher with the configured momentum.
5. Evaluates source and adapted models on the target validation/test loaders.

The DINOv3 integration supplies the selected feature extractor to these stages; it does not replace this trainer. DINOv3 weights are frozen, while the adapter projections and classifier remain trainable.

## Outputs

The existing scripts save source-only checkpoints under `src_only_saved_models/<dataset>/` and experiment logs/results under the configured `save_dir` (default: `experiments_logs/`). Generated checkpoints, logs, and datasets are local experiment artifacts and should not be added to the GitHub submission unless explicitly required.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `DINOv3_SLARDA requires --dinov3_repo` | Pass the local DINOv3 source directory containing `hubconf.py`. |
| Checkpoint not found | Pass the full path to the downloaded ViT-S/16 checkpoint file; the source-code repository alone is not the checkpoint. |
| Expected 113 sensor channels | Confirm the prepared windows and layout handling produce `[B,113,T]` for Opp_HAR. |
| No domain IDs are discovered | Confirm `data/Opp_HAR/` contains files named `train_<id>.pt`, and run from `Autorgressive_Adaptation`. |
| Dataset loader cannot read the raw files | Convert raw recordings into windowed `.pt` files with `samples` and `labels`; raw `.dat` files are not the loader input. |

## Validation status

The modified model and argument files passed Python syntax compilation. The local checkpoint and prepared dataset are not currently present in the project workspace, so a DINOv3 forward pass and a full training run remain pending. No metrics are claimed.
