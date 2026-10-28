# Self-Supervised Pretraining and Fine-Tuning

This directory contains the separate self-supervised time-series pretraining pipeline included with SLARDA. It is distinct from the course-project DINOv3 change. The DINOv3 encoder is integrated in [`Autorgressive_Adaptation`](../Autorgressive_Adaptation/README.md); this directory's model definitions remain the original CNN/CPC pipeline.

## Pipeline overview

For the Opportunity HAR model, `models/Opp_HAR_models.py` uses a 1D CNN encoder and a CPC forecasting objective in self-supervised mode. In fine-tuning mode, the encoded sequence is pooled and passed to a class-prediction head.

```text
Sensor window
    → 1D CNN encoder
    ├── self_supervised mode → CPC prediction loss
    └── fine_tune mode       → temporal pooling → classifier → class logits
```

`main.py` iterates through the configured dataset's domain IDs, loads each domain's train/validation/test loaders, runs pretraining, and then attempts fine-tuning from the saved CNN checkpoint. In the checked-in script, the domain loop is explicitly set to `a`, `b`, `c`, and `d`, and the data path points to `../Autorgressive_Adaptation/data/<dataset>/`.

## Main files

| Path | Responsibility |
| --- | --- |
| `main.py` | Runs self-supervised pretraining followed by fine-tuning for each configured domain |
| `models/` | Dataset-specific CNN encoders, classifiers, and CPC modules |
| `trainer/trainer.py` | Training/evaluation loop and checkpoint saving |
| `dataloader/dataloader.py` | Loads the prepared PyTorch split files |
| `config_files/` | Dataset-specific model and training hyperparameters |

## Data and checkpoint expectations

The data loader expects prepared files named `train_<domain>.pt`, `val_<domain>.pt`, and `test_<domain>.pt` under the shared `Autorgressive_Adaptation/data/<dataset>/` directory. Each split is a dictionary with `samples` and `labels`. Raw sensor recordings must be preprocessed into the window format expected by the selected dataset configuration.

The fine-tuning stage loads a source checkpoint from `saved_models/` using the filename pattern `last_<dataset>_CNN_AR_src_<domain>_.pt`. A compatible pretraining checkpoint must therefore exist before fine-tuning can start.

## Relationship to the DINOv3 assignment

This folder was not modified for the DINOv3 backbone assignment. The selected DINOv3 model is available through the autoregressive adaptation entrypoint instead:

```powershell
cd ..\Autorgressive_Adaptation
python train_CD.py --selected_dataset Opp_HAR --base_model DINOv3_SLARDA `
  --dinov3_repo "C:\path\to\dinov3-main" `
  --dinov3_weights "C:\path\to\dinov3_vits16_checkpoint.pth" `
  --device cuda --num_runs 1
```

See the [DINOv3 implementation guide](../README_DINOV3.md) for the architecture, checkpoint setup, and validation status. The self-supervised CNN pretraining pipeline does not load the DINOv3 model.

## Note about running this legacy entrypoint

The checked-in `main.py` imports an `args` module, but this extracted directory does not contain an `args.py` file. The project also includes a `Self-supervised training.rar` archive. Before running this separate pipeline, check whether the archive contains the required argument/configuration module and extract it as intended by the original project. This caveat does not affect the DINOv3 autoregressive entrypoint documented above.
