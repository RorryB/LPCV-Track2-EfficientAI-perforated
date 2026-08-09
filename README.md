# LPCV Track2 EfficientAI

## Perforated Hackathon Project

Two scripts were added to apply PerforatedAI dendritic optimization to the distilled student model.

**`references/video_classification/train_distill_r2plus1d_from_slowfast_mmaction_perforated.py`**
Distillation training script with PAI integration. Loads a Kinetics-400 pretrained student model, fine-tunes it via knowledge distillation from the SlowFast-R101 teacher, and applies PerforatedAI dendrite optimization during training.

```bash
./ENV/bin/python -m pdb ./references/video_classification/train_distill_r2plus1d_from_slowfast_mmaction.py \
  --data-path ./Dataset/QEVD_sup_full \
  --teacher-config ./mmaction/mmaction2/configs/recognition/slowfast/slowfast_r101_16x4_QEVD_sup.py \
  --teacher-checkpoint ./models/slowfast_r101_16x4_QEVD_sup.pth \
  --student-weights KINETICS400_V1 \
  --clip-len 16 --student-clip-len 16 \
  --teacher-train-resize-size 256 256 --teacher-train-crop-size 224 224 \
  --student-train-resize-size 128 171 --student-train-crop-size 112 112 \
  --student-val-resize-size 128 171 --student-val-crop-size 112 112 \
  --distill-alpha 0.5 --distill-temp 2.0 \
  --cache-dataset --epochs 30 --batch-size 32 --lr 0.005 \
  --output-dir ./checkpoint --amp --print-freq 1000
```

**`references/video_classification/benchmark_inference.py`**
Standalone inference benchmark that evaluates and compares the baseline model (trained weights before dendrites were added) against the PAI-optimized model (with dendrites). Reports parameter count, throughput, and accuracy for each.

```bash
./ENV/bin/python -m pdb references/video_classification/benchmark_inference.py
```

The trained models and inference dataset can be found [here](https://drive.google.com/drive/u/1/folders/1bCamFCXcOWJbiwMVEwvakfqBFPQvxKM2).  For the full training dataset see download instructions in the original README below the ARM Hackathon section.

**Results:**

```
==============================================================
  COMPARISON SUMMARY
==============================================================
  Model                          Params    Clips/s      Acc@1
  ---------------------- -------------- ---------- ----------
  Baseline Model             31,347,321       2.261    94.490%
  Perforated Model           31,441,989       2.265    95.058%
==============================================================
```

The perforated model reduces top-1 error for video classification by ~0.57 percentage points (a ~10% relative error reduction) with a parameter increase of only 94k (~0.3%). The throughput difference is well within the run-to-run noise you'd expect on a laptop.

### Perforated Windows ARM Setup

**Prerequisites:**
- Windows 11 ARM64
- Python 3.14+ (tested with 3.14.6)
- PowerShell with script execution enabled
- ARM Snapdragon CPU

**1. Enable PowerShell Script Execution**

If you see execution policy errors, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**2. Create Virtual Environment**

```powershell
# Create venv
python -m venv ENV

# Activate it
.\ENV\Scripts\Activate.ps1
```

**3. Install Dependencies**

```powershell
# Install PyTorch (latest ARM64 compatible version)
pip install torch torchvision

# Install torchcodec (required for video decoding)
pip install torchcodec

# Install other dependencies
pip install onnx onnxruntime onnxscript h5py av scipy pycocotools wheel packaging tqdm
```

**Important: Install FFmpeg**

TorchCodec requires FFmpeg 7 or 8 (versions 4-8 are supported, but not 9). On Windows:

1. Download FFmpeg 7.1 "full-shared" build from: https://www.gyan.dev/ffmpeg/builds/
   - Direct link: https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-full_build-shared.7z
   - Extract to `C:\Users\YourUsername\` (the archive contains a nested folder structure)
   
2. Add FFmpeg to PATH permanently:
   ```powershell
   # Replace YourUsername with your actual username
   [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Users\YourUsername\ffmpeg-7.1-full_build-shared\ffmpeg-7.1-full_build-shared\bin", [System.EnvironmentVariableTarget]::User)
   ```
   
3. **Close and reopen your terminal** for the PATH change to take effect

4. Reactivate your environment: `.\ENV\Scripts\Activate.ps1`

5. Verify FFmpeg is available: `ffmpeg -version`

6. **Copy FFmpeg DLLs to torchcodec** (required for Windows):
   ```powershell
   Copy-Item "C:\Users\YourUsername\ffmpeg-7.1-full_build-shared\ffmpeg-7.1-full_build-shared\bin\*.dll" -Destination ".\ENV\Lib\site-packages\torchcodec\" -Force
   ```

**4. Install PerforatedAI Package**

Clone and install the PerforatedAI package from GitHub:

```powershell
# Clone the PerforatedAI repository
git clone https://github.com/PerforatedAI/PerforatedAI.git
Set-Location PerforatedAI

# Checkout the develop branch
git checkout develop

# Install the package
pip install -e .

# Return to project directory
Set-Location ..
```

**5. Verify Installation**

Test that PerforatedAI is installed correctly:
```powershell
python -c "from perforatedai import utils_perforatedai as UPA; from perforatedai import network_perforatedai as NPA; print('PerforatedAI imports successful!')"
```

**6. Download Models and Data**

The perforated scripts require:
- **Dataset**: `./Dataset/QEVD_sup_full` - See the main README section for dataset setup instructions
- **PAI checkpoints**: `./r2plus1d_dendritic/` directory containing:
  - `beforeSwitch_0_pai.pt` - Baseline model (trained, no dendrites)
  - `best_model_pai.pt` - Perforated model (with dendrites)

These model files can be generated by running the training script first or our pretrained models can be found [here](https://drive.google.com/drive/u/1/folders/1bCamFCXcOWJbiwMVEwvakfqBFPQvxKM2).

**7. Run Scripts**

**Running in Your Terminal:**

Open PowerShell in VS Code (`Ctrl + ` ` `), navigate to the project directory, and activate the environment:

```powershell
cd C:\Users\rabre\LPCV-Track2-EfficientAI-perforated
.\ENV\Scripts\Activate.ps1
```

You should see `(ENV)` in your prompt. Now you can run the scripts:

**Benchmark Inference** (compares baseline vs perforated models):
```powershell
python references/video_classification/benchmark_inference.py --device cpu
```

**Note:** On Windows ARM64, you must use `--device cpu` because PyTorch is CPU-only. The script defaults to 4 workers for parallel data loading.

**Training with Perforation** (generates the PAI checkpoints):
```powershell
python references/video_classification/train_distill_r2plus1d_from_slowfast_mmaction_perforated.py --data-path ./Dataset/QEVD_sup_full --teacher-config mmaction/QEVD_config/slowfast_r101_16x4_QEVD_sup.py --teacher-checkpoint models/slowfast_r101_16x4_QEVD_sup.pth --student-weights KINETICS400_V1 --clip-len 16 --student-clip-len 16 --output-dir checkpoint
```

**Notes:**
- The benchmark script has defaults configured for `./Dataset/QEVD_sup_full` and `./r2plus1d_dendritic`
- For multi-line commands, use backtick (`` ` ``) at the end of each line for continuation

---

This repository contains our LPCV Track 2 video classification solution. The
workflow has three stages:

1. Train a SlowFast-R101 teacher with MMAction2.
2. Distill the teacher into a lightweight TorchVision video model.
3. Export the distilled model for Qualcomm AI Hub deployment.


# Original README below with additional instructions

## Released Assets

- Supplemental dataset: [shuangtianxiaoye/QEVD_sup](https://huggingface.co/datasets/shuangtianxiaoye/QEVD_sup)
- Model checkpoints: [shuangtianxiaoye/LPCV-Track2-EfficientAI](https://huggingface.co/shuangtianxiaoye/LPCV-Track2-EfficientAI)

The model repository contains:

- `slowfast_r101_16x4_QEVD_sup.pth`: MMAction2 SlowFast-R101 teacher checkpoint.
- `r2plus1d_r18_16x4_QEVD_sup.pth`: distilled TorchVision R(2+1)D-18 student checkpoint.

The original Qualcomm QEVD videos are not redistributed in this repository or
in the Hugging Face dataset. Users must download the original QEVD dataset from
Qualcomm after accepting its license.

## 1. Train The MMAction2 Teacher

### Install MMAction2

```bash
cd mmaction
git clone https://github.com/open-mmlab/mmaction2.git

conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install mmcv
pip install pytorchvideo

cd mmaction2
pip install -v -e .
```

<!-- If `mmaction/mmaction2` already exists, skip the `git clone` command. -->

Some `pytorchvideo` / `torchvision` combinations may fail with:

```text
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

If this happens, edit:

```text
<conda-env>/lib/python3.8/site-packages/pytorchvideo/transforms/augmentations.py
```

and replace:

```python
import torchvision.transforms.functional_tensor as F_t
```

with:

```python
import torchvision.transforms.functional as F_t
```

### Copy QEVD Files Into MMAction2

Run the following commands from `mmaction/mmaction2`:

```bash
cp ../QEVD_datasets/video_dataset_QEVD_sup.py ./mmaction/datasets/
cp ../QEVD_datasets/VideoSampler.py ./mmaction/datasets/
cp ../QEVD_config/slowfast_r101_16x4_QEVD_sup.py ./configs/recognition/slowfast/
```

The config uses `custom_imports`, so `mmaction/datasets/__init__.py` does not
need to be modified.

### Prepare Data

We provide a supplemental `QEVD_sup` dataset on Hugging Face:

```bash
mkdir -p datasets
hf download shuangtianxiaoye/QEVD_sup \
    --repo-type dataset \
    --local-dir datasets/QEVD_sup
```

You must also download the original QEVD dataset separately from Qualcomm after
accepting its license. Please follow the instructions on the official
[QEVD dataset page](https://www.qualcomm.com/developer/software/qevd-dataset).

This repository does not redistribute the original Qualcomm QEVD videos. The
full training dataset is reconstructed locally by merging the original QEVD
`train` / `val` folders with our supplemental videos.

The recommended local layout under the repository root is:

```text
datasets/
├── QEVD_sup/                 # redistributed supplemental videos
│   ├── train/
│   └── meta/
│       └── task_map.txt
└── QEVD_sup_full/            # local merged dataset, not redistributed
    ├── train/
    ├── val/
    └── meta/
```

Create the merged dataset locally:

```bash
mkdir -p datasets/QEVD_original datasets/QEVD_sup_full

# Put the original QEVD train/val folders into datasets/QEVD_sup_full first.
cp -a /path/to/original_qevd/train datasets/QEVD_sup_full/
cp -a /path/to/original_qevd/val datasets/QEVD_sup_full/

# Merge our supplemental videos.
rsync -a datasets/QEVD_sup/train/ datasets/QEVD_sup_full/train/

# Regenerate metadata for the merged dataset.
mkdir -p datasets/QEVD_sup_full/meta
cp datasets/QEVD_sup/meta/task_map.txt datasets/QEVD_sup_full/meta/

python datasets/update_qevd_sup_meta.py \
    --dataset-root datasets/QEVD_sup_full \
    --meta-root datasets/QEVD_sup_full/meta
```

<!-- For MMAction2 teacher training, copy or symlink the merged dataset into
`mmaction/mmaction2/data`:

```bash
cd mmaction/mmaction2
mkdir -p data cache
ln -s ../../datasets/QEVD_sup_full data/QEVD_sup
mkdir -p data/cache/qevd
```

The teacher config expects this layout under `mmaction/mmaction2`:

```text
data/
├── QEVD_sup/
│   ├── train/
│   │   ├── class_1/
│   │   │   └── video.mp4
│   │   └── class_2/
│   ├── val/
│   │   ├── class_1/
│   │   └── class_2/
│   └── meta/
│       ├── task_map.txt
│       ├── train.txt
│       └── val.txt
└── cache/
    └── qevd/
``` -->

The annotation files use MMAction2 video-list format:

```text
relative/path/to/video.mp4 label_id
```

Example:

```text
class_1/video_0001.mp4 0
class_2/video_0002.mp4 1
```

The custom QEVD dataset uses `torchvision.datasets.Kinetics`, so class labels
are inferred from the class folders under `data/QEVD_sup`. Keep folder names
and annotation labels consistent with `meta/task_map.txt`.

### Prepare The Pretrained Weight

Download the Kinetics-400 SlowFast-R101 checkpoint from the official MMAction2
model source and place it at:

```text
mmaction/mmaction2/checkpoints/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth
```

If you use another path, update `load_from` in:

```text
configs/recognition/slowfast/slowfast_r101_16x4_QEVD_sup.py
```

Also update the dataset paths in the same config if your merged dataset is not
available at the default `data/QEVD_sup` location.

If you want to skip teacher training and use our released teacher checkpoint,
download it from Hugging Face:

```bash
mkdir -p models
hf download shuangtianxiaoye/LPCV-Track2-EfficientAI \
    slowfast_r101_16x4_QEVD_sup.pth \
    --local-dir models
```

### Train The Teacher

Run from `mmaction/mmaction2`:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/recognition/slowfast/slowfast_r101_16x4_QEVD_sup.py \
    --work-dir work_dirs/slowfast_r101_16x4_QEVD_sup
```

## 2. Distill The Student Model

The distillation script trains a lightweight TorchVision `r2plus1d_18` student
model using logits from the MMAction2 SlowFast teacher.

### Install Dependencies

Use the same `openmmlab` environment:

```bash
conda activate openmmlab
pip install -r requirements.txt
```

### Prepare Inputs

Before distillation, prepare:

- The local merged QEVD data in Kinetics-style folders, for example `datasets/QEVD_sup_full`.
- The trained teacher checkpoint, for example `models/slowfast_r101_16x4_QEVD_sup.pth`.
- The copied MMAction2 teacher config at `mmaction/mmaction2/configs/recognition/slowfast/slowfast_r101_16x4_QEVD_sup.py`.
- Optional dataset caches generated by the teacher dataset under `datasets/cache/qevd`.

When `--qevd-cache-dir` is provided, the script expects cache files named like:

```text
datasets/cache/qevd/16-4-train.pt
datasets/cache/qevd/16-4-val.pt
```

where `16` is `--clip-len` and `4` is `--frame-rate`.

### Run Distillation

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python references/video_classification/train_distill_r2plus1d_from_slowfast_mmaction.py \
    --data-path datasets/QEVD_sup_full \
    --teacher-config mmaction/mmaction2/configs/recognition/slowfast/slowfast_r101_16x4_QEVD_sup.py \
    --teacher-checkpoint models/slowfast_r101_16x4_QEVD_sup.pth \
    --student-weights KINETICS400_V1 \
    --clip-len 16 \
    --student-clip-len 16 \
    --teacher-train-resize-size 256 256 \
    --teacher-train-crop-size 224 224 \
    --student-train-resize-size 128 171 \
    --student-train-crop-size 112 112 \
    --student-val-resize-size 128 171 \
    --student-val-crop-size 112 112 \
    --distill-alpha 0.5 \
    --distill-temp 2.0 \
    --cache-dataset \
    --qevd-cache-dir datasets/cache/qevd \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.005 \
    --output-dir checkpoint \
    --amp
```

The script writes checkpoints and validation logs to `checkpoint/`. The latest
student checkpoint is saved as:

```text
checkpoint/checkpoint.pth
```

Alternatively, download our released distilled student checkpoint:

```bash
mkdir -p models
hf download shuangtianxiaoye/LPCV-Track2-EfficientAI \
    r2plus1d_r18_16x4_QEVD_sup.pth \
    --local-dir models
```

## 3. Export The Distilled Model

The export script packages the distilled `r2plus1d_18` student model for
Qualcomm AI Hub.

### Install Export Environment

Use a separate Python 3.10 environment for export:

```bash
conda create --name lpcvc python=3.10 -y
conda activate lpcvc
pip install -r requirements_export.txt
```

### Export

Run from the repository root:

```bash
python example_export.py \
    --checkpoint-path ./models/r2plus1d_r18_16x4_QEVD_sup.pth \
    --num-frames 16 --precision w8a8 \
    --compile-options="--qairt_version=default"  \
    --profile-options="--max_profiler_iterations 100 --qairt_version=default"
```

By default, `example_export.py` targets `Dragonwing IQ-9075 EVK`. Use the
export script arguments to select a different AI Hub device, runtime, precision,
or output format.
