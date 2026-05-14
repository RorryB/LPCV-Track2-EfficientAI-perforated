---
license: other
task_categories:
- video-classification
pretty_name: QEVD_sup
---

# QEVD_sup

This dataset repository contains our supplemental QEVD videos for LPCV Track 2
video classification.

## Important Notice

This dataset does not redistribute the original Qualcomm QEVD videos. Users must
download the original QEVD dataset from Qualcomm after accepting its license,
then merge it locally with this supplemental dataset.

Official QEVD page: https://www.qualcomm.com/developer/software/qevd-dataset

## Structure

```text
train/<class_name>/*.mp4
meta/task_map.txt
```

## Reconstruct The Full Local Dataset

After downloading the original QEVD dataset, merge it with this supplemental
dataset and regenerate metadata:

```bash
python datasets/update_qevd_sup_meta.py \
    --dataset-root datasets/QEVD_sup_full \
    --meta-root datasets/QEVD_sup_full/meta
```

## Annotation Format

Generated annotation files follow:

```text
relative/path/to/video.mp4 label_id
```
