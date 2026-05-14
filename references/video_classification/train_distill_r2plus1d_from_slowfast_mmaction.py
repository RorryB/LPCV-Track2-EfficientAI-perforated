import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import warnings
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler

import presets
import utils
from datasets import KineticsWithVideoId

LOCAL_MMACTION_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmaction2")
if os.path.isdir(LOCAL_MMACTION_ROOT) and LOCAL_MMACTION_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_MMACTION_ROOT)

try:
    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmaction.registry import MODELS
    from mmaction.structures import ActionDataSample
    from mmaction.utils import register_all_modules
except ImportError as exc:
    Config = None
    load_checkpoint = None
    MODELS = None
    ActionDataSample = None
    register_all_modules = None
    MMACTION_IMPORT_ERROR = exc
else:
    MMACTION_IMPORT_ERROR = None


def resolve_weights(model_name, weights_name):
    if weights_name is None or str(weights_name).lower() == "none":
        return None
    weights_enum = torchvision.models.get_model_weights(model_name)
    short_name = weights_name.split(".")[-1]
    if not hasattr(weights_enum, short_name):
        available = [w.name for w in weights_enum]
        raise ValueError(
            f"Invalid weights '{weights_name}' for model '{model_name}'. Available weights: {available}"
        )
    return getattr(weights_enum, short_name)


def build_student_model(student_model_name, num_classes, weights_name):
    supported_models = {"r2plus1d_18", "r3d_18", "mc3_18"}
    if student_model_name not in supported_models:
        raise ValueError(f"Unsupported student model '{student_model_name}'. Available: {sorted(supported_models)}")

    weights = resolve_weights(student_model_name, weights_name) if weights_name else None
    model_builder = getattr(torchvision.models.video, student_model_name)
    model = model_builder(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _extract_student_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict must be a dict-like object.")
    if not any(str(k).startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {
        (k[len("module."):] if str(k).startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_student_init_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_extract_student_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if utils.is_main_process():
        print(
            "[student-init] "
            f"checkpoint='{checkpoint_path}' "
            f"missing_keys={len(missing)} "
            f"unexpected_keys={len(unexpected)}"
        )


def _get_cache_path(filepath, args):
    value = f"{filepath}-{args.clip_len}-{args.kinetics_version}-{args.frame_rate}"
    h = hashlib.sha1(value.encode()).hexdigest()
    cache_root = args.cache_dir
    os.makedirs(cache_root, exist_ok=True)
    return os.path.join(cache_root, h[:10] + ".pt")


def _build_relpath_to_label_from_dataset(dataset, split_root):
    mapping = {}
    for sample in dataset.samples:
        if len(sample) < 2:
            continue
        sample_path = sample[0]
        sample_label = int(sample[1])
        rel_path = os.path.relpath(sample_path, split_root).replace("\\", "/")
        mapping[rel_path] = sample_label
    return mapping


def _build_relpath_to_label_from_ann_file(ann_file):
    mapping = {}
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            mapping[path.replace("\\", "/")] = int(label)
    return mapping


def build_teacher_to_student_reorder_index(cfg, data_path, dataset_train, dataset_val):
    train_ann = cfg.get("ann_file_train", None)
    val_ann = cfg.get("ann_file_val", None)
    if train_ann is None or val_ann is None:
        raise RuntimeError("Teacher config is missing ann_file_train/ann_file_val; cannot align label spaces.")
    if not os.path.exists(train_ann):
        raise FileNotFoundError(f"Teacher train annotation file not found: '{train_ann}'.")
    if not os.path.exists(val_ann):
        raise FileNotFoundError(f"Teacher val annotation file not found: '{val_ann}'.")

    teacher_to_student = {}

    split_specs = [
        ("train", dataset_train, train_ann),
        ("val", dataset_val, val_ann),
    ]
    for split_name, dataset, ann_file in split_specs:
        split_root = os.path.join(data_path, split_name)
        dataset_map = _build_relpath_to_label_from_dataset(dataset, split_root)
        ann_map = _build_relpath_to_label_from_ann_file(ann_file)
        common_keys = set(dataset_map) & set(ann_map)
        if not common_keys:
            raise RuntimeError(f"No overlapping videos found between dataset and ann_file for split '{split_name}'.")
        for rel_path in common_keys:
            teacher_label = ann_map[rel_path]
            student_label = dataset_map[rel_path]
            if teacher_label in teacher_to_student and teacher_to_student[teacher_label] != student_label:
                raise RuntimeError(
                    f"Inconsistent label mapping for teacher label {teacher_label}: "
                    f"{teacher_to_student[teacher_label]} vs {student_label} on '{rel_path}'"
                )
            teacher_to_student[teacher_label] = student_label

    num_classes = len(dataset_train.classes)
    if len(teacher_to_student) != num_classes:
        missing = sorted(set(range(num_classes)) - set(teacher_to_student.keys()))
        raise RuntimeError(
            f"Teacher->student label mapping incomplete: found {len(teacher_to_student)} / {num_classes}. "
            f"Missing teacher labels: {missing[:10]}"
        )

    student_to_teacher = [None] * num_classes
    for teacher_label, student_label in teacher_to_student.items():
        if student_to_teacher[student_label] is not None and student_to_teacher[student_label] != teacher_label:
            raise RuntimeError(
                f"Student label {student_label} maps to multiple teacher labels: "
                f"{student_to_teacher[student_label]} and {teacher_label}"
            )
        student_to_teacher[student_label] = teacher_label

    if any(x is None for x in student_to_teacher):
        missing = [i for i, x in enumerate(student_to_teacher) if x is None]
        raise RuntimeError(f"Missing teacher label for student labels: {missing[:10]}")

    if utils.is_main_process():
        print(
            "[label-align] "
            f"built reorder index for {num_classes} classes; "
            f"sample student->teacher {student_to_teacher[:10]}"
        )

    return torch.tensor(student_to_teacher, dtype=torch.long)


def _load_cached_dataset(cache_path, split):
    try:
        return torch.load(cache_path, weights_only=False)
    except Exception as exc:
        warnings.warn(
            f"Failed to load cached {split} dataset from '{cache_path}': {exc}. "
            "Removing stale cache and rebuilding it.",
            RuntimeWarning,
        )
        try:
            os.remove(cache_path)
        except OSError as remove_exc:
            warnings.warn(
                f"Unable to remove stale cache '{cache_path}': {remove_exc}",
                RuntimeWarning,
            )
        return None


def _load_qevd_cached_dataset(cache_dir, split, clip_len, frame_rate):
    filename = f"{clip_len}-{frame_rate}-{split}.pt"
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"QEVD dataset cache not found: '{cache_path}'")
    dataset, _ = torch.load(cache_path, weights_only=False)
    dataset.transform = None
    if utils.is_main_process():
        print(f"[dataset-cache] loaded QEVD {split} cache from {cache_path}")
    return dataset


def create_datasets(args):
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    train_cache_path = _get_cache_path(train_dir, args)
    val_cache_path = _get_cache_path(val_dir, args)

    if args.qevd_cache_dir:
        dataset_train = _load_qevd_cached_dataset(args.qevd_cache_dir, "train", args.clip_len, args.frame_rate)
        dataset_val = _load_qevd_cached_dataset(args.qevd_cache_dir, "val", args.clip_len, args.frame_rate)
        return dataset_train, dataset_val

    cached_train = None
    if args.cache_dataset and os.path.exists(train_cache_path):
        cached_train = _load_cached_dataset(train_cache_path, "train")
    if cached_train is not None:
        dataset_train, _ = cached_train
    else:
        dataset_train = KineticsWithVideoId(
            args.data_path,
            frames_per_clip=args.clip_len,
            num_classes=args.kinetics_version,
            split="train",
            step_between_clips=1,
            transform=None,
            frame_rate=args.frame_rate,
            extensions=("avi", "mp4"),
            output_format="TCHW",
        )
        if args.cache_dataset:
            utils.mkdir(os.path.dirname(train_cache_path))
            utils.save_on_master((dataset_train, train_dir), train_cache_path)

    cached_val = None
    if args.cache_dataset and os.path.exists(val_cache_path):
        cached_val = _load_cached_dataset(val_cache_path, "val")
    if cached_val is not None:
        dataset_val, _ = cached_val
    else:
        dataset_val = KineticsWithVideoId(
            args.data_path,
            frames_per_clip=args.clip_len,
            num_classes=args.kinetics_version,
            split="val",
            step_between_clips=1,
            transform=None,
            frame_rate=args.frame_rate,
            extensions=("avi", "mp4"),
            output_format="TCHW",
        )
        if args.cache_dataset:
            utils.mkdir(os.path.dirname(val_cache_path))
            utils.save_on_master((dataset_val, val_dir), val_cache_path)

    return dataset_train, dataset_val


def _tchw_to_cthw(video):
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(video.shape)}")
    if video.shape[0] == 3:
        return video
    if video.shape[1] == 3:
        return video.permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"Unable to infer channel dimension from video shape {tuple(video.shape)}")


def iter_npy_paths(root: str):
    all_paths = []
    for cls in os.listdir(root):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith(".npy"):
                all_paths.append(os.path.join(cls_dir, fname))
    yield from sorted(all_paths)


def enforce_frames(x: np.ndarray, target_t: int) -> np.ndarray:
    if x.ndim != 5:
        raise ValueError(f"Expected 5-D tensor (N, C, T, H, W), got shape {x.shape}")

    current_t = x.shape[2]
    if current_t < target_t:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (0, target_t - current_t), (0, 0), (0, 0)),
            mode="edge",
        )
    elif current_t > target_t:
        x = x[:, :, :target_t, :, :]
    return x


def load_logits_from_h5(h5_path: str) -> np.ndarray:
    logits = []
    with h5py.File(h5_path, "r") as f:
        grp = f["data/0"]
        sorted_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))
        for k in sorted_keys:
            logits.append(grp[k][...].squeeze())
    return np.stack(logits, axis=0)


def load_labels_from_manifest(manifest_path: str, class_to_idx: dict) -> List[int]:
    labels = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            label = record["label"]
            if label not in class_to_idx:
                raise KeyError(
                    f"Label '{label}' from manifest not found in class map. "
                    "Check that the correct class_map.json is being used."
                )
            labels.append(class_to_idx[label])
    return labels


def topk_accuracy(preds: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[torch.Tensor]:
    maxk = max(topk)
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [
        (correct[:k].reshape(-1).float().sum(0) / preds.size(0)) * 100.0
        for k in topk
    ]

class MMActionSlowFastTeacher(nn.Module):
    def __init__(self, recognizer, mean, std, reorder_index=None):
        super().__init__()
        self.recognizer = recognizer
        self.register_buffer("mean", mean.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, -1, 1, 1, 1), persistent=False)
        if reorder_index is not None:
            self.register_buffer("reorder_index", reorder_index.clone().long(), persistent=False)
        else:
            self.reorder_index = None

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected teacher input as NCTHW, got shape {tuple(x.shape)}")
        x = x.float()
        x = (x - self.mean) / self.std
        x = x.unsqueeze(1)
        if ActionDataSample is None:
            raise RuntimeError("ActionDataSample is unavailable; mmaction2 structures were not imported correctly.")
        data_samples = [ActionDataSample() for _ in range(x.shape[0])]

        logits = self.recognizer(x, data_samples=data_samples, mode="tensor", stage="head")

        if logits.ndim != 2:
            raise RuntimeError(f"Expected teacher logits/scores as [N, num_classes], got shape {tuple(logits.shape)}")
        if self.reorder_index is not None:
            logits = logits.index_select(1, self.reorder_index.to(logits.device))
        return logits


def _infer_mmaction_num_classes(cfg, checkpoint_path):
    cls_head = cfg.model.get("cls_head", None)
    if cls_head is not None and "num_classes" in cls_head:
        return int(cls_head["num_classes"])

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    for key in ("cls_head.fc_cls.weight", "module.cls_head.fc_cls.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Unable to infer num_classes from MMAction2 config/checkpoint")


def _resolve_checkpoint_path(checkpoint_path):
    if os.path.basename(checkpoint_path) != "last_checkpoint":
        return checkpoint_path
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        resolved = f.read().strip()
    if not resolved:
        raise RuntimeError(f"last_checkpoint file is empty: {checkpoint_path}")
    if not os.path.isabs(resolved):
        resolved = os.path.join(os.path.dirname(checkpoint_path), resolved)
    return resolved


def build_teacher_slowfast_mmaction(config_path, checkpoint_path, reorder_index=None):
    if Config is None or MODELS is None or load_checkpoint is None or register_all_modules is None:
        raise ImportError(
            "MMAction2/MMEngine is required for the teacher model. "
            "Please install mmaction2, mmengine and their dependencies first."
        ) from MMACTION_IMPORT_ERROR

    register_all_modules(init_default_scope=True)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.copy()
    recognizer = MODELS.build(model_cfg)
    load_msg = load_checkpoint(recognizer, checkpoint_path, map_location="cpu")
    if hasattr(load_msg, "missing_keys") and hasattr(load_msg, "unexpected_keys") and utils.is_main_process():
        print(
            "[teacher-load] "
            f"missing_keys={len(load_msg.missing_keys)} "
            f"unexpected_keys={len(load_msg.unexpected_keys)}"
        )

    num_classes = _infer_mmaction_num_classes(cfg, checkpoint_path)
    # The QEVD torchvision presets already convert videos to normalized tensors,
    # and the corresponding SlowFast QEVD configs use identity preprocessing.
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)
    if utils.is_main_process():
        print(f"[teacher-normalize] fixed identity mean={mean.tolist()} std={std.tolist()}")

    teacher = MMActionSlowFastTeacher(
        recognizer,
        mean=mean,
        std=std,
        reorder_index=reorder_index,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, num_classes, cfg


def build_collate_fns(args):
    if args.student_clip_len <= 0:
        raise ValueError(f"student_clip_len must be positive, got {args.student_clip_len}")
    if args.clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {args.clip_len}")
    if args.clip_len % args.student_clip_len != 0:
        raise ValueError(
            "clip_len must be divisible by student_clip_len for temporal downsampling: "
            f"got clip_len={args.clip_len}, student_clip_len={args.student_clip_len}"
        )

    teacher_train_tf = presets.VideoClassificationPresetTrain(
        crop_size=tuple(args.teacher_train_crop_size),
        resize_size=tuple(args.teacher_train_resize_size),
    )
    student_train_tf = presets.VideoClassificationPresetTrain(
        crop_size=tuple(args.student_train_crop_size),
        resize_size=tuple(args.student_train_resize_size),
    )
    student_eval_tf = presets.VideoClassificationPresetEval(
        crop_size=tuple(args.student_val_crop_size),
        resize_size=tuple(args.student_val_resize_size),
    )
    teacher_eval_tf = presets.VideoClassificationPresetEval(
        crop_size=tuple(args.teacher_train_crop_size),
        resize_size=tuple(args.teacher_train_resize_size),
    )

    def _split_batch(batch):
        videos, _, targets, video_idx = zip(*batch)
        return list(videos), list(targets), list(video_idx)


    t_stride = args.clip_len // args.student_clip_len

    def collate_train(batch):
        videos, targets, video_idx = _split_batch(batch)

        student_v = torch.stack([_tchw_to_cthw(student_train_tf(v))[:, ::t_stride, :, :] for v in videos], dim=0)
        teacher_v = torch.stack([_tchw_to_cthw(teacher_train_tf(v)) for v in videos], dim=0)
        
        targets = default_collate(targets)
        video_idx = default_collate(video_idx)
        return student_v, teacher_v, targets, video_idx

    def collate_val(batch):
        videos, targets, video_idx = _split_batch(batch)
        student_v = torch.stack([_tchw_to_cthw(student_eval_tf(v))[:, ::t_stride, :, :] for v in videos], dim=0)
        targets = default_collate(targets)
        video_idx = default_collate(video_idx)
        return student_v, targets, video_idx

    def collate_teacher_val(batch):
        videos, targets, video_idx = _split_batch(batch)
        teacher_v = torch.stack([_tchw_to_cthw(teacher_eval_tf(v)) for v in videos], dim=0)
        targets = default_collate(targets)
        video_idx = default_collate(video_idx)
        return teacher_v, targets, video_idx

    return collate_train, collate_val, collate_teacher_val


def train_one_epoch(student, teacher, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, alpha, temperature, scaler=None, args=None):
    student.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))
    header = f"Epoch: [{epoch}]"

    for step, (student_v, teacher_v, target, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        student_v = student_v.to(device, non_blocking=True)
        teacher_v = teacher_v.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_out = teacher(teacher_v)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            student_out = student(student_v)
            ce = F.cross_entropy(student_out, target)
            t = float(temperature)
            kl = F.kl_div(
                F.log_softmax(student_out / t, dim=1),
                F.softmax(teacher_out / t, dim=1),
                reduction="batchmean",
            )
            loss = (1.0 - alpha) * ce + alpha * (t * t) * kl

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        acc1, acc5 = utils.accuracy(student_out, target, topk=(1, 5))
        batch_size = student_v.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / max(time.time() - start_time, 1e-6))


def _get_sampler_size(data_loader):
    if isinstance(data_loader.sampler, DistributedSampler):
        return len(data_loader.sampler.dataset)
    return len(data_loader.sampler)


def _print_eval_loader_debug(name, data_loader, num_processed_samples, seen_counts=None):
    if not utils.is_main_process():
        return
    dataset = data_loader.dataset
    dataset_len = len(dataset) if hasattr(dataset, "__len__") else "unknown"
    samples_len = len(dataset.samples) if hasattr(dataset, "samples") else "unknown"
    sampler_len = _get_sampler_size(data_loader)
    loader_len = len(data_loader)
    msg = (
        f"[eval-debug] {name}: "
        f"dataset_len={dataset_len} samples_len={samples_len} "
        f"sampler_len={sampler_len} loader_len={loader_len} "
        f"processed={num_processed_samples}"
    )
    if seen_counts is not None:
        msg += f" seen_videos={int((seen_counts > 0).sum().item())}"
    print(msg)


def evaluate(student, data_loader, device, args=None):
    student.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0

    num_videos = len(data_loader.dataset.samples)
    num_classes = len(data_loader.dataset.classes)
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos,), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for student_v, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            student_v = student_v.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out = student(student_v)
            preds = torch.softmax(out, dim=1)
            for b in range(student_v.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()
            acc1, acc5 = utils.accuracy(out, target, topk=(1, 5))
            batch_size = student_v.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    num_data_from_sampler = _get_sampler_size(data_loader)

    if (
        hasattr(data_loader.dataset, "__len__")
        and num_data_from_sampler != num_processed_samples
        and utils.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} samples were used "
            "for the validation, which might bias the results."
        )

    metric_logger.synchronize_between_processes()
    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )

    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.all_reduce(agg_preds, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(agg_targets, op=torch.distributed.ReduceOp.MAX)
    _print_eval_loader_debug("student", data_loader, num_processed_samples)
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(" * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(acc1=agg_acc1, acc5=agg_acc5))
    metrics = {
        "clip_acc1": float(metric_logger.acc1.global_avg),
        "clip_acc5": float(metric_logger.acc5.global_avg),
        "video_acc1": float(agg_acc1.item()),
        "video_acc5": float(agg_acc5.item()),
    }
    return metrics


def evaluate_teacher(teacher, data_loader, device):
    teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Teacher Test:"
    num_processed_samples = 0

    num_videos = len(data_loader.dataset.samples)
    num_classes = len(data_loader.dataset.classes)
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos,), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for teacher_v, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            teacher_v = teacher_v.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out = teacher(teacher_v)
            preds = torch.softmax(out, dim=1)
            for b in range(teacher_v.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()
            acc1, acc5 = utils.accuracy(out, target, topk=(1, 5))
            batch_size = teacher_v.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    num_data_from_sampler = _get_sampler_size(data_loader)

    if (
        hasattr(data_loader.dataset, "__len__")
        and num_data_from_sampler != num_processed_samples
        and utils.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} samples were used "
            "for the teacher validation, which might bias the results."
        )

    metric_logger.synchronize_between_processes()
    print(
        " * Teacher Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )

    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.all_reduce(agg_preds, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(agg_targets, op=torch.distributed.ReduceOp.MAX)
    _print_eval_loader_debug("teacher", data_loader, num_processed_samples)
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(" * Teacher Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(acc1=agg_acc1, acc5=agg_acc5))
    return {
        "clip_acc1": float(metric_logger.acc1.global_avg),
        "clip_acc5": float(metric_logger.acc5.global_avg),
        "video_acc1": float(agg_acc1.item()),
        "video_acc5": float(agg_acc5.item()),
    }


def append_eval_result_txt(output_dir, metrics, epoch=None):
    if not output_dir or not utils.is_main_process():
        return
    log_path = os.path.join(output_dir, "eval_results.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f"epoch={epoch}" if epoch is not None else "test_only"
    line = (
        f"[{timestamp}] {tag} "
        f"clip_acc1={metrics['clip_acc1']:.4f} "
        f"clip_acc5={metrics['clip_acc5']:.4f} "
        f"video_acc1={metrics['video_acc1']:.4f} "
        f"video_acc5={metrics['video_acc5']:.4f}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    if args.student_checkpoint_init and args.resume:
        raise ValueError("--student-checkpoint-init and --resume cannot be used together.")

    device = torch.device(args.device)
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset_train, dataset_val = create_datasets(args)
    num_classes = len(dataset_train.classes)

    reorder_index = None
    if args.label_reorder == "ann-file":
        teacher_cfg = Config.fromfile(args.teacher_config)
        reorder_index = build_teacher_to_student_reorder_index(
            teacher_cfg,
            args.data_path,
            dataset_train,
            dataset_val,
        )
    elif utils.is_main_process():
        print("[label-align] disabled; using teacher logits order as-is")

    teacher, teacher_classes, _ = build_teacher_slowfast_mmaction(
        args.teacher_config,
        args.teacher_checkpoint,
        reorder_index=reorder_index,
    )
    if teacher_classes != num_classes:
        raise RuntimeError(f"Teacher classes ({teacher_classes}) != dataset classes ({num_classes})")
    teacher.to(device)

    student = build_student_model(args.student_model, num_classes, args.student_weights)
    if args.student_checkpoint_init:
        load_student_init_checkpoint(student, args.student_checkpoint_init)
    student.to(device)

    if args.distributed and args.sync_bn:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)

    collate_train, collate_val, collate_teacher_val = build_collate_fns(args)

    train_sampler = RandomClipSampler(dataset_train.video_clips, args.clips_per_video)
    val_sampler = UniformClipSampler(dataset_val.video_clips, args.clips_per_video)
    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)
        val_sampler = DistributedSampler(val_sampler, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_train,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_val,
    )
    data_loader_teacher_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_teacher_val,
    )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader_train)
    lr_milestones = [iters_per_epoch * (m - args.lr_warmup_epochs) for m in args.lr_milestones]
    lr_milestones = [m for m in lr_milestones if m > 0]
    main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        method = args.lr_warmup_method.lower()
        if method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(f"Invalid warmup lr method '{args.lr_warmup_method}'.")

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    student_without_ddp = student
    if args.distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
        student_without_ddp = student.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        student_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        eval_metrics = evaluate(student, data_loader_val, device=device, args=args)
        append_eval_result_txt(args.output_dir, eval_metrics, epoch=None)
        return

    if args.eval_teacher_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        teacher_metrics = evaluate_teacher(teacher, data_loader_teacher_val, device=device)
        if utils.is_main_process():
            print(
                "[teacher-only] "
                f"clip_acc1={teacher_metrics['clip_acc1']:.4f} "
                f"clip_acc5={teacher_metrics['clip_acc5']:.4f} "
                f"video_acc1={teacher_metrics['video_acc1']:.4f} "
                f"video_acc5={teacher_metrics['video_acc5']:.4f}"
            )
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            student,
            teacher,
            optimizer,
            lr_scheduler,
            data_loader_train,
            device,
            epoch,
            args.print_freq,
            args.distill_alpha,
            args.distill_temp,
            scaler,
            args,
        )
        
        eval_metrics = evaluate(student, data_loader_val, device=device, args=args)
        append_eval_result_txt(args.output_dir, eval_metrics, epoch=epoch)
        if args.output_dir:
            checkpoint = {
                "model": student_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Distill MMAction2 SlowFast teacher into torchvision video student with official preset transforms",
        add_help=add_help,
    )
    parser.add_argument("--data-path", default="./full_dataset/", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (cuda or cpu)")
    parser.add_argument("--kinetics-version", default="400", type=str, choices=["400", "600"])
    parser.add_argument("--clip-len", default=16, type=int, metavar="N", help="teacher/DataLoader clip length")
    parser.add_argument("--student-clip-len", default=16, type=int, help="student clip length after temporal sampling")
    parser.add_argument("--frame-rate", default=4, type=int, metavar="N")
    parser.add_argument("--clips-per-video", default=1, type=int, metavar="N")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=10, type=int, metavar="N")
    parser.add_argument("-j", "--workers", default=10, type=int, metavar="N")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--lr-warmup-epochs", default=1, type=int)
    parser.add_argument("--lr-warmup-method", default="linear", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float)
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--student-checkpoint-init", default="", type=str, help="initialize student weights from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    parser.add_argument("--cache-dir", default="./kinetics_cache", type=str)
    parser.add_argument(
        "--qevd-cache-dir",
        default="",
        type=str,
        help=(
            "Load KineticsWithVideoId datasets from MMAction VideoDataset_QEVD cache "
            "(expects train.pt and test.pt). This cache is assumed to use clip_len=16 and frame_rate=4."
        ),
    )
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument("--test-only", dest="test_only", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval-teacher-only", action="store_true")

    parser.add_argument("--teacher-config", required=True, type=str, help="MMAction2 config path for SlowFast teacher")
    parser.add_argument("--teacher-checkpoint", required=True, type=str, help="MMAction2 checkpoint path for SlowFast teacher")
    parser.add_argument(
        "--label-reorder",
        default="none",
        choices=["none", "ann-file"],
        help=(
            "How to align teacher logits to student labels. Use 'none' when the teacher was trained with "
            "the same KineticsWithVideoId label order as this script; use 'ann-file' to build a mapping "
            "from teacher ann_file_train/ann_file_val."
        ),
    )
    parser.add_argument("--student-model", default="r2plus1d_18", choices=["r2plus1d_18", "r3d_18", "mc3_18"])
    parser.add_argument("--student-weights", default="KINETICS400_V1", type=str)
    parser.add_argument("--distill-alpha", default=0.5, type=float)
    parser.add_argument("--distill-temp", default=2.0, type=float)

    parser.add_argument("--student-train-resize-size", default=(128, 171), nargs="+", type=int)
    parser.add_argument("--student-train-crop-size", default=(112, 112), nargs="+", type=int)
    parser.add_argument("--student-val-resize-size", default=(128, 171), nargs="+", type=int)
    parser.add_argument("--student-val-crop-size", default=(112, 112), nargs="+", type=int)

    parser.add_argument("--teacher-train-resize-size", default=(256, 256), nargs="+", type=int)
    parser.add_argument("--teacher-train-crop-size", default=(224, 224), nargs="+", type=int)

    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
