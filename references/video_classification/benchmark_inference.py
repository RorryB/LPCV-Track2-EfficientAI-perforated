"""
Inference benchmark: compare original vs perforated r2plus1d_18 models.

Three models are evaluated in order:
  1. Untrained baseline      -- fresh r2plus1d_18 (random weights, no dendrites)
  2. Pre-perforation model   -- beforeSwitch_0 + beforeSwitch_0_pai loaded via UPA (no dendrites yet)
  3. PAI-optimized model     -- best_model + best_model_pai loaded via UPA (with dendrites)

Example:
  CUDA_VISIBLE_DEVICES=0 ./ENV/bin/python \
    ./references/video_classification/benchmark_inference.py \
    --data-path ./Dataset/QEVD_sup_full \
    --pai-folder ./r2plus1d_dendritic \
    --batch-size 32 \
    --workers 10 \
    --cache-dataset \
    --cache-dir ./kinetics_cache
"""

import argparse
import os
import sys
import time
import warnings

import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import UniformClipSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import presets
import utils
from datasets import KineticsWithVideoId

from perforatedai import utils_perforatedai as UPA
from perforatedai import network_perforatedai as NPA


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_base_model(num_classes, weights=None):
    model = torchvision.models.video.r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model



# ---------------------------------------------------------------------------
# Dataset / dataloader
# ---------------------------------------------------------------------------

def build_val_loader(args, num_classes):
    val_dir = os.path.join(args.data_path, "val")
    cache_path = None
    if args.cache_dataset:
        import hashlib
        val_key = f"{val_dir}-16-400-4"
        h = hashlib.sha1(val_key.encode()).hexdigest()
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(args.cache_dir, h[:10] + ".pt")

    dataset_val = None
    if cache_path and os.path.exists(cache_path):
        try:
            dataset_val, _ = torch.load(cache_path, weights_only=False)
            print(f"[dataset] loaded val cache from {cache_path}")
        except Exception as e:
            warnings.warn(f"Cache load failed: {e}, rebuilding.")
            dataset_val = None

    if dataset_val is None:
        dataset_val = KineticsWithVideoId(
            args.data_path,
            frames_per_clip=16,
            num_classes="400",
            split="val",
            step_between_clips=1,
            transform=None,
            frame_rate=4,
            extensions=("avi", "mp4"),
            output_format="TCHW",
        )
        if cache_path:
            torch.save((dataset_val, val_dir), cache_path)

    student_eval_tf = presets.VideoClassificationPresetEval(
        crop_size=(112, 112),
        resize_size=(128, 171),
    )

    def _tchw_to_cthw(video):
        if video.shape[0] == 3:
            return video
        return video.permute(1, 0, 2, 3).contiguous()

    def collate_val(batch):
        videos, _, targets, video_idx = zip(*batch)
        student_v = torch.stack([_tchw_to_cthw(student_eval_tf(v)) for v in videos])
        targets = default_collate(targets)
        video_idx = default_collate(video_idx)
        return student_v, targets, video_idx

    val_sampler = UniformClipSampler(dataset_val.video_clips, 1)
    loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_val,
    )
    return loader, dataset_val


# ---------------------------------------------------------------------------
# Evaluation with timing
# ---------------------------------------------------------------------------

def run_eval(model, loader, dataset_val, device, label):
    model.eval()
    model.to(device)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(model)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters:    {params:,}")

    num_videos = len(dataset_val.samples)
    num_classes = len(dataset_val.classes)
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos,), dtype=torch.int32, device=device)

    total_clips = 0
    total_batches = len(loader)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (student_v, target, video_idx) in enumerate(loader):
            student_v = student_v.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out = model(student_v)
            # s3d outputs [N, C, 1, 1, 1] — flatten to [N, C]
            if out.ndim > 2:
                out = out.flatten(1)
            preds = torch.softmax(out, dim=1)
            for b in range(student_v.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()
            total_clips += student_v.size(0)
            elapsed_so_far = time.perf_counter() - t0
            cps = total_clips / max(elapsed_so_far, 1e-6)
            print(f"\r  [{batch_idx+1}/{total_batches}]  clips: {total_clips}  clips/s: {cps:.1f}  elapsed: {elapsed_so_far:.1f}s", end="", flush=True)
    print()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    _, top1_pred = agg_preds.topk(1, dim=1)
    correct1 = top1_pred.squeeze(1).eq(agg_targets.long()).sum().item()
    num_videos_eval = (agg_targets != -1).sum().item()

    video_acc1 = 100.0 * correct1 / num_videos_eval
    clips_per_sec = total_clips / elapsed
    params = sum(p.numel() for p in model.parameters())

    print(f"  Clips/sec:     {clips_per_sec:.1f}")
    print(f"  Total time:    {elapsed:.1f}s  ({total_clips} clips)")
    print(f"  Video Acc@1:   {video_acc1:.3f}%")

    return {
        "label": label,
        "params": params,
        "clips_per_sec": clips_per_sec,
        "elapsed": elapsed,
        "video_acc1": video_acc1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    loader, dataset_val = build_val_loader(args, num_classes=400)
    num_classes = len(dataset_val.classes)

    results = []

    # ------------------------------------------------------------------
    # 1. Baseline model (beforeSwitch_0_pai.pt — trained, no dendrites)
    # ------------------------------------------------------------------
    pre_switch_path = os.path.join(args.pai_folder, "beforeSwitch_0_pai.pt")
    if os.path.exists(pre_switch_path):
        print("\n>>> Building baseline model (beforeSwitch_0_pai.pt)...")
        model1 = build_base_model(num_classes, weights=None)
        model1 = NPA.load_pai_model(model1, pre_switch_path)
        results.append(run_eval(model1, loader, dataset_val, device, "Baseline Model"))
        del model1
    else:
        print(f"\n[skip] {pre_switch_path} not found")

    # ------------------------------------------------------------------
    # 2. Perforated model (best_model_pai.pt — with dendrites)
    # ------------------------------------------------------------------
    best_model_path = os.path.join(args.pai_folder, "best_model_pai.pt")
    print("\n>>> Building perforated model (best_model_pai.pt)...")
    model2 = build_base_model(num_classes, weights=None)
    model2 = NPA.load_pai_model(model2, best_model_path)
    results.append(run_eval(model2, loader, dataset_val, device, "Perforated Model"))
    del model2

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    W = 62
    print(f"\n\n{'='*W}")
    print(f"  {'COMPARISON SUMMARY'}")
    print(f"{'='*W}")
    print(f"  {'Model':<22} {'Params':>14} {'Clips/s':>10} {'Acc@1':>10}")
    print(f"  {'-'*22} {'-'*14} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['label']:<22} {r['params']:>14,} {r['clips_per_sec']:>10.1f} {r['video_acc1']:>9.3f}%")
    print(f"{'='*W}\n")


def get_args_parser():
    parser = argparse.ArgumentParser(description="Inference benchmark for original vs PAI-optimized r2plus1d_18")
    parser.add_argument("--data-path", default="./Dataset/QEVD_sup_full", type=str)
    parser.add_argument("--pai-folder", default="./r2plus1d_dendritic", type=str,
                        help="PAI save folder containing best_model.pt, best_model_pai.pt, beforeSwitch_0.pt etc.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-j", "--workers", default=10, type=int)
    parser.add_argument("--cache-dataset", default=True, action="store_true")
    parser.add_argument("--cache-dir", default="./kinetics_cache", type=str)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
