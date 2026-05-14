# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Callable, List, Optional, Tuple, Union

import torch
import torchvision_backup
from mmengine.fileio import exists, list_from_file
from torch import Tensor
from torch import nn
from torchvision_backup.transforms import transforms

from mmaction.registry import DATASETS
from mmaction.structures import ActionDataSample
from mmaction.utils import ConfigType
from .base import BaseActionDataset


def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert video tensor from (T, C, H, W) to (C, T, H, W)."""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        trans = [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size, antialias=False),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomCrop(crop_size),
            ConvertBCHWtoCBHW(),
        ])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
    ):
        self.transforms = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size, antialias=False),
            transforms.Normalize(mean=mean, std=std),
            transforms.CenterCrop(crop_size),
            ConvertBCHWtoCBHW(),
        ])

    def __call__(self, x):
        return self.transforms(x)


class KineticsWithVideoId(torchvision_backup.datasets.Kinetics):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, video_idx


@DATASETS.register_module()
class VideoDataset_QEVD_sup(BaseActionDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        delimiter (str): Delimiter for the annotation file.
            Defaults to ``' '`` (whitespace).
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 delimiter: str = ' ',
                 qevd_split: str = "train",
                 clip_len: int = 16,
                 frame_rate: int = 4,
                 cache_root: Optional[str] = None,
                 data_path: Optional[str] = None,
                 **kwargs) -> None:
        self.delimiter = delimiter
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            **kwargs)
        
        if not cache_root:
            raise ValueError('cache_root must be set to a valid directory.')
        if not data_path:
            raise ValueError('data_path must point to the QEVD dataset root.')

        os.makedirs(cache_root, exist_ok=True)
        cache_path = os.path.join(cache_root, f"{clip_len}-{frame_rate}-{qevd_split}.pt")

        transform_train = VideoClassificationPresetTrain(
            crop_size=(224, 224),
            resize_size=(256, 256),
        )

        transform_test = VideoClassificationPresetEval(
            crop_size=(224, 224),
            resize_size=(256, 256),
        )
        
        if qevd_split == "train":
            self.clip_method = "random"
            # video, audio, label, video_idx
            if os.path.exists(cache_path):
                dataset, _ = torch.load(cache_path, weights_only=False)
                dataset.transform = transform_train
            else:
                dataset = KineticsWithVideoId(
                    data_path,
                    frames_per_clip=clip_len,
                    num_classes="400",
                    split="train",
                    step_between_clips=1,
                    transform=transform_train,
                    frame_rate=frame_rate,
                    extensions=("avi", "mp4"),
                    output_format="TCHW",
                )
                train_dir = os.path.join(data_path, f"train")
                save_on_master((dataset, train_dir), cache_path)
        elif qevd_split == "test":
            self.clip_method = "uni"
            if os.path.exists(cache_path):
                dataset, _ = torch.load(cache_path, weights_only=False)
                dataset.transform = transform_test
            else:
                dataset = KineticsWithVideoId(
                    data_path,
                    frames_per_clip=clip_len,
                    num_classes="400",
                    split="val",
                    step_between_clips=1,
                    transform=transform_test,
                    frame_rate=frame_rate,
                    # frame_rate=6,
                    extensions=("avi", "mp4"),
                    output_format="TCHW",
                )
                test_dir = os.path.join(data_path, f"val")
                save_on_master((dataset, test_dir), cache_path)

        self.qevd_dataset = dataset

    def __getitem__(self, idx: int) -> dict:
        packed_results = {}

        qevd_inputs = self.qevd_dataset[idx]
        inputs = qevd_inputs[0].unsqueeze(0)
        packed_results["inputs"] = inputs

        data_sample = ActionDataSample()

        data_sample.set_gt_label(qevd_inputs[2])

        # Set meta keys
        img_meta = {"img_shape": (qevd_inputs[0].shape[-2], qevd_inputs[0].shape[-1])}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __len__(self) -> int:
        return len(self.qevd_dataset)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
                filename, label = line_split[0], line_split[1:]
                label = list(map(int, label))
            # add fake label for inference datalist without label
            elif len(line_split) == 1:
                filename, label = line_split[0], -1
            else:
                filename, label = line_split
                label = int(label)
            if self.data_prefix['video'] is not None:
                filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list
