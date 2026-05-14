_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmaction.datasets.video_dataset_QEVD_sup',
        'mmaction.datasets.VideoSampler',
    ],
    allow_failed_imports=False)

dataset_type = 'VideoDataset_QEVD_sup'
data_root = '<your_home_directory>/LPCV-Track2-EfficientAI/datasets/QEVD_sup_full'
data_root_train = f'{data_root}/train'
data_root_val = f'{data_root}/val'
cache_root = '<your_home_directory>/LPCV-Track2-EfficientAI/datasets/cache/qevd'
train_dir = data_root
val_dir = data_root
test_dir = data_root
ann_file_train = '<your_home_directory>/LPCV-Track2-EfficientAI/datasets/QEVD_sup_full/meta/train.txt'
ann_file_val = '<your_home_directory>/LPCV-Track2-EfficientAI/datasets/QEVD_sup_full/meta/val.txt'
ann_file_test = '<your_home_directory>/LPCV-Track2-EfficientAI/datasets/QEVD_sup_full/meta/test.txt'

model = dict(
    backbone=dict(
        fast_pathway=dict(depth=101),
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(depth=101, fusion_kernel=7)),
    cls_head=dict(num_classes=92,
                  loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)))

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='VideoSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root_train),
        pipeline=train_pipeline,
        qevd_split="train",
        clip_len=16,
        frame_rate=4,
        cache_root=cache_root,
        data_path=train_dir))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='VideoSampler_UNI', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
        qevd_split='test',
        clip_len=16,
        frame_rate=4,
        cache_root=cache_root,
        data_path=val_dir))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='VideoSampler_UNI', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
        qevd_split='test',
        clip_len=16,
        frame_rate=4,
        cache_root=cache_root,
        data_path=test_dir))

val_evaluator = dict(type='AccMetric')
# val_evaluator = dict(type='ConfusionMatrix')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2),
    paramwise_cfg=dict(
            custom_keys={
            'backbone': dict(lr_mult=1, decay_mult=1),
            'cls_head': dict(lr_mult=1, decay_mult=1),
        }),
    )

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=3)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3), logger=dict(interval=100))

load_from = '<your_home_directory>/LPCV-Track2-EfficientAI/mmaction/mmaction2/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth'
