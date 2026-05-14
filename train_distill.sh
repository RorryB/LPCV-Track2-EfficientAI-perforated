

CUDA_VISIBLE_DEVICES=0 python references/video_classification/train_distill_r2plus1d_from_slowfast_mmaction.py \
    --data-path /mnt/ssd1/datasets/QEVD_organized \
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
    --epochs 30 \
    --batch-size 32 \
    --lr 0.005 \
    --output-dir checkpoint \
    --qevd-cache-dir datasets/cache/qevd \
    --amp

python export_r2plus1d.py \
    --device "Dragonwing IQ-9075 EVK" \
    --output-dir ./output \
    --checkpoint-path models/r2plus1d_r18_16x4_QEVD_sup.pth

python example_export.py  \
    --checkpoint-path models/r2plus1d_r18_16x4_QEVD_sup.pth \
    --num-frames 16 \
    --precision w8a8