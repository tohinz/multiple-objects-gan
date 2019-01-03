#!/bin/bash
DATASET="$1"
GPU="$2"

if [ "$DATASET" = "mnist" ]; then
    echo "Starting training on the Multi-MNIST data set."
    cd code/multi-mnist
    python main.py --cfg cfg/mnist_train.yml --gpu "$GPU"
    cd ../../
elif [ "$DATASET" = "clevr" ]; then
    echo "Starting training on the CLEVR data set."
    cd code/clevr
    python main.py --cfg cfg/clevr_train.yml --gpu "$GPU"
    cd ../../
elif [ "$DATASET" = "coco-stackgan-1" ]; then
    echo "Starting training on the MS-COCO data set."
    cd code/coco/stackgan
    python main.py --cfg cfg/coco_s1_train.yml --gpu "$GPU"
    cd ../../../
elif [ "$DATASET" = "coco-stackgan-2" ]; then
    echo "Starting training on the MS-COCO data set."
    cd code/coco/stackgan
    python main.py --cfg cfg/coco_s2_train.yml --gpu "$GPU"
    cd ../../../
elif [ "$DATASET" = "coco-attngan" ]; then
    echo "Starting training on the MS-COCO data set."
    cd code/coco/attngan
    python main.py --cfg cfg/coco_train.yml --gpu "$GPU"
    cd ../../../
else
    echo "Dataset argument must be either \"mnist\", \"clevr\", \"coco-stackgan-1\", \"coco-stackgan-2\", or \"coco-attngan\"."
fi
