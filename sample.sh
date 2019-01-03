#!/bin/bash
DATASET="$1"

if [ "$DATASET" = "mnist" ]; then
    echo "Sampling from the Multi-MNIST data set."
    echo "Going to Multi-MNIST folder."
    cd code/multi-mnist
    python main.py --cfg cfg/mnist_eval.yml
elif [ "$DATASET" = "clevr" ]; then
    echo "Sampling from the CLEVR data set."
    echo "Going to CLEVR folder."
    cd code/clevr
    python main.py --cfg cfg/clevr_eval.yml
elif [ "$DATASET" = "coco-stackgan-1" ]; then
    echo "Starting training on the MS-COCO data set."
    echo "Going to MS-COCO folder."
    cd code/coco/stackgan
    python main.py --cfg cfg/coco_s1_eval.yml
elif [ "$DATASET" = "coco-stackgan-2" ]; then
    echo "Starting training on the MS-COCO data set."
    echo "Going to MS-COCO folder."
    cd code/coco/stackgan
    python main.py --cfg cfg/coco_s2_eval.yml
elif [ "$DATASET" = "coco-attngan" ]; then
    echo "Starting training on the MS-COCO data set."
    echo "Going to MS-COCO folder."
    cd code/coco/attngan
    python main.py --cfg cfg/coco_eval.yml
else
    echo "Only one argument allowed. Must be either \"mnist\", \"clevr\", \"coco-stackgan-1\", \"coco-stackgan-2\", or \"coco-attngan\"."
fi
