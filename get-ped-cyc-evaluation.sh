#!/bin/bash

rm -rf /workspace/data/kitti-3d-object-detection/training/pred_2/*

# extract detections and format as KITTI labels:
./load_pred_from_point_pillars_pred.py \
    -i /workspace/data/kitti-3d-object-detection/ \
    -m /workspace/assignments/06-deep-detection/models/trained_v1.6/16/ped_cycle \
    -o /workspace/data/kitti-3d-object-detection/training/pred_2/

# evaluate:
./build/kitti_eval_node \
    /workspace/data/kitti-3d-object-detection/training/label_2/ \
    /workspace/data/kitti-3d-object-detection/training/pred_2

# collect results:
cp \
    /workspace/data/kitti-3d-object-detection/training/pred_2/plot/pedestrian_* \
    /workspace/data/kitti-3d-object-detection/training/pred_2/plot/cyclist_* \
    /workspace/data/kitti-3d-object-detection/training/pred_2/stats_pedestrian_* \
    /workspace/data/kitti-3d-object-detection/training/pred_2/stats_cyclist_* \
    /workspace/assignments/06-deep-detection/doc/eval-on-pointpillars-trained

