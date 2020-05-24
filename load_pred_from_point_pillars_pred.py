#!/opt/conda/envs/deep-detection/bin/python

import argparse

import os
import glob
import shutil

import pickle
import second.data.kitti_dataset as kitti_ds

import pandas as pd

import progressbar

def get_model_detections(input_dir, model_dir, output_dir):
    """ 
    Create KITTI 3D object detection results from point pillars detections

    """    
    # init kitti dataset handler:
    kitti_dataset = kitti_ds.KittiDataset(
        root_path = input_dir,
        info_path = os.path.join(input_dir, 'kitti_infos_val.pkl'),
        class_names = ['Car', 'Pedestrian', 'Cyclist'], 
    )

    # load detection results:
    checkpoints = os.listdir(
        os.path.join(model_dir, 'results')
    )
    latest_checkpoint = max(
        checkpoints, key = lambda x: int(x.split('_')[-1])
    )
    latest_result = os.path.join(
        model_dir, 'results', latest_checkpoint, 'result.pkl'
    )

    print(f'[Point Pillar Evaluation]: load detections {latest_result}...')
    with open(latest_result, 'rb') as f:
        detections = pickle.load(f)

    # format as annotations:
    print('[Point Pillar Evaluation]: convert to annotations...')
    annos = kitti_dataset.convert_detection_to_kitti_annos(detections)

    # write intermediate results to temp output folder:
    print('[Point Pillar Evaluation]: save intermediate results...')
    kitti_ds.kitti_anno_to_label_file(annos, output_dir)

def generate_detection_results(input_dir, output_dir):
    """ 
    Create KITTI 3D object detection results from labels

    """
    # create output dir:
    os.mkdir(
        os.path.join(output_dir, 'data')
    )

    # get input point cloud filename:
    print('[Point Pillar Evaluation]: generate KITTI labels...')
    for input_filename in progressbar.progressbar(
        glob.glob(
            os.path.join(input_dir, '*.txt')
        )
    ):
        # read data:
        try:
            label = pd.read_csv(input_filename, sep=' ', header=None)
        except:
            continue
            
        label.columns = [
            'category',
            'truncation', 'occlusion', 
            'alpha',
            '2d_bbox_left', '2d_bbox_top', '2d_bbox_right', '2d_bbox_bottom', 
            'height', 'width', 'length', 
            'location_x', 'location_y', 'location_z',
            'rotation',
            'score'
        ]
        # add score:
        label['score'] *= 100.0
        # create output:
        output_filename = os.path.join(
            output_dir, 'data', os.path.basename(input_filename)
        )
        idx = label.groupby(['category'])['score'].transform(max) == label['score']
        if len(idx) > 0:
            label[idx].to_csv(output_filename, sep=' ', header=False, index=False)

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Generate KITTI 3D Object Detection result from ground truth labels.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of ground truth labels.",
        required=True, type=str
    )
    required.add_argument(
        "-m", dest="model", help="Input path of point pillars model.",
        required=True, type=str
    )
    required.add_argument(
        "-o", dest="output", help="Output path of detection results.",
        required=True, type=str
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    # init intermediate result buffer:
    if os.path.exists('pred_intermediate'):
        shutil.rmtree('pred_intermediate') 
    os.makedirs('pred_intermediate')

    get_model_detections(
        arguments.input,
        arguments.model,
        'pred_intermediate'      
    )

    # perform non-maximum supression:
    generate_detection_results('pred_intermediate', arguments.output)
    
    # remove intermediate result buffer:
    shutil.rmtree('pred_intermediate') 