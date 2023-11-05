'''Script to run Depth Completion on Synthetic and Real datasets, visualizing the results and computing the error metrics.
This will save all intermediate outputs like surface normals, etc, create a collage of all the inputs and outputs and
create pointclouds from the input, modified input and output depth images.
'''
import argparse
import csv
import glob
import os
import shutil
import sys
import itertools

# Importing Pytorch before Open3D can cause unknown "invalid pointer" error
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import depth_completion_api_rftrans
from api import utils as api_utils

import attrdict
import imageio
import termcolor
import yaml
import math
import torch
import cv2
import numpy as np
from PIL import Image
from utils import img_clip

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    parser.add_argument('-m', '--maskInputDepth', action="store_true", help='Whether we should mask out objects in input depth')
    args = parser.parse_args()

    # Load Config File
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = attrdict.AttrDict(config_yaml)

    # Create directory to save results
    RESULTS_ROOT_DIR = config.resultsDir
    runs = sorted(glob.glob(os.path.join(RESULTS_ROOT_DIR, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(results_dir):
        if len(os.listdir(results_dir)) > 1:
            # Min 1 file always in folder: copy of config file
            results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id + 1))
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)
    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('\nSaving results to folder: ' + termcolor.colored('"{}"\n'.format(results_dir), 'green'))

    # Init depth completion API
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)
    # if config.files.camera_intrinsics is not None or config.files.camera_intrinsics is not '':
    #     print(colored('\nUsing camera intrinsics from yaml file of real camera!\n'), 'green')
    #     CONFIG_FILE_PATH = config.files.camera_intrinsics
    #     if not os.path.isfile(CONFIG_FILE_PATH):
    #         print('\nError: Camera Intrinsics yaml does not exist: {}\n'.format(CONFIG_FILE_PATH))
    #         exit()
    #     with open(CONFIG_FILE_PATH) as fd:
    #         config_intr_yaml = yaml.safe_load(fd)
    #         camera_params = attrdict.AttrDict(config_intr_yaml)
    #     fx = (float(config.depth2depth.xres) / camera_params.xres) * camera_params.fx
    #     fy = (float(config.depth2depth.yres) / camera_params.yres) * camera_params.fy
    #     cx = (float(config.depth2depth.xres) / camera_params.xres) * camera_params.cx
    #     cy = (float(config.depth2depth.yres) / camera_params.yres) * camera_params.cy
    # else:
    #     fx = int(config.depth2depth.fx)
    #     fy = int(config.depth2depth.fy)
    #     cx = int(config.depth2depth.cx)
    #     cy = int(config.depth2depth.cy)

    depthcomplete = depth_completion_api_rftrans.DepthToDepthCompletion(rgb2flowWeightsFile=config.rgb2flow.pathWeightsFile,
                                                                flow2normalWeightsFile=config.flow2normal.pathWeightsFile,
                                                                outlinesWeightsFile=config.outlines.pathWeightsFile,
                                                                masksWeightsFile=config.masks.pathWeightsFile,
                                                                rgb2flowModel=config.rgb2flow.model,
                                                                flow2normalModel=config.flow2normal.model,
                                                                outlinesModel=config.outlines.model,
                                                                masksModel=config.masks.model,
                                                                depth2depthExecutable=config.depth2depth.pathExecutable,
                                                                outputImgHeight=outputImgHeight,
                                                                outputImgWidth=outputImgWidth,
                                                                fx=int(config.depth2depth.fx),
                                                                fy=int(config.depth2depth.fy),
                                                                cx=int(config.depth2depth.cx),
                                                                cy=int(config.depth2depth.cy),
                                                                filter_d=config.outputDepthFilter.d,
                                                                filter_sigmaColor=config.outputDepthFilter.sigmaColor,
                                                                filter_sigmaSpace=config.outputDepthFilter.sigmaSpace,
                                                                maskinferenceHeight=config.masks.inferenceHeight,
                                                                maskinferenceWidth=config.masks.inferenceWidth,
                                                                normalsInferenceHeight=config.flow2normal.inferenceHeight,
                                                                normalsInferenceWidth=config.flow2normal.inferenceWidth,
                                                                outlinesInferenceHeight=config.outlines.inferenceHeight,
                                                                outlinesInferenceWidth=config.outlines.inferenceWidth,
                                                                min_depth=config.depthVisualization.minDepth,
                                                                max_depth=config.depthVisualization.maxDepth,
                                                                tmp_dir=results_dir)

    # Create lists of input data
    rgb_file_list = []
    depth_file_list = []
    segmentation_masks_list = []
    gt_depth_file_list = []
    for dataset in config.files:
        EXT_COLOR_IMG = ['-transparent-rgb-img.jpg', '-rgb.jpg', '.jpg', '.png']  #'-rgb.jpg' - includes normals-rgb.jpg
        EXT_DEPTH_IMG = ['-depth-rectified.exr', '-transparent-depth-img.exr', '.exr', '.png']
        EXT_DEPTH_GT = ['-depth-rectified.exr', '-opaque-depth-img.exr', '.exr', '.png']
        EXT_MASK = ['.png']
        for ext in EXT_COLOR_IMG:
            rgb_file_list += (sorted(glob.glob(os.path.join(dataset.image, '*' + ext)), key=lambda x:int(x.split('rgb_')[1].split('.')[0])))
        for ext in EXT_DEPTH_IMG:
            depth_file_list += (sorted(glob.glob(os.path.join(dataset.depth, '*' + ext)), key=lambda x:int(x.split('depth_')[1].split('.')[0])))
        assert len(rgb_file_list) == len(depth_file_list), (
            'number of rgb ({}) and depth images ({}) are not equal'.format(len(rgb_file_list), len(depth_file_list)))

        if dataset.masks is not None and dataset.masks != '':
            for ext in EXT_MASK:
                if 'binary' in dataset.masks:
                    segmentation_masks_list += (sorted(glob.glob(os.path.join(dataset.masks, '*' + ext)), key=lambda x:int(x.split('binary_mask/')[1].split('_')[0])))
                else:
                    segmentation_masks_list += (sorted(glob.glob(os.path.join(dataset.masks, '*' + ext)), key=lambda x:int(x.split('mask_')[1].split('.')[0])))
            assert len(rgb_file_list) == len(segmentation_masks_list), (
                'number of rgb ({}) and masks ({}) are not equal'.format(len(rgb_file_list),
                                                                         len(segmentation_masks_list)))
        if dataset.gt_depth is not None and dataset.gt_depth != '':
            for ext in EXT_DEPTH_GT:
                gt_depth_file_list += (sorted(glob.glob(os.path.join(dataset.gt_depth, '*' + ext)), key=lambda x:int(x.split('depth_')[1].split('.')[0])))
            assert len(rgb_file_list) == len(gt_depth_file_list), (
                'number of rgb ({}) and gt depth ({}) are not equal'.format(len(rgb_file_list),
                                                                            len(gt_depth_file_list)))
    segmentation_masks_list = segmentation_masks_list[:config.depth2depth.num_images]
    rgb_file_list = rgb_file_list[:config.depth2depth.num_images]
    depth_file_list = depth_file_list[:config.depth2depth.num_images]
    gt_depth_file_list = gt_depth_file_list[:config.depth2depth.num_images]

    print('Total Num of rgb_files:', len(rgb_file_list))
    print('Total Num of depth_files:', len(depth_file_list))
    print('Total Num of gt_depth_files:', len(gt_depth_file_list))
    print('Total Num of segmentation_masks:', len(segmentation_masks_list))
    assert len(rgb_file_list) > 0, ('No files found in given directories')

    # Create CSV File to store error metrics
    csv_filename = 'computed_errors.csv'
    field_names = ["Image Num", "RMSE", "REL", "MAE", "Delta 1.25", "Delta 1.25^2", "Delta 1.25^3"]
    with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()

    # Iterate over all the files and run depth completion
    a1_mean = 0.0
    a2_mean = 0.0
    a3_mean = 0.0
    rmse_mean = 0.0
    abs_rel_mean = 0.0
    mae_mean = 0.0
    sq_rel_mean = 0.0
    nan_number = 0

    for i in range(len(rgb_file_list)):

        # Run Depth Completion
        color_img = Image.open(rgb_file_list[i]).convert('RGB')
        color_img = np.array(color_img)
        color_img = img_clip(color_img)

        if depth_file_list[i][-4:] == '.exr':
            input_depth = api_utils.exr_loader(depth_file_list[i], ndim=1).astype(np.float32)
            input_depth = input_depth / config.depth2depth.input_depth_scale
        elif depth_file_list[i][-4:] == '.png':
            input_depth = cv2.imread(depth_file_list[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            input_depth = input_depth / config.depth2depth.input_depth_scale

        else:
            raise ValueError("Unknown Suffix")

        #  To solve the problem of depth error caused by being too close to the camera (tmp)
        input_depth = img_clip(input_depth)
        tmp_gt = cv2.imread(gt_depth_file_list[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / config.depth2depth.gt_depth_scale
        tmp_mask = imageio.imread(segmentation_masks_list[i])
        tmp_mask = tmp_mask[..., 1] / 255
        input_depth[tmp_mask == 0] = tmp_gt[tmp_mask == 0]
        input_depth = cv2.resize(input_depth, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)

        # NOTE: If no gt_depth present, it means the depth itself is gt_depth (syn data). We mask out all objects in input depth so depthcomplete can't cheat.
        if len(gt_depth_file_list) == 0 and len(segmentation_masks_list) > 0:
            if args.maskInputDepth:
                _masks = imageio.imread(segmentation_masks_list[i])
                masks = _masks[..., 1] / 255
                masks = cv2.resize(masks, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
                input_depth[masks > 0] = 0.0

        try:
            output_depth, filtered_output_depth = depthcomplete.depth_completion(
                color_img,
                input_depth,
                inertia_weight=float(config.depth2depth.inertia_weight),
                smoothness_weight=float(config.depth2depth.smoothness_weight),
                tangent_weight=float(config.depth2depth.tangent_weight),
                mode_modify_input_depth=config.modifyInputDepth.mode,
                dilate_mask=True)

        except depth_completion_api_rftrans.DepthCompletionError as e:
            print('Depth Completion Failed:\n  {}\n  ...skipping image {}'.format(e, i))
            continue

        # Compute Errors in Depth Estimation over the Masked Area
        # If a folder of masks is given, use it to calc error only over masked regions, else over entire image
        if segmentation_masks_list:
            seg_mask = imageio.imread(segmentation_masks_list[i])
            seg_mask = cv2.resize(seg_mask, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            seg_mask = seg_mask[..., 1] / 255
        else:
            seg_mask = np.full((outputImgHeight, outputImgWidth), True, dtype=float)

        # If Ground Truth depth folder is given, use that to compute errors. In case of Synthetic data, input depth is GT depth.
        if gt_depth_file_list:
            if gt_depth_file_list[i][-4:] == '.exr':
                depth_gt = api_utils.exr_loader(gt_depth_file_list[i], ndim=1)
            elif gt_depth_file_list[i][-4:] == '.png':
                depth_gt = cv2.imread(gt_depth_file_list[i], cv2.IMREAD_UNCHANGED).astype(np.float32)  # 0 to 3 meters, in int16
                depth_gt = depth_gt / config.depth2depth.gt_depth_scale
        else:
            if depth_file_list[i][-4:] == '.exr':
                depth_gt = api_utils.exr_loader(depth_file_list[i], ndim=1)
            elif depth_file_list[i][-4:] == '.png':
                depth_gt = cv2.imread(depth_file_list[i], cv2.IMREAD_UNCHANGED).astype(np.float32)  # 0 to 3 meters, in int16
                depth_gt = depth_gt / config.depth2depth.gt_depth_scale

        depth_gt = cv2.resize(depth_gt, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
        depth_gt[np.isnan(depth_gt)] = 0
        depth_gt[np.isinf(depth_gt)] = 0
        mask_valid_region = (depth_gt > 0)

        mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
        # mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)
        # output_depth[~seg_mask] = input_depth[~seg_mask]

        metrics = depthcomplete.compute_errors(depth_gt, output_depth, mask_valid_region)

        print('\nImage {:09d} / {}:'.format(i, len(rgb_file_list) - 1))
        print('{:>15}:'.format('rmse'), metrics['rmse'])
        print('{:>15}:'.format('abs_rel'), metrics['abs_rel'])
        print('{:>15}:'.format('mae'), metrics['mae'])
        print('{:>15}:'.format('a1.05'), metrics['a1'])
        print('{:>15}:'.format('a1.10'), metrics['a2'])
        print('{:>15}:'.format('a1.25'), metrics['a3'])

        if math.isnan(metrics['rmse']) or math.isnan(metrics['abs_rel']) or math.isnan(metrics['mae']) or math.isnan(metrics['a1']) or math.isnan(metrics['a2']) or math.isnan(metrics['a3']):
            nan_number += 1
            continue

        # Write the data into a csv file
        with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
            row_data = [
                i, metrics["rmse"], metrics["abs_rel"], metrics["mae"], metrics["a1"], metrics["a2"], metrics["a3"]
            ]
            writer.writerow(dict(zip(field_names, row_data)))

        a1_mean += metrics['a1']
        a2_mean += metrics['a2']
        a3_mean += metrics['a3']
        rmse_mean += metrics['rmse']
        abs_rel_mean += metrics['abs_rel']
        mae_mean += metrics['mae']
        sq_rel_mean += metrics['sq_rel']

        # Save Results of Depth Completion
        error_output_depth, error_filtered_output_depth = depthcomplete.store_depth_completion_outputs(
            root_dir=results_dir,
            files_prefix=i,
            min_depth=config.depthVisualization.minDepth,
            max_depth=config.depthVisualization.maxDepth,
            calc_normal=False)
        # print('    Mean Absolute Error in output depth (if Synthetic Data)   = {:.4f} cm'.format(error_output_depth))
        # print('    Mean Absolute Error in filtered depth (if Synthetic Data) = {:.4f} cm'.format(error_filtered_output_depth))

    # Calculate Mean Errors over entire Dataset
    a1_mean = round(a1_mean / (len(rgb_file_list) - nan_number), 2)
    a2_mean = round(a2_mean / (len(rgb_file_list) - nan_number), 2)
    a3_mean = round(a3_mean / (len(rgb_file_list) - nan_number), 2)
    rmse_mean = round(rmse_mean / (len(rgb_file_list) - nan_number), 3)
    abs_rel_mean = round(abs_rel_mean / (len(rgb_file_list) - nan_number), 3)
    mae_mean = round(mae_mean / (len(rgb_file_list) - nan_number), 3)
    sq_rel_mean = round(sq_rel_mean / (len(rgb_file_list) - nan_number), 3)

    print('\n\nMean Error Stats for Entire Dataset:')
    print('{:>15}:'.format('rmse_mean'), rmse_mean)
    print('{:>15}:'.format('abs_rel_mean'), abs_rel_mean)
    print('{:>15}:'.format('mae_mean'), mae_mean)
    print('{:>15}:'.format('a1.05_mean'), a1_mean)
    print('{:>15}:'.format('a1.10_mean'), a2_mean)
    print('{:>15}:'.format('a1.25_mean'), a3_mean)

    # Write the data into a csv file
    with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        row_data = ['MEAN', rmse_mean, abs_rel_mean, mae_mean, a1_mean, a2_mean, a3_mean]
        writer.writerow(dict(zip(field_names, row_data)))