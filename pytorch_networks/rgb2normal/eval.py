import argparse
import csv
import errno
import os
import glob
import io
import shutil
import yaml
import numpy as np
from attrdict import AttrDict
import imageio
from PIL import Image
import cv2
import loss_functions
from termcolor import colored
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
from utils import utils
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm
from modeling import deeplab, UNet
from rgb2normal_model import RGB2NormalModel
import dataloader

# Enable Multi-GPU evaling

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6


parser = argparse.ArgumentParser(description='Run evaling of rgb2normal model')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)  # Returns an ordered dict. Used for printing
config = AttrDict(config_yaml)

# Create directory to save results
SUBDIR_RESULT = 'results'
SUBDIR_NORMALS = 'normal_files'
SUBDIR_FLOWS = 'flow_files'

results_root_dir = config.eval.resultsDir
runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
if os.path.isdir(os.path.join(results_dir, SUBDIR_RESULT)):
    NUM_FILES_IN_EMPTY_FOLDER = 0
    if len(os.listdir(os.path.join(results_dir, SUBDIR_RESULT))) > NUM_FILES_IN_EMPTY_FOLDER:
        prev_run_id += 1
        results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
        os.makedirs(results_dir)
else:
    os.makedirs(results_dir)

try:
    os.makedirs(os.path.join(results_dir, SUBDIR_RESULT))
    os.makedirs(os.path.join(results_dir, SUBDIR_NORMALS))
    os.makedirs(os.path.join(results_dir, SUBDIR_FLOWS))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' + colored('"{}"\n'.format(results_dir), 'blue'))

# Create CSV File to store error metrics
csv_filename = 'computed_errors_exp_{:03d}.csv'.format(prev_run_id)
field_names = ["Image Num", "Loss", "<11.25", "<22.5", "<30"]
with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
    writer.writeheader()

###################### DataLoader #############################
augs_eval = iaa.Sequential([
    iaa.Resize({
        "height": config.eval.imgHeight,
        "width": config.eval.imgWidth
    }, interpolation='nearest'),
])

# Make new dataloaders
db_eval_list = []
if config.eval.datasetsReal is not None:
    for dataset in config.eval.datasetsReal:
        print('Creating Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.RGB2NormalDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  flow_dir=dataset.flows,
                                                  mask_dir=dataset.masks,
                                                  transform=augs_eval,
                                                  input_only=None)
            db_eval_list.append(db)

if config.eval.datasetsSynthetic is not None:
    for dataset in config.eval.datasetsSynthetic:
        print('Creating Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.RGB2NormalDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  flow_dir=dataset.flows,
                                                  mask_dir=dataset.masks,
                                                  transform=augs_eval,
                                                  input_only=None)
            db_eval_list.append(db)

dataloaders_dict = {}
if db_eval_list:
    db_eval = torch.utils.data.ConcatDataset(db_eval_list)
    evalLoader = DataLoader(db_eval, batch_size=config.eval.batchSize,
                                     shuffle=False,
                                     num_workers=config.eval.numWorkers,
                                     drop_last=False)
    dataloaders_dict.update({'loader': evalLoader})

assert (len(dataloaders_dict) > 0), 'No valid datasets given in config.yaml to run inference on!'

###################### ModelBuilder #############################
if config.eval.rgb2flow.model == 'drn':
    r2f_model = deeplab.DeepLab(num_classes=2, backbone='drn', sync_bn=True,
                                        freeze_bn=False)
    R2F_CHECKPOINT = torch.load(config.eval.rgb2flow.pathWeightsFile, map_location='cpu')
    r2f_config_checkpoint_dict = R2F_CHECKPOINT['config']
    r2f_config_checkpoint = AttrDict(r2f_config_checkpoint_dict)
    # r2f_model.load_state_dict(R2F_CHECKPOINT['model_state_dict'])
    print("load checkpoint for rgb2flow model")
    r2f_model.load_state_dict({k.replace('module.', ''): v for k, v in R2F_CHECKPOINT['model_state_dict'].items()})
else:
    raise ValueError("unexpected model type")

if config.eval.flow2normal.model == 'simple_unet':
    f2n_model = UNet(n_channels=3, n_classes=3)
    F2N_CHECKPOINT = torch.load(config.eval.flow2normal.pathWeightsFile, map_location='cpu')
    f2n_config_checkpoint_dict = F2N_CHECKPOINT['config']
    f2n_config_checkpoint = AttrDict(f2n_config_checkpoint_dict)
    # f2n_model.load_state_dict(F2N_CHECKPOINT['model_state_dict'])
    print("load checkpoint for flow2normal model")
    f2n_model.load_state_dict({k.replace('module.', ''): v for k, v in F2N_CHECKPOINT['model_state_dict'].items()})
else:
    raise ValueError("unexpected model type")

model = RGB2NormalModel(r2f_model, f2n_model)

print("Let's use", torch.cuda.device_count(), "GPUs!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# define criterion
r2f_criterion = nn.MSELoss(reduction='mean')
f2n_criterion = loss_functions.loss_fn_cosine

for key in dataloaders_dict:
    print('Running inference on {} dataset:'.format(key))
    print('=' * 30)

    running_loss = 0.0
    running_mean = []
    running_median = []
    running_percentage1 = []
    running_percentage2 = []
    running_percentage3 = []

    evalLoader = dataloaders_dict[key]
    for ii, sample_batched in enumerate(tqdm(evalLoader)):
        # NOTE: In raw data, invalid surface normals are represented by [-1, -1, -1]. However, this causes
        #       problems during normalization of vectors. So they are represented as [0, 0, 0] in our dataloader output.

        inputs, flows, labels, masks = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        flows = flows.to(device)

        with torch.no_grad():
            predicted_flows, rgb_flows, predicted_normals = model(inputs)

        predicted_normals = nn.functional.normalize(predicted_normals.double(), p=2, dim=1)

        r2f_loss = model.flow_loss(predicted_flows, flows, r2f_criterion)
        f2n_loss = model.normal_loss(predicted_normals, labels, f2n_criterion)
        loss = r2f_loss + 0.02 * f2n_loss
        running_loss += loss.item()

        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = predicted_normals.detach().cpu()
        label_tensor = labels.detach().cpu()
        flow_tensor = rgb_flows.detach().cpu()
        mask_tensor = masks.squeeze(1)

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, flow_tensor, output_tensor, label_tensor, mask_tensor)):
            img, flow, output, label, mask = sample_batched

            # Save grid image with input, prediction and label
            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels = loss_functions.metric_calculator(
                output, label, mask=mask)
            running_mean.append(loss_deg_mean.item())
            running_median.append(loss_deg_median.item())
            running_percentage1.append(percentage_1.item())
            running_percentage2.append(percentage_2.item())
            running_percentage3.append(percentage_3.item())

            # Write the data into a csv file
            with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                row_data = [((ii * config.eval.batchSize) + iii),
                            loss_deg_mean.item(),
                            loss_deg_median.item(),
                            percentage_1.item(),
                            percentage_2.item(),
                            percentage_3.item()]
                writer.writerow(dict(zip(field_names, row_data)))

            mask_3d = torch.stack((mask, mask, mask), dim=0).type(torch.float32)
            img = img / 255.
            output_rgb = utils.normal_to_rgb(output)
            flow_rgb = flow / 255.
            label_rgb = label / 255.

            grid_image = make_grid([img, flow_rgb, output_rgb, label_rgb, mask_3d], 5, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)

            result_path = os.path.join(results_dir, SUBDIR_RESULT,
                                       '{:09d}-normals-result.jpg'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(result_path, numpy_grid)

            output_path_flow = os.path.join(results_dir, SUBDIR_FLOWS,
                                       '{:09d}-flows-result.jpg'.format(ii * config.eval.batchSize + iii))
            output_flow = (flow_rgb * 255).numpy().astype(np.uint8).transpose(1, 2, 0)
            output_flow = cv2.resize(output_flow, (512, 512), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(output_path_flow, output_flow)

            output_rgb = (output_rgb * 255).numpy().astype(np.uint8).transpose(1, 2, 0)
            output_rgb = cv2.resize(output_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            output_path_rgb = os.path.join(results_dir, SUBDIR_NORMALS,
                                           '{:09d}-normals.jpg'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(output_path_rgb, output_rgb)

    num_batches = len(evalLoader)  # Num of batches
    num_images = len(evalLoader.dataset)  # Num of total images
    print('\nnum_batches:', num_batches)
    print('num_images:', num_images)
    epoch_loss = running_loss / num_batches
    print('Test Mean Loss: {:.4f}'.format(epoch_loss))

    epoch_mean = sum(running_mean) / num_images
    epoch_median = sum(running_median) / num_images
    epoch_percentage1 = sum(running_percentage1) / num_images
    epoch_percentage2 = sum(running_percentage2) / num_images
    epoch_percentage3 = sum(running_percentage3) / num_images
    print(
        '\nTest Metrics - Mean: {:.2f}deg, Median: {:.2f}deg, P1: {:.2f}%, P2: {:.2f}%, p3: {:.2f}%, num_images: {}\n\n'
        .format(epoch_mean, epoch_median, epoch_percentage1, epoch_percentage2, epoch_percentage3, num_images))




