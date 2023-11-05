'''Train unet for refractive flows
'''
import argparse
import csv
import errno
import os
import glob
import io
import shutil

from termcolor import colored
import yaml
from attrdict import AttrDict
import imageio
import numpy as np
import h5py
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

from modeling import deeplab
import dataloader
from utils import utils

# Enable Multi-GPU training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

print('Inference of Refractive Flow Estimation model. Loading checkpoint...')

parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

###################### Load Config File #############################
CONFIG_FILE_PATH = args.configFile  #'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

###################### Load Checkpoint and its data #############################
if not os.path.isfile(config.eval.pathWeightsFile):
    raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
        config.eval.pathWeightsFile))

# Read config file stored in the model checkpoint to re-use it's params
CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location='cpu')
if 'model_state_dict' in CHECKPOINT:
    print(colored('Loaded data from checkpoint {}'.format(config.eval.pathWeightsFile), 'green'))

    config_checkpoint_dict = CHECKPOINT['config']
    config_checkpoint = AttrDict(config_checkpoint_dict)
else:
    raise ValueError('The checkpoint file does not have model_state_dict in it.\
                     Please use the newer checkpoint files!')

# Create directory to save results
SUBDIR_RESULT = 'results'
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
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": config.eval.imgHeight,
        "width": config.eval.imgWidth
    }, interpolation='nearest'),
])

# Make new dataloaders for each synthetic dataset
db_test_list_synthetic = []
if config.eval.datasetsSynthetic is not None:
    for dataset in config.eval.datasetsSynthetic:
        print('Creating Synthetic Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.RefractionFlowsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  mask_dir=dataset.masks,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list_synthetic.append(db)

# Make new dataloaders for each real dataset
db_test_list_real = []
if config.eval.datasetsReal is not None:
    for dataset in config.eval.datasetsReal:
        print('Creating Real Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.RefractionFlowsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  mask_dir=dataset.masks,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list_real.append(db)

# Create pytorch dataloaders from datasets
dataloaders_dict = {}
if db_test_list_synthetic:
    db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
    testLoader_synthetic = DataLoader(db_test_synthetic,
                                      batch_size=config.eval.batchSize,
                                      shuffle=False,
                                      num_workers=config.eval.numWorkers,
                                      drop_last=False)
    dataloaders_dict.update({'synthetic': testLoader_synthetic})

if db_test_list_real:
    db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)
    testLoader_real = DataLoader(db_test_real,
                                 batch_size=config.eval.batchSize,
                                 shuffle=False,
                                 num_workers=config.eval.numWorkers,
                                 drop_last=False)
    dataloaders_dict.update({'real': testLoader_real})


assert (len(dataloaders_dict) > 0), 'No valid datasets given in config.yaml to run inference on!'

###################### ModelBuilder #############################
if config.eval.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='xception', sync_bn=True,
                            freeze_bn=False)
elif config.eval.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='resnet', sync_bn=True,
                            freeze_bn=False)
elif config.eval.model == 'drn':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='drn', sync_bn=True,
                            freeze_bn=False)

#model.load_state_dict(CHECKPOINT['model_state_dict'])

model.load_state_dict({k.replace('module.', ''): v for k, v in CHECKPOINT['model_state_dict'].items()})


# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

### Select Loss Func ###
criterion = nn.MSELoss(reduction='mean')

### Run Validation and Test Set ###
print('\nInference - Refractive Flow Estimation')
print('-' * 50 + '\n')
print(colored('Results will be saved to: {}\n'.format(config.eval.resultsDir), 'green'))

for key in dataloaders_dict:
    print('Running inference on {} dataset:'.format(key))
    print('=' * 30)

    running_loss = 0.0

    testLoader = dataloaders_dict[key]
    for ii, sample_batched in enumerate(tqdm(testLoader)):
        # NOTE: In raw data, invalid surface normals are represented by [-1, -1, -1]. However, this causes
        #       problems during normalization of vectors. So they are represented as [0, 0, 0] in our dataloader output.

        inputs, labels, masks = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        #print(loss)
        running_loss += loss.item()

        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = outputs.detach().cpu()
        label_tensor = labels.detach().cpu()
        mask_tensor = masks.squeeze(1)

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor, mask_tensor)):
            img, output, label, mask = sample_batched
            # Save grid image with input, prediction and label
            if len(mask.shape) == 2:
                masks_3d = torch.stack((mask, mask, mask), dim=0).type(torch.float32)
            else:
                masks_3d = mask
            label_rgb = utils.flow_to_rgb(label.numpy()) / 255.
            output_rgb = utils.flow_to_rgb(output.numpy()) / 255.

            grid_image = make_grid([img, output_rgb, label_rgb, masks_3d], 4, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)

            result_path = os.path.join(results_dir, SUBDIR_RESULT,
                                       '{:09d}-flows-result.jpg'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(result_path, numpy_grid)

            # # Write Predicted Surface Normal as hdf5 file for depth2depth
            # # NOTE: The hdf5 expected shape is (3, height, width), dtype float32
            # output_path_hdf5 = os.path.join(results_dir, SUBDIR_NORMALS,
            #                                 '{:09d}-normals.h5'.format(ii * config.eval.batchSize + iii))
            # with h5py.File(output_path_hdf5, "w") as f:
            #     dset = f.create_dataset('/result', data=output.numpy())

            # Save PNG and EXR Output
            output_rgb = (output_rgb * 255).numpy().astype(np.uint8).transpose(1, 2, 0)
            output_rgb = cv2.resize(output_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            output_path_rgb = os.path.join(results_dir, SUBDIR_FLOWS,
                                           '{:09d}-flows.png'.format(ii * config.eval.batchSize + iii))
            output_path_flo = os.path.join(results_dir, SUBDIR_FLOWS,
                                           '{:09d}-flows.flo'.format(ii * config.eval.batchSize + iii))
            output_path_valid_mask = os.path.join(results_dir, SUBDIR_FLOWS,
                                                  '{:09d}-valid-mask.png'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(output_path_rgb, output_rgb)
            _, h, w = output.shape
            utils.flow_saver(output.numpy(), output_path_flo)

    num_batches = len(testLoader)  # Num of batches
    num_images = len(testLoader.dataset)  # Num of total images
    print('\nnum_batches:', num_batches)
    print('num_images:', num_images)
    epoch_loss = running_loss / num_batches
    print('Test Mean Loss: {:.4f}'.format(epoch_loss))

