import argparse
import errno
import glob
import io
import os
import random
import shutil

import cv2
cv2.setNumThreads(0)
import imgaug as ia
import numpy as np
import oyaml
import torch
import torch.nn as nn
from attrdict import AttrDict
from imgaug import augmenters as iaa
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from torchvision import transforms
from modeling import deeplab, UNet
from tqdm import tqdm
import loss_functions
import dataloader
from utils import utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run training of flow2normal model')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.safe_load(fd)  # Returns an ordered dict. Used for printing

config = AttrDict(config_yaml)

###################### Logs (TensorBoard)  #############################
# Create directory to save results
SUBDIR_RESULT = 'checkpoints'

results_root_dir = config.train.logsDir
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
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

MODEL_LOG_DIR = results_dir
CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, SUBDIR_RESULT)
shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' + colored('"{}"'.format(results_dir), 'blue'))

# Create a tensorboard object and Write config to tensorboard
writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

string_out = io.StringIO()
oyaml.dump(config_yaml, string_out, default_flow_style=False)
config_str = string_out.getvalue().split('\n')
string = ''
for line in config_str:
    string = string + '    ' + line + '\n\r'
writer.add_text('Config', string, global_step=None)

###################### DataLoader #############################
# Train Dataset - Create a dataset object for each dataset in our list, Concatenate datasets, select subset for training

augs = iaa.Sequential([
    iaa.Resize({
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
])

db_synthetic_lst = []
if config.train.datasetsTrain is not None:
    for dataset in config.train.datasetsTrain:
        if dataset.images:
            db_synthetic = dataloader.Flow2NormalDataset(input_dir=dataset.images,
                                                            label_dir=dataset.labels,
                                                            mask_dir=dataset.masks,
                                                            transform=augs,
                                                            input_only=None)
            db_synthetic_lst.append(db_synthetic)
    db_synthetic = torch.utils.data.ConcatDataset(db_synthetic_lst)


db_train_list = []
if config.train.datasetsTrain is not None:
    if config.train.datasetsTrain[0].images:
        train_size_synthetic = int(config.train.percentageDataForTraining * len(db_synthetic))
        db, _ = torch.utils.data.random_split(db_synthetic,
                                                (train_size_synthetic, len(db_synthetic) - train_size_synthetic))
        db_train_list.append(db)

db_train = torch.utils.data.ConcatDataset(db_train_list)
trainLoader = DataLoader(db_train,
                         batch_size=config.train.batchSize,
                         shuffle=True,
                         num_workers=config.train.numWorkers,
                         drop_last=True,
                         pin_memory=True)


db_val_list = []
if config.train.datasetsVal is not None:
    for dataset in config.train.datasetsVal:
        if dataset.images:
            db = dataloader.Flow2NormalDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  mask_dir=dataset.masks,
                                                  transform=augs,
                                                  input_only=None)

            train_size = int(config.train.percentageDataForValidation * len(db))
            db = torch.utils.data.Subset(db, range(train_size))
            db_val_list.append(db)


if db_val_list:
    db_val = torch.utils.data.ConcatDataset(db_val_list)

if db_val_list:
    assert (config.train.validationBatchSize <= len(db_val)), \
        ('validationBatchSize ({}) cannot be more than the ' +
         'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val))

    validationLoader = DataLoader(db_val,
                                  batch_size=config.train.validationBatchSize,
                                  shuffle=True,
                                  num_workers=config.train.numWorkers,
                                  drop_last=True)


def resize_tensor(input_tensor, height, width):
    augs_label_resize = iaa.Sequential([iaa.Resize({"height": height, "width": width}, interpolation='nearest')])
    det_tf = augs_label_resize.to_deterministic()
    input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)
    resized_array = det_tf.augment_images(input_tensor)
    resized_array = torch.from_numpy(resized_array.transpose(0, 3, 1, 2))
    resized_array = resized_array.type(torch.DoubleTensor)

    return resized_array


if config.train.continueTraining:
    print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
    if not os.path.isfile(config.train.pathPrevCheckpoint):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(config.train.pathPrevCheckpoint))

    CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')

    if 'model_state_dict' in CHECKPOINT:
        # Our weights file with various dicts
        model.load_state_dict(CHECKPOINT['model_state_dict'])
    elif 'state_dict' in CHECKPOINT:
        # Original Author's checkpoint
        model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
    else:
        # Our old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)


###################### ModelBuilder #############################
if config.train.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='xception', sync_bn=True,
                            freeze_bn=False)
elif config.train.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='resnet', sync_bn=True,
                            freeze_bn=False)
elif config.train.model == 'drn':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='drn', sync_bn=True,
                            freeze_bn=False)  # output stride is 8 for drn
elif config.train.model == 'resnet':
    model = resnet.ResNet()
elif config.train.model == 'simple_unet':
    model = UNet(n_channels=3, n_classes=3)
else:
    raise ValueError(
        'Invalid model "{}" in config file. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet", "resnet", "simple_unet"]'
        .format(config.train.model))


print("Let's use", torch.cuda.device_count(), "GPUs!")
device_ids = [0]
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

###################### Setup Optimizer #############################
if config.train.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(config.train.optimSgd.learningRate),
                                momentum=float(config.train.optimSgd.momentum),
                                weight_decay=float(config.train.optimSgd.weight_decay))
elif config.train.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config.train.optimAdam.learningRate),
                                 weight_decay=float(config.train.optimAdam.weightDecay))
else:
    raise ValueError(
        'Invalid optimizer "{}" in config file. Must be one of ["SGD","Adam"]'
        .format(config.train.optimizer))

if not config.train.lrScheduler:
    pass
elif config.train.lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.train.lrSchedulerStep.step_size,
                                                   gamma=float(config.train.lrSchedulerStep.gamma))
elif config.train.lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=float(config.train.lrSchedulerPlateau.factor),
                                                              patience=config.train.lrSchedulerPlateau.patience,
                                                              verbose=True)
elif config.train.lrScheduler == 'lr_poly':
    pass

else:
    raise ValueError(
        "Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(
            config.train.lrScheduler))

# Continue Training from prev checkpoint if required
if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
    if 'optimizer_state_dict' in CHECKPOINT:
        optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
    else:
        print(
            colored(
                'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))

### Select Loss Func ###
if config.train.lossFunc == 'cosine':
    criterion = loss_functions.loss_fn_cosine
elif config.train.lossFunc == 'radians':
    criterion = loss_functions.loss_fn_radians
else:
    raise ValueError("Invalid lossFunc from config file. Can only be ['cosine', 'radians']. " +
                     "Value passed is: {}".format(config.train.lossFunc))

###################### Train Model #############################
# Set total iter_num (number of batches seen by model, used for logging)
total_iter_num = 0
START_EPOCH = 0
END_EPOCH = config.train.numEpochs

if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
    if 'model_state_dict' in CHECKPOINT:
        # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
        total_iter_num = CHECKPOINT['total_iter_num'] + 1
        START_EPOCH = CHECKPOINT['epoch'] + 1
        END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
    else:
        print(
            colored(
                'Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                       Starting from epoch num 0', 'red'))

for epoch in range(START_EPOCH, END_EPOCH):
    print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
    print('-' * 30)

    # Log the current Epoch Number
    writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

    ###################### Training Cycle #############################
    print('Train:')
    print('=' * 10)

    running_loss = 0.0
    running_mean = 0
    running_median = 0

    for iter_num, batch in enumerate(tqdm(trainLoader)):

        model.train()
        total_iter_num += 1

        # Get data
        inputs, labels, masks = batch

        tmp = torch.zeros((inputs.shape[0], 3, inputs.shape[2], inputs.shape[3]))
        tmp[:, :2, :, :] = inputs

        inputs = tmp.to(device)
        labels = labels.to(device)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors.float(), p=2, dim=1)
        # assert torch.isnan(outputs).sum() == 0, print(outputs)

        outputs = normal_vectors_norm
        if config.train.model in ["drn", "unet", "deeplab_xception", "deeplab_resnet", "refinenet", "resnet", "simple_unet"]:
            loss = criterion(outputs, labels)
        else:
            raise ValueError(
                'Invalid model "{}" in config file. Must be one of ["drn", "unet", "deeplab_xception", "deeplab_resnet", "refinenet", "resnet", "simple_unet"]'
                .format(config.train.model))

        loss.backward()
        optimizer.step()

        normal_vectors_norm = normal_vectors_norm.detach().cpu()
        inputs = inputs.detach().cpu()
        outputs = normal_vectors_norm
        labels = labels.detach().cpu()
        mask_tensor = masks.squeeze(1)

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, labels, mask_tensor)
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()
        running_loss += loss.item()

        writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)
        writer.add_scalar('data/Train Mean Error (deg)', loss_deg_mean.item(), total_iter_num)
        writer.add_scalar('data/Train Median Error (deg)', loss_deg_median.item(), total_iter_num)

        # Log 10 images every N iters
        if (iter_num % config.train.saveImageIntervalIter) == 0:

            flows_rgb = utils.flow_to_rgb(inputs[:, :2, :, :])
            grid_image = utils.create_grid_image(flows_rgb, outputs.float(), labels, max_num_images_to_save=16)
            writer.add_image('Train', grid_image, total_iter_num)

    num_samples = (len(trainLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    print('Train Epoch Loss: {:.4f}'.format(epoch_loss))
    epoch_mean = running_mean / num_samples
    epoch_median = running_median / num_samples
    print('Train Epoch Mean Error (deg): {:.4f}'.format(epoch_mean))
    print('Train Epoch Median Error (deg): {:.4f}'.format(epoch_median))

    # Update Learning Rate Scheduler
    if config.train.lrScheduler == 'StepLR':
        lr_scheduler.step()
    elif config.train.lrScheduler == 'ReduceLROnPlateau':
        lr_scheduler.step(epoch_loss)
    elif config.train.lrScheduler == 'lr_poly':
        if epoch % config.train.epochSize == config.train.epochSize - 1:
            lr_ = utils.lr_poly(float(config.train.optimSgd.learningRate), int(epoch - START_EPOCH),
                                int(END_EPOCH - START_EPOCH), 0.9)

            train_params = model.parameters()
            if model == 'drn':
                train_params = [{
                    'params': model.get_1x_lr_params(),
                    'lr': config.train.optimSgd.learningRate
                }, {
                    'params': model.get_10x_lr_params(),
                    'lr': config.train.optimSgd.learningRate * 10
                }]

            optimizer = torch.optim.SGD(train_params,
                                        lr=lr_,
                                        momentum=float(config.train.optimSgd.momentum),
                                        weight_decay=float(config.train.optimSgd.weight_decay))
    # Log Current Learning Rate
    # TODO: NOTE: The lr of adam is not directly accessible. Adam creates a loss for every parameter in model.
    #    The value read here will only reflect the initial lr value.
    current_learning_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_learning_rate, total_iter_num)

    if (epoch % config.train.saveImageInterval) == 0:
        flow_rgb = utils.flow_to_rgb(inputs[:, :2, :, :])

        grid_image = utils.create_grid_image(flow_rgb.detach().cpu(),
                                             outputs.detach().cpu().type(torch.float32),
                                             labels.detach().cpu(),
                                             max_num_images_to_save=16)
        writer.add_image('Train', grid_image, total_iter_num)

    if (epoch % config.train.saveModelInterval) == 0:
        filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
        if torch.cuda.device_count() > 1:
            model_params = model.module.state_dict()  # Saving nn.DataParallel model
        else:
            model_params = model.state_dict()

        torch.save(
            {
                'model_state_dict': model_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_iter_num': total_iter_num,
                'epoch_loss': epoch_loss,
                'config': config_yaml
            }, filename)

    ###################### Validation Cycle #############################
    if db_val_list:
        print('\nValidation:')
        print('=' * 10)

        model.eval()
        running_loss = 0.0
        running_mean = 0
        running_median = 0
        for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
            inputs, labels, masks = sample_batched

            inputs = inputs.to(device)
            assert torch.all(inputs[:, 2, :, :] == 0)
            labels = labels.to(device)

            with torch.no_grad():
                normal_vectors = model(inputs)

            normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
            outputs = normal_vectors_norm

            if config.train.model in ["drn", "unet", "deeplab_xception", "deeplab_resnet", "refinenet", "resnet", "simple_unet"]:
                loss = criterion(outputs, labels)
            else:
                raise ValueError("Invalid model")

            running_loss += loss.item()

            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
                normal_vectors_norm, labels.double())
            running_mean += loss_deg_mean.item()
            running_median += loss_deg_median.item()

        # Log Epoch Loss
        num_samples = (len(validationLoader))
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
        print('Validation Epoch Loss: {:.4f}'.format(epoch_loss))
        epoch_mean = running_mean / num_samples
        epoch_median = running_median / num_samples
        print('Val Epoch Mean: {:.4f}'.format(epoch_mean))
        print('Val Epoch Median: {:.4f}'.format(epoch_median))
        writer.add_scalar('data/Val Epoch Mean Error (deg)', epoch_mean, total_iter_num)
        writer.add_scalar('data/Val Epoch Median Error (deg)', epoch_median, total_iter_num)

        # Log 10 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            inputs = inputs.detach().cpu()
            flows_rgb = utils.flow_to_rgb(inputs[:, :2, :, :])

            grid_image = utils.create_grid_image(flows_rgb,
                                                outputs.detach().cpu().type(torch.float32),
                                                labels.detach().cpu(),
                                                max_num_images_to_save=10)

            writer.add_image('Validation', grid_image, total_iter_num)

writer.close()



