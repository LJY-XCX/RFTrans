'''Train unet for refractive flow
'''

import argparse
import errno
import glob
import io
import os
import random
import shutil
import time

# Enable Multi-GPU training

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
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
from tqdm import tqdm

import dataloader
from modeling import deeplab
from utils import utils

###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run training of refractive flow prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.safe_load(fd)  # Returns an ordered dict. Used for printing

config = AttrDict(config_yaml)
# print(colored('Config being used for training:\n{}\n\n'.format(oyaml.dump(config_yaml)), 'green'))

###################### Logs (TensorBoard)  #############################
# Create directory to save results
SUBDIR_RESULT = 'checkpoints'

results_root_dir = config.train.logsDir
runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id+1))
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
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),

    # Bright Patches
    iaa.Sometimes(
        0.1,
        iaa.blend.Alpha(factor=(0.2, 0.7),
                        first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                          upscale_method='cubic',
                                                          iterations=(1, 2)),
                        name="simplex-blend")),

    # Color Space Mods
    iaa.Sometimes(
        0.3,
        iaa.OneOf([
            iaa.Add((20, 20), per_channel=0.7, name="add"),
            iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                               name="hue"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                               name="sat"),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
            iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
        ])),

    # Blur and Noise
    iaa.Sometimes(
        0.2,
        iaa.SomeOf((1, None), [
            iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                       iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
            iaa.OneOf([
                iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
            ]),
        ],
                   random_order=True)),

    # Colored Blocks
    iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
])
input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]

db_synthetic_lst = []
if config.train.datasetsTrain is not None:
    for dataset in config.train.datasetsTrain:
        if dataset.images:
            db_synthetic = dataloader.RefractionFlowsDataset(input_dir=dataset.images,
                                                             label_dir=dataset.labels,
                                                             mask_dir=dataset.masks,
                                                             transform=augs_train,
                                                             input_only=input_only)
            db_synthetic_lst.append(db_synthetic)
    db_synthetic = torch.utils.data.ConcatDataset(db_synthetic_lst)

# Validation Dataset
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
])

db_val_list = []
if config.train.datasetsVal is not None:
    for dataset in config.train.datasetsVal:
        if dataset.images:
            db = dataloader.RefractionFlowsDataset(input_dir=dataset.images,
                                                   label_dir=dataset.labels,
                                                   mask_dir=dataset.masks,
                                                   transform=augs_test,
                                                   input_only=None)
            train_size = int(config.train.percentageDataForValidation * len(db))
            db = torch.utils.data.Subset(db, range(train_size))
            db_val_list.append(db)

if db_val_list:
    db_val = torch.utils.data.ConcatDataset(db_val_list)

# Test Dataset - Real
db_test_list = []
if config.train.datasetsTestReal is not None:
    for dataset in config.train.datasetsTestReal:
        if dataset.images:
            mask_dir = dataset.masks if hasattr(dataset, 'masks') and dataset.masks else ''

            db = dataloader.RefractionFlowsDataset(input_dir=dataset.images,
                                                   label_dir=dataset.labels,
                                                   mask_dir=mask_dir,
                                                   transform=None,
                                                   input_only=None)
            db_test_list.append(db)
    if db_test_list:
        db_test = torch.utils.data.ConcatDataset(db_test_list)


# Test Dataset - Synthetic
db_test_synthetic_list = []
if config.train.datasetsTestSynthetic is not None:
    for dataset in config.train.datasetsTestSynthetic:
        if dataset.images:
            db = dataloader.RefractionFlowsDataset(input_dir=dataset.images,
                                                   label_dir=dataset.labels,
                                                   transform=augs_test,
                                                   input_only=None)
            db_test_synthetic_list.append(db)
    if db_test_synthetic_list:
        db_test_synthetic = torch.utils.data.ConcatDataset(db_test_synthetic_list)

# Create dataloaders
if db_val_list:
    assert (config.train.validationBatchSize <= len(db_val)), \
        ('validationBatchSize ({}) cannot be more than the ' +
         'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val))

    validationLoader = DataLoader(db_val,
                                  batch_size=config.train.validationBatchSize,
                                  shuffle=True,
                                  num_workers=16,
                                  drop_last=True)
if db_test_list:
    assert (config.train.testBatchSize <= len(db_test)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test))

    testLoader = DataLoader(db_test,
                            batch_size=config.train.testBatchSize,
                            shuffle=False,
                            num_workers=16,
                            drop_last=True)
if db_test_synthetic_list:
    assert (config.train.testBatchSize <= len(db_test_synthetic)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test_synthetic_list))

    testSyntheticLoader = DataLoader(db_test_synthetic,
                                     batch_size=config.train.testBatchSize,
                                     shuffle=True,
                                     num_workers=16,
                                     drop_last=True)


# Resize Tensor
def resize_tensor(input_tensor, height, width):
    augs_label_resize = iaa.Sequential([iaa.Resize({"height": height, "width": width}, interpolation='nearest')])
    det_tf = augs_label_resize.to_deterministic()
    input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)
    resized_array = det_tf.augment_images(input_tensor)
    resized_array = torch.from_numpy(resized_array.transpose(0, 3, 1, 2))
    resized_array = resized_array.type(torch.DoubleTensor)

    return resized_array


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
else:
    raise ValueError(
        'Invalid model "{}" in config file. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet"]'
        .format(config.train.model))

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
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
        model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
    else:
        # Our old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)

print("Let's use", torch.cuda.device_count(), "GPUs!")
# device_ids = [0, 1, 2, 3]
# model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if config.train.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(config.train.optimSgd.learningRate),
                                momentum=float(config.train.optimSgd.momentum),
                                weight_decay=float(config.train.optimSgd.weight_decay))
elif config.train.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config.train.optimAdam.learningRate),
                                 weight_decay=float(config.train.optimAdam.weightDecay))
else:
    raise ValueError(
            'Invalid optimizer "{}" in config file. Must be one of ["SGD", "Adam"]'
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

criterion = nn.MSELoss(reduction='mean')
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
                             num_workers=32,
                             drop_last=True)

    running_loss = 0.0

    model.train()
    for iter_num, batch in enumerate(tqdm(trainLoader)):
        total_iter_num += 1

        # Get data
        inputs, labels, masks = batch
        masks = torch.stack((masks, masks, masks), dim=1)

        # labels = labels * masks
        if config.train.model == 'refinenet':
            labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
            labels_resized = labels_resized.to(device)
        if config.train.model == 'densenet':
            labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
            labels_resized = labels_resized.to(device)

        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Get Model Graph
        # if epoch == 0 and iter_num == 0:
        # writer.add_graph(model, inputs, False)

        # Forward + Backward Prop
        optimizer.zero_grad()
        # torch.set_grad_enabled(True)

        outputs = model(inputs)
        # outputs = outputs * masks
        # assert torch.isnan(outputs).sum() == 0, print(outputs)

        if config.train.model == 'unet':
            loss = criterion(outputs, labels.long().squeeze(1))
        elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
            loss = criterion(outputs, labels)
        else:
            raise ValueError(
                'Invalid model "{}" in config file. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet"]'
                .format(config.train.model))

        loss.backward()

        optimizer.step()

        outputs = outputs.detach().cpu()
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)
        if (iter_num % config.train.saveImageIntervalIter) == 0:
            rgb_outputs = utils.flow_to_rgb(outputs)
            rgb_labels = utils.flow_to_rgb(labels)
            grid_image = utils.create_grid_image(inputs, rgb_outputs, rgb_labels, max_num_images_to_save=16)
            writer.add_image('Train', grid_image, total_iter_num)

    # Log Epoch Loss
    num_samples = (len(trainLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    print('Train Epoch Loss: {:.4f}'.format(epoch_loss))

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

    # Log 10 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        if config.train.model == 'refinenet':
            outputs = resize_tensor(outputs.detach().cpu(), int(outputs.shape[2] * 4),
                                    int(outputs.shape[3] * 4))
        elif config.train.model == 'densenet':
            outputs = resize_tensor(outputs.detach().cpu(), int(outputs.shape[2] * 2),
                                    int(outputs.shape[3] * 2))
        else:
            outputs = outputs.detach().cpu()

        grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                             outputs.float(),
                                             labels.detach().cpu(),
                                             max_num_images_to_save=16)
        writer.add_image('Train', grid_image, total_iter_num)

    # Save the model checkpoint every N epochs
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

        for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
            inputs, labels, masks = sample_batched

            # Forward pass of the mini-batch
            if config.train.model == 'refinenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                labels_resized = labels_resized.to(device)
            if config.train.model == 'densenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                labels_resized = labels_resized.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            if config.train.model == 'unet':
                loss = criterion(outputs, labels.double())
            elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                loss = criterion(outputs, labels.double())
                loss /= config.train.batchSize
            elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                outputs = outputs.detach().cpu()
                labels_resized = labels_resized.detach().cpu()
                # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
                loss = criterion(outputs, labels_resized, reduction='sum')
                loss /= config.train.batchSize

            running_loss += loss.item()

        # Log Epoch Loss
        num_samples = (len(validationLoader))
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
        print('Validation Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log 10 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            if config.train.model == 'refinenet':
                outputs = resize_tensor(outputs.detach().cpu(), int(outputs.shape[2] * 4),
                                        int(outputs.shape[3] * 4))
            elif config.train.model == 'densenet':
                outputs = resize_tensor(outputs.detach().cpu(), int(outputs.shape[2] * 2),
                                        int(outputs.shape[3] * 2))
            else:
                outputs = outputs.detach().cpu()
            grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                                outputs.float(),
                                                labels.detach().cpu(),
                                                max_num_images_to_save=10)
            writer.add_image('Validation', grid_image, total_iter_num)

    ###################### Test Cycle - Real #############################
    if db_test_list:
        print('\nTesting:')
        print('=' * 10)

        model.eval()

        running_loss = 0
        img_tensor_list = []
        output_tensor_list = []
        label_tensor_list = []

        for iter_num, sample_batched in enumerate(tqdm(testLoader)):
            inputs, labels, masks = sample_batched


            # Forward pass of the mini-batch
            if config.train.model == 'refinenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                labels_resized = labels_resized.to(device)
            if config.train.model == 'densenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                labels_resized = labels_resized.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            # Save output images, one at a time, to results
            img_tensor = inputs.detach().cpu()
            output_tensor = outputs.detach().cpu()
            label_tensor = labels.detach().cpu()
            mask_tensor = masks.squeeze(1)

            img_tensor_list.append(img_tensor)
            output_tensor_list.append(output_tensor)
            label_tensor_list.append(label_tensor)

            running_loss = 0

            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor, mask_tensor)):
                img, output, label, mask = sample_batched
                if config.train.model == 'unet':
                    loss = criterion(output, label.long().squeeze(1))
                elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                    loss = criterion(output, label)
                running_loss += loss.item()

        # Log Epoch Loss
        num_samples = len(testLoader.dataset)
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Test Real Epoch Loss', epoch_loss, total_iter_num)
        print('\Test Real Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            grid_image = utils.create_grid_image(torch.cat(img_tensor_list, dim=0),
                                                 torch.cat(output_tensor_list, dim=0),
                                                 torch.cat(label_tensor_list, dim=0),
                                                 max_num_images_to_save=200)
            writer.add_image('Test Real', grid_image, total_iter_num)

    ###################### Test Cycle - Synthetic #############################
    if db_test_synthetic_list:
        print('\nTest Synthetic:')
        print('=' * 10)

        model.eval()

        running_loss = 0.0

        for iter_num, sample_batched in enumerate(tqdm(testSyntheticLoader)):
            inputs, labels, masks = sample_batched


            # Forward pass of the mini-batch
            if config.train.model == 'refinenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                labels_resized = labels_resized.to(device)
            if config.train.model == 'densenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                labels_resized = labels_resized.to(device)

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            if config.train.model == 'unet':
                loss = criterion(outputs, labels.double())
            elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                loss = criterion(outputs, labels.double())
                loss /= config.train.batchSize
            elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
                loss = criterion(outputs, labels_resized.double())
                loss /= config.train.batchSize

            running_loss += loss.item()

        # Log Epoch Loss
        num_samples = (len(testSyntheticLoader))
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Test Synthetic Epoch Loss', epoch_loss, total_iter_num)
        print('\Test Synthetic Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            if config.train.model == 'refinenet':
                outputs = resize_tensor(outputs.detach().cpu(), int(outputs.shape[2] * 4),
                                        int(outputs.shape[3] * 4))
            elif config.train.model == 'densenet':
                outputs = resize_tensor(outputs.detach().cpu(), int(outputs.shape[2] * 2),
                                        int(outputs.shape[3] * 2))
            else:
                outputs = outputs.detach().cpu()

            grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                                outputs.float(),
                                                labels.detach().cpu(),
                                                max_num_images_to_save=16)
            writer.add_image('Test Synthetic', grid_image, total_iter_num)

writer.close()
