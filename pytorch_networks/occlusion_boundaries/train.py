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
from modeling import deeplab
from tqdm import tqdm
import dataloader
from utils import utils

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
            db_synthetic = dataloader.OutlinesDataset(input_dir=dataset.images,
                                                            label_dir=dataset.labels,
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
            db = dataloader.OutlinesDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
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
        'Invalid model "{}" in config file. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet", "resnet", "simple_unet"]'
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
        model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
    else:
        # Our old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Let's use", torch.cuda.device_count(), "GPUs!")
# device_ids = [0]
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

###################### Setup Optimizer #############################
if config.train.model == 'unet':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train.optimAdam.learningRate,
                                 weight_decay=config.train.optimAdam.weightDecay)
    criterion = nn.CrossEntropyLoss(reduction='sum')

elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.train.optimSgd.learningRate),
                                momentum=float(config.train.optimSgd.momentum),
                                weight_decay=float(config.train.optimSgd.weight_decay))
    # optimizer = torch.optim.Adam(model.parameters(),
    #                               lr=config.train.optimAdam.learningRate,
    #                               weight_decay=config.train.optimAdam.weightDecay)
    criterion = utils.cross_entropy2d


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

    # Update Learning Rate Scheduler
    if config.train.lrScheduler == 'StepLR':
        lr_scheduler.step()
    elif config.train.lrScheduler == 'ReduceLROnPlateau':
        lr_scheduler.step(epoch_loss)

    model.train()

    running_loss = 0.0
    total_iou = 0.0

    for iter_num, batch in enumerate(tqdm(trainLoader)):

        total_iter_num += 1

        # Get data
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        outputs = model(inputs)

        predictions = torch.max(outputs, 1)[1]

        # assert torch.isnan(outputs).sum() == 0, print(outputs)

        if config.train.model == 'unet':
            loss = criterion(outputs, labels.long().squeeze(1))
        elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)
        _total_iou, per_class_iou, num_images_per_class = utils.get_iou(predictions,
                                                                        labels,
                                                                        n_classes=config.train.numClasses)
        total_iou += _total_iou

    # Log Epoch Loss
    epoch_loss = running_loss / (len(trainLoader))
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    print('\nTrain Epoch Loss: {:.4f}'.format(epoch_loss))

    # Log mIoU
    miou = total_iou / len(trainLoader.dataset)
    writer.add_scalar('data/Train mIoU', miou, total_iter_num)
    print('Train mIoU: {:.4f}'.format(miou))

    # Log Current Learning Rate
    # TODO: NOTE: The lr of adam is not directly accessible. Adam creates a loss for every parameter in model.
    #    The value read here will only reflect the initial lr value.
    current_learning_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_learning_rate, total_iter_num)

    # Log 3 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                             outputs.detach().cpu(),
                                             labels.detach().cpu(),
                                             max_num_images_to_save=15)
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
        total_iou = 0.0
        for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
            inputs, labels = sample_batched

            # Forward pass of the mini-batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)

            loss = criterion(outputs, labels.long().squeeze(1))
            # if config.train.model == 'unet':
            #     loss = criterion(outputs, labels, reduction='sum')
            # elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet':
            #     loss = criterion(outputs, labels, reduction='sum')
            #     loss /= config.train.batchSize

            running_loss += loss.item()

            predictions = torch.max(outputs, 1)[1]
            _total_iou, per_class_iou, num_images_per_class = utils.get_iou(predictions,
                                                                            labels,
                                                                            n_classes=config.train.numClasses)
            total_iou += _total_iou

            # Pring loss every 20 Batches
            # if (iter_num % 20) == 0:
            #     print('Epoch{} Batch{} BatchLoss: {:.4f} '.format(epoch, iter_num, loss.item()))

        # Log Epoch Loss
        epoch_loss = running_loss / (len(validationLoader))
        writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
        print('\nValidation Epoch Loss: {:.4f}'.format(epoch_loss))

        # Log mIoU
        miou = total_iou / len(validationLoader.dataset)
        writer.add_scalar('data/Validation mIoU', miou, total_iter_num)
        print('Validation mIoU: {:.4f}'.format(miou))

        # Log 10 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                                 outputs.detach().cpu(),
                                                 labels.detach().cpu(),
                                                 max_num_images_to_save=10)
            writer.add_image('Validation', grid_image, total_iter_num)

writer.close()



