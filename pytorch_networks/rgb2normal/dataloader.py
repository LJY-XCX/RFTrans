#!/usr/bin/env python3

import os
import glob
import sys
from PIL import Image
import Imath
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio

from utils.utils import rgb_to_binary, scale_and_padding, flow_to_rgb, flow_loader, img_clip


class RGB2NormalDataset(Dataset):
    """
    Dataset class for training model on estimation of refraction flows.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(
            self,
            input_dir,
            flow_dir='',
            label_dir='',
            mask_dir='',
            transform=None,
            input_only=None,
    ):

        super().__init__()

        self.images_dir = input_dir
        self.flows_dir = flow_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_flow = []
        self._datalist_label = []
        self._datalist_mask = []
        self._extension_input = ['.png', '.jpg']  # The file extension of input images
        self._extension_flow = ['.npy']
        self._extension_label = ['.png', '.exr']
        self._extension_mask = ['.png']
        self._create_lists_filenames(self.images_dir, self.flows_dir, self.labels_dir, self.masks_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None)
        '''

        # Open input imgs

        image_path = self._datalist_input[index]

        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)
        _img = img_clip(_img)

        if self.flows_dir:
            flow_path = self._datalist_flow[index]
            _flow = np.load(flow_path)
            _flow = img_clip(_flow)
            _flow = _flow.transpose((2, 0, 1))  # To Shape: (2, H, W)


        # Open labels
        if self.labels_dir:
            label_path = self._datalist_label[index]
            _label = imageio.imread(label_path)
            _label = _label.astype(np.float32)

            _label = img_clip(_label)
            _label = _label.transpose((2, 0, 1))
            _label = (_label - 127.5) / 127.5  # reshape to (-1, 1)

        if self.masks_dir:
            mask_path = self._datalist_mask[index]
            _mask = imageio.imread(mask_path)  # 512 512 3
            _mask = _mask[..., 1] / 255
            _mask = img_clip(_mask)

            if self.labels_dir:
                _label[:, _mask == 0] = 0.0

            if self.flows_dir:
                _flow[:, _mask == 0] = 0.0


        # Apply image augmentations and convert to Tensor

        if self.transform:
            det_tf = self.transform.to_deterministic()

            _img = det_tf.augment_image(_img)

            if self.flows_dir:
                mask = np.all(_flow == -512.0, axis=0)

                _flow = _flow.transpose((1, 2, 0))  # To Shape: (H, W, 2)
                _flow[:, mask] = 0.0
                _flow = _flow.transpose((2, 0, 1))  # To Shape: (2, H, W)

            if self.labels_dir:
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_label == -1.0, axis=0)

                _label[:, mask] = 0.0

                _label = _label.transpose((1, 2, 0))  # To Shape: (H, W, 2)
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
                _label = _label.transpose((2, 0, 1))  # To Shape: (2, H, W)

            if self.masks_dir:
                _mask = det_tf.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        if self.flows_dir:
            _flow_tensor = torch.from_numpy(_flow).type(torch.float32)
        else:
            _flow_tensor = torch.zeros((2, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label).type(torch.float32)
        else:
            _label_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        if self.masks_dir:
            _mask_tensor = transforms.ToTensor()(_mask)
        else:
            _mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        ## correct shape
        return _img_tensor, _flow_tensor, _label_tensor, _mask_tensor

    def _create_lists_filenames(self, images_dir, flows_dir, labels_dir, masks_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored
            masks_dir (str): Path to the dir where masks are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        for ext in self._extension_input:
            imageSearchStr = os.path.join(images_dir, '*' + ext)
            imagepaths = sorted(glob.glob(imageSearchStr), key=lambda x: int(x.split('rgb_')[1].split('.')[0]))
            self._datalist_input = self._datalist_input + imagepaths

        numImages = len(self._datalist_input)

        if numImages == 0:
            raise ValueError('No images found in given directory. Searched in dir: {} '.format(images_dir))

        if flows_dir:
            assert os.path.isdir(flows_dir), ('Dataloader given flows directory that does not exist: "%s"' %
                                              (flows_dir))
            for ext in self._extension_flow:
                flowSearchStr = os.path.join(flows_dir, '*' + ext)

                flowpaths = sorted(glob.glob(flowSearchStr),
                                   key=lambda x: int(x.split('flow_')[1].split('.')[0]))

                self._datalist_flow = self._datalist_flow + flowpaths

            numFlows = len(self._datalist_flow)

            if numFlows == 0:
                raise ValueError('No flows found in given directory. Searched in dir: {} '.format(flows_dir))

            if numFlows != numImages:
                raise ValueError('Number of images and flows do not match. Images: %d, Flows: %d' % (numImages, numFlows))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"' %
                                               (labels_dir))
            for ext in self._extension_label:
                labelSearchStr = os.path.join(labels_dir, '*' + ext)
                print(labelSearchStr)
                if 'masked_normal' in labelSearchStr:
                    labelpaths = sorted(glob.glob(labelSearchStr),
                                        key=lambda x: int(x.split('masked_normal/')[1].split('_')[0]))
                elif 'normal' in labelSearchStr:
                    labelpaths = sorted(glob.glob(labelSearchStr), key=lambda x:int(x.split('normal_')[1].split('.')[0]))
                self._datalist_label = self._datalist_label + labelpaths

            numLabels = len(self._datalist_label)


            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,' +
                                 'found {} images and {} labels in dirs:\n'.format(numImages, numLabels) +
                                 'images: {}\nlabels: {}\n'.format(images_dir, labels_dir))
        if masks_dir:
            assert os.path.isdir(masks_dir), ('Dataloader given masks directory that does not exist: "%s"' %
                                               (masks_dir))
            for ext in self._extension_mask:
                maskSearchStr = os.path.join(masks_dir, '*' + ext)
                if 'binary_mask' in maskSearchStr:
                    maskpaths = sorted(glob.glob(maskSearchStr),
                                       key=lambda x: int(x.split('binary_mask/')[1].split('_')[0]))
                elif 'mask' in maskSearchStr:
                    maskpaths = sorted(glob.glob(maskSearchStr), key=lambda x: int(x.split('mask_')[1].split('.')[0]))

                self._datalist_mask = self._datalist_mask + maskpaths

            numMasks = len(self._datalist_mask)

            if numMasks == 0:
                raise ValueError('No masks found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numMasks:
                raise ValueError('The number of images and masks do not match. Please check data,' +
                                 'found {} images and {} masks in dirs:\n'.format(numImages, numMasks) +
                                 'images: {}\nmasks: {}\n'.format(images_dir, masks_dir))

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    augs = None  # augs_train
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = RGB2NormalDataset(input_dir='/ssd1/jiyu/data/unity/train/RGB',
                                    flow_dir='/ssd1/jiyu/data/unity/train/flow',
                                    label_dir='/ssd1/jiyu/data/unity/train/normal',
                                    mask_dir='/ssd1/jiyu/data/unity/train/mask',
                                    transform=augs,
                                    input_only=input_only)

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch

        img, flow, label, mask = batch

        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)
        print('image max: ', img.max())
        print('image min: ', img.min())
        print('flow max: ', flow.max())
        print('flow min: ', flow.min())
        print('label max: ', label.max())
        print('label min: ', label.min())

        flow_img = flow_to_rgb(flow) / 255

        label_img = (label + 1) / 2

        # Show Batch
        plt.imshow(label_img[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("label_image.png")
        plt.show()

        plt.imshow(flow_img[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("flow_image.png")
        plt.show()

        plt.imshow(img[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("rgb_image.png")
        plt.show()

        sample = torch.cat((img, label_img.type(torch.float32)), 2)
        im_vis = torchvision.utils.make_grid(sample[:8], nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.axis("off")  # remove axis ticks and labels
        plt.savefig("grid_image.png", bbox_inches="tight")  # save the figure with tight
        plt.show()

        break
