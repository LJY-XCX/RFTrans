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

from utils.utils import flow_loader, rgb_to_binary, flow_to_rgb, flowToColor, scale_and_padding, img_clip


class RefractionFlowsDataset(Dataset):
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
            label_dir='',
            mask_dir='',
            transform=None,
            input_only=None,
    ):

        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []
        self._datalist_mask = []
        self._extension_input = ['.png', '.jpg']  # The file extension of input images
        self._extension_label = ['.npy']
        self._extension_mask = ['.png']
        self._create_lists_filenames(self.images_dir, self.labels_dir, self.masks_dir)

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

        # Open labels
        if self.labels_dir:
            label_path = self._datalist_label[index]
            _label = np.load(label_path)
            _label = img_clip(_label)
            _label = _label.transpose((2, 0, 1))  # To Shape: (2, H, W)

        if self.masks_dir:
            mask_path = self._datalist_mask[index]
            _mask = imageio.imread(mask_path)  # 512 512 3
            _mask = _mask[..., 1] / 255
            _mask = img_clip(_mask)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = _img.astype(np.uint8)
            _img = det_tf.augment_image(_img)

            if self.labels_dir:
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_label == -512.0, axis=0)

                _label[:, mask] = 0.0

                _label = _label.transpose((1, 2, 0))  # To Shape: (H, W, 2)
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
                _label = _label.transpose((2, 0, 1))  # To Shape: (2, H, W)
                #  print('shape2', _label.shape)

            if self.masks_dir:
                _mask = _mask.astype(np.uint8)
                _mask = det_tf.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

                if self.labels_dir:
                    _label[:, _mask == 0] = 0.0


        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label).type(torch.float32)
        else:
            _label_tensor = torch.zeros((2, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        if self.masks_dir:
            _mask_tensor = transforms.ToTensor()(_mask)
        else:
            _mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor, _mask_tensor

    def _create_lists_filenames(self, images_dir, labels_dir, masks_dir):
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
            imagepaths = sorted(glob.glob(imageSearchStr), key=lambda x:int(x.split('rgb_')[1].split('.')[0]))
            self._datalist_input = self._datalist_input + imagepaths

        numImages = len(self._datalist_input)


        if numImages == 0:
            raise ValueError('No images found in given directory. Searched in dir: {} '.format(images_dir))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"' %
                                               (labels_dir))
            for ext in self._extension_label:
                flowSearchStr = os.path.join(labels_dir, '*' + ext)

                flowpaths = sorted(glob.glob(flowSearchStr),
                                   key=lambda x: int(x.split('flow_')[1].split('.')[0]))

                self._datalist_label = self._datalist_label + flowpaths

            numLabels = len(self._datalist_label)

            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(flowSearchStr))
            if numImages != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,' +
                                 'found {} images and {} labels in dirs:\n'.format(numImages, numLabels) +
                                 'images: {}\nlabels: {}\n'.format(images_dir, labels_dir))
        if masks_dir:
            assert os.path.isdir(masks_dir), ('Dataloader given masks directory that does not exist: "%s"' %
                                               (masks_dir))
            for ext in self._extension_mask:
                maskSearchStr = os.path.join(masks_dir, '*' + ext)
                maskpaths = sorted(glob.glob(maskSearchStr), key=lambda x:int(x.split('mask_')[1].split('.')[0]))
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

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

    augs = None  # augs_train
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = RefractionFlowsDataset(input_dir='/ssd1/jiyu/data/unity/train/RGB',
                                     label_dir='/ssd1/jiyu/data/unity/train/flow',
                                     mask_dir='/ssd1/jiyu/data/unity/train/mask',
                                    transform=augs,
                                    input_only=input_only)

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch

        img, label, mask = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)
        print('image max: ', img.max())
        print('image min: ', img.min())

        print('label max: ', label.max())
        print('label min: ', label.min())


        # Show Batch
        label_img = flow_to_rgb(label.numpy()) / 255.  # (B, C, H, W)

        plt.imshow(label_img[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("label_image.png")
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
