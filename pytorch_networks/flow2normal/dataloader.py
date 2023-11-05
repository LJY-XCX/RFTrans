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
from utils.utils import flow_loader, rgb_to_binary, flow_to_rgb, flowToColor, exr_loader, exr_saver, normal_to_rgb, img_clip


class Flow2NormalDataset(Dataset):

    def __init__(
            self,
            input_dir,
            label_dir='',
            mask_dir='',
            transform=None,
            input_only=None
    ):

        super().__init__()

        self.flows_dir = input_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []
        self._datalist_mask = []
        self._extension_input = ['.npy']  # The file extension of input images
        self._extension_label = ['.png', '.exr']
        self._extension_mask = ['.png']
        self._create_lists_filenames(self.flows_dir, self.labels_dir, self.masks_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):

        flow_path = self._datalist_input[index]

        _flo = np.load(flow_path)
        _flo = img_clip(_flo)

        if self.labels_dir:
            label_path = self._datalist_label[index]
            _label = imageio.imread(label_path)
            _label = _label.astype(np.float32)

            _label = img_clip(_label)
            _label = _label.transpose((2, 0, 1))
            _label = (_label - 127.5) / 127.5  # reshape to (-1, 1)


        if self.masks_dir:
            mask_path = self._datalist_mask[index]
            _mask = imageio.imread(mask_path)
            _mask = _mask[..., 1] / 255.0
            _mask = img_clip(_mask)

            if self.labels_dir:
                _label[:, _mask == 0] = 0.0

            if self.flows_dir:
                _flo = _flo.transpose((2, 0, 1))
                _flo[:, _mask == 0] = 0.0
                _flo = _flo.transpose((1, 2, 0))

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()

            _flo = det_tf.augment_image(_flo)

            if self.labels_dir:
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_label == -1.0, axis=0)
                _label[:, mask] = 0.0

                _label = _label.transpose((1, 2, 0))  # To Shape: (H, W, 3)
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
                _label = _label.transpose((2, 0, 1))  # To Shape: (3, H, W)

            if self.masks_dir:
                _mask = det_tf.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _flo_tensor = transforms.ToTensor()(_flo).type(torch.float32)

        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label).type(torch.float32)
        else:
            _label_tensor = torch.zeros((3, _flo_tensor.shape[1], _flo_tensor.shape[2]), dtype=torch.float32)

        if self.masks_dir:
            _mask_tensor = transforms.ToTensor()(_mask)
        else:
            _mask_tensor = torch.ones((1, _flo_tensor.shape[1], _flo_tensor.shape[2]), dtype=torch.float32)

        return _flo_tensor, _label_tensor, _mask_tensor

    def _create_lists_filenames(self, flows_dir, labels_dir, masks_dir):

        assert os.path.isdir(flows_dir), 'Dataloader given images directory that does not exist: "%s"' % (flows_dir)
        for ext in self._extension_input:
            flowSearchStr = os.path.join(flows_dir, '*' + ext)
            for ext in self._extension_input:
                flowSearchStr = os.path.join(flows_dir, '*' + ext)

                flowpaths = sorted(glob.glob(flowSearchStr),
                                   key=lambda x: int(x.split('flow_')[1].split('.')[0]))

                self._datalist_input = self._datalist_input + flowpaths

        numFlows = len(self._datalist_input)

        if numFlows == 0:
            raise ValueError('No .npy or .flo file found in given directory. Searched in dir: {} '.format(flows_dir))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"' %
                                               (labels_dir))
            for ext in self._extension_label:
                labelSearchStr = os.path.join(labels_dir, '*' + ext)
                if 'masked_normal' in labelSearchStr:
                    labelpaths = sorted(glob.glob(labelSearchStr),
                                        key=lambda x: int(x.split('masked_normal/')[1].split('_')[0]))
                elif 'normal' in labelSearchStr:
                    labelpaths = sorted(glob.glob(labelSearchStr), key=lambda x:int(x.split('normal_')[1].split('.')[0]))
                self._datalist_label = self._datalist_label + labelpaths

            numLabels = len(self._datalist_label)

            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(imageSearchStr))
            if numFlows != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,' +
                                 'found {} images and {} labels in dirs:\n'.format(numFlows, numLabels) +
                                 'images: {}\nlabels: {}\n'.format(flows_dir, labels_dir))
        if masks_dir:
            assert os.path.isdir(masks_dir), ('Dataloader given masks directory that does not exist: "%s"' %
                                               (masks_dir))
            for ext in self._extension_mask:
                maskSearchStr = os.path.join(masks_dir, '*' + ext)
                if 'binary_mask' in maskSearchStr:
                    maskpaths = sorted(glob.glob(maskSearchStr), key=lambda x:int(x.split('binary_mask/')[1].split('_')[0]))
                elif 'mask' in maskSearchStr:
                    maskpaths = sorted(glob.glob(maskSearchStr), key=lambda x:int(x.split('mask_')[1].split('.')[0]))

                self._datalist_mask = self._datalist_mask + maskpaths

            numMasks = len(self._datalist_mask)
            if numMasks == 0:
                raise ValueError('No masks found in given directory. Searched for {}'.format(flowSearchStr))
            if numFlows != numMasks:
                raise ValueError('The number of images and masks do not match. Please check data,' +
                                 'found {} images and {} masks in dirs:\n'.format(numFlows, numMasks) +
                                 'images: {}\nmasks: {}\n'.format(flows_dir, masks_dir))

    def _activator_masks(self, images, augmenter, parents, default):

        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    augs = None
    input_only = None
    db_train = Flow2NormalDataset(input_dir='/ssd1/jiyu/data/unity/train/flow',
                                    label_dir='/ssd1/jiyu/data/unity/train/normal',
                                    mask_dir='/ssd1/jiyu/data/unity/train/mask',
                                    transform=augs,
                                    input_only=input_only)

    batch_size = 16
    testloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch

        flo, label, mask = batch

        print('flow shape, type: ', flo.shape, flo.dtype)
        print('label shape, type: ', label.shape, label.dtype)
        print('flow max: ', flo.max())
        print('flow min: ', flo.min())
        print('label max: ', label.max())
        print('label min: ', label.min())

        flow_img = flow_to_rgb(flo) / 255
        label_img = (label + 1) / 2

        plt.imshow(flow_img[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("flo_image.png")
        plt.show()

        plt.imshow(label[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("label_image.png")
        plt.show()

        plt.imshow(mask[0].numpy().transpose(1, 2, 0), cmap='Greys')
        plt.axis("off")
        plt.savefig("mask_image.png")
        plt.show()

        sample = torch.cat((flow_img, label_img.type(torch.float32)), 2)
        im_vis = torchvision.utils.make_grid(sample[:8], nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.axis("off")  # remove axis ticks and labels
        plt.savefig("grid_image.png", bbox_inches="tight")  # save the figure with tight
        plt.show()

        break


