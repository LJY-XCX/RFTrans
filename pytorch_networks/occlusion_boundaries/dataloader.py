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
import cv2
import utils.utils as utils


class OutlinesDataset(Dataset):

    def __init__(
            self,
            input_dir,
            label_dir='',
            transform=None,
            input_only=None
    ):

        super().__init__()

        self.imgs_dir = input_dir
        self.labels_dir = label_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []
        self._extension_input = ['.png']  # The file extension of input images
        self._extension_label = ['.png']
        self._create_lists_filenames(self.imgs_dir, self.labels_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):

        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)
        _img = utils.img_clip(_img)

        if self.labels_dir:
            label_path = self._datalist_label[index]
            # mask = imageio.imread(label_path)
            # mask = utils.img_clip(mask)
            # if 'binary_mask' not in label_path:
            #     mask = utils.rgb_to_binary(mask)
            # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # _label = np.zeros(_img.shape, dtype=np.uint8)
            # _label = cv2.drawContours(_label, contours, contourIdx=-1, color=(255, 255, 255), thickness=8, hierarchy=hierarchy)
            # _label = utils.rgb_to_binary(_label)
            _label = imageio.imread(label_path)


            # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()

            if self.labels_dir:
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))


        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)
        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label.astype(np.float32))
            _label_tensor = torch.unsqueeze(_label_tensor, 0)

        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor

    def _create_lists_filenames(self, imgs_dir, labels_dir):

        assert os.path.isdir(imgs_dir), 'Dataloader given images directory that does not exist: "%s"' % (imgs_dir)
        for ext in self._extension_input:
            imgSearchStr = os.path.join(imgs_dir, '*' + ext)

            imgpaths = sorted(glob.glob(imgSearchStr),
                               key=lambda x: int(x.split('rgb_')[1].split('.')[0]))
            self._datalist_input = self._datalist_input + imgpaths

        numimgs = len(self._datalist_input)
        if numimgs == 0:
            raise ValueError('No .png file found in given directory. Searched in dir: {} '.format(imgs_dir))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"' %
                                               (labels_dir))
            for ext in self._extension_label:
                labelSearchStr = os.path.join(labels_dir, '*' + ext)
                # if 'binary_mask' in labelSearchStr:
                #     labelpaths = sorted(glob.glob(labelSearchStr),
                #                         key=lambda x: int(x.split('binary_mask/')[1].split('_')[0]))
                # else:
                #     labelpaths = sorted(glob.glob(labelSearchStr),
                #                         key=lambda x: int(x.split('mask_')[1].split('.')[0]))
                labelpaths = sorted(glob.glob(labelSearchStr),
                                    key=lambda x: int(x.split('outline_')[1].split('.')[0]))
                self._datalist_label = self._datalist_label + labelpaths

            numLabels = len(self._datalist_label)

            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(imageSearchStr))
            if numimgs != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,' +
                                 'found {} images and {} labels in dirs:\n'.format(numImages, numLabels) +
                                 'images: {}\nlabels: {}\n'.format(imgs_dir, labels_dir))

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

    augs = iaa.Sequential([
        iaa.Resize({
            "height": 512,
            "width": 512
        }, interpolation='nearest'),
    ])
    input_only = None
    db_train = OutlinesDataset(input_dir='/ssd1/jiyu/data/unity/train/RGB',
                                    label_dir='/ssd1/jiyu/data/unity/train/boundary',
                                    transform=augs,
                                    input_only=input_only)

    batch_size = 16
    testloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch

        img, label = batch
        print(label.shape)
        label = torch.cat([label]*3, dim=1)
        print(label.shape)

        print('img shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)
        print('img max: ', img.max())
        print('img min: ', img.min())
        print('label max: ', label.max())
        print('label min: ', label.min())

        plt.imshow(img[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("image.png")
        plt.show()

        plt.imshow(label[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.savefig("mask.png")
        plt.show()

        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample[:8], nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.axis("off")  # remove axis ticks and labels
        plt.savefig("grid_image.png", bbox_inches="tight")  # save the figure with tight
        plt.show()

        break


