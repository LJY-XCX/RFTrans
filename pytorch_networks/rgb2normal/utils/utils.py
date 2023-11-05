'''Functions for reading and saving EXR images using OpenEXR.
'''

import sys

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
from matplotlib.colors import hsv_to_rgb
import torch.nn as nn
import os

sys.path.append('../..')
from api import utils as api_utils

def img_clip(img):
    if len(img.shape) == 3:
        h, w, c = img.shape
        if h == w:
            clipped_img = img
        elif h > w:
            clip_size = h - w
            clipped_img = img[clip_size//2: h - clip_size//2, :, :]
        else:
            clip_size = w - h
            clipped_img = img[:, clip_size//2: w - clip_size//2, :]
    else:
        h, w = img.shape
        if h == w:
            clipped_img = img
        elif h > w:
            clip_size = h - w
            clipped_img = img[clip_size//2: h - clip_size//2, :]
        else:
            clip_size = w - h
            clipped_img = img[:, clip_size//2: w - clip_size//2]

    return clipped_img


def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    return camera_normal_rgb


def rgb_to_binary(img):
    # Convert to grayscale
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # Apply threshold to convert to binary
    binary = np.where(gray > 1, 1, 0)

    return binary


def flow_loader(path):
    assert os.path.isfile(path) is True, "file does not exist %r" % str(file)
    assert path[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.int16, count=2 * w * h)
    data = np.resize(data, (h, w, 2))

    return data


def flow_saver(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    from https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/optflow.py
    """

    if flow.shape[0] == 2:
        flow = flow.transpose((1, 2, 0))
    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(202021.25, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.uint16).tofile(f)


def flow_to_rgb(flows):
    '''converts a refractive flow array into an RGB image tensor.
        Note that if you want to convert a .flo file to an RGB image tensor,
        you need to use function flow_loader first.
    '''

    if torch.is_tensor(flows):
        flows = flows.numpy()
    if len(flows.shape) == 4:
        if flows.shape[1] == 2:
            flows = flows.transpose((0, 2, 3, 1))

        B, H, W, _ = flows.shape

        img_list = []

        for index in range(B):
            rgb = flowToColor(flows[index])
            img_list.append(rgb)

        rgb_img = torch.tensor(np.stack(img_list), dtype=torch.uint8).permute((0, 3, 1, 2))

    elif len(flows.shape) == 3:
        if flows.shape[0] == 2:
            flows = flows.transpose((1, 2, 0))
        H, W, _ = flows.shape
        rgb = flowToColor(flows)

        rgb_img = torch.tensor(rgb, dtype=torch.uint8).permute((2, 0, 1))

    else:
        raise ValueError("Incorrect flow shape :{}".format(flows.shape))

    return rgb_img


def flowToColor(flow, normalize=True, info=None, flow_mag_max=None):
    """Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.
    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
        info: Text to superimpose on image (typically, the epe for the predicted flow)
        flow_mag_max: Max flow to map to 255
    Returns:
        img: viewable representation of the dense optical flow in RGB format
        flow_avg: optionally, also return average flow magnitude
    Ref:
        - OpenCV 3.0.0-dev documentation » OpenCV-Python Tutorials » Video Analysis »
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def create_grid_image(inputs, flows, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        flows (Tensor): Batch Tensor of shape (B x 3 x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]

    output_tensor = outputs[:max_num_images_to_save]

    flow_tensor = flows[:max_num_images_to_save]

    label_tensor = labels[:max_num_images_to_save]

    img_tensor_rgb = img_tensor

    output_tensor_rgb = normal_to_rgb(output_tensor)

    label_tensor_rgb = normal_to_rgb(label_tensor)

    mask_invalid_pixels = torch.all(label_tensor == 0, dim=1, keepdim=True)
    mask_invalid_pixels = (torch.cat([mask_invalid_pixels] * 3, dim=1)).byte().bool()

    label_tensor_rgb[mask_invalid_pixels] = 0

    mask_invalid_pixels_rgb = torch.ones_like(img_tensor)
    mask_invalid_pixels_rgb[mask_invalid_pixels] = 0

    images = torch.cat((img_tensor_rgb, flow_tensor, output_tensor_rgb, label_tensor_rgb, mask_invalid_pixels_rgb), dim=3)

    grid_image = make_grid(images, 1, normalize=False, scale_each=False)

    return grid_image


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter)**power)


def scale_and_padding(img, new_scale=512):
    height, width = img.shape[:2]
    if height == width:
        img = cv2.resize(img, dsize=(new_scale, new_scale), interpolation=cv2.INTER_LINEAR)

    elif height < width:
        pad_szie = (width - height) // 2
        img = np.pad(img, ((pad_szie, pad_szie), (0, 0), (0, 0)), 'constant')
        img = cv2.resize(img, dsize=(new_scale, new_scale), interpolation=cv2.INTER_LINEAR)

    else:
        pad_size = (height - width) // 2
        img = np.pad(img, ((0, 0), (pad_size, pad_size), (0, 0)), 'constant')
        img = cv2.resize(img, dsize=(new_scale, new_scale), interpolation=cv2.INTER_LINEAR)

    return img
