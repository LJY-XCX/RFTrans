def compute_errors(gt, pred, mask):
    """Compute error for depth as required for paper (RMSE, REL, etc)
    Args:
        gt (numpy.ndarray): Ground truth depth (metric). Shape: [B, H, W], dtype: float32
        pred (numpy.ndarray): Predicted depth (metric). Shape: [B, H, W], dtype: float32
        mask (numpy.ndarray): Mask of pixels to consider while calculating error.
                              Pixels not in mask are ignored and do not contribute to error.
                              Shape: [B, H, W], dtype: bool

    Returns:
        dict: Various measures of error metrics
    """

    safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))

    gt = torch.from_numpy(gt)
    pred = torch.from_numpy(pred)
    mask = torch.from_numpy(mask).bool()
    mask = torch.logical_and(mask, gt > 0)

    gt = gt[mask]
    pred = pred[mask]

    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.05).float().mean()
    a2 = (thresh < 1.10).float().mean()
    a3 = (thresh < 1.25).float().mean()

    rmse = ((gt - pred) ** 2).mean().sqrt()
    rmse_log = ((safe_log(gt) - safe_log(pred)) ** 2).mean().sqrt()
    log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
    abs_rel = ((gt - pred).abs() / gt).mean()
    mae = (gt - pred).abs().mean()
    sq_rel = ((gt - pred) ** 2 / gt).mean()

    measures = {
        'a1': round(a1.item() * 100, 5),
        'a2': round(a2.item() * 100, 5),
        'a3': round(a3.item() * 100, 5),
        'rmse': round(rmse.item(), 5),
        'rmse_log': round(rmse_log.item(), 5),
        'log10': round(log10.item(), 5),
        'abs_rel': round(abs_rel.item(), 5),
        'sq_rel': round(sq_rel.item(), 5),
        'mae': round(mae.item(), 5),
    }
    return measures

def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr

import OpenEXR, Imath
import glob
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

output_depth_path = '/home/jiyu/ClearGrasp/eval_depth_completion/results/exp-023/output-depth'
gt_depth_path = '/home/jiyu/ClearGrasp/eval_depth_completion/results/exp-023/gt-depth'

extension = '.exr'
output_depth_files = glob.glob(output_depth_path + '/*' + extension)
gt_depth_files = glob.glob(gt_depth_path + '/*' + extension)
assert len(output_depth_files) == len(gt_depth_files)
total_a1 = 0
total_a2 = 0
total_a3 = 0
total_rmse = 0
total_abs_rel = 0
total_sq_rel = 0
total_mae = 0
count = 0

for i in range(len(output_depth_files)):
    output_depth_path = output_depth_files[i]
    output_depth = exr_loader(output_depth_path, ndim=1)
    index = os.path.basename(output_depth_path).split('-')[0]
    gt_depth = exr_loader(os.path.join(gt_depth_path, index + '-gt-depth.exr'), ndim=1)

    # gt_depth_rgb = (gt_depth / 3) * 255
    # gt_depth_rgb = gt_depth_rgb.astype(np.uint8)
    # gt_depth_rgb = cv2.applyColorMap(gt_depth_rgb, cv2.COLORMAP_JET)
    # output_depth_rgb = (output_depth / 3) * 255
    # output_depth_rgb = output_depth_rgb.astype(np.uint8)
    # output_depth_rgb = cv2.applyColorMap(output_depth_rgb, cv2.COLORMAP_JET)

    mask = cv2.imread(os.path.join('/ssd1/jiyu/data/unity/cgreal/mask', f'mask_{int(index)}.png'))
    rgb = cv2.imread(os.path.join('/ssd1/jiyu/data/unity/cgreal/RGB', f'rgb_{int(index)}.png'))

    mask = mask[:, :, 1] / 255
    bi_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    bi_mask = bi_mask.astype(np.uint8)

    if np.all(bi_mask == 0):
        continue

    fig, ax = plt.subplots(2, 4)
    ax[0][0].imshow(gt_depth)
    ax[0][1].imshow(output_depth)
    ax[0][2].imshow(bi_mask)
    ax[0][3].imshow(rgb)

    true_mask = np.logical_and(bi_mask, gt_depth > 0)
    thresh = np.maximum(gt_depth/output_depth, output_depth/gt_depth)
    thresh[np.logical_not(true_mask)] = 1.2  # just a magic number
    ax[1][0].imshow(thresh)
    ax[1][1].imshow(thresh < 1.05)
    ax[1][2].imshow(thresh < 1.10)
    ax[1][3].imshow(thresh < 1.15)
    plt.tight_layout()
    plt.show()

    metrics = compute_errors(gt_depth, output_depth, bi_mask)
    total_a1 += metrics['a1']
    total_a2 += metrics['a2']
    total_a3 += metrics['a3']
    total_rmse += metrics['rmse']
    total_abs_rel += metrics['abs_rel']
    total_sq_rel += metrics['sq_rel']
    total_mae += metrics['mae']
    count += 1
    print(metrics)


print('a1: ', total_a1 / count)
print('a2: ', total_a2 / count)
print('a3: ', total_a3 / count)
print('rmse: ', total_rmse / count)
print('abs_rel: ', total_abs_rel / count)
# print('sq_rel: ', total_sq_rel / count)
print('mae: ', total_mae / count)