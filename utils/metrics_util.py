import matplotlib.pyplot as plt
import numpy as np
import torchvision
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter
import torch


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def apply_cmap_to_volume(torch_tensor):
    cmap_vol = np.apply_along_axis(cm.viridis, 0, torch_tensor.numpy())  # converts prediction to cmap!
    return torch.from_numpy(np.squeeze(cmap_vol))


def write_image_grid(writer: SummaryWriter, input, target, prediction, step, sample='Training', size=6):
    writer.add_image(f'{sample} Inputs', torchvision.utils.make_grid(input[:size, :, :, :]), global_step=step)
    # writer.add_image(f'{sample} Ground Truth', torchvision.utils.make_grid(apply_cmap_to_volume(target[:size, :, :, :])), global_step=step)
    # writer.add_image(f'{sample} Prediction', torchvision.utils.make_grid(apply_cmap_to_volume(prediction[:size, :, :, :])), global_step=step)
    writer.add_image(f'{sample} Ground Truth', torchvision.utils.make_grid(target[:size, :, :, :]), global_step=step)
    writer.add_image(f'{sample} Prediction', torchvision.utils.make_grid(prediction[:size, :, :, :]), global_step=step)
    writer.flush()
