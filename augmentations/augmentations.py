from __future__ import division
import torch
import numpy as np
import numbers
import scipy.ndimage as ndimage
import random
from skimage import transform


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, tar):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        if h1 == th and w1 == tw:
            return inputs, tar
        x1 = int(round((w1 - th) / 2.))
        y1 = int(round((h1 - tw) / 2.))

        inputs = inputs[y1: y1 + th, x1: x1 + tw]
        target = tar[y1: y1 + th, x1: x1 + tw]
        if inputs.shape[0] == 0:
            print('center crop issue')
        return inputs, target


class RandomCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, tar):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        width = random.randint(tw, w1 - 2)
        height = round((h1 / w1) * width)
        x1 = int(round((w1 - width) / 2.))
        y1 = int(round((h1 - height) / 2.))

        inputs = inputs[y1: y1 + height, x1: x1 + width]
        target = tar[y1: y1 + height, x1: x1 + width]
        return inputs, target


class Scale(object):

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target_depth):
        h, w, _ = inputs.shape

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs, target_depth

        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = np.stack((ndimage.interpolation.zoom(inputs[:, :, 0], ratio, order=self.order),
                           ndimage.interpolation.zoom(inputs[:, :, 1], ratio, order=self.order), \
                           ndimage.interpolation.zoom(inputs[:, :, 2], ratio, order=self.order)), axis=2)

        target_depth = ndimage.interpolation.zoom(target_depth, ratio, order=self.order)
        return inputs, target_depth


class ScaleExact(object):

    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, inputs, target_depth):
        h, w, _ = inputs.shape
        req_h, req_w = self.size

        if w == req_w and h == req_h:
            return inputs, target_depth

        return transform.resize(inputs, (req_h, req_w)), transform.resize(target_depth, (req_h, req_w))


class Compose(object):

    def __init__(self, co_transforms):
        self.coff_transforms = co_transforms

    def __call__(self, input, target_depth):
        for i, trans in enumerate(self.coff_transforms):
            input, target_depth = trans(input, target_depth)
            if input.shape[0] == 0:
                print('some fuck up')
        return input, target_depth


class ArrayToTensor(object):
    def __call__(self, arr, depth):
        tensor_img, tensor_dep = torch.from_numpy(arr).permute(2, 0, 1), torch.from_numpy(np.expand_dims(depth, axis=2)).permute(2, 0, 1)
        return tensor_img.float(), tensor_dep.float()


class NormalizeData(object):
    def __call__(self, img, depth):
        return img / np.max(img), depth


class AorB(object):
    def __init__(self, transform_A, transform_B, probA=0.5):
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.probA = probA

    def __call__(self, input, target_depth):
        if random.random() < self.probA:
            return self.transform_A(input, target_depth)
        else:
            return self.transform_B(input, target_depth)


class Lambda(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, input, tar):
        return self.lambd(input, tar)


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target_depth):
        h, w, _ = inputs.shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, target_depth

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return inputs[y1: y1 + th, x1: x1 + tw], target_depth[y1: y1 + th, x1: x1 + tw]


class RandomHorizontalFlip(object):
    def __call__(self, inputs, target_depth):
        if random.random() < 0.5:
            inputs = np.flip(inputs, axis=0).copy()
            target_depth = np.flip(target_depth, axis=0).copy()
        return inputs, target_depth


class RandomVerticalFlip(object):
    def __call__(self, inputs, target):
        if random.random() < 0.5:
            # inputs[0] = np.flipud(inputs[0])
            inputs = np.flipud(inputs)
            target = np.flipud(target)
            # target[:, :, 1] *= -1
        return inputs, target


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.trans = (int(translation), int(translation))
        else:
            self.trans = translation

    def __call__(self, inputs, target):
        height, weight, _ = inputs[0].shape
        theight, tweight = self.trans
        tweight = random.randint(-tweight, tweight)
        theight = random.randint(-theight, theight)
        if tweight == 0 and theight == 0:
            return inputs, target
        x1, x2, x3, x4 = max(0, tweight), min(weight + tweight, weight), max(0, -tweight), min(weight - tweight, weight)
        y1, y2, y3, y4 = max(0, theight), min(height + theight, height), max(0, -theight), min(height - theight, height)

        inputs[0] = inputs[0][y1:y2, x1:x2]
        inputs[1] = inputs[1][y3:y4, x3:x4]
        target = target[y1:y2, x1:x2]
        target[:, :, 0] += tweight
        target[:, :, 1] += theight

        return inputs, target


class RandomRotate(object):
    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, inputs, target_depth):
        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180

        inputs = ndimage.interpolation.rotate(inputs, angle1, reshape=self.reshape, order=self.order)
        target_depth = ndimage.interpolation.rotate(target_depth, angle1, reshape=self.reshape, order=self.order)
        # target_label = ndimage.interpolation.rotate(target_label, angle1, reshape=self.reshape, order=self.order)

        return inputs, target_depth


class RandomCropRotate(object):
    def __init__(self, angle, size, diff_angle=0, order=2):
        self.angle = angle
        self.order = order
        self.diff_angle = diff_angle
        self.size = size

    def __call__(self, inputs, target):
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2

        angle1_rad = angle1 * np.pi / 180
        angle2_rad = angle2 * np.pi / 180

        h, w, _ = inputs[0].shape

        def rotate_flow(i, j, k):
            return -k * (j - w / 2) * (diff * np.pi / 180) + (1 - k) * (i - h / 2) * (diff * np.pi / 180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=True, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=True, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=True, order=self.order)

        target_ = np.array(target, copy=True)
        target[:, :, 0] = np.cos(angle1_rad) * target_[:, :, 0] - np.sin(angle1_rad) * target_[:, :, 1]
        target[:, :, 1] = np.sin(angle1_rad) * target_[:, :, 0] + np.cos(angle1_rad) * target_[:, :, 1]

        angle1_rad = np.pi / 2 - np.abs(angle1_rad % np.pi - np.pi / 2)
        angle2_rad = np.pi / 2 - np.abs(angle2_rad % np.pi - np.pi / 2)

        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c2 = np.cos(angle2_rad)
        s2 = np.sin(angle2_rad)
        c_diag = h / np.sqrt(h * h + w * w)
        s_diag = w / np.sqrt(h * h + w * w)

        ratio = c_diag / max(c1 * c_diag + s1 * s_diag, c2 * c_diag + s2 * s_diag)

        crop = CenterCrop((int(h * ratio), int(w * ratio)))
        scale = Scale(self.size)
        inputs, target = crop(inputs, target)
        return scale(inputs, target)
