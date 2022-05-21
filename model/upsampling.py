import torch.nn as nn
import torch
from torch.nn.functional import pad, conv_transpose2d
import numpy as np


class Unpool(nn.Module):
    """
    The unpooling layer
    """

    def __init__(self, stride=2):
        super(Unpool, self).__init__()
        self.stride = stride
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:, :, 0, 0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return conv_transpose2d(x, self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1), stride=self.stride, groups=num_channels)


class UpConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(UpConvolution, self).__init__()
        padding = (kernel_size - 1) // 2
        assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
        self.upconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.upconv(input)


class FastUpConvolution(nn.Module):
    """
    Fast Convolutions
    """

    def __init__(self, in_channels, out_channels):
        """Initialize the Fast convolution using the in and out channels"""
        super(FastUpConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2, 2))

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    # https://github.com/iapatil/depth-semantic-fully-conv
    def prepare_indices(self, before, row, col, after, dims):
        x0, x1, x2, x3 = np.meshgrid(before, row, col, after)
        x_0 = torch.from_numpy(x0.reshape([-1]))
        x_1 = torch.from_numpy(x1.reshape([-1]))
        x_2 = torch.from_numpy(x2.reshape([-1]))
        x_3 = torch.from_numpy(x3.reshape([-1]))
        if torch.cuda.is_available():
            x_0 = x_0.cuda()
            x_1 = x_1.cuda()
            x_2 = x_2.cuda()
            x_3 = x_3.cuda()

        linear_indices = x_3 + dims[3] * x_2 + 2 * dims[2] * dims[3] * x_0 * 2 * dims[1] + 2 * dims[2] * dims[3] * x_1
        return linear_indices

    # # interleaving operation
    # def interleave_helper(self, tensors, axis):
    #     tensor_shape = list(tensors[0].size())
    #     tensor_shape[axis + 1] *= 2
    #     return torch.stack(tensors, axis + 1).view(tensor_shape)

    # https://github.com/iapatil/depth-semantic-fully-conv
    def interleave_tensors(self, out1, out2, out3, out4, batch_size):
        out1 = out1.permute(0, 2, 3, 1)
        out2 = out2.permute(0, 2, 3, 1)
        out3 = out3.permute(0, 2, 3, 1)
        out4 = out4.permute(0, 2, 3, 1)

        dims = out1.size()
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = np.arange(0, dim1, 2, dtype=np.int64)

        A_col_indices = np.arange(0, dim2, 2, dtype=np.int64)
        B_row_indices = np.arange(1, dim1, 2, dtype=np.int64)
        B_col_indices = np.arange(0, dim2, 2, dtype=np.int64)
        C_row_indices = np.arange(0, dim1, 2, dtype=np.int64)
        C_col_indices = np.arange(1, dim2, 2, dtype=np.int64)
        D_row_indices = np.arange(1, dim1, 2, dtype=np.int64)
        D_col_indices = np.arange(1, dim2, 2, dtype=np.int64)

        all_indices_before = np.arange(int(batch_size), dtype=np.int64)
        all_indices_after = np.arange(int(dims[3]), dtype=np.int64)

        A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after, dims)
        B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after, dims)
        C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after, dims)
        D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after, dims)

        A_flat = (out1.permute(1, 0, 2, 3)).contiguous().view(-1)
        B_flat = (out2.permute(1, 0, 2, 3)).contiguous().view(-1)
        C_flat = (out3.permute(1, 0, 2, 3)).contiguous().view(-1)
        D_flat = (out4.permute(1, 0, 2, 3)).contiguous().view(-1)

        size_ = A_linear_indices.size()[0] + B_linear_indices.size()[0] + C_linear_indices.size()[0] + D_linear_indices.size()[0]

        Y_flat = torch.FloatTensor(size_).zero_()
        if torch.cuda.is_available():
            # print("running on gpu")
            Y_flat = Y_flat.cuda()

        Y_flat.scatter_(0, A_linear_indices.squeeze(), A_flat.data)
        Y_flat.scatter_(0, B_linear_indices.squeeze(), B_flat.data)
        Y_flat.scatter_(0, C_linear_indices.squeeze(), C_flat.data)
        Y_flat.scatter_(0, D_linear_indices.squeeze(), D_flat.data)

        Y = Y_flat.view(-1, dim1, dim2, dims[3])
        return torch.autograd.Variable(Y.permute(0, 3, 1, 2))

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor : Tensor of shape N, C, H, W

        Returns
        -------
        output_tensor : Returns an output tensor of shape N, C, 2*H, 2*W
        """
        conved33 = self.conv1(pad(input_tensor, (1, 1, 1, 1)))
        conved23 = self.conv2(pad(input_tensor, (1, 1, 1, 0)))
        conved32 = self.conv3(pad(input_tensor, (1, 0, 1, 1)))
        conved22 = self.conv4(pad(input_tensor, (1, 0, 1, 0)))
        out_interleaved = self.interleave_tensors(conved33, conved23, conved32, conved22, input_tensor.shape[0])
        normed = self.batch_norm(out_interleaved)
        return self.relu(normed)


class FastUpProjection(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FastUpProjection, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor : Tensor of shape N, C, H, W

        Returns
        -------
        output_tensor : Returns an output tensor of shape N, C, 2*H, 2*W
        """
        batch_size = input_tensor.shape[0]
        out1_1 = self.conv1_1(nn.functional.pad(input_tensor, (1, 1, 1, 1)))
        out1_2 = self.conv1_2(nn.functional.pad(input_tensor, (1, 1, 1, 0)))  # author's interleaving padding
        out1_3 = self.conv1_3(nn.functional.pad(input_tensor, (1, 0, 1, 1)))  # author's interleaving padding
        out1_4 = self.conv1_4(nn.functional.pad(input_tensor, (1, 0, 1, 0)))  # author's interleaving padding

        out2_1 = self.conv2_1(nn.functional.pad(input_tensor, (1, 1, 1, 1)))
        out2_2 = self.conv2_2(nn.functional.pad(input_tensor, (1, 1, 1, 0)))  # author's interleaving padding
        out2_3 = self.conv2_3(nn.functional.pad(input_tensor, (1, 0, 1, 1)))  # author's interleaving padding
        out2_4 = self.conv2_4(nn.functional.pad(input_tensor, (1, 0, 1, 0)))  # author's interleaving padding

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out
