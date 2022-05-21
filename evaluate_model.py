import argparse
import os.path
import time
from pathlib import Path

import numpy as np
import torch

from model.loss_function import inverseHuberLoss, rmseLoss
from model.model_architecture import Resnet50BasedModel, Resnet50BasedUpProjModel, AlexNetBasedModel, EfficientNet, VGG16, VGG19
from utils.dataset_util import get_nyuv2_test_train_dataloaders, load_test_train_ids, get_cityscape_val_train_dataloader
from utils.train_util import validate, load_weights
import matplotlib.pyplot as plt


def argparser():
    parser = argparse.ArgumentParser(description='DeepSense_DL_Project')
    parser.add_argument('-m', '--model-class', required=True, type=str)
    parser.add_argument('-w', '--weights-path', required=True, type=str)
    parser.add_argument('-o', '--output-path', required=False, type=str, default=None)
    parser.add_argument('-b', '--batch-size', default=16, required=False, type=int)
    parser.add_argument('-d', '--dataset', default='nyu', required=False, type=str)

    return parser


def load_weights_for_eval(model, weights_path):
    if Path(weights_path).is_file():
        model.load_state_dict(torch.load(weights_path))
        model.eval()
    else:
        raise ValueError(f"Weights couldn't be loade at path {weights_path}")


def main():
    args = argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_class == 'AlexNetBasedModel':
        model = AlexNetBasedModel(device=device)
    elif args.model_class == 'EfficientNet':
        model = EfficientNet(device=device)
    elif args.model_class == 'VGG-16Based-UpConvolution':
        model = VGG16(device=device)
    elif args.model_class == 'VGG-19Based-UpConvolution':
        model = VGG19(device=device)
    elif args.model_class == 'Resnet50BasedModel':
        model = Resnet50BasedModel(device=device)
    elif args.model_class == 'Resnet50BasedUpProjModel':
        model = Resnet50BasedUpProjModel(device=device)
    else:
        raise ValueError(f'Model class specified in the config args is not implemented => {args.model_class}')

    if torch.cuda.is_available():
        model = model.cuda()

    print('Loading weights for model...')
    load_weights(model, args.weights_path)

    print('Loading dataset for evaluation...')
    if args.dataset == 'cityscape':
        train_loader, val_loader = get_cityscape_val_train_dataloader('datasets/cityscapes/data', augment=False, batch_size=args.batch_size, shuffle=False)
    else:
        train_ids, val_ids, test_ids = load_test_train_ids('datasets/Nyu_v2/train_val_test_ids.json')
        train_loader, val_loader, test_loader = get_nyuv2_test_train_dataloaders('datasets/Nyu_v2/nyu_depth_v2_labeled.mat', train_ids, val_ids, test_ids, batch_size=args.batch_size, apply_augmentations=False, shuffle=False)

    start = time.time()
    print('Evaluation started...')
    # Calculating MSE Error
    mse_loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        mse_loss = mse_loss.cuda()
    mse_loss_value = validate(0, val_loader, model, mse_loss)
    print(f'Val Final MSE loss ==> {mse_loss_value.item():.4f}')

    rmse_loss_value = validate(0, val_loader, model, rmseLoss)
    print(f'Val Final RMSE loss ==> {rmse_loss_value.item():.4f}')

    inv_hub_loss_val = validate(0, val_loader, model, inverseHuberLoss)
    print(f'Val Final Inverse Hubert loss ==> {inv_hub_loss_val.item():.4f}')

    # generate training images and ground truth for validaaton data
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        root_path = os.path.join(args.output_path, 'val')
        Path(root_path).mkdir(exist_ok=True)
        data, target = next(iter(val_loader))
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model(data).squeeze()
            if torch.cuda.is_available():
                out = out.cpu().numpy()
                data = data.permute(0, 2, 3, 1).cpu().numpy()
                target = target.squeeze().cpu().numpy()
            else:
                out = out.numpy()
                data = data.permute(0, 2, 3, 1).numpy()
                target = target.squeeze().numpy()
            for i in range(args.batch_size):
                plt.imsave(os.path.join(root_path, f'{i}.jpg'), data[i])
                plt.imsave(os.path.join(root_path, f'{i}_gt.jpg'), (target[i] / np.max(target[i])), cmap='viridis')
                plt.imsave(os.path.join(root_path, f'{i}_pred.jpg'), out[i] / np.max(out[i]), cmap='viridis')

    mse_loss_value = validate(0, train_loader, model, mse_loss)
    print(f'Train Final MSE loss ==> {mse_loss_value.item():.4f}')

    rmse_loss_value = validate(0, train_loader, model, rmseLoss)
    print(f'Train Final RMSE loss ==> {rmse_loss_value.item():.4f}')

    inv_hub_loss_val = validate(0, train_loader, model, inverseHuberLoss)
    print(f'Train Final Inverse Hubert loss ==> {inv_hub_loss_val.item():.4f}')

    # generate training images and ground truth for tarining data
    if args.output_path:
        root_path = os.path.join(args.output_path, 'train')
        Path(root_path).mkdir(exist_ok=True)
        data, target = next(iter(train_loader))
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            out = model(data).squeeze()
            if torch.cuda.is_available():
                out = out.cpu().numpy()
                data = data.permute(0, 2, 3, 1).cpu().numpy()
                target = target.squeeze().cpu().numpy()
            else:
                out = out.numpy()
                data = data.permute(0, 2, 3, 1).numpy()
                target = target.squeeze().numpy()
            for i in range(args.batch_size):
                plt.imsave(os.path.join(root_path, f'{i}.jpg'), data[i])
                plt.imsave(os.path.join(root_path, f'{i}_gt.jpg'), target[i] / np.max(target[i]), cmap='viridis')
                plt.imsave(os.path.join(root_path, f'{i}_pred.jpg'), out[i] / np.max(out[i]), cmap='viridis')

    print(f'Time taken for Epoch => {time.time() - start} seconds')


if __name__ == '__main__':
    main()
