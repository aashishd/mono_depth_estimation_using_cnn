import argparse
import copy
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from model.loss_function import inverseHuberLoss, rmseLoss
from model.model_architecture import Resnet50BasedModel, Resnet50BasedUpProjModel, AlexNetBasedModel, EfficientNet, VGG16, VGG19
from utils.dataset_util import get_nyuv2_test_train_dataloaders, load_test_train_ids, get_cityscape_val_train_dataloader
from utils.train_util import adjust_learning_rate, train, validate, save_json, load_weights


def argparser():
    parser = argparse.ArgumentParser(description='DeepSense_DL_Project')
    parser.add_argument('-c', '--config', default='./config_resnet.yaml')
    return parser


def main():
    args = argparser().parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

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

    load_weights(model, args.weights_path)

    writer = SummaryWriter(f'runs/{args.model.lower()}')
    if torch.cuda.is_available():
        model = model.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        print('Using Adam optimzier...')
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.reg)
        print('Using SGD optimzier...')
    else:
        print('Using Adam optimzier...')
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # criterion = torch.nn.MSELoss()  # inverseHuberLoss
    if args.loss_type == "MSE":
        print('Using MSE Loss...')
        criterion = torch.nn.MSELoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
    elif args.loss_type == "RMSE":
        print('Using RMSE Loss...')
        criterion = rmseLoss
    else:
        print('Using Inverese Hubert Loss...')
        criterion = inverseHuberLoss

    '''if torch.cuda.is_available():
        criterion = criterion.cuda()'''
    best = float('inf')
    train_ids, val_ids, test_ids = load_test_train_ids('datasets/Nyu_v2/train_val_test_ids.json')
    if args.dataset == 'cityscape':
        train_loader, val_loader = get_cityscape_val_train_dataloader('datasets/cityscapes/data', batch_size=args.batch_size)
        # test_loader = val_loader
    else:
        train_loader, val_loader, test_loader = get_nyuv2_test_train_dataloaders('datasets/Nyu_v2/nyu_depth_v2_labeled.mat', train_ids, val_ids, test_ids, batch_size=args.batch_size, apply_augmentations=args.apply_augmentations)

    train_losses = []
    val_losses = []
    learning_rates = []
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f'------------------ Learning Rate for EPOCH {epoch} => {lr} ------------------')
        learning_rates.append(lr)

        # train loop
        # print(f'********* Training Started for epoch {epoch} *********')
        train_loss = train(epoch, train_loader, model, optimizer, criterion, writer)
        train_losses.append(train_loss.item())
        print(f'Training loss ==> {train_loss:.4f}')

        # validation loop
        # print(f'********* Validation Started for epoch {epoch} *********')
        loss = validate(epoch, val_loader, model, criterion, writer)
        val_losses.append(loss.item())
        print(f'Validation loss ==> {loss:.4f}')

        if loss < best:
            best = loss
            best_model = copy.deepcopy(model)

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': train_loss, 'Validation': loss},
                           global_step=epoch)
        writer.add_scalars('Learning Rate',
                           {'Training': lr},
                           global_step=epoch)
        save_json(train_losses, f'metrics/train_{args.model.lower()}_berHu_default.json')
        save_json(val_losses, f'metrics/val_{args.model.lower()}_berHu_default.json')
        save_json(learning_rates, f'metrics/learning_rate_{args.model.lower()}_default.json')
        if args.save_best and epoch % 10 == 0:
            torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + f'-{epoch}.pth')

        print(f'Time taken for Epoch => {time.time() - start} seconds')

    print('Best Loss: {:.4f}'.format(best))
    torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + f'-{epoch}-final.pth')


if __name__ == '__main__':
    main()
