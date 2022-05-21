import json
import pickle
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.metrics_util import write_image_grid


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    lr = args.learning_rate
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    for idx, step in enumerate(reversed(args.steps)):
        if epoch > step:
            loss_multiplier = args.stepdown_factor ** (len(args.steps) - idx)
            lr = args.learning_rate * loss_multiplier
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


def train(epoch, data_loader, model, optimizer, criterion, writer: SummaryWriter):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # for train_features, train_labels in next(iter(data_loader))
    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            # tensor_dtype = torch.cuda.FloatTensor
            # print('chanding dtype of input & target tensors')
            data = data.cuda()
            target = target.cuda()

        out = model.forward(data)
        loss = criterion(out, target)

        if idx == 0:
            write_image_grid(writer, data, target, out, step=epoch, sample='Training')

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss, out.shape[0])
        # batch_acc = accuracy(out, target)
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))
    return losses.avg


def validate(epoch, val_loader, model, criterion, writer: SummaryWriter = None):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # evaluation loop
    model.eval()
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

            if idx == 0:
                if writer:
                    write_image_grid(writer, data, target, out, step=epoch, sample='Validation')
        losses.update(loss, out.shape[0])
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 5 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    return losses.avg


def load_weights(model, weights_path):
    if Path(weights_path).is_file():
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        print(f"Weights couldn't be loade at path {weights_path}")


def save_pickle(data, file_path):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def save_json(json_data, file_path):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(json.dumps(json_data, indent=4))


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.loads(f.read())
