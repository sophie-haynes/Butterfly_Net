from __future__ import print_function

import math
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torchvision import transforms
from torchvision.transforms import v2

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

# tensor loader class
class TensorData(Dataset):
  def __init__(self, img, img_label):
    self.img = img  #img path
    self.label = img_label  #mask path
    self.len = len(os.listdir(self.img))

  def __getitem__(self, index):
    ls_img = sorted(os.listdir(self.img))
    ls_label = sorted(os.listdir(self.label))

    img_file_path = os.path.join(self.img, ls_img[index])
    img_tensor = torch.load(img_file_path)

    label_file_path = os.path.join(self.label, ls_label[index])
    label_tensor = torch.load(label_file_path)

    return img_tensor, label_tensor

  def __len__(self):
    return self.len


# custom writer to make hparams and scalars appear in same run in tensorboard
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


crop_dict = {
    # data      mean         std
    'cxr14': [[162.7414], [44.0700]],
    'openi': [[157.6150], [41.8371]],
    'jsrt': [[161.7889], [41.3950]],
    'padchest': [[160.3638], [44.8449]],
}

lung_seg_dict = {
    # data       mean        std
    'cxr14': [[60.6809], [68.9660]],
    'openi': [[60.5483], [66.5276]],
    'jsrt': [[66.5978], [72.6493]],
    'padchest': [[60.5482], [66.5276]],
}
arch_seg_dict = {
    # data       mean        std
    'cxr14': [[128.2716], [76.7148]],
    'openi': [[127.7211], [69.7704]],
    'jsrt': [[139.9666], [72.4017]],
    'padchest': [[129.5006], [72.6308]],
}

def get_cxr_train_transforms(crop_size,normalise):
    cxr_transform_list = [
        v2.ToImage(),
        # added since RandomGrayscale was removed
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([
            # reduced saturation and contrast - prevent too much info loss + removed hue
            v2.ColorJitter(0.4, 0.2, 0.2,0)
        ], p=0.8),
        # moved after transforms to preserve resolution, reduced scale to increase likelihood of indicator presence
        v2.RandomResizedCrop(size=crop_size, scale=(0.6, 1.),antialias=True),
        # required for normalisation
        v2.ToDtype(torch.float32, scale=True),
        normalise
    ]
    return cxr_transform_list

def get_cxr_eval_transforms(crop_size,normalise):
    cxr_transform_list = [
        v2.ToImage(),
        v2.Resize(size=crop_size,antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        normalise
    ]
    return cxr_transform_list


cifar_sc_transform_list = [
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
]

cifar_ce_transform_list = [
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

cxr_sc_transform_list = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.2, 0.2,0)
    ], p=0.8),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(size=128, scale=(0.6, 1.)),
    transforms.ToTensor(),
]
