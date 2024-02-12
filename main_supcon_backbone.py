from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.transforms import v2

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNetW1,SupConResNetW2, SupConDenseNetW1,SupConSwinV2TW1
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, choices = ['resnet50','densenet121','swin_v2_t'], help="backbone for classification")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cxr14','jsrt','padchest','openi','path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # seed for reproducibility
    parser.add_argument('--seed', type=int, default=3, help='seed')
    parser.add_argument('--weight_version', type=str, default="v1", choices = ["v1", "v2"], help='ImageNet weights version to use')
    parser.add_argument('--rand_weights', action='store_true', help='Randomly initialised weights')
    parser.add_argument('--bbox', type=str, default=None, help='path to bounding box annnotations')
    parser.add_argument('--cxr_proc', type=str,choices=['crop', 'lung_seg','arch_seg'],help='CXR processing method applied')
    parser.add_argument('--fully_frozen', action='store_true',help="Freeze backbone and use as feature extractor")
    parser.add_argument('--half_frozen', action='store_true',help="Freeze half the backbone tune later layers")
    parser.add_argument('--save_out', type=str, default=None, help='path to save to')
    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # if running CXR experiment, make sure processing is specified
    try:
        if (opt.dataset == 'cxr14' or opt.dataset == 'jsrt' or opt.dataset == 'padchest' or opt.dataset == 'openi'):
            assert opt.cxr_proc == "crop" or opt.cxr_proc == "lung_seg" or opt.cxr_proc == "arch_seg";
    except AssertionError as e:
        print("CXR pre-processing not specified! Ensure you select 'crop', 'lung_seg' or 'arch_seg' when running on CXR images.")

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    if opt.save_out is None:
        opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
        opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)
    else:
        opt.model_path = '{}/{}_models'.format(opt.save_out,opt.dataset)
        opt.tb_path = '{}/{}_tensorboard'.format(opt.save_out,opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'backbone_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.weight_version, opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    # CXR normalisation values - single channel
    elif opt.dataset == 'cxr14':
        if opt.cxr_proc == "crop":
            mean = [162.7414]
            std = [44.0701]
        elif opt.cxr_proc == "lung_seg":
            mean = [128.2716]
            std = [76.7147]
        elif opt.cxr_proc == "arch_seg":
            mean = [128.2717]
            std = [76.7147]
        else:
            raise ValueError('cxr14 preprocessing unspecified!')
    elif opt.dataset == 'padchest':
        raise NotImplementedError("{} not yet implemented!".format(opt.dataset))
    elif opt.dataset == 'jsrt':
        raise NotImplementedError("{} not yet implemented!".format(opt.dataset))
    elif opt.dataset == 'openi':
        raise NotImplementedError("{} not yet implemented!".format(opt.dataset))
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    v2Normalise = v2.Normalize(mean=mean, std=std)

    cifar_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    if opt.bbox:
        # custom dataset
        # RandomIoUCrop
        raise NotImplementedError("BBox support is not yet implemented!")
    else:
        cxr_v2_train_transform = v2.Compose([
        # added since RandomGrayscale was removed
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([
        # reduced saturation and contrast - prevent too much info loss + removed hue
        v2.ColorJitter(0.4, 0.2, 0.2,0)
        ], p=0.8),
        # moved after transforms to preserve resolution, reduced scale to increase likelihood of indicator presence
        v2.RandomResizedCrop(size=opt.size, scale=(0.6, 1.)),
        # required for normalisation
        v2.ToDtype(torch.float32, scale=True),
        v2Normalise
        ])

        cxr_train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.2, 0.2,0)
            ], p=0.8),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.ToTensor(),
            normalize,
        ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(cifar_train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(cifar_train_transform),
                                          download=True)
    elif opt.dataset == 'cxr14':
        if opt.bbox:
            raise NotImplementedError("BBox support is not yet implemented!")
        else:
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                transform=TwoCropTransform(cxr_v2_train_transform))
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    if opt.model == "resnet50":
        if opt.weight_version == "v2":
            model = SupConResNetW2(name=opt.model)
        elif opt.weight_version == "v1":
            if opt.rand_weights:
                model = SupConResNetW1(name=opt.model,rand_init=opt.rand_weights)
                opt.model_name = "rand_"+opt.model_name
            else:
                if opt.fully_frozen:
                    opt.model_name = "FF_"+opt.model_name
                elif opt.half_frozen:
                    opt.model_name = "HF_"+opt.model_name
                else:
                    opt.model_name = "FN_"+opt.model_name

                model = SupConResNetW1(name=opt.model, frozen=opt.fully_frozen,half=opt.half_frozen)

        else:
            raise ValueError("Weight version provided is not available")
    elif opt.model == "densenet121":
        if opt.weight_version == "v2":
            raise NotImplementedError("Weight version provided is not available")
        elif opt.weight_version == "v1":
            model = SupConDenseNetW1(name=opt.model)
        else:
            raise ValueError("Weight version provided is not available")
    elif opt.model == "swin_v2_t":
        if opt.weight_version == "v2":
            raise NotImplementedError("Weight version provided is not available")
        elif opt.weight_version == "v1":
            model = SupConSwinV2TW1(name=opt.model)
    else:
        raise ValueError("Model backbone type invalid.")
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        model = model.to(mps_device)
        criterion = criterion.to(mps_device)

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        metal_flag = True
    else:
        metal_flag = False
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if metal_flag:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
        elif torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    #set seeds for reprod.
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
