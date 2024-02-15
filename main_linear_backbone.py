from __future__ import print_function

import sys
import argparse
import time
import math
import os

import torch
import torch.backends.cudnn as cudnn

from main_ce_cxr import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from util import SummaryWriter
from networks.resnet_big import SupConResNetW1,SupConResNetW2, SupConDenseNetW1,SupConSwinV2TW1,LinearClassifier
from util import crop_dict, lung_seg_dict, arch_seg_dict

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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, choices = ['resnet50','densenet121','swin_v2_t'], help="backbone for classification")
    parser.add_argument('--dataset', type=str, choices=['cxr14','jsrt','padchest','openi'], help='dataset')
    parser.add_argument('--cxr_proc', type=str,choices=['crop', 'lung_seg','arch_seg'],help='CXR processing method applied')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--seed', type=int, default=3, help='seed')
    parser.add_argument('--save_out', type=str, default=None, help='path to save to')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    opt = parser.parse_args()

    try:
        if (opt.dataset == 'cxr14' or opt.dataset == 'jsrt' or opt.dataset == 'padchest' or opt.dataset == 'openi'):
            assert opt.cxr_proc == "crop" or opt.cxr_proc == "lung_seg" or opt.cxr_proc == "arch_seg";
    except AssertionError as e:
        print("CXR pre-processing not specified! Ensure you select 'crop', 'lung_seg' or 'arch_seg' when running on CXR images.")

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    if opt.save_out is None:
        opt.model_path = './save/SupCon/linear/{}_models'.format(opt.dataset)
        opt.tb_path = './save/SupCon/linear/{}_tensorboard'.format(opt.dataset)
    else:
        opt.model_path = '{}/linear/{}_models'.format(opt.save_out,opt.dataset)
        opt.tb_path = '{}/linear/{}_tensorboard'.format(opt.save_out,opt.dataset)


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'backbone_{}_{}_bsz_{}'.\
        format(opt.dataset, opt.model,
               opt.batch_size)

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
    opt.model_name = '{}_trial_{}'.format(opt.model_name,opt.trial)

    if (opt.dataset == 'cxr14' or opt.dataset == 'jsrt' or opt.dataset == 'padchest' or opt.dataset == 'openi'):
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    if opt.model == "resnet50":
        model = SupConResNetW1(name=opt.model)
    elif opt.model == "densenet121":
        model = SupConDenseNetW1(name=opt.model)
    elif opt.model == "swin_v2_t":
        model = SupConSwinV2TW1(name=opt.model)
    else:
        raise ValueError("Model backbone type invalid.")

    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        print("Loading model to GPU")
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels)
        top1.update(acc1[0].item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            # images = images.float().cuda()
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1= accuracy(output, labels)
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    #set seeds for reprod.
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build data loader
    # patch for handling no bbox
    opt.bbox = None
    train_loader, val_loader,external_loaders = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    logger = SummaryWriter(opt.tb_folder)
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, acc))
        logger.add_scalar("train_loss",loss, epoch)
        logger.add_scalar("train_acc",acc, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        logger.add_scalar('{}_test_loss'.format(opt.dataset), loss, epoch)
        logger.add_scalar('{}_test_acc'.format(opt.dataset), val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        for ds_name in external_loaders.keys():
            loss, val_acc = validate(external_loaders[ds_name], model, classifier, criterion, opt)
            logger.add_scalar('{}_val_loss'.format(ds_name), loss, epoch)
            logger.add_scalar('{}_val_acc'.format(ds_name), val_acc, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
