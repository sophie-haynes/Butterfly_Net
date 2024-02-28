from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
from util import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.transforms import v2

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import crop_dict, lung_seg_dict, arch_seg_dict
from util import get_cxr_train_transforms, cifar_ce_transform_list, TensorData
from networks.resnet_big import SupCEResNet,SupCEResNetW1

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass




def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type = int, default = 10,
                        help = 'print frequency')
    parser.add_argument('--save_freq', type = int, default = 50,
                        help = 'save frequency')
    parser.add_argument('--batch_size', type = int, default = 256,
                        help = 'batch_size')
    parser.add_argument('--num_workers', type = int, default = 16,
                        help = 'num of workers to use')
    parser.add_argument('--epochs', type = int, default = 500,
                        help = 'number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type = float, default = 0.2,
                        help = 'learning rate')
    parser.add_argument('--lr_decay_epochs', type = str, default = '350,400,450',
                        help = 'where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type = float, default = 0.1,
                        help = 'decay rate for learning rate')
    parser.add_argument('--weight_decay', type = float, default = 1e-4,
                        help = 'weight decay')
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'momentum')

    # model dataset
    parser.add_argument('--model', type = str, default = 'resnet50')
    parser.add_argument('--dataset', type = str, default = 'cifar10',
                        choices=['cifar10', 'cifar100', 'cxr14', 'jsrt', \
                        'padchest','openi','path'], help = 'dataset')
    parser.add_argument('--mean', type = str, \
                            help = 'mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type = str, help = 'std of dataset ')
    parser.add_argument('--data_folder', type = str, default = None, \
                            help = 'path to custom dataset')
    parser.add_argument('--tensor_path', type = str, default = None, \
                            help = 'Path to load augmented tensors')
    parser.add_argument('--size', type = int, default = 224, \
                            help = 'parameter for RandomResizedCrop')
    parser.add_argument('--n_cls', type = int, default = 1, \
                            help = 'Number of Classes to Predict')

    # other setting
    parser.add_argument('--cosine', action = 'store_true',
                        help = 'using cosine annealing')
    parser.add_argument('--syncBN', action = 'store_true',
                        help = 'using synchronized batch normalization')
    parser.add_argument('--warm', action = 'store_true',
                        help = 'warm-up for large batch training')
    parser.add_argument('--trial', type = str, default = '0',
                        help = 'id for recording multiple runs')

    # other setting
    parser.add_argument('--seed', type = int, default = 3, help = 'seed')
    parser.add_argument('--weight_version', type = str, default = "v1", \
                        choices = ["v1", "v2"], \
                        help = 'ImageNet weights version to use')
    parser.add_argument('--grey_path', type = str, default = None, \
                        help = 'Path to load greyscale model')
    parser.add_argument('--cxr_proc', type = str, \
                        choices=['crop', 'lung_seg','arch_seg'], \
                        help = 'CXR processing method applied')
    parser.add_argument('--fully_frozen', action = 'store_true',default = False, \
                        help = "Freeze backbone and use as feature extractor")
    parser.add_argument('--half_frozen', action = 'store_true', default = False, \
                        help = "Freeze half the backbone tune later layers")
    parser.add_argument('--save_out', type = str, default = None, \
                        help = 'path to save model to')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None
    # if running CXR experiment, make sure processing is specified
    try:
        if (opt.dataset == 'cxr14' or opt.dataset == 'jsrt' or \
            opt.dataset == 'padchest' or opt.dataset == 'openi'):
            assert opt.cxr_proc == "crop" or opt.cxr_proc == "lung_seg" \
                or opt.cxr_proc == "arch_seg";
    except AssertionError as e:
        print("CXR pre-processing not specified!")


    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    # set the path according to the environment
    if opt.save_out is None:
        opt.model_path = './save/CrossEnt/{}_models'.format(opt.dataset)
        opt.tb_path = './save/CrossEnt/CE/{}_tensorboard'.format(opt.dataset)
    else:
        opt.model_path = '{}/{}_models'.format(opt.save_out,opt.dataset)
        opt.tb_path = '{}/{}_tensorboard'.format(opt.save_out,opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'CE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

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
    """
    :returns: train_loader, val_loader, external_dict{loaders}
    """
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.cxr_proc == "crop":
        mean, std = crop_dict[opt.dataset]
    elif opt.cxr_proc == "lung_seg":
        mean, std = lung_seg_dict[opt.dataset]
    elif opt.cxr_proc == "arch_seg":
        mean, std = arch_seg_dict[opt.dataset]
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    # Create normalisation layers for image transforms
    normalize = transforms.Normalize(mean = mean, std = std)
    v2Normalise = v2.Normalize(mean = mean, std = std)


    if opt.dataset == 'cifar10':
        cifar_ce_transform_list.append(normalize)
        train_dataset = datasets.CIFAR10(root = opt.data_folder,
                                         transform = TwoCropTransform(\
                                            transforms.Compose(\
                                                cifar_ce_transform_list)),
                                         download=True)
    elif opt.dataset == 'cifar100':
        cifar_ce_transform_list.append(normalize)
        train_dataset = datasets.CIFAR100(root = opt.data_folder,
                                          transform = TwoCropTransform(\
                                            transforms.Compose(\
                                                cifar_ce_transform_list)),
                                          download=True)
    elif opt.cxr_proc is not None:
        if not opt.tensor_path:
            cxr_v2_train_transform = v2.Compose(\
                get_cxr_train_transforms(opt.size,v2Normalise))
            cxr_v2_val_transform = v2.Compose(\
                get_cxr_train_transforms(opt.size,v2Normalise))

            train_dataset = datasets.ImageFolder(\
                            root = os.path.join(opt.data_folder,"train"),
                            transform = cxr_v2_train_transform)
            val_dataset = datasets.ImageFolder(\
                            root = os.path.join(opt.data_folder,"test"),
                            transform = cxr_v2_val_transform)
            external_loaders = {}

        ext_names = ['cxr14','padchest','openi','jsrt']
        ext_names.remove(opt.dataset)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = opt.batch_size, \
        shuffle = (train_sampler is None), num_workers = opt.num_workers, \
        pin_memory = True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = opt.batch_size, shuffle = False,
        num_workers = opt.num_workers, pin_memory = True)

    # external validation on the fly
    for ds_name in ext_names:
        ext_pth = opt.data_folder.replace(opt.dataset,ds_name)

        ext_ds =  datasets.ImageFolder(root = os.path.join(ext_pth,"test"),
                                       transform = cxr_v2_val_transform)
        print("ext val loader... {}".format(os.path.join(ext_pth,"test")))
        external_loader = torch.utils.data.DataLoader(
            ext_ds, batch_size = opt.batch_size, shuffle = False,
            num_workers = opt.num_workers, pin_memory = True)
        external_loaders[ds_name] = external_loader

    return train_loader, val_loader, external_loaders


def set_model(opt):
    if opt.model == "resnet50":
        model = SupCEResNetW1(name = opt.model, num_classes = opt.n_cls, \
        frozen = opt.fully_frozen, half = opt.half_frozen, grey = opt.grey_path)
    else:
        raise NotImplementedError("Only ResNet50 is currently supported!")

    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    # CUDA GPU Loading
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        # cudnn.benchmark = True
    # Metal GPU Loading
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
    top1 = AverageMeter()

    end = time.time()
    # check for metal GPU
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        metal_flag = True
    else:
        metal_flag = False

    # load to GPU if available
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if opt.tensor_path:
            images = torch.cat([images[0], images[1]], dim = 0)
        if metal_flag:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
        else:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        if opt.tensor_path:
            f1, f2 = torch.split(output, [bsz, bsz], dim = 0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
            loss = criterion(output,labels)
            acc1 = accuracy(f1, labels)
            top1.update(acc1[0], bsz)
            acc2 = accuracy(f2, labels)
            top1.update(acc2[0], bsz)
        else:
            loss = criterion(output, labels)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc {top1.val[0]:.3f} ({top1.avg[0]:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time = batch_time,
                   data_time = data_time, loss = losses, top1 = top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        # check for metal  GPU
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            metal_flag = True
        else:
            metal_flag = False

        # load to GPU
        for idx, (images, labels) in enumerate(val_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            if metal_flag:
                images = images.float().to(mps_device)
                labels = labels.to(mps_device)
            else:
                images = images.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.tensor_path:
                output = model(images)
                f1, f2 = torch.split(output, [bsz, bsz], dim = 0)
                output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
                loss = criterion(output, labels)
                acc1 = accuracy(f1, labels)
                top1.update(acc1[0], bsz)
                acc2 = accuracy(f2, labels)
                top1.update(acc2[0], bsz)
            else:
                output = model(images)
                loss = criterion(output, labels)
                acc1 = accuracy(output, labels)
                top1.update(acc1[0], bsz)
            # update metric
            losses.update(loss.item(), bsz)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val[0]:.3f} ({top1.avg[0]:.3f})'.format(
                       idx, len(val_loader), batch_time = batch_time,
                       loss = losses, top1 = top1))

    print(' * Acc {top1.avg:.3f}'.format(top1 = top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    #set seeds for reprod.
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not opt.tensor_path:
        # build data loader
        train_loader, val_loader, external_loaders = set_loader(opt)
    else:
        ext_names = ['cxr14','padchest','jsrt']
        ext_names.remove(opt.dataset)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = SummaryWriter(opt.tb_folder)
    # training routine
    curr_epoch = 1
    for epoch in range(1, opt.epochs + 1):
        if curr_epoch > 30:
            curr_epoch -= 30
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        if opt.tensor_path:
            train_dataset = TensorData(\
                        os.path.join(opt.tensor_path,str(curr_epoch),'img'),
                        os.path.join(opt.tensor_path,str(curr_epoch),'label'))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size = opt.batch_size, shuffle = True,
                            num_workers = opt.num_workers, pin_memory = True)

        loss, train_acc = train(train_loader, model, \
                                criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.add_scalar("train_loss",loss, epoch)
        logger.add_scalar("train_acc",train_acc, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # logger.log_value('train_loss', loss, epoch)
        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        if opt.tensor_path:
            val_pth = opt.tensor_path.replace("train_loader","test_loader")
            val_dataset = TensorData(\
                            os.path.join(val_pth,'img'),
                            os.path.join(val_pth,'label'))
            val_loader = torch.utils.data.DataLoader(val_dataset,
                            batch_size = opt.batch_size,
                            shuffle = True,
                            num_workers = opt.num_workers, pin_memory = True)
        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        logger.log_value('{}_test_loss'.format(opt.dataset), loss, epoch)
        logger.log_value('{}_test_acc'.format(opt.dataset), val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        # external validation
        if opt.tensor_path:
            for ds_name in ext_names:
                ext_pth = opt.tensor_path.replace("train_loader","test_loader")\
                                            .replace(opt.dataset,ds_name)
                ext_dataset = TensorData(os.path.join(ext_pth,'img'),
                                            os.path.join(ext_pth, 'label'))
                ext_loader = torch.utils.data.DataLoader(ext_dataset,
                                batch_size = opt.batch_size,
                                shuffle = True,
                                num_workers = opt.num_workers, pin_memory = True)
                loss, val_acc = validate(ext_loader, model, criterion, opt)
                logger.log_value('{}_val_loss'.format(ds_name), loss, epoch)
                logger.log_value('{}_val_acc'.format(ds_name), val_acc, epoch)

        else:
            for ds_name in external_loaders.keys():
                loss, val_acc = validate(external_loaders[ds_name], model, \
                                            criterion, opt)
                logger.log_value('{}_val_loss'.format(ds_name), loss, epoch)
                logger.log_value('{}_val_acc'.format(ds_name), val_acc, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch = epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
