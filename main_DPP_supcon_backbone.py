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

# wrapper for python multiprocessing
import torch.multiprocessing as mp
# distribute data to GPUs
from torch.utils.data.distributed import DistributedSampler
# new, more efficient, and scalable to multiple end nodes with single GPUs
from torch.nn.parallel import DistributedDataParallel as DDP
# manage GPU group
from torch.distributed import init_process_group, destroy_process_group

from util import TwoCropTransform, AverageMeter, TensorData
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import crop_dict, lung_seg_dict, arch_seg_dict
from util import get_cxr_train_transforms, cifar_sc_transform_list

from networks.resnet_big import SupConResNetW1,SupConResNetW2, SupConDenseNetW1,SupConSwinV2TW1
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def ddp_setup(rank, world_size):
    """
    Method to initialise a GPU "group" to allow for mutual communication
    between the GPUs.
    :param rank: Unique id for each process.
    :param world_size: Total number of processes in the group.
    """
    # IP Address of the head node/machine running process with rank 0
    # NOTE: "localhost" only works on single-node instances, needs set on cluster!
    os.environ["MASTER_ADDR"] = "localhost"
    # Any free port on head node
    os.environ["MASTER_PORT"] = "12355"
    # initialise GPU group - nccl is nvidia CUDA distribution backend
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

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
    parser.add_argument('--tensor_path', type = str, default = None, \
                            help = 'Path to load augmented tensors')
    parser.add_argument('--grey_path', type = str, default = None, \
                        help = 'Path to load greyscale model')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

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
    parser.add_argument('--bbox', type=str, default=None, help='path to bounding box annnotations')
    parser.add_argument('--cxr_proc', type=str,choices=['crop', 'lung_seg','arch_seg'],help='CXR processing method applied')
    parser.add_argument('--fully_frozen', action='store_true',help="Freeze backbone and use as feature extractor")
    parser.add_argument('--half_frozen', action='store_true',help="Freeze half the backbone tune later layers")
    parser.add_argument('--save_out', type=str, default=None, help='path to save to')

    parser.add_argument('--feat_dim', type=int, default=128, help="Dimension for projection network output")
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

    normalize = transforms.Normalize(mean=mean, std=std)
    v2Normalise = v2.Normalize(mean=mean, std=std)

    if opt.dataset == 'cifar10':
        cifar_sc_transform_list.append(normalize)
        train_dataset = datasets.CIFAR10(root = opt.data_folder,
                                         transform = TwoCropTransform(\
                                            transforms.Compose(\
                                                cifar_sc_transform_list)),
                                         download=True)
    elif opt.dataset == 'cifar100':
        cifar_sc_transform_list.append(normalize)
        train_dataset = datasets.CIFAR100(root = opt.data_folder,
                                          transform = TwoCropTransform(\
                                            transforms.Compose(\
                                                cifar_sc_transform_list)),
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
        raise ValueError('dataset not currently supported: {}'.format(opt.dataset))

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
            if opt.fully_frozen:
                opt.model_name = "FF_"+opt.model_name
            elif opt.half_frozen:
                opt.model_name = "HF_"+opt.model_name
            else:
                opt.model_name = "FN_"+opt.model_name

            model = SupConResNetW1(name=opt.model,feat_dim=opt.feat_dim, frozen=opt.fully_frozen,half=opt.half_frozen,grey = opt.grey_path)

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
        # cudnn.benchmark = True
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

    if not opt.tensor_path:
        # build data loader
        train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    curr_epoch = 1
    for epoch in range(1, opt.epochs + 1):
        # tracking for tensor loading
        if curr_epoch > 30:
            curr_epoch -= 30
        print("Epoch {}\n".format(epoch))
        adjust_learning_rate(opt, optimizer, epoch)
        # load in current epoch data
        if opt.tensor_path:
            train_dataset = TensorData(\
                        os.path.join(opt.tensor_path,str(curr_epoch),'img'),
                        os.path.join(opt.tensor_path,str(curr_epoch),'label'))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size = opt.batch_size, shuffle = True,
                            num_workers = opt.num_workers, pin_memory = True)

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
        curr_epoch += 1
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
