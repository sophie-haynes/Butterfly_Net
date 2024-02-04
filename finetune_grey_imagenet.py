import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = Trueplt.ion()  
# interactive mode
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def parse_option():  
  parser = argparse.ArgumentParser('argument for training')
  parser.add_argument('--model', type=str, default='resnet50',\
                      choices=['resnet50','densenet121','swin_v2_t'])  
  parser.add_argument('--data_dir', type=str, default='datasets/imagenet',                      help='dataset path')  parser.add_argument('--learning_rate', type=float, default=0.001,                        help='learning rate')  parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',                        help='where to decay lr, can be a list')  parser.add_argument('--lr_decay_rate', type=float, default=0.2,                        help='decay rate for learning rate')  parser.add_argument('--weight_decay', type=float, default=0,                        help='weight decay')  parser.add_argument('--momentum', type=float, default=0.9,                        help='momentum')    parser.add_argument('--epochs', type=int, default=100, \                      help='number of training epochs')  parser.add_argument('--batch_size', type=int, default=256,\                      help='batch_size')  parser.add_argument('--num_workers', type=int, default=2,\                      help='num of workers to use')  parser.add_argument('--seed', type=int, default=3, help='seed')  opt = parser.parse_args()def main():  best_acc = 0  opt = parse_option()  #set seeds for reprod.  torch.manual_seed(opt.seed)  torch.backends.cudnn.deterministic = True  torch.backends.cudnn.benchmark = False  # data loading  data_transforms = {    'train': transforms.Compose([      transforms.RandomResizedCrop(224),      transforms.RandomHorizontalFlip(),      transforms.ToTensor(),      transforms.Normalize([0.449], [0.236]) # avg 3-channel to single channel      #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    ]),    'val': transforms.Compose([      transforms.Resize(256),      transforms.CenterCrop(224),      transforms.ToTensor(),      transforms.Normalize([0.449], [0.236]) # avg 3-channel to single      #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    ]),  }  data_dir = opt.data_dir  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),                                            data_transforms[x])                    for x in ['train', 'val']}  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,                                                shuffle=True, num_workers=opt.num_workers)                 for x in ['train','val']}  dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}  class_names = image_datasets['train'].classes  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # model tuning  if(opt.model == "resnet50"):    model_ft = models.resnet50(weights='IMAGENET1K_V1')  elif(opt.model == 'densenet121'):    model_ft = models.densenet121(weights='IMAGENET1K_V1')  elif (opt.model == 'swin_v2_t'):    #TODO: Implement SWIN finetuning    raise NotImplementedError('SWIN support unavailable')  else:    raise ValueError('Invalid model name passed in')  model_ft = model_ft.to_device(device)  #TODO: enable contrastive loss finetuning  criterion = nn.CrossEntropyLoss()  optimizer_ft = optim.SGD(model_ft.parameters(), lr=opt.learning_rate,momentum=opt.momentum)  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)      
