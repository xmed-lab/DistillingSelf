#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from ast import arg
import builtins
import imp
from itertools import zip_longest
from logging import logThreads
import math
import os
from symbol import if_stmt
from tty import IFLAG
from turtle import Turtle
# gpustr = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpustr
import random
import shutil
import time
import warnings
# from thop import profile
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models 
import torch.nn.functional as F
from tools import moco_dataset, temporal_dataset,all_dataset, multipositive_dataset
import moco.loader
import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
f_path = os.path.abspath('..')
# print(f_path.split('shadow_code'))
# print(f_path)
# sss
root_path = f_path.split('surgical_code')[0]

dataset_pth = root_path+'/datasets/surgical/{}/'
# datasets/
# dataset_pth = root_path+'/datasets/workflow/camma.u-strasbg.fr/datasets/'
parser.add_argument('--data', type=str, default=dataset_pth,
                    help='path to dataset')
parser.add_argument('--data_type', type=str, default='workflow',
                    help='data_type:workflow, 71_heart')
parser.add_argument('--supervised', action="store_true",help="imagenet supervised or not.")
parser.add_argument('--onlyfc', action="store_true",help="only train fc or not.")
parser.add_argument('--decoupling', action="store_true",help="decoupling or not.")
# parser.add_argument('--imagenet', action="store_true",help="imagenet pretrain or not.")
parser.add_argument('--imagenet', type=int, default=0,
                    help='imagenet pre-train type, 0=none,1=supervised,2=moco')
parser.add_argument('--distill', type=int, default=0,
                    help='distill type, 0=none,1=supervised,2=moco')
parser.add_argument('--concatenate', type=int, default=0,
                    help='concatenate type')

parser.add_argument('--moco_pre', action="store_true",help="moco imagenet pretrain or not.")
parser.add_argument('--method', type=str, default='base',
                    help='path to dataset')
parser.add_argument('--sample_rate', type=int, default=5,
                    help='sample_rate of frame')
parser.add_argument('--predictor', type=int, default=0,
                    help='if predictor')
parser.add_argument('--dis_weight', type=float, default=0,
                    help='sem_weight of semantic')
parser.add_argument('--interval', type=int, default=1,
                    help='interval of positive frame')
parser.add_argument('--num_instance', type=int, default=2,
                    help='num_instance')
parser.add_argument('--dataset', type=str, default='cholec80',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    final_model_name = 'mocov2'+args.method+str(args.interval)
    log_path = 'logs/{}/{}.txt'.format(args.dataset,final_model_name)
    
    open(log_path, 'w').write(str(args) + '\n\n')
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
         models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.generative,args.predictor,args.semantic,args.extra,\
            args.supervised,args.decoupling,args.concatenate,args.distill)
    # print(model)
    model_t = moco.builder.MoCo(
         models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.generative,args.predictor,args.semantic,args.extra,\
            args.supervised,args.decoupling,args.concatenate,args.distill)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model_t.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
            model_t = torch.nn.parallel.DistributedDataParallel(model_t, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            model_t.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_t = torch.nn.parallel.DistributedDataParallel(model_t)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_t = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    if args.distill:
        # state_dict = torch.load(root_path+"/surgical_code/ssl_surgical/saved_models/cholec80/mocov2_un200_zero/200.pth.tar")
        loc = 'cuda:{}'.format(args.gpu)
        
        t_path = "mocov2base_b0_s25_g0_w0.0_p0_sem0_w0.0_e0_dFalse_INpre1_onlyfcTrue_supervisedFalse_ms2048_numinstance2_cat0"
      
        checkpoint = torch.load(root_path+"/surgical_code/ssl_surgical/saved_models/cholec80/{}/199.pth.tar".format(t_path))
        # for l,l2 in zip(checkpoint['state_dict'].keys(),model.state_dict()):
        #     print(l,l2)
        
        [misskeys, unexpkeys] = model_t.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Missing keys: {}'.format(misskeys))
        print('Unexpect keys: {}'.format(unexpkeys))
       
       
        for name, param in (model_t.module.named_parameters()):
        
           
            param.requires_grad = False  # not update by gradient
            # else:
            #     param.requires_grad = True
            #     print(name)
        # parameters = list(filter(lambda p: p.requires_grad==True, model.parameters()))
        # print(len(parameters))
        parameters = model.parameters()

    elif args.imagenet==1:
        state_dict = torch.load(root_path+"/IN_supervised/resnet50-0676ba61.pth")
        new_dict = {}
        for name, param in model.named_parameters():
            # print(name)

            if 'fc' not in name:
                # param.requires_grad = False
                for k in state_dict.keys(): 
                    # sss
                    if k == name[len('module.encoder_q.'):]:
                        print(k,name)
                        new_dict[name] = state_dict[k]
                        break
        model.module.encoder_q.fc[0].weight.data.normal_(mean=0.0, std=0.01)
        model.module.encoder_q.fc[0].bias.data.zero_()
        model.module.encoder_q.fc[2].weight.data.normal_(mean=0.0, std=0.01)
        model.module.encoder_q.fc[2].bias.data.zero_()

        model.module.encoder_k.fc[0].weight.data.normal_(mean=0.0, std=0.01)
        model.module.encoder_k.fc[0].bias.data.zero_()
        model.module.encoder_k.fc[2].weight.data.normal_(mean=0.0, std=0.01)
        model.module.encoder_k.fc[2].bias.data.zero_()
        msg = model.load_state_dict(new_dict, strict=False)
        print(msg)
        # model.module.encoder_k.fc.0.weight.data.normal_(mean=0.0, std=0.01)
        # model.module.encoder_k.fc.0.bias.data.zero_()
        # model.module.encoder_k.fc.2.weight.data.normal_(mean=0.0, std=0.01)
        # model.module.encoder_k.fc.2.bias.data.zero_()
        # for name, param in model.named_parameters():
            # print(name,param.requires_grad)
        # parameters = list(filter(lambda p: p.requires_grad==True, model.parameters()))
        # parameters = model.parameters()
        if args.onlyfc:
            for name, param in (model.module.named_parameters()):
           
                if 'fc' not in name:
                    param.requires_grad = False  # not update by gradient
                # else:
                #     param.requires_grad = True
                #     print(name)
            parameters = list(filter(lambda p: p.requires_grad==True, model.parameters()))
            print(len(parameters))
        else:
            parameters = model.parameters()
        # print(parameters)
  
    else:
        
        parameters = model.parameters()
    # print(parameters)
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, '')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    
    blacklist = []
    for i in range(1,41):
        # blacklist.append('%02d'%i)
        blacklist.append(i)

    
        video_folder = 'train_dataset'
      
            
    framewise_traindataset = moco_dataset.FramewiseDataset(args.dataset, dataset_pth.format(args.data_type)+'/{}'.format(args.dataset),video_folder=video_folder,blacklist=blacklist,sample_rate=args.sample_rate,\
            transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
        
   
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(framewise_traindataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        framewise_traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    final_model_name = 'mocov2{}_INpre{}_onlyfc{}_super{}_ms{}_numin{}_cat{}_dis{}_w{}'.\
        format(args.method,(args.sample_rate),args.imagenet,args.onlyfc,args.supervised,\
                args.moco_k,args.num_instance,args.concatenate,args.distill,args.dis_weight)
   
    log_path = 'logs/{}/{}.txt'.format(args.dataset,final_model_name)

    if not os.path.exists('logs/{}'.format(args.dataset)):
            os.makedirs('logs/{}'.format(args.dataset))
    criterion2 = nn.MSELoss().cuda()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        
       
        train(train_loader, model, model_t, criterion,criterion2, optimizer, epoch, args,log_path)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            
            model_root_path = 'saved_models/{}/{}/'.format(args.dataset,final_model_name)
            if (epoch+1) %50 == 0:
                if not os.path.exists(model_root_path):
                    os.makedirs(model_root_path)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='saved_models/{}/{}/{}.pth.tar'.format(args.dataset,final_model_name,epoch))



def train(train_loader, model, model_t,criterion, criterion2, optimizer, epoch, args,log_path):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    if args.distill:
        all_losses = AverageMeter('All_loss', ':.2e')
        losses_distill = AverageMeter('Loss_distill', ':.2e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, all_losses, losses, losses_distill, top1, top5],log_path,
            prefix="Epoch: [{}]".format(epoch))
   
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],log_path,
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, (images, _, mask1, mask2) in enumerate(train_loader):
    # for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(target)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            
     
        if args.distill:
            output, target = model(im_q=images[0], im_k=images[1])
            with torch.no_grad():  # no gradient to keys
                output_t, target_t = model_t(im_q=images[0], im_k=images[1])
            output_t = output_t.detach()
            log_simi = F.log_softmax(output, dim=1)
            simi_knowledge = F.softmax(output_t, dim=1)
            
            loss_distill = F.kl_div(log_simi, simi_knowledge, \
                        reduction='batchmean') 
            # print(loss_distill)
            # sssss

        else:
            output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

       


        losses.update(loss.item(), images[0].size(0))
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # if args.method == "multi":
        #     losses_multi = AverageMeter('Loss_multi', ':.2e')


        if args.distill:
            loss += loss + args.dis_weight * loss_distill
        # print(output.size(),target.size())
  
       
        if args.distill:
            losses_distill.update(loss_distill.item(), images[0].size(0))
            all_losses.update(loss.item(), images[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
           
           

def save_checkpoint(state, is_best, filename='./savedmodels/checkpoint.pth.tar'):
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, log_path, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_path = log_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        # return entries
        open(self.log_path, 'a').write('\t'.join(entries)+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
