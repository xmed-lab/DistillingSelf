import os
import sys
import numpy as np
from .dataset import label2phase, phase2label_dicts
from matplotlib import pyplot as plt
from matplotlib import *
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
import torch.nn.functional as F
import random
from PIL import ImageFilter
min_len = 20
min_diff = 0.5
num_k = 5

def divide(input,weight,k_size,allindx,start,ans):
    v_len = input.size(0)
    if v_len<min_len or ans>3:
        return
    input = input.unsqueeze(0).unsqueeze(0)
    x = F.conv2d(input,weight,stride=1,padding=k_size//2)
    x = x.squeeze()
    # print(x)
    # x = x * torch.eye(v_len,v_len).cuda() + input * (torch.ones(v_len,v_len).cuda() - torch.eye(v_len,v_len).cuda())
    # x = x.squeeze()
    dia_tensor = torch.diagonal(x)
    # print(dia_tensor.max(),dia_tensor.min(),dia_tensor.mean())
    dia_tensor[0] = 0.0 
    dia_tensor[-1] = 0.0 
    
    top_v,indx = torch.topk(dia_tensor,k=num_k)
    # print(top_v)
    if top_v[0] - top_v[1:].mean() < min_diff*5*ans:
       
        return
    allindx.append(start+indx[0].cpu())
    input = input.squeeze()
    ans += 1
    divide(input[:indx[0],:indx[0]],weight,k_size, allindx, start,ans)
    divide(input[indx[0]+1:,indx[0]+1:],weight,k_size, allindx, start+indx[0]+1,ans)


def boundarg_gen(input,k_size):
    # k_size = 5
    kk = np.zeros((k_size,k_size))
    v_len = input.size(0)
    # img = torch.rand(1,1,v_len,v_len).cuda()
    img = input.unsqueeze(0).unsqueeze(0).cuda()
    
    # print(img)
    for i in range(k_size):
        for j in range(k_size):
            if i<k_size//2 and j<k_size//2:
                kk[i][j] = 1
            elif i<k_size//2 and j>k_size//2:
                kk[i][j] = -1
            elif i>k_size//2 and j<k_size//2:
                kk[i][j] = -1
            elif i>k_size//2 and j>k_size//2:
                kk[i][j] = 1
    kernel = torch.FloatTensor(kk).unsqueeze(0).unsqueeze(0)
    # print(kernel)
    weight = nn.Parameter(data=kernel, requires_grad=False).cuda()
    x = F.conv2d(img,weight,stride=1,padding=k_size//2)
    x = x.squeeze()
    # print(x)
    x = x * torch.eye(v_len,v_len).cuda() + img * (torch.ones(v_len,v_len).cuda() - torch.eye(v_len,v_len).cuda())
    # # print(x)
    x = x.squeeze()
    # dia_tensor = torch.diagonal(x)
    # # print(dia_tensor.max(),dia_tensor.min(),dia_tensor.mean())
    # dia_tensor[0] = 0.0 
    # dia_tensor[-1] = 0.0 
    # allindx = []
    # top_v,indx = torch.topk(dia_tensor,k=num_k)
    
    # upper_v = top_v[0]
    # mean_v = top_v[1:].mean()
    # # print(upper_v,mean_v)
    # if upper_v - mean_v < min_diff or v_len < min_len:
    #     return
    # allindx.append(indx[0].cpu())
    # img = img.squeeze()
    # divide(img[:indx[0],:indx[0]],weight,k_size,allindx,start=0,ans=1)
    # divide(img[indx[0]+1:,indx[0]+1:],weight,k_size,allindx,start=indx[0]+1,ans=1)
    
    # # print(top_v)
    # # # sxsss
    # # # final = final.squeeze()
    # # # print(final.size())
    # print(allindx)
    # allindx = indx
    return x

def save_predctions(args, save_path, video_name, preds, pro, save_prob=False):
    gt_dir = args.dataset_path+'/test_dataset/gt-phase'
    
    
    results_dir = sys.path[0]+'/results/{}/'.format(save_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # print(sys.path[0])
    target_video_file = video_name[0].split('.')[0] + '_pred.txt'
    # print(video_name)
    if args.dataset == 'cholec80':
        gt_file = video_name[0].split('.')[0] + '.txt'
    if args.dataset == 'm2cai16':
        gt_file = video_name[0].split('.')[0] + '.txt'
        
    predicted = preds.squeeze(0).tolist()

    pro = pro.squeeze(0).tolist()
    predicted_phases_expand = []
    probabilty_phases_expand = []
    
    for i in predicted:
        predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 25)) # 25 is the resampled resolution
        # print([p]*2)

        
      
    probabilty_phases_expand = np.array(probabilty_phases_expand)
    # print(gt_file)
    
    g_ptr = open(os.path.join(gt_dir, gt_file), "r")
    f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
    
    gt = g_ptr.readlines()##
    # print(len(gt),len(predicted_phases_expand))
    # sss
    # if args.dataset == 'cholec80':
    gt = gt[1:]
    predicted_phases_expand = predicted_phases_expand[0:len(gt)]
  
    assert len(predicted_phases_expand) == len(gt)
    predicted_phases_expand = label2phase(predicted_phases_expand,phase2label_dicts[args.dataset] )
    # f_ptr.write("Frame\tPhase\tGT\tProbabilty\n")
    f_ptr.write("Frame\tPhase\n")
    index=0
    if not save_prob:
        probabilty_phases_expand=gt
    for line,pro,gt_l in zip(predicted_phases_expand,probabilty_phases_expand,gt):
        p_str=''
        # print(pro)
        gt_l= gt_l.strip('\n')
        gt_label = gt_l.split('\t')[1]
        
        if save_prob:
            for ppp in pro:
                p_str+=' '+str(round(ppp,2))
            f_ptr.write('{}\t{}\t{}\t{}\n'.format(index, int(line), gt_label,p_str))
        else:
            f_ptr.write('{}\t{}\n'.format(index, str(line)))
        index += 1
    f_ptr.close()
def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)
    
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels)
    color_map = plt.cm.tab10

#     axprops = dict(xticks=[], yticks=[0,0.5,1], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=15)
    fig = plt.figure(figsize=(15, (num_pics+1) * 1.5))
    plt.clf()
    interval = 1 / (num_pics+2)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0.1, 1-i*interval, 0.8, interval - interval/num_pics]))
#         ax1.imshow([label], **barprops)
    titles = ['Ground Truth','Causal-TCN', 'Causal-TCN + PKI', 'Causal-TCN + MS-GRU']
    for i, label in enumerate(labels):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].imshow([label], **barprops)
#         axes[i].set_title(titles[i])
    
    ax99 = fig.add_axes([0.1, 0.05, 0.8, interval - interval/num_pics])
#     ax99.set_xlim(-len(confidence_score)/15, len(confidence_score) + len(confidence_score)/15)
    ax99.set_xlim(0, len(confidence_score))
    ax99.set_ylim(-0.2, 1.2)
    ax99.set_yticks([0,0.5,1])
    ax99.set_xticks([])
 
     
    ax99.plot(range(len(confidence_score)), confidence_score)

    # if save_path is not None:
    print(save_path)
    plt.savefig(save_path)
    
    # sss
    # else:
    #     plt.show()

    plt.close()
    
def PKI(confidence_seq, prediction_seq, transition_prior_matrix, alpha, beta, gamma): # fix the predictions that do not meet priors
    initital_phase = 0
    previous_phase = 0
    alpha_count = 0
    assert len(confidence_seq) == len(prediction_seq)
    refined_seq = []
    for i, prediction in enumerate(prediction_seq):
        if prediction == initital_phase:
            alpha_count = 0
            refined_seq.append(initital_phase)
        else:
            if prediction != previous_phase or confidence_seq[i] <= beta:
                alpha_count = 0
            
            if confidence_seq[i] >= beta:
                alpha_count += 1
            
            if transition_prior_matrix[initital_phase][prediction] == 1:
                refined_seq.append(prediction)
            else:
                refined_seq.append(initital_phase)
            
            if alpha_count >= alpha and transition_prior_matrix[initital_phase][prediction] == 1:
                initital_phase = prediction
                alpha_count = 0
                
            if alpha_count >= gamma:
                initital_phase = prediction
                alpha_count = 0
        previous_phase = prediction

    
    assert len(refined_seq) == len(prediction_seq)
    return refined_seq

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x, async_op=False)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class DistributedShuffle:
    @staticmethod
    def forward_shuffle(x, epoch=None):
        x_all = dist_collect(x)
        forward_inds, backward_inds = DistributedShuffle.get_shuffle_ids(x_all.shape[0], epoch)
        forward_inds_local = DistributedShuffle.get_local_id(forward_inds)
        return x_all[forward_inds_local], backward_inds

    @staticmethod
    def backward_shuffle(x, backward_inds, return_local=True):
        x_all = dist_collect(x)
        if return_local:
            backward_inds_local = DistributedShuffle.get_local_id(backward_inds)
            return x_all[backward_inds], x_all[backward_inds_local]
        else:
            return x_all[backward_inds]

    @staticmethod
    def get_local_id(ids):
        return ids.chunk(dist.get_world_size())[dist.get_rank()]

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        if epoch is not None:
            torch.manual_seed(epoch)
        forward_inds = torch.randperm(bsz).long().cuda()
        if epoch is None:
            torch.distributed.broadcast(forward_inds, src=0)
        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds


def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model.eval()
    model.apply(set_bn_train_helper)


@torch.no_grad()
def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data = p2.data * m + p1.data * (1 - m)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def load_pretrained(args, model, logger=print):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    print(ckpt.keys())
    if len(ckpt) == 3:  # moco initialization
        ckpt = {k[17:]: v for k, v in ckpt['state_dict'].items() if k.startswith('module.encoder_q')}
        for fc in ('fc_inter', 'fc_intra', 'fc_order', 'fc_tsn'):
            ckpt[fc + '.0.weight'] = ckpt['fc.0.weight']
            ckpt[fc + '.0.bias'] = ckpt['fc.0.bias']
            ckpt[fc + '.2.weight'] = ckpt['fc.2.weight']
            ckpt[fc + '.2.bias'] = ckpt['fc.2.bias']
    else:
        ckpt = ckpt['model']
    [misskeys, unexpkeys] = model.load_state_dict(ckpt, strict=False)
    logger('Missing keys: {}'.format(misskeys))
    logger('Unexpect keys: {}'.format(unexpkeys))
    logger("==> loaded checkpoint '{}'".format(args.pretrained_model))


def load_checkpoint(args, model, model_ema, contrast, contrast_tsn, optimizer, scheduler, logger=print):
    logger("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    contrast_tsn.load_state_dict(checkpoint['contrast_tsn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, model_ema, contrast,contrast2, contrast_tsn, optimizer, scheduler, logger=print):
    logger('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'contrast2': contrast2.state_dict(),
        'contrast_tsn': contrast_tsn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


def get_transform(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.cropsize, scale=(args.crop, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # train_dataset = build_dataset(args.dataset_name, args.dataset, transform=train_transform, split = args.datasplit)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True,
    #     sampler=train_sampler, drop_last=True)
    return train_transform

def get_loader(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.cropsize, scale=(args.crop, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = build_dataset(args.dataset_name, args.dataset, transform=train_transform, split = args.datasplit)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return train_loader

import argparse
def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    # parser.add_argument('--data_dir', type=str, required=True, help='root director of dataset')
    parser.add_argument('--dataset', type=str, default='cholec80', help='dataset to training')
    parser.add_argument('--datasplit', type=str, default='train')
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--cropsize', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--num_positive', type=int, default=3, help='frames within segment')
    parser.add_argument('--sample_rate', type=int, default=1, help='sample rate')
    # model
    parser.add_argument('--model_mlp', action='store_true', default=False)
    parser.add_argument("--seg_num", type=int, default=10, help='manual seed')
    # loss function
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce_k', type=int, default=800, help='num negative sampler')
    parser.add_argument('--nce_t', type=float, default=0.10, help='NCE temperature')
    parser.add_argument('--nce_t_intra', type=float, default=0.10, help='NCE temperature')

    # optimization
    parser.add_argument('--base_lr', type=float, default=0.1,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup_epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained_model', default='', type=str, metavar='PATH',
                        help='path to pretrained weights like imagenet (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--broadcast_buffer", action='store_true', default=False, help='broadcast_buffer for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=-1, help='manual seed')
    
    args = parser.parse_args()
    return args