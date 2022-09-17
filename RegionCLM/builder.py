# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torch.nn import init
from mmcv.cnn import kaiming_init, normal_init


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class RegionCLNonLinearNeckV1(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(RegionCLNonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x, randStartW=None, randStartH=None, randWidth=None, randHeight=None, randperm=None, unShuffle=None):
        assert len(x) == 1
        x = x[0]
        if randStartW is None:
            if self.with_avg_pool:
                x = self.avgpool(x)
            return self.mlp(x.view(x.size(0), -1))
        else:
            mask = torch.ones_like(x, device=x.device)
            if randHeight is None:
                randHeight = randWidth
            mix_mean_shuffle = torch.mean(x[:, :, randStartH:randStartH+randHeight, randStartW:randStartW+randWidth], [2, 3])
            mask[:, :, randStartH:randStartH+randHeight, randStartW:randStartW+randWidth] = 0.
            origin_mean = torch.sum(x * mask, dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
            feature = torch.cat([origin_mean, mix_mean_shuffle], 0)
            origin_mean, mix_mean_shuffle = torch.chunk(self.mlp(feature), 2)
            origin_mean_shuffle = origin_mean[randperm].clone()
            mix_mean = mix_mean_shuffle[unShuffle].clone()
            return origin_mean, origin_mean_shuffle, mix_mean, mix_mean_shuffle

class RegionCLM(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False,generative=False, predictor=False,cutMixUpper=4,
                 cutMixLower=1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(RegionCLM, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.generative = generative
        self.predictor = predictor
        self.cutMixUpper = cutMixUpper
        self.cutMixLower = cutMixLower
        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_k = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)
        
        neck = RegionCLNonLinearNeckV1(in_channels=2048,
                                                hid_channels=2048,
                                                out_channels=128,
                                                with_avg_pool=True)
        self.encoder_q = nn.Sequential(
                        base_encoder(num_classes=dim, generative=generative,region=True), neck)
        self.encoder_k = nn.Sequential(
                         base_encoder(num_classes=dim, generative=generative,region=True), neck)
        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     # print('dim_mlp',dim_mlp)
        #     # else:
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if predictor:
                pred_dim = 128
            
                self.predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim, bias=False),
                                                        nn.BatchNorm1d(pred_dim),
                                                        nn.ReLU(inplace=True), # hidden layer
                                                        nn.Linear(pred_dim, dim)) # output layer
            
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):



            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def RegionSwapping(self, img):
        '''
        RegionSwapping(img)
        Args:
        :param img: [B, C, H, W]

        Return:
        :param img_mix: [B, C, H, W]
        '''

        B, C, H, W = img.shape
        randperm = torch.arange(B - 1, -1, -1)
        unshuffle = torch.argsort(randperm)
        randWidth = (32 * torch.randint(self.cutMixLower, self.cutMixUpper, (1,)).float())
        randHeight = (32 * torch.randint(self.cutMixLower, self.cutMixUpper, (1,)).float())

        randStartW = torch.randint(0, W, (1,)).float()
        randStartW = torch.round(randStartW / 32.) * 32.
        randStartW = torch.minimum(randStartW, W - 1 - randWidth)

        randStartH = torch.randint(0, H, (1,)).float()
        randStartH = torch.round(randStartH / 32.) * 32.
        randStartH = torch.minimum(randStartH, H - 1 - randHeight)

        randStartW = randStartW.long()
        randStartH = randStartH.long()
        randWidth = randWidth.long()
        randHeight = randHeight.long()

        img_mix = img.clone()
        img_mix[:, :, randStartH:randStartH + randHeight, randStartW:randStartW + randWidth] = img[randperm, :, randStartH:randStartH + randHeight, randStartW:randStartW + randWidth]

        return img_mix, randStartW.float() / 32., randStartH.float() / 32., randWidth.float() / 32., randHeight.float() / 32., randperm, unshuffle
    
    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        im_q_swapped, randStartW, randStartH, randWidth, randHeight, randperm, unShuffle = self.RegionSwapping(im_q)
        # compute query features
        q = self.encoder_q[0](im_q)
        # print(q.size())
        q = self.encoder_q[1]([q])

        q_swapped = self.encoder_q[0](im_q_swapped)
        q_canvas, q_canvas_shuffle, q_paste, q_paste_shuffle = self.encoder_q[1]([q_swapped], randStartW.long(), randStartH.long(), randWidth.long(), randHeight.long(), randperm, unShuffle)    # queries: NxC

        q = nn.functional.normalize(q, dim=1)
        q_canvas = nn.functional.normalize(q_canvas, dim=1)
        q_canvas_shuffle = nn.functional.normalize(q_canvas_shuffle, dim=1)
        q_paste = nn.functional.normalize(q_paste, dim=1)
        q_paste_shuffle = nn.functional.normalize(q_paste_shuffle, dim=1)
        # if self.generative:
            
        #     q,g_q = self.encoder_q(im_q)  # queries: NxC
          
        # else:
        #     q = self.encoder_q(im_q)
          
        # if self.predictor:
        #         # ssss
        #         # print(q.size())
        #         q = self.predictor(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k[0](im_k)
            # print(k.size())
            k = self.encoder_k[1]([k])  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos_instance = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_pos_region_canvas = torch.einsum('nc,nc->n', [q_canvas, k]).unsqueeze(-1)
        l_pos_region_paste = torch.einsum('nc,nc->n', [q_paste, k]).unsqueeze(-1)

        l_pos_region = torch.cat([l_pos_region_canvas, l_pos_region_paste], dim=0)



        # negative logits: NxK
        queue = self.queue.clone().detach()

        l_neg_instance = torch.einsum('nc,ck->nk', [q, queue])

        l_neg_canvas_inter = torch.einsum('nc,ck->nk', [q_canvas, queue])
        l_neg_canvas_intra = torch.einsum('nc,nc->n', [q_canvas, q_paste_shuffle.detach()]).unsqueeze(-1)
        l_neg_canvas = torch.cat([l_neg_canvas_intra, l_neg_canvas_inter], dim=1)

        l_neg_paste_inter = torch.einsum('nc,ck->nk', [q_paste, queue])
        l_neg_paste_intra = torch.einsum('nc,nc->n', [q_paste, q_canvas_shuffle.detach()]).unsqueeze(-1)
        l_neg_paste = torch.cat([l_neg_paste_intra, l_neg_paste_inter], dim=1)

        l_neg_region = torch.cat([l_neg_canvas, l_neg_paste], dim=0)

        # logits: Nx(1+K)
        logits_instance = torch.cat([l_pos_instance, l_neg_instance], dim=1)
        logits_region =  torch.cat([l_pos_region, l_neg_region], dim=1)
        # apply temperature
        logits_instance /= self.T
        logits_region /= self.T

        # labels: positive key indicators
        labels_instance = torch.zeros(logits_instance.shape[0], dtype=torch.long).cuda()
        labels_region = torch.zeros(logits_region.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        # print(logits_instance.size(),logits_region.size(),labels_instance.size())
        return logits_instance,labels_instance, logits_region,  labels_region

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
