# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from email.policy import strict
from re import L
from numpy import concatenate
import torch
import torch.nn as nn
import models
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, c_in, c_out):
        super(Head, self).__init__()
        self.pre_mlp = nn.Conv2d(c_in,c_out,1,1)
        
    def forward(self, x):
        return self.pre_mlp(x)

class Decoder(nn.Module):
    def __init__(self, c_in):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Conv2d(c_in,512,7),
                                        nn.ConvTranspose2d(512,512,7),
                                        nn.ConvTranspose2d(512,256,2,2),
                                        nn.ConvTranspose2d(256,128,2,2),
                                        nn.ConvTranspose2d(128,64,2,2),
                                        nn.ConvTranspose2d(64,32,2,2),
                                        nn.ConvTranspose2d(32,16,2,2),
                                        nn.ConvTranspose2d(16,3,1,1),
                                        )
    def forward(self, x):
        return self.decoder(x)
                    

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, generative=False, predictor=False,semantic=0,extra=0,supervised=False,decoupling=False,concatenate=0,distill=0):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.generative = generative
        self.predictor = predictor
        self.semantic = semantic
        self.extra = extra
        self.supervised = supervised
        self.decoupling = decoupling
        self.concatenate = concatenate
        self.distill = distill
        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_k = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim, generative=generative,semantic=semantic,decoupling=decoupling,concatenate=concatenate)
        self.encoder_q = base_encoder(num_classes=dim, generative=generative,semantic=semantic,decoupling=decoupling,concatenate=concatenate)
        #

        
        #####        
        if semantic or concatenate:
            self.encoder_semantic = self._get_moco_in()
        if concatenate == 3:
            self.pre_mlp = Head(2048,64)
        elif concatenate == 4 or concatenate == 5:
            self.decoder = Decoder(2048)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            
            # print('dim_mlp',dim_mlp)
            # else:
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

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
    def _get_moco_in(self):
        
        mocov2_un200 = models.resnet50(pretrained=False,semantic=True)
        # ssss
        if self.supervised:
            model_path = "./saved_models/{}/{}/{}".format('cholec80',"IN_supervised_zero","resnet50-0676ba61.pth")
            pre_dict = torch.load(model_path)
            mocov2_un200.load_state_dict(pre_dict,strict=False)
        else:
            model_path = "./saved_models/{}/{}/{}".format('cholec80',"mocov2_un200_zero","200.pth.tar")
            pre_dict = torch.load(model_path)['state_dict']
            for k in list(pre_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    # print(k)
                    
                    if k.startswith('module.encoder_q'):
                        # remove prefix
                    
                        # print(k[len("module.encoder_q."):])
                        pre_dict[k[len("module.encoder_q."):]] = pre_dict[k]
                    # delete renamed or unused k
                    del pre_dict[k]


            mocov2_un200.load_state_dict(pre_dict,strict=False)
        
        for name, param in mocov2_un200.named_parameters():
            # print(name)
            param.requires_grad = False
               
                
                
        return mocov2_un200

   
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
        # print(self.K, batch_size)
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

    def forward(self, im_q, im_k,img_orig = None, multipositive=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.concatenate==1 or self.concatenate==2:
            _,feature_ls_q,_ = self.encoder_semantic(im_q)
            _,feature_ls_k,_ = self.encoder_semantic(im_k)
            # feat_q = feature_ls_q[-1]
            # feat_k = feature_ls_k
            q = self.encoder_q(im_q,feature_ls_q)
            # print(feat_q.size(),feat_k.size())
        elif self.concatenate==3:
            _,feature_ls_q,_ = self.encoder_semantic(im_q)
            _,feature_ls_k,_ = self.encoder_semantic(im_k)
            feat_q = feature_ls_q[-1]
            feat_k = feature_ls_k[-1]
            feat_q = self.pre_mlp(feat_q)
            feat_k = self.pre_mlp(feat_k)
            feat_q = F.upsample_bilinear(feat_q,im_q.size(2))
            feat_k = F.upsample_bilinear(feat_k,im_k.size(2))
            # print(feat_q.size(),im_q.size())
            im_q = torch.cat((im_q,feat_q),dim=1)
            im_k = torch.cat((im_k,feat_k),dim=1)
            # print(im_q.size())
        elif self.concatenate==4:
            # q,g_q = self.encoder_q(im_q)  # queries: NxC
            _,feature_ls_q,_ = self.encoder_semantic(im_q)
            _,feature_ls_k,_ = self.encoder_semantic(im_k)
            feat_q = feature_ls_q[-1]
            feat_k = feature_ls_k[-1]
            # print(feat_q.size())
            feat_q = self.decoder(feat_q)
            feat_k = self.decoder(feat_k)
            # print(feat_q.size())
            im_q = torch.cat((im_q,feat_q),dim=1)
            im_k = torch.cat((im_k,feat_k),dim=1)

        elif self.concatenate==5:
            # q,g_q = self.encoder_q(im_q)  # queries: NxC
            _,feature_ls_q,_ = self.encoder_semantic(im_q)
            _,feature_ls_k,_ = self.encoder_semantic(im_k)
            feat_q = feature_ls_q[-1]
            feat_k = feature_ls_k[-1]
            # print(feat_q.size())
            feat_q = self.decoder(feat_q)
            feat_k = self.decoder(feat_k)
            # print(feat_q.size())
            im_q = 0.8*im_q + 0.2*feat_k
            im_k = 0.8*im_k + 0.2*feat_k

        elif self.concatenate==6:
            # with torch.no_grad():
            _,feature_ls_q_s,_ = self.encoder_semantic(im_q)
            _,feature_ls_k_s,_ = self.encoder_semantic(im_k)
            # for i in feature_ls_k_s:
            #     print(i.size())
            q = self.encoder_q(im_q,feature_ls_q_s)
            # q = self.encoder_q(im_q)
            # sss
        # compute query features
        elif self.generative:
            
            q,g_q = self.encoder_q(im_q)  # queries: NxC
          
      
        elif self.semantic:
            
            if self.extra==1:  
                q,_,g_q = self.encoder_q(im_q)  # queries: NxC     
                _,feature_ls,_  = self.encoder_semantic(im_q)
                _,feature_ls_moco,_ = self.encoder_semantic(g_q)
                # ssss
            elif self.extra==2:
                q,feature_ls,g_q = self.encoder_q(im_q)  # queries: NxC     
                _,feature_ls_moco,_  = self.encoder_semantic(im_q)
                # _,feature_ls_moco,_ = self.encoder_semantic(im_q)
            # elif self.extra==0:
                # q,_,g_q = self.encoder_q(im_q)  # queries: NxC     
                # _,feature_ls_moco,_ = self.encoder_semantic(im_q)
            
            # for i in feature_ls_s:
            #     print(i.size())
        elif self.decoupling:
            q, q_recon = self.encoder_q(im_q)
            online_k, online_k_recon = self.encoder_q(im_k)
            orig_features,_ = self.encoder_q(img_orig)
            online_k = nn.functional.normalize(online_k, dim=1)
            orig_features = nn.functional.normalize(orig_features, dim=1)
        
        # elif self.distill:
        #     with torch.no_grad():  # no gradient to keys
        #         t_q = self.t_encoder(im_q)
       
        else:
            q = self.encoder_q(im_q)
        
        if self.predictor:
                # ssss
                # print(q.size())
                q = self.predictor(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            if multipositive is not None:
                # print(multipositive.size())
                multipositive = self.encoder_k(multipositive.view(-1,multipositive.size(2),multipositive.size(3),multipositive.size(4)))
                # print(multipositive.size())
            if self.generative:
                k, g_k = self.encoder_k(im_k)  # keys: NxC
            elif self.semantic:
                k,_, g_k = self.encoder_k(im_k)  # keys: NxC
            elif self.decoupling:
                target_q, target_q_recon = self.encoder_k(im_q)
                k, k_recon = self.encoder_k(im_k)
                target_q = nn.functional.normalize(target_q, dim=1)
            
            elif self.concatenate==2:
                feat_k = feature_ls_k[-1]
                k = self.encoder_k(im_k,feat_k)
                
            elif self.concatenate==6:
                k = self.encoder_k(im_k,feature_ls_k_s)
                # k = self.encoder_k(im_k)
            else:
                k = self.encoder_k(im_k)
            
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        
        if multipositive == None:
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

           
        else:
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            # l_pos = torch.cat([l_pos, l_pos_multi],dim=0)
            # print(l_pos.size(),l_neg.size())
            logits = torch.cat([l_pos, l_neg], dim=1)



            bs = q.size(0)
            feat_dim = q.size(-1)
            # k = k.unsqueeze(dim=1)
            multipositive = multipositive.view(bs,-1 ,multipositive.size(-1))
            # print(k.size(), multipositive.size())
            # ssss
            # print(multipositive.size())
            # k_r = k.unsqueeze(dim=1)
            # k_r = torch.cat([k_r,multipositive],dim=1)  #NxMxK
            m = multipositive.size(1)
            # k_r = k_r.view(-1, feat_dim)
            # m = sss.size(1)
            # k = k.repeat(1,m,1)
            # k = k.view(-1, k.size(-1))
            # print(q.size(),k.size())
            q = q.repeat(1,m,1)
            
            q = q.view(-1, feat_dim)
            multipositive = multipositive.view(-1, feat_dim)
            
            l_pos_mutli = torch.einsum('nc,nc->n', [q, multipositive])
            # print(l_pos_mutli.size())
            l_pos_mutli = l_pos_mutli.view(bs,m).mean(dim=1,keepdim=True)
            # print(l_pos.size())
            # l_pos = l_pos.view()
            
            logits_multi = torch.cat([l_pos_mutli, l_neg], dim=1)

            # print(logits_multi.size(),logits.size())
            # print(q.size())
            logits_multi /= self.T
        
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)


        if self.generative:
            return logits, labels, g_q, g_k
        elif self.semantic:
            return logits, labels, feature_ls, feature_ls_moco,g_q

        elif self.decoupling:
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos_k = torch.einsum('nc,nc->n', [online_k, target_q]).unsqueeze(-1)
            # negative logits: NxK
            l_neg_k = torch.einsum('nc,ck->nk', [online_k, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits_k = torch.cat([l_pos_k, l_neg_k], dim=1)

            # apply temperature
            logits_k /= self.T

            # labels: positive key indicators
            labels_k = torch.zeros(logits_k.shape[0], dtype=torch.long).cuda()
            bs = logits_k.shape[0]

            mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)


            logits_io = torch.mm(q, orig_features.t()) /self.T
            logits_jo = torch.mm(k, orig_features.t()) / self.T
            probs_io = F.softmax(logits_io[torch.logical_not(mask)], -1)
            probs_jo = F.log_softmax(logits_jo[torch.logical_not(mask)], -1)
            # kl_div_loss1 = F.kl_div(probs_io, probs_jo, log_target=True, reduction="sum")
            kl_div_loss1 = F.kl_div(probs_jo,probs_io , log_target=True, reduction="sum")

            logits_iok = torch.mm(online_k, orig_features.t()) /self.T
            logits_jok = torch.mm(target_q, orig_features.t()) / self.T
            probs_iok = F.softmax(logits_iok[torch.logical_not(mask)], -1)
            probs_jok = F.log_softmax(logits_jok[torch.logical_not(mask)], -1)
            # kl_div_loss2 = F.kl_div(probs_iok, probs_jok, log_target=True, reduction="sum")
            kl_div_loss2 = F.kl_div(probs_jok, probs_iok, log_target=True, reduction="sum")

            return logits, labels, logits_k ,labels_k, q_recon, online_k_recon, kl_div_loss1+kl_div_loss2
       
        elif multipositive is not None:
            return logits, labels,logits_multi
        else:

            return logits, labels


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
