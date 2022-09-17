"""The functions for MoCoV2 contrastive loss
Code partially borrowed from
https://github.com/YihengZhang-CV/SeCo-Sequence-Contrastive-Learning/blob/main/seco/Contrast.py.

MIT License
Copyright (c) 2020 YihengZhang-CV
"""

import torch
import torch.nn as nn
import math


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        loss = 0.0
        if isinstance(x, dict):
            for k,v in x.items():
                label = torch.zeros([v.shape[0]]).long().to(x.device)
                loss = loss + self.criterion(x, label)
        else:
            label = torch.zeros([x.shape[0]]).long().to(x.device)
            loss  = self.criterion(x, label)
        return loss


class MemorySeCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, feature_dim, queue_size, temperature=0.10, temperature_intra=0.10,num_label=10,class_bank=False):
        super(MemorySeCo, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.temperature_intra = temperature_intra
       
        self.feature_dim = feature_dim
        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        self.class_bank = class_bank
        if self.class_bank:
            memory = torch.rand(num_label,self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
            # self.memory_ls = {}
            self.num_label = num_label
            self.index = {}
            # print(num_label)
            for i in range(0,num_label):
                self.index[i] = 0
                # memory = torch.rand(num_label,self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
                # memory_ls.append(memory)
                # print(i)
                # self.memory_ls[i] = memory
            # self.register_buffer('memory', self.memory_ls[i])
        else:
            self.index = 0
            memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        

    def _select_pos_by_label(self, q, k, i, label ,pos_label , pos_set,neg_label,neg_set, k_all_label,k_all):
        ## selcet q by label 
        # print(label.size())
        # print(label == i)
        # sss
        indices = (label == i).long().nonzero()
        l_q = torch.index_select(q, 0, indices.squeeze())

        ## selcet k by label 
        indices = (label == i).nonzero()
        l_k = torch.index_select(k, 0, indices.squeeze())

        # selcet pos_set  by label 
        # print(pos_label.size(),pos_set.size())
        indices = (pos_label.reshape(-1) == i).nonzero()
        l_pos_set = torch.index_select(pos_set.reshape(-1,self.feature_dim), 0, indices.squeeze())
        # print(l_pos_set.size(),l_pos_set.get_device())


        ## selcet neg_set  by label 
        indices = (neg_label.reshape(-1) == i).nonzero()
        l_neg_set = torch.index_select(neg_set.reshape(-1,self.feature_dim), 0, indices.squeeze())
        # print(l_neg_set.size(),l_neg_set.get_device())

        ## selcet k_all  by label 
        indices = (k_all_label == i).nonzero()
        l_k_all = torch.index_select(k_all, 0, indices.squeeze())
        # print(l_k_all.size(),l_k_all.get_device())

        # pos_bank = torch.cat([l_pos_set,l_neg_set,l_k_all],dim=0)
        # print(pos_bank.size(),self.memory_ls[i].size())
        return l_q, l_k, l_pos_set, l_neg_set, l_k_all


    def forward(self, q, k, pos_set, neg_set, k_all, inter=True, label=None, k_all_label = None, pos_label=None, neg_label=None):
        # print(q.size(),k_all_label.size(),k_all.size(),self.memory_ls[1].size())
        # ssss
        # feat_dim = q.size(1)
        num_positive = pos_set.size(1)
        if self.class_bank:
            q_dict = {}
            k_dict = {}
            pos_dict = {}
            neg_dict = {}
            all_dict = {}
            out_ls = {}
            for i in range(0,self.num_label+1):

                q_dict[i], k_dict[i], pos_dict[i], neg_dict[i], all_dict[i] = self._select_pos_by_label(q, k, i, label ,pos_label , pos_set, neg_label, neg_set, k_all_label, k_all)
            
                # print(feat_label_dict[i].get_device(),feat_label_dict[i].size(),k_all.size())
            neg_ls = []
            for i in range(0,self.num_label): 
                if q_dict[i].size(0)==0:
                    continue
                for j in range(0,self.num_label):
                    if i==j or neg_dict[j].size(0)==0:
                        continue
                   
                    neg_ls.append(neg_dict[j])
                neg_set = torch.cat(neg_ls,dim=0)
                # print(neg_ls.size())
                # print(self.memory[i-1].size(),self.memory[i-1].get_device())
                neg_memory = torch.cat((neg_set, self.memory[i-1]),dim=0)
                # print(neg_memory.size())
                
                #positve socres
                q = q_dict[i]
                k = k_dict[i]
                pos_set_l = pos_dict[i]
                # print(q.size(),k.size(),pos_set_l.size())
                l_pos_k =  (q * k.detach()).sum(dim=-1, keepdim=True)  # shape: (num_label, 1)
                l_pos_set = (q.repeat(num_positive,1)* pos_set_l.detach()).sum(dim=-1, keepdim=True) # shape: (num_positive*num_label, 1)

                #negative socres
                if inter:
                    # neg_set =  neg_set.reshape(-1,self.feature_dim)
                    # neg_memory = torch.cat((neg_set,self.memory),dim=0)
                    l_neg = torch.mm(q, neg_memory.clone().detach().t())  # shape: (num_label, memory_banksize+negative)
                    # print(l_pos_k.size(),l_pos_set.size(),l_neg.size())
                    pos_all = torch.cat((l_pos_k, l_pos_set), dim=0)
                    neg_all = l_neg.repeat(pos_all.size(0)//l_neg.size(0), 1)
                    out = torch.cat((pos_all, neg_all), dim=1)
                    # sss
                    out = torch.div(out, self.temperature).contiguous()
                    with torch.no_grad():
                        all_size = all_dict[i].shape[0]
                        out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index[i], self.queue_size)
                        self.memory[i-1].index_copy_(0, out_ids, all_dict[i])
                        self.index[i] = (self.index[i] + all_size) % self.queue_size
                    # print(out.size())
                else:
                    # out intra-frame similarity
                    # pos_set = pos_set.reshape(-1, self.feature_dim)
                    # print(q.unsqueeze(dim=1).size(),pos_set.size())
                    l_pos_neg = torch.matmul(q.unsqueeze(dim=1), pos_set_l.clone().detach().transpose(1,2)).squeeze(dim=1) ##shape: (batchSize, num_positive)
                    l_neg = torch.matmul(q.unsqueeze(dim=1), neg_set.clone().detach().transpose(1,2)).squeeze(dim=1) ##shape: (batchSize, num_negative)
                    print(l_neg.size(),l_pos_neg.size())
                    # ssss
                    out = torch.div(torch.cat((l_pos_k, torch.cat((l_pos_neg, l_neg), dim=1)), dim=-1),
                                    self.temperature_intra).contiguous()
                out_ls[i]=out
            #     print(out.size())
            # out_ls = torch.cat(out_ls,dim=0)
            # print(out_ls.size())
            # sss
        else:
            l_pos_k =  (q * k.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
            # q = q.unsqueeze(dim=1)
           
            l_pos_set = (q.unsqueeze(dim=1) * pos_set.detach()).sum(dim=-1, keepdim=True).mean(dim=1)  # shape: (batchSize, num_positive, 1) -> (batchSize, 1)
            # l_pos_set = l_pos_set.reshape(-1,1)           # shape: (batchSize*num_positive,1)
            # print()
            # l_pos_sf = (q * k_sf.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
            # l_pos_df1 = (q * k_df1.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
            # l_pos_df2 = (q * k_df2.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
            if inter:
                neg_set =  neg_set.reshape(-1,self.feature_dim)
                neg_memory = torch.cat((neg_set,self.memory),dim=0)
                l_neg = torch.mm(q, neg_memory.clone().detach().t())
                # print(l_pos_k.size(),l_pos_set.size(),l_neg.size())
                out = torch.cat((torch.cat((l_pos_k, l_pos_set), dim=0), l_neg.repeat(2, 1)), dim=1)
                out = torch.div(out, self.temperature).contiguous()
                with torch.no_grad():
                    all_size = k_all.shape[0]
                    out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
                    self.memory.index_copy_(0, out_ids, k_all)
                    self.index = (self.index + all_size) % self.queue_size
                # print(out.size())
            else:
                # out intra-frame similarity
                # pos_set = pos_set.reshape(-1, self.feature_dim)
                # print(q.unsqueeze(dim=1).size(),pos_set.size())
                l_pos_neg = torch.matmul(q.unsqueeze(dim=1), pos_set.clone().detach().transpose(1,2)).squeeze(dim=1) ##shape: (batchSize, num_positive)
                l_neg = torch.matmul(q.unsqueeze(dim=1), neg_set.clone().detach().transpose(1,2)).squeeze(dim=1) ##shape: (batchSize, num_negative)
                # print(l_neg.size())
                out = torch.div(torch.cat((l_pos_k, torch.cat((l_pos_neg, l_neg), dim=1)), dim=-1),
                                self.temperature_intra).contiguous()
                # print(out.size())
        return out

# class MemorySeCo(nn.Module):
#     """Fixed-size queue with momentum encoder"""

#     def __init__(self, feature_dim, queue_size, temperature=0.10, temperature_intra=0.10):
#         super(MemorySeCo, self).__init__()
#         self.queue_size = queue_size
#         self.temperature = temperature
#         self.temperature_intra = temperature_intra
#         self.index = 0

#         # noinspection PyCallingNonCallable
#         self.register_buffer('params', torch.tensor([-1]))
#         stdv = 1. / math.sqrt(feature_dim / 3)
#         memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
#         self.register_buffer('memory', memory)

#     def forward(self, q, k_sf, k_df1, k_df2, k_all, inter=True):
#         l_pos_sf = (q * k_sf.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
#         l_pos_df1 = (q * k_df1.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
#         l_pos_df2 = (q * k_df2.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
#         if inter:
#             l_neg = torch.mm(q, self.memory.clone().detach().t())
#             out = torch.cat((torch.cat((l_pos_sf, l_pos_df1, l_pos_df2), dim=0), l_neg.repeat(3, 1)), dim=1)
#             out = torch.div(out, self.temperature).contiguous()
#             with torch.no_grad():
#                 all_size = k_all.shape[0]
#                 out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
#                 self.memory.index_copy_(0, out_ids, k_all)
#                 self.index = (self.index + all_size) % self.queue_size
#         else:
#             # out intra-frame similarity
#             out = torch.div(torch.cat((l_pos_sf.repeat(2, 1), torch.cat((l_pos_df1, l_pos_df2), dim=0)), dim=-1),
#                             self.temperature_intra).contiguous()

#         return out

class MemoryVCLR(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.10):
        super(MemoryVCLR, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    def forward(self, q, k, k_all):
        l_pos = (q * k.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()
        with torch.no_grad():
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.index = (self.index + all_size) % self.queue_size

        return out
