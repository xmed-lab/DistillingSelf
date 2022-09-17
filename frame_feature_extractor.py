# This file contains the code about:
# 1. Extract inceptionv3 features for the video frame.
# 2. Find out hard frames.
# This step is not necessary if you download the extracted feature we provided.
import imp
import os
from numpy.lib.function_base import bartlett
# gpustr = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpustr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models, transforms
from models.resnet_mlp import resnet50
# from models.resnet import resnet50
import models as mymodels
from tools.utils import load_pretrained
import argparse
import numpy as np
import random
from tqdm import tqdm
from tools.utils import get_transform
from tools import dataset
from moco.builder import Head, Decoder
# from model import inception_v3
# import clip
from PIL import Image
# import os
f_path = os.path.abspath('..')
# print(f_path.split('shadow_code'))
root_path = f_path.split('surgical_code')[0]
dataset_pth = root_path+'datasets/surgical/workflow/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', choices=['train', 'extract', 'test', 'hard_frame'], default='train')
parser.add_argument('--dataset', default="cholec80", choices=['cholec80','m2cai16'])
parser.add_argument('--target', type=str, default='train_set')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--best_ep', type=str, default="183.pth.tar")
parser.add_argument('--imagenet', action="store_true",help="imagenet pretrain or not.")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10)
parser.add_argument('--sample_rate', type=int, default=1)
parser.add_argument('--cropsize', type=int, default=224)
parser.add_argument('--crop', type=float, default=0.2)
parser.add_argument('--pretrained_model', type=str,default='', help="pretrained model path")
parser.add_argument('--k', type=int, default=-100)
parser.add_argument('--linear', action="store_true",help="imagenet pretrain or not.")
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()
if args.linear:
    learning_rate = 1e-2
else:
    learning_rate = 1e-4

epochs = args.epochs
loss_layer = nn.CrossEntropyLoss()
# if args.dataset == "cholec80":
dataset_pth = root_path+'datasets/surgical/workflow/'
# if args.dataset == "m2cai16":
#     dataset_pth = root_path+'datasets/workflow/camma.u-strasbg.fr/m2cai2016/datasets/workflow/'
    
def train(model, model_name, save_dir, train_loader, validation_loader,linear=False,pre_mlp=None):
    global learning_rate
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # model.to(device)
    best_epoch = 0.0
    if 'cat' in model_name and 'cat0' not in model_name:
        model_moco = _get_moco_in(model_name)
        model_moco = (model_moco).cuda()
        pre_mlp = (pre_mlp).cuda()
    pre_mlp
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        if epoch % 2 == 0:
            learning_rate = learning_rate * 0.5
        model.train()

        correct = 0
        total = 0
        loss_item = 0

        if linear:
            if 'vclr' in model_name:
               
                optimizer = torch.optim.Adam(model.module.fc_inter.parameters(), learning_rate, weight_decay=1e-5)
            else:
                optimizer = torch.optim.Adam(model.module.fc.parameters(), learning_rate, weight_decay=1e-5)
            print("linear")
        else:
            optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)

        for (imgs, labels, img_names) in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            # print(imgs.size())
            
          
            res = model(imgs) # of shape 64 x 7
                
          
            # print(res.size())
            loss = loss_layer(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        accuracy = test(model, validation_loader, model_name, pre_mlp)
        if accuracy > best_accuracy:
            best_accuracy =accuracy
            torch.save(model.state_dict(), save_dir + "/best_{}.model".format(epoch))
            # torch.save(pre_mlp.state_dict(), save_dir + "/pre_mlp_{}.model".format(epoch))

    print('Training done!')


def test(model, test_loader,model_name,pre_mlp=None):
    print('Testing...')
    model.eval()
    # model.to(device)
    correct = 0
    total = 0
    loss_item = 0
    if 'cat' in model_name and 'cat0' not in model_name:
        model_moco = _get_moco_in(model_name)
        model_moco = (model_moco).cuda()
        pre_mlp = (pre_mlp).cuda()
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
           
          
                
            
            res = model(imgs)  # of shape 64 x 7
                
            loss = loss_layer(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)
    print('Test: Acc {}, Loss {}'.format(correct / total, loss_item / total))
    accuracy = correct / total
    return accuracy


def extract(model, loader, save_path, model_name, pre_mlp= None):
    model.eval()
    model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
        
    err_dict = {}
   

    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(loader):
            # assert len(img_names) == 1 # batch_size = 1
            # print(img_names)
            video_list = []
            img_in_video_list = []
            for img_name in img_names:
                video, img_in_video = img_name.split('/')[-2], img_name.split('/')[-1] # video63 5730.jpg
                video_list.append(video)
                img_in_video_list.append(img_in_video)
            # sss
            # video, img_in_video = img_names[0].split('/')[-2], img_names[0].split('/')[-1] # video63 5730.jpg
           
            imgs, labels = imgs.to(device), labels.to(device)
            # for k, v
            # print(*list(model.children())[0:-1])
            # sss
            
           
            
            resnet50_feature_extractor=nn.Sequential(*list(model.children())[:-1])
            features = resnet50_feature_extractor(imgs)
         

            # print(features.size())
            features = features.to('cpu').numpy() # of shape 1 x 2048
            video_folder_list = []
            feature_save_path_list = []
            assert len(video_list) == features.shape[0]
            for idex,video in enumerate(video_list):
                video_folder = os.path.join(save_path, video)
                video_folder_list.append(video_folder)
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                feature_save_path = os.path.join(video_folder, img_in_video_list[idex].split('.')[0] + '.npy')
                # print(feature_save_path)
                feature_save_path_list.append(feature_save_path)
            
            # if os.path.exists(feature_save_path):
            #     continue
                # print(features[idex].shape)
                np.save(feature_save_path, features[idex])
    
    return err_dict



def imgf2videof(source_folder, target_folder, blacklist):
    '''
        Merge the extracted img feature to video feature.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # for video in os.listdir(source_folder):
    for video in blacklist:
        video = str(video)
        video_feature_save_path = os.path.join(target_folder, video + '.npy')
        video_abs_path = os.path.join(source_folder, video)
        imgs_list = []
        # print(video_abs_path)
        
        for img in os.listdir(video_abs_path):
            idex = img.split('.')[0]
            imgs_list.append(int(idex))
        imgs_list.sort()
        # for i in (imgs_list):
        #     print(i)
        # ssss
        # nums_of_imgs = len(os.listdir(video_abs_path))
        video_feature = []
        # print()
        for i in (imgs_list):
            
            img_abs_path = os.path.join(video_abs_path, str(i) + '.npy')
            print(np.load(img_abs_path).shape,img_abs_path)
            # sss
            feat = np.load(img_abs_path)
            # feat = np.expand_dims(feat, axis=1)
            # feat = np.expand_dims(feat, axis=1)
            # print(feat.shape)
            video_feature.append(feat)

        video_feature = np.concatenate(video_feature, axis=1)
        
        np.save(video_feature_save_path, video_feature)
        print('{} done!'.format(video),video_feature.shape)

def _get_moco_in(model_name):
        
        # mocov2_un200 = models.resnet50(pretrained=False)
        mocov2_un200 = mymodels.resnet50(pretrained=False,semantic=True)
        # ssss
        if 'superFalse' in model_name:
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
        elif 'superTrue' in model_name:
            model_path = "./saved_models/{}/{}/{}".format('cholec80',"IN_supervised","resnet50-0676ba61.pth")
            pre_dict = torch.load(model_path)
       
        mocov2_un200.load_state_dict(pre_dict,strict=False)
        
        return  mocov2_un200


if __name__ == '__main__':
    # inception = inception_v3(pretrained=True, aux_logits=False)
    # print(args.imagenet)
    # sss
    resnet_50 = models.resnet50(pretrained=args.imagenet)
    fc_features = resnet_50.fc.in_features
    resnet_50.fc = nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset]))
    model_name= args.model
    pre_mlp = None
    if model_name=='resnet50' and not args.imagenet:
        model_name += '_noIN'
    # if args.linear:
    #     model_name+='_linear'
    if args.action == 'train':
        isself = False
       

        if 'mocov2' in model_name:
            # isself = True
            model_path = "./saved_models/{}/{}/{}.pth.tar".format('cholec80',model_name,args.best_ep)
            pre_dict = torch.load(model_path)['state_dict']
           
            for k in list(pre_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                # print(k)
                
                if k.startswith('module.encoder_q'):
                    # remove prefix
                    if 'fc' in k:
                        # del pre_dict[k]
                        continue
                    # print(k[len("module.encoder_q."):])
                    pre_dict[k[len("module.encoder_q."):]] = pre_dict[k]
                # delete renamed or unused k
                del pre_dict[k]
            resnet_50.fc.weight.data.normal_(mean=0.0, std=0.01)
            resnet_50.fc.bias.data.zero_()
        
            [misskeys, unexpkeys] = resnet_50.load_state_dict(pre_dict,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
            print("==> loaded checkpoint '{}'".format(args.pretrained_model))
            
           
            resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()
        elif 'Region' in model_name:
            # isself = True
            model_path = "./saved_models/{}/{}/{}".format('cholec80',model_name,args.best_ep)
            pre_dict = torch.load(model_path)['state_dict']
            new_state={}
            for k in list(pre_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                # print(k)
                if k.startswith('module.encoder_q.0'):
                    # remove prefix
                    if 'fc' in k:
                        # del pre_dict[k]
                        continue
                    # print(k[len("module.encoder_q."):])
                    new_state[k[len("'module.encoder_q.0"):]] = pre_dict[k]
                # delete renamed or unused k
                del pre_dict[k]
            resnet_50.fc.weight.data.normal_(mean=0.0, std=0.01)
            resnet_50.fc.bias.data.zero_()
            [misskeys, unexpkeys] = resnet_50.load_state_dict(new_state,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
            print("==> loaded checkpoint '{}'".format(args.pretrained_model))
           
            resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()

        elif 'simsiam' in model_name:
            model_path = "./saved_models/{}/{}/{}".format('cholec80',model_name,args.best_ep)
            pre_dict = torch.load(model_path)['state_dict']
            for k in list(pre_dict.keys()):
            #     # retain only encoder_q up to before the embedding layer
               
                
                if k.startswith('module.encoder'):
                    # remove prefix
                  
                    # print(k[len("module.encoder."):])
                    pre_dict[k[len("module.encoder."):]] = pre_dict[k]
                # delete renamed or unused k
                del pre_dict[k]
           
            resnet_50.fc.weight.data.normal_(mean=0.0, std=0.01)
            resnet_50.fc.bias.data.zero_()
            [misskeys, unexpkeys] = resnet_50.load_state_dict(pre_dict,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
            print("==> loaded checkpoint '{}'".format(args.pretrained_model))
            
           
            resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()




    
        # 
        elif "swav" in model_name:
           
            if os.path.isfile(args.pretrained_model):
                
                state_dict = torch.load(args.pretrained_model)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # remove prefixe "module."
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                for k, v in resnet_50.state_dict().items():
                    # if k in list(state_dict):
                    #     print(k)
                    if k not in list(state_dict):
                       print('key "{}" could not be found in provided state dict'.format(k))
                    elif state_dict[k].shape != v.shape:
                        print('key "{}" is of different shape in model and provided state dict'.format(k))
                        state_dict[k] = v
                # print(state_dict)
                # sss
                resnet_50.fc.weight.data.normal_(mean=0.0, std=0.01)
                resnet_50.fc.bias.data.zero_()
                # resnet_50.load_state_dict(pre_dict,strict=False)
                [misskeys, unexpkeys] = resnet_50.load_state_dict(state_dict,strict=False)
                print('Missing keys: {}'.format(misskeys))
                print('Unexpect keys: {}'.format(unexpkeys))
                print("==> loaded checkpoint '{}'".format(args.pretrained_model))
                # resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()
                # print("Load pretrained model with msg: {}".format(resnet_50))
                resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()
                # print(resnet_50.parameters())
       
        else:
             resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()

        ##get transforms
        # train_transforms = get_transform(args)
        blacklist = []
        for i in range(args.start,args.end):
            # blacklist.append('%02d'%i)
            blacklist.append(i)
        # if args.dataset == 'cholec80':
        framewise_traindataset = dataset.FramewiseDataset(args.dataset, dataset_pth+'/{}'.format(args.dataset),\
            video_folder="train_dataset",sample_rate=args.sample_rate,blacklist=blacklist,transform=None)
    # else:
        #     framewise_traindataset = dataset.FramewiseDataset(args.dataset, dataset_pth,label_folder= 'train_dataset',\
        #         video_folder="train_dataset",sample_rate=args.sample_rate,blacklist=blacklist,transform=train_transforms)
        framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=64, shuffle=True, drop_last=False)

        blacklist = []
        for i in range(41,81):
            blacklist.append(i)
        # if args.dataset == 'cholec80':
        framewise_testdataset = dataset.FramewiseDataset(args.dataset, dataset_pth+'/{}'.format(args.dataset),\
            video_folder="test_dataset",sample_rate=args.sample_rate,blacklist=blacklist, split='test')
        # else:
        #     framewise_testdataset = dataset.FramewiseDataset(args.dataset, dataset_pth,label_folder= 'test_dataset',\
        #         video_folder="test_dataset",sample_rate=args.sample_rate,blacklist=blacklist,split='test')
        framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=256, shuffle=True, drop_last=False)
       
        if args.linear:
            model_name+='_linear'
        epochsss =  args.best_ep.split('.')[0]
        model_name += '_epoch'+epochsss
        train(resnet_50, model_name,'saved_models/{}/{}_s{}_e{}'.format(args.dataset,model_name,args.start,args.end), framewise_train_dataloader, framewise_test_dataloader,args.linear,pre_mlp)
    
    
    if args.action == 'extract': # extract inception feature
        pre_mlp = None
        model_path = "./saved_models/{}/{}/best_{}.model".format(args.dataset,model_name,args.best_ep)
        # model_path = "./saved_models/{}/{}/{}.pth.tar".format(args.dataset,model_name,args.best_ep)
#         model_path = 'models/{}/inceptionv3/2.model'.format(args.dataset) # use your own model
       
        if 'Region' in model_name:
            # isself = True
            model_path = "./saved_models/{}/{}/{}.pth.tar".format('cholec80',model_name,args.best_ep)
            pre_dict = torch.load(model_path)['state_dict']
            new_state={}
            for k in list(pre_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                # print(k)
                if k.startswith('module.encoder_q.0'):
                    # remove prefix
                    if 'fc' in k:
                        # del pre_dict[k]
                        continue
                    # print(k[len("module.encoder_q."):])
                    new_state[k[len("'module.encoder_q.0"):]] = pre_dict[k]
                # delete renamed or unused k
                del pre_dict[k]
            resnet_50.fc.weight.data.normal_(mean=0.0, std=0.01)
            resnet_50.fc.bias.data.zero_()
            [misskeys, unexpkeys] = resnet_50.load_state_dict(new_state,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
            print("==> loaded checkpoint '{}'".format(args.pretrained_model))
           
            resnet_50 = torch.nn.DataParallel(resnet_50).cuda().train()
        elif 'linear' in model_name:
            resnet_50 = resnet_50.cuda()
       
            pre_dict = torch.load(model_path)
            new_pre = {}
                
            for k,v in pre_dict.items():
                    # print(k)
                    name = k[7:]
                    if 'fc' in k:
                        continue
                    # name = k
                    # print(name)
                    new_pre[name] = v
            [misskeys, unexpkeys] = resnet_50.load_state_dict(new_pre,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
            # ssss
        elif 'zero' in model_name:
            model_path = "./saved_models/{}/{}/{}.pth.tar".format(args.dataset,model_name,args.best_ep)
            
            resnet_50 = models.resnet50(pretrained=args.imagenet)
           
            pre_dict = torch.load(model_path)['state_dict']
            
            for k in list(pre_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                # print(k)
                
                if k.startswith('module.encoder_q') and not 'fc' in k:
                    # remove prefix
                  
                    # print(k[len("module.encoder_q."):])
                    
                    pre_dict[k[len("module.encoder_q."):]] = pre_dict[k]
               
                # delete renamed or unused k
                del pre_dict[k]
            # fc_features = resnet_50.fc.in_features
            # resnet_50.fc = nn.Sequential(nn.Linear(fc_features*2, fc_features),nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset])))
            # resnet_50.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            # resnet_50.fc[0].bias.data.zero_()
            # resnet_50.fc[1].weight.data.normal_(mean=0.0, std=0.01)
            # resnet_50.fc[1].bias.data.zero_()
            # for k in list(pre_dict.keys()):
            #     print(k)
            
            [misskeys, unexpkeys] = resnet_50.load_state_dict(pre_dict,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
        
        else:
            resnet_50 = resnet_50.cuda()
       
            pre_dict = torch.load(model_path)
            new_pre = {}
                
            for k,v in pre_dict.items():
                    # print(k)
                    name = k[7:]
                    if 'fc' in k:
                        continue
                    # name = k
                    # print(name)
                    new_pre[name] = v
            
            # for name in resnet_50.state_dict():
                # print(name)
            # ssss
            # if not 'concat' in model_name:
            # resnet_50.load_state_dict(new_pre)
            
            [misskeys, unexpkeys] = resnet_50.load_state_dict(new_pre,strict=False)
            print('Missing keys: {}'.format(misskeys))
            print('Unexpect keys: {}'.format(unexpkeys))
        # resnet_50 = models.resnet50(pretrained=args.imagenet)
        # fc_features = resnet_50.fc.in_features
        # resnet_50.fc = nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset]))

        # resnet_50.register_forward_hook(make_hook(name, 'forward'))

        if args.target == 'train_set':
            blacklist = []
            frame_path = 'features/'+model_name+'/{}/train_dataset/frame_feature@2020@{}/'.format(args.dataset,args.sample_rate)
            video_path = 'features/'+model_name+'/{}/train_dataset/video_feature@2020@{}/'.format(args.dataset,args.sample_rate)
            for i in range(args.start,args.end):
                # blacklist.append('%02d'%i)
                blacklist.append(i)
            framewise_traindataset = dataset.FramewiseDataset(args.dataset, dataset_pth+'/{}'.format(args.dataset),\
                video_folder="train_dataset",sample_rate=args.sample_rate,blacklist=blacklist,transform=None)
            framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=256, shuffle=False, drop_last=False)

            extract(resnet_50, framewise_train_dataloader, frame_path,model_name, pre_mlp)
            imgf2videof(frame_path, video_path,blacklist)
        elif args.target == 'test_set':
            blacklist = []
            frame_path = 'features/'+model_name+'/{}/test_dataset/frame_feature@2020@{}/'.format(args.dataset,args.sample_rate)
            video_path = 'features/'+model_name+'/{}/test_dataset/video_feature@2020@{}/'.format(args.dataset,args.sample_rate)
            for i in range(args.start,args.end):
                # blacklist.append('%02d'%i)
                blacklist.append(i)
            framewise_testdataset = dataset.FramewiseDataset(args.dataset, dataset_pth+'/{}'.format(args.dataset),\
                video_folder="test_dataset",sample_rate=args.sample_rate,blacklist=blacklist,transform=None,split='test')

            framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=256, shuffle=False, drop_last=False)
        
            extract(resnet_50, framewise_test_dataloader, frame_path,model_name,pre_mlp)
            imgf2videof(frame_path, video_path, blacklist)
        
   
        
    
        
        

    
   


