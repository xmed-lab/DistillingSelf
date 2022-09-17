
from enum import Flag
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection  import KFold


from models import ms_tcn
from tools.utils import segment_bars_with_confidence_score, PKI
from tools.dataset import VideoDataset
phase2label_dicts = {
    'cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'm2cai16':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}

}


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125
# print(device)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='hierarch_train')
parser.add_argument('--dataset', default="cholec80")
parser.add_argument('--dataset_path', default="/datasets/fixed/m2cai16/")
parser.add_argument('--backbone', default="")
parser.add_argument('--sample_rate', default=5, type=int)
parser.add_argument('--fps', default=5, type=int)
parser.add_argument('--best_ep', default=5, type=int)
parser.add_argument('--test_sample_rate', default=5, type=int)
parser.add_argument('--k', default=-100, type=int) # for cross validate type
parser.add_argument('--refine_model', default='gru')
parser.add_argument('--masked', default=False)
parser.add_argument('--softdtw', default=False)
parser.add_argument('--num_classes', default=8)
parser.add_argument('--dtw_rate', default=1,type=int)
parser.add_argument('--model', default="Base")
# parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--epochs', default=100)
parser.add_argument('--gpu', default="3", type=str)
parser.add_argument('--combine_loss', default=False, type=bool)
parser.add_argument('--ms_loss', default=True, type=bool)
parser.add_argument('--lc_loss', default=False, type=bool)
parser.add_argument('--gl_loss', default=False, type=bool)
parser.add_argument('--fpn', default=False, type=bool)
parser.add_argument('--output', default=False, type=bool)
parser.add_argument('--feature', default=False, type=bool)
parser.add_argument('--trans', default=False, type=bool)
parser.add_argument('--prototype', default=False, type=bool)
parser.add_argument('--last', default=False, type=bool)
parser.add_argument('--first', default=False, type=bool)
parser.add_argument('--hier', default=False, type=bool)
####ms-tcn2
parser.add_argument('--num_layers_PG', default="11", type=int)
parser.add_argument('--num_layers_R', default="10", type=int)
parser.add_argument('--num_R', default="3", type=int)

##Transformer
parser.add_argument('--head_num', default=8)
parser.add_argument('--embed_num', default=512)
parser.add_argument('--block_num', default=1)
parser.add_argument('--positional_encoding_type', default="learned", type=str, help="fixed or learned")
args = parser.parse_args()
# print(args.combine_loss)
learning_rate = 5e-4
epochs = 100
refine_epochs = 40

f_path = os.path.abspath('..')
root_path = f_path.split('surgical_code')[0]
feat_path = root_path+'ssl_surgical/features'
if args.dataset == 'm2cai16':
    refine_epochs = 15 # early stopping
   
args.sample_rate=args.fps
loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')


num_stages = 3  # refinement stages
if args.dataset == 'm2cai16':
    num_stages = 2 # for over-fitting
num_layers = 12 # layers of prediction tcn e
num_f_maps = 64
dim = 2048
sample_rate = args.sample_rate
test_sample_rate = args.test_sample_rate
num_classes = len(phase2label_dicts[args.dataset])
args.num_classes = num_classes
# print(args.num_classes)
dtw_rate=args.dtw_rate
num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R

print(args)



def base_train(model, train_loader, validation_loader, save_dir = 'models/base_tcn', debug = False):
    global learning_rate, epochs
    model.to(device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_epoch = 0
    best_acc = 0
    model.train()
    for epoch in range(1, epochs + 1):
        if epoch % 30 == 0:
            learning_rate = learning_rate * 0.5
        
        correct = 0
        total = 0
        loss_item = 0
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
        for (video, labels, mask, video_name ) in (train_loader):
        
              
                labels = torch.Tensor(labels).long()
                mask = torch.Tensor(mask).float()

                

                video, labels = video.to(device), labels.to(device)
            
                # print(video.size(), labels.size())
                # ssss
                mask = mask.to(device)

                outputs = model(video)
               
                loss = 0
                loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1)) # cross_entropy loss
                loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16)) # smooth loss

                loss_item += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

              

                _, predicted = torch.max(outputs.data, 1)
                
                correct += ((predicted == labels).sum()).item()
                total += labels.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        if debug:
            test_acc,all_preds=base_test(model, validation_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                # pred_name = "/home/xmli/xpding/code/casual_tcn/results/{}.pkl".format(best_epoch)
                # with open(pred_name, 'wb') as f:
                #     pickle.dump(all_preds, f)
                torch.save(model.state_dict(), save_dir + '/best_{}_{}.model'.format(sample_rate,epoch))
        print('Best Test: Acc {}, Epoch {}'.format(best_acc, best_epoch))
        
        

def base_test(model, test_loader, save_prediction=False, random_mask=False):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        all_preds = []
        for (video, labels, mask, video_name ) in (test_loader):
            labels = torch.Tensor(labels).long()
            if random_mask:
                # random_mask
                mask = np.random.choice(2, len(mask), replace=True, p=[0.3,0.7])
                mask = torch.from_numpy(mask).float().to(device)
            else:
                mask = torch.Tensor(mask).float().to(device)
                
            video, labels = video.to(device), labels.to(device)
            
            mask = mask.to(device)
            # print(mask.size())
           
            outputs = model(video)
            
           

            _, predicted = torch.max(outputs.data, 1)
            
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
           
    
        print('Test: Acc {}'.format(correct / total))
        return correct / total, all_preds

def base_predict(model, test_loader, argdataset, sample_rate, pki = False,split='test'):

    phase2label_dicts = {
    'cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'm2cai16':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}
    }
    model.to(device)
    model.eval()
    
    pic_save_dir = 'results/{}/{}/vis/'.format(args.dataset,args.backbone)
    results_dir = 'results/{}/{}/prediction_{}/'.format(args.dataset,args.backbone,args.sample_rate)
    # /home/xdingaf/share/datasets/surgical/workflow/m2cai16/phase_annotations
    gt_dir = root_path+'/datasets/surgical/workflow/{}/phase_annotations/'.format(args.dataset)
    # gt_phase_dir = '/home/xmli/xpding/datasets/fixed/cholec80/features/test_dataset/gt-phase'
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with torch.no_grad():
        correct =0
        total =0 
        for (video, labels, mask, video_name) in tqdm(test_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            print(video.size(),video_name,labels.size())
            video = video.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            re = model(video)
           
            
            confidence, predicted = torch.max(F.softmax(re.data,1), 1)
           

            # _, predicted = torch.max(re.data, 1)

            # print(predicted,labels)
            
            # correct += ((predicted == labels).sum()).item()
            # total += labels.shape[0]

            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]


        
            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()
            
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted])

            # if pki:
            #     # best hyper by grid search
            #     alpha = 3
            #     beta = 0.95
            #     gamma = 30            
            #     predicted, _ = PKI(confidence, predicted, transtion_prior_matrix, alpha, beta, gamma)
                        
            
            # predicted_phases_txt = label2phase(predicted, phase2label_dict=phase2label_dicts[argdataset])
            predicted_phases_expand = []
            # predicted_phases_expand = predicted
            # print(sample_rate)
            for i in predicted:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] )) # we downsample the framerate from 25fps to 5fps
            
            # for i in predicted_phases_txt:
            #     predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 5 * sample_rate)) # we downsample the framerate from 25fps to 5fps
            if args.dataset == 'm2cai16':
                v_n = int(video_name[0].split('.')[0])
                target_video_file = "%02d_pred.txt"%(v_n)
            else:
                target_video_file = video_name[0].split('.')[0] + '_pred.txt'
            if args.dataset == 'm2cai16':
                v_n = int(video_name[0].split('.')[0])
               
                gt_file = 'test_workflow_video_%02d.txt'%(v_n)
            else:
                v_n = int(video_name[0].split('.')[0])
                gt_file = 'video%02d-phase.txt'%(v_n)
            # print(gt_file)
            g_ptr = open(os.path.join(gt_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
            # g_phase_ptr = open(os.path.join(gt_phase_dir, target_video_file), 'w')
 
            gt = g_ptr.readlines()[1:] ##
            # gt = g_ptr.readlines() ##
            gt = gt[::args.sample_rate]
            print(len(gt), len(predicted_phases_expand))
            # ssss
            if len(gt) >  len(predicted_phases_expand):
                lst = predicted_phases_expand[-1]
                print(len(gt) - len(predicted_phases_expand))
                for i in range(0,len(gt) - len(predicted_phases_expand)):
                    predicted_phases_expand=np.append(predicted_phases_expand,lst)
            else:
                predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            print(len(gt), len(predicted_phases_expand))
            assert len(predicted_phases_expand) == len(gt)
 
            f_ptr.write("Frame\tPhase\n")
            for index, line in enumerate(predicted_phases_expand):
                # print(int(line),args.dataset)
                phase_dict = phase2label_dicts[args.dataset]
                p_phase = ''
                for k,v in phase_dict.items():
                    if v==int(line):
                        p_phase = k
                        break

                # line = phase2label_dicts[args.dataset][int(line)]
                # f_ptr.write('{}\t{}\n'.format(index, int(line)))
                f_ptr.write('{}\t{}\n'.format(index, p_phase))
            f_ptr.close()

            # g_phase_ptr.write("Frame\tPhase\n")
            # for line in (gt):
            #     line = line.strip('\n')
            #     index, pp = line.split('\t')
            #     pp = phase2label_dicts[args.dataset][pp]
            #     g_phase_ptr.write('{}\t{}\n'.format(index, pp))
            # g_phase_ptr.close()
        print(correct/total)


    
    
if args.model=="Base":
    base_model = ms_tcn.BaseCausalTCN(num_layers, num_f_maps, dim, num_classes)
    


annotation_path = '/datasets/surgical/workflow/{}/phase_annotations'


if args.action == 'base_train':
    
    video_train_folder = root_path+'/datasets/surgical/workflow/{}/train_dataset'
    video_train_feature_folder = 'surgical_code/ssl_surgical/features/{}/{}/train_dataset/video_feature@2020@{}'
    video_traindataset = VideoDataset(args.dataset, root_path, root_path+annotation_path.format(args.dataset),\
            video_train_feature_folder.format(args.backbone,args.dataset,args.sample_rate),\
            video_train_folder.format(args.dataset),split='train',sample_rate=args.sample_rate)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_test_folder = root_path+'/datasets/surgical/workflow/{}/test_dataset'
    video_test_feature_folder = 'surgical_code/ssl_surgical/features/{}/{}/test_dataset/video_feature@2020@{}'
    video_testdataset = VideoDataset(args.dataset, root_path, root_path+annotation_path.format(args.dataset),\
            video_test_feature_folder.format(args.backbone,args.dataset,args.sample_rate),\
            video_test_folder.format(args.dataset),split='test',sample_rate=args.sample_rate)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    
    model_save_dir = 'models/{}/{}'.format(args.dataset,args.backbone)
    base_train(base_model, video_train_dataloader, video_test_dataloader, save_dir=model_save_dir, debug=True)


if args.action == 'base_predict':
    
    model_path = 'models/{}/{}/best_{}_{}.model'.format(args.dataset,args.backbone,args.sample_rate, args.best_ep)
    base_model.load_state_dict(torch.load(model_path))
    video_folder = root_path+'/datasets/surgical/workflow/{}/test_dataset'
    video_feature_folder = 'surgical_code/ssl_surgical/features/{}/{}/test_dataset/video_feature@2020@{}'
    video_testdataset = VideoDataset(args.dataset, root_path, root_path+annotation_path.format(args.dataset),\
        video_feature_folder.format(args.backbone,args.dataset,args.sample_rate),\
        video_folder.format(args.dataset),split='test',sample_rate=args.sample_rate)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)

    # sssss
    # base_test(base_model, video_test_dataloader)
    base_predict(base_model, video_test_dataloader,args.dataset, test_sample_rate)
    


        

    
    
