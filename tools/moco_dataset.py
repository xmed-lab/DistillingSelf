from cv2 import phase
from numpy.lib.function_base import append
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
# from tools.utils import get_transform
from torchvision import transforms
import os
import numpy as np
import torch
import re
import copy
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
    
#     'm2cai16':{
#     'Preparation':0,
#     'CalotTriangleDissection':1,
#     'ClippingCutting':2,
#     'GallbladderDissection':3,
#     'GallbladderPackaging':4,
#     'CleaningCoagulation':5,
#     'GallbladderRetraction':6}
}


transtion_prior_matrix = [
    [1,1,0,0,0,0,0],
    [0,1,1,0,0,0,0],
    [0,0,1,1,0,0,0],
    [0,0,0,1,1,1,0],
    [0,0,0,0,1,1,1],
    [0,0,0,0,1,1,1],
    [0,0,0,0,0,1,1],    
]

# transtion_prior_matrix = [
#     [1,1,0,0,0,0,0],
#     [1,1,1,1,0,0,0],
#     [0,1,1,1,0,0,0],
#     [0,0,1,1,1,1,0],
#     [0,0,0,1,1,1,1],
#     [0,0,0,1,1,1,1],
#     [0,0,0,0,1,1,1],    
# ]

# transtion_prior_matrix = [
#     [1,1,1,0,0,0,0],
#     [1,1,1,0,0,0,0],
#     [1,1,1,0,0,0,0],
#     [0,0,1,1,1,1,1],
#     [0,0,0,1,1,1,1],
#     [0,0,0,1,1,1,1],
#     [0,0,0,1,1,1,1],    
# ]


def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else len(phase2label_dict) for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases

class FramewiseDataset(Dataset):
    def __init__(self, dataset, root, label_folder='phase_annotations', video_folder='cutMargin', blacklist=[],sample_rate=1,transform=None,split='train'):
        self.dataset = dataset
        self.blacklist= blacklist
        self.imgs = []
        self.labels = []
        self.transform = transform
        label_folder = os.path.join(root, label_folder)
        video_folder = os.path.join(root, video_folder)
        
        # for v in os.listdir(video_folder):
        for v in blacklist:
            # print(v)
            # print(type(v))
            # sss
            # print(int(v))z

            # if int(v) not in blacklist:
            #     # print(v)
            #     continue
            
            v_abs_path = os.path.join(video_folder, str(v))
            if self.dataset == "cholec80":
                v_label_file_abs_path = os.path.join(label_folder, 'video%02d-phase.txt'%(int(v)) )
            else:
                # v_abs_path = os.path.join(video_folder, '%02d_25'%v)
                if split=='train':
                    v_label_file_abs_path = os.path.join(label_folder, 'workflow_video_%02d.txt'%(int(v)) )
                else:
                    v_label_file_abs_path = os.path.join(label_folder, 'test_workflow_video_%02d.txt'%(int(v)) )

            labels = self.read_labels(v_label_file_abs_path)
           
            images = os.listdir(v_abs_path)
            # print(len(images))
            # min_len = min(len(labels),len(images))
            # images = images[:min_len]
            # labels = labels[:min_len]
            # print(len(labels),len(images))
            # assert len(
            #     labels
            # ) == len(images)
            image_list = []
            # label_list = []
            for image in images:
                image_index = int(image.split('.')[0])
                image_list.append(image_index)
            image_list.sort()
            length  = len(image_list)
            images = []
            
          
            for i in range(0,length,sample_rate):
                # print(v,str(image_list[i])+'.jpg')
                images.append(str(image_list[i])+'.jpg')
            # ssss
            for image in images:
                image_index = int(image.split('.')[0])
                # print(image_index)
                self.imgs.append(os.path.join(v_abs_path, image))
                # self.labels.append(labels[image_index])
                try:
                    # print(image_index)
                    self.labels.append(labels[image_index])
                except:
                    print(image_index,v_abs_path,v_label_file_abs_path)
                    # ssss
        # print(len(self.labels),len(self.imgs))
      
            

        print('FramewiseDataset: Load dataset {} with {} images.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):

        
        (img, img2) = self.transform(default_loader(self.imgs[item]))
        orig_img = self.get_transform()(default_loader(self.imgs[item]))
        labels = self.labels[item]
        mask1 = copy.deepcopy(img)
        mask2 = copy.deepcopy(img2)
        return (img, img2, orig_img), labels, mask1, mask2

    

    def read_labels(self, label_file):
        # print(label_file)
        # label_file = '/home/xdingaf/datasets/surgical/workflow//m2cai16/test_dataset_25/13'
        num =0 
        with open(label_file,'r') as f:
            # print(label_file)
            # phases = []
            phases_dict = {}
            labels = []
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                for k, v in phase2label_dicts[self.dataset].items():
                    if k in line:
                        labels.append(v)
                        break

                # print(line)
                num+=1
                # try:
                #     frame_index, phases = line.strip().split('\t')
                #     # phases[]append(line.strip().split('\t')[1])
                #     phases_dict[int(frame_index)] = phase2label_dicts[self.dataset][phases] 
                # except:
                #     # print(line.strip().split('\t'))
                #     # print(re.findall(r"[A-Za-z]+", line.strip()))
                   
                #     phase_str =  re.findall(r"[A-Za-z]+",line.strip())[0]
                #     frame_index =  frame_index.strip()
                #     filter(str.isdigit, frame_index)
                #     # print(frame_index)
                #     if phase_str not in phase2label_dicts[self.dataset].keys():
                #         print(phase_str,label_file)
                #         ssss
                #     phases_dict[int(frame_index)] = phase2label_dicts[self.dataset][phase_str] 
                    #  phase_str =  re.findall(r"[A-Za-z]+",line.strip())[0]
                    # phases.append(phase_str)
                # print(line.strip().split('\t'))
            # phases = [line.strip().split('\t')[1] for line in f.readlines()]
            # print(phases[0])
            # phases = phases[1:]
            # print(phases[1:])
            # ssss
            # labels = phase2label(phases, phase2label_dicts[self.dataset])
        # print(num,len(labels))
        # ssss
        # print(phases_dict[51425])
        return labels

