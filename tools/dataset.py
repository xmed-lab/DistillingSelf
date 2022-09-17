from random import sample
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
    def __init__(self, dataset, root, label_folder='phase_annotations', video_folder='cutMargin', blacklist=[],sample_rate=1,transform=None,split='train',isself=False):
        self.dataset = dataset
        self.blacklist= blacklist
        self.imgs = []
        self.labels = []
        self.isself = isself
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
        if self.transform is None:
            # ssss
            self.transform = self.get_transform()

        print('FramewiseDataset: Load dataset {} with {} images.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):

       

        img, label, img_path = self.transform(default_loader(self.imgs[item])), self.labels[item], self.imgs[item]
     
        
        return  img, label, img_path
        # return img, label, img_path

    def get_transform(self):
        return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
        ])
        # return 

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

class VideoDataset(Dataset):
    def __init__(self, dataset, root, label_folder, video_feature_folder, video_folder,split='train',sample_rate=5):
        self.dataset = dataset
    
        self.videos = []
        self.labels = []
        #
        self.sup_labels=[]
        self.unsup_labels = []
        self.sup_videos = []
        self.unsup_videos = []
      
        ###
        self.hard_frames = []
        self.video_names = []
        self.mach_score = []
        if dataset =='cholec80':
            self.hard_frame_index = 7
        if dataset == 'm2cai16':
            self.hard_frame_index = 8 

        video_feature_folder = os.path.join(root, video_feature_folder)
        self.label_folder = label_folder
      
        video_folder = os.path.join(root , video_folder)
        for v_f in os.listdir(video_feature_folder):
           
            # if dataset == 'm2cai16' and 'test_dataset' in video_feature_folder:
            #     v_f = 'test_'+v_f
            
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            if self.dataset == "cholec80":
                v_label_file_abs_path = os.path.join(label_folder, 'video%02d-phase.txt'%(int(v_f.split('.')[0])) )
            else:
                # v_abs_path = os.path.join(video_folder, '%02d_25'%v)
                if split=='train':
                    v_label_file_abs_path = os.path.join(label_folder, 'workflow_video_%02d.txt'%(int(v_f.split('.')[0])) )
                else:
                    v_label_file_abs_path = os.path.join(label_folder, 'test_workflow_video_%02d.txt'%(int(v_f.split('.')[0])) )
           
            

            labels = self.read_labels(v_label_file_abs_path) 
            # 
            # print(root, video_folder)
            
           
            v_abs_path = os.path.join(video_folder, v_f.split('.')[0])
            # print(v_abs_path)
            # sss
            images = os.listdir(v_abs_path)
            # print(len(images))
            new_labels = []
            imgs_list = []
            for img in images:
                idex = img.split('.')[0]
                imgs_list.append(int(idex))
            imgs_list.sort()
            imgs_list = imgs_list[::sample_rate]
            # num= 0
            # for i,j  in zip(imgs_list,labels):
            #    if i==j:
            #        continue
            #    else:
            #        print(i,j)
            # print(num)
                
            # print(v_abs_path,v_label_file_abs_path)
            # print(len(labels),len(imgs_list))
            # ssss
            for index in imgs_list:
                # print(index,v_abs_path)
                try:
                    new_labels.append(labels[index])
                except:
                    # print(index,v_abs_path)
                    new_labels
            # sss
            videos = np.load(v_f_abs_path)
            # print(v_f_abs_path)
            print(videos.shape)
            # print(videos.shape[1],len(new_labels))
            v_len = min(videos.shape[1],len(new_labels))
            videos = videos[:,:v_len,:]
            new_labels = new_labels[:v_len]
            # print(videos.shape,len(new_labels))
            # ssss
            # ssss
            # masks = self.read_hard_frames(v_hard_frame_abs_path,  self.hard_frame_index)
            # masks = masks[::sample_rate]
           
                # unsup_labels.append([i-1])
            # assert len(labels) == len(masks)
            
            self.videos.append(videos)
            # match_score_start, match_score_end = self._get_train_label(torch.tensor(labels))
            # match_score = torch.cat((match_score_start.unsqueeze(0), match_score_end.unsqueeze(0)), 0)
            # match_score,_ = torch.max(match_score, 0)
            # self.mach_score.append(match_score)
            self.labels.append(new_labels)

            # self.hard_frames.append(masks)
            self.video_names.append(v_f)

        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.videos)

        # return len(self.unsup_labels)
    def _get_labels_start_end_time(self,target_tensor, bg_class):
        labels = []
        starts = []
        ends = []
        target=target_tensor.numpy()
        last_label = target[0]
        if target[0] not in bg_class:
            labels.append(target[0])
            starts.append(0)

        for i in range(np.shape(target)[0]):
            if target[i] != last_label:
                if target[i] not in bg_class:
                    labels.append(target[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = target[i]

        if last_label not in bg_class:
            ends.append(np.shape(target)[0]-1)
        return labels, starts, ends

    def _ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min) 
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.0)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def _get_train_label(self, target_tensor):

        total_frame = target_tensor.size()[0]
        temporal_scale = total_frame
        temporal_gap = 1.0 / temporal_scale
        anchor_xmin = [temporal_gap * i for i in range(temporal_scale)]
        anchor_xmax = [temporal_gap * i for i in range(1, temporal_scale + 1)]
       
        gt_label, gt_starts, gt_ends = self._get_labels_start_end_time(target_tensor, [0])  # original length
        gt_label, gt_starts, gt_ends = np.array(gt_label), np.array(gt_starts), np.array(gt_ends)
        gt_starts, gt_ends = gt_starts.astype(np.float), gt_ends.astype(np.float)
        gt_starts, gt_ends = gt_starts / total_frame, gt_ends / total_frame  # length to 0~1

        gt_lens = gt_ends - gt_starts
        gt_len_small = np.maximum(temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_starts - gt_len_small / 2, gt_starts + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_ends - gt_len_small / 2, gt_ends + gt_len_small / 2), axis=1)

        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_start, match_score_end

    def __getitem__(self, item):
        video, label, mask, video_name = self.videos[item], self.labels[item], self.labels[item], self.video_names[item]
        return video, label, mask, video_name
    
    # def __getitem__(self, item):
    #     sup_video, unsup_video, sup_label, unsup_label, mask, video_name = self.sup_videos[item], self.unsup_videos[item], self.sup_labels[item],  self.unsup_labels[item], self.hard_frames[item], self.video_names[item]
    #     return sup_video, unsup_video, sup_label, unsup_label, mask, video_name

    def read_labels(self, label_file):
        # print(label_file)
        # label_file = '/home/xdingaf/datasets/surgical/workflow//m2cai16/test_dataset_25/13'
        # print(label_file)

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

    def read_hard_frames(self, hard_frame_file, hard_frame_index):
        with open(hard_frame_file, 'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
#         return labels
         
        masks = np.array(labels)
        masks[masks != hard_frame_index] = 1
        masks[masks == hard_frame_index] = 0
        return masks.tolist()
    
    def merge(self, videodataset_a):
        self.videos += videodataset_a.videos
        self.labels += videodataset_a.labels
        self.hard_frames += videodataset_a.hard_frames
        self.video_names += videodataset_a.video_names
        
        print('After merge: ', self.__len__())


        
if __name__ == '__main__':
    '''
        UNIT TEST
    '''
    framewisedataset_cholec80 = FramewiseDataset('cholec80','cholec80/train_dataset', 5)
    framewisedataloader_cholec80 = DataLoader(framewisedataset_cholec80, batch_size=64, shuffle=True, drop_last=False)

    videodataset_cholec80 = VideoDataset('cholec80', 'cholec80/train_dataset', 5)
    videodataloader_cholec80 = DataLoader(videodataset_cholec80, batch_size=1, shuffle=True, drop_last=False)
