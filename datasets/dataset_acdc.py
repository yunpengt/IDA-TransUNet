import os
import random
from typing import DefaultDict
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import glob
import cv2
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample['image']=image
        sample['label']=label.long()
        #sample = {'image': image, 'label': label.long(),}
        return sample


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.case2tag={}
        for index,i in enumerate(os.listdir(base_dir+'/gt')):
            self.case2tag[i.split('.')[0]]=index
            
        #self.case2tag={'1': 0, '2': 1, '3': 2, '5': 3, '8': 4, '10': 5, '13': 6, '15': 7, '19': 8, '20': 9, '21': 10, '22': 11, '31': 12, '32': 13}
        self.data_list=[]
        random.seed(2022)
        self.data_list=glob.glob('{}/datas/*/*'.format(base_dir))
        self.data_list = random.sample(self.data_list,len(self.data_list))[:]
        
    def __len__(self):
        

        return len(self.data_list)

    def __getitem__(self, idx):
       
        slice_name = self.data_list[idx]
       
       
        image = np.load(slice_name)
        label_path = slice_name.replace('datas','gt')
        label = np.load(label_path)
       
        if self.split=='train':
            tag=self.case2tag[slice_name.split('/')[-2]]
            sample = {'image': image, 'label': label,'tag':tag}
        else:
            x, y = image.shape
            if x != 224 or y != 224:
                image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
                label = zoom(label, (224 / x, 224 / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] =slice_name.split('/')[-2]
        return sample

class Datafor_metalearning():
    def __init__(self, base_dir, list_dir, split, batchsize,transform=None):
       
        self.batchsize=batchsize
        Dataset=ACDC_dataset(base_dir, list_dir, split, transform)
        self.dataloader=get_generator(torch.utils.data.DataLoader(

            Dataset,

            batch_size=batchsize,

            shuffle=False,

            num_workers=4))

    def __getitem__(self, item):
        sample = next(self.dataloader)
        s_list=[]
        dic=DefaultDict(int)
        for i in range(self.batchsize):
            casename=sample['case_name'][i].split('_')[0]
            if dic[casename]==0:
                dic[casename]+=1
                s_list.append(casename)
       
        length=len(s_list)
        meta_train=s_list[:length*3//4]
        meta_test=s_list[length*3//4:]
        #print(meta_train,meta_test)
        imgs_train=[]
        masks_train=[]
        tags_train=[]
        imgs_test = []
        masks_test = []
        tags_test=[]
        
        for i in range(self.batchsize):
            img=sample['image'][i].unsqueeze(0)
            mask=sample['label'][i].unsqueeze(0)
            tag=sample['tag'][i].unsqueeze(0)
            casename=sample['case_name'][i].split('_')[0]
            if casename in meta_train:
                imgs_train.append(img)
                masks_train.append(mask)
                tags_train.append(tag)
            elif casename in meta_test:
                imgs_test.append(img)
                masks_test.append(mask)
                tags_test.append(tag)
        
        imgs_train=torch.tensor(torch.cat(imgs_train,dim=0),requires_grad=True)
     
        masks_train=torch.cat(masks_train,dim=0)
        tags_train=torch.cat(tags_train,dim=0)
        imgs_test = torch.cat(imgs_test, dim=0)
        masks_test = torch.cat(masks_test, dim=0)
        tags_test=torch.cat(tags_test,dim=0)
        return imgs_train,masks_train,tags_train,imgs_test,masks_test,tags_test


def get_generator(a):
    for sample in a:
        yield sample

if __name__ == '__main__':
    from torchvision import transforms
    root='/data/ssd1/typ/project/transformer/project_TransUNet/data/Synapse/train_npz'
    db_train = Datafor_metalearning(base_dir=root, list_dir='./lists/lists_Synapse', split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[224, 224])]))
    for imgs_train,masks_train,imgs_test,masks_test,tags_train,tags_test in db_train:
        print(imgs_test)
