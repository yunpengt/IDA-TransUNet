import os
import random
from typing import DefaultDict
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


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


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
       
        self.sample_list = open(os.path.join(list_dir, self.split)).readlines()
        self.data_dir = base_dir
        self.case2tag={'case0030': 0, 'case0031': 1, 'case0005': 2, 'case0010': 3, 'case0033': 4, 'case0034': 5, 'case0007': 6, 'case0024': 7, 'case0040': 8, 'case0027': 17, 'case0006': 9, 'case0039': 10, 'case0023': 11, 'case0037': 12, 'case0026': 13, 'case0021': 14, 'case0028': 15, 'case0009': 16}
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        slice_name = self.sample_list[idx].strip('\n')
        tag=self.case2tag[slice_name.split('_')[0]]
        data_path = os.path.join(self.data_dir, slice_name)
        try:
            data = np.load(data_path)
        except:
            print(1)
        image, label = data['image'], data['label']


        sample = {'image': image, 'label': label,'tag':tag}
   
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class Datafor_metalearning():
    def __init__(self, base_dir, list_dir,  batchsize,transform=None):
       
        self.batchsize=batchsize
        self.dataloader={}
        for i in os.listdir(list_dir):
            Dataset=Synapse_dataset(base_dir, list_dir, i, transform)
            dataloader=get_generator(torch.utils.data.DataLoader(

                Dataset,

                batch_size=batchsize,

                shuffle=True,

                num_workers=4))
            self.dataloader[i]=dataloader

    def __getitem__(self, item):
        sample_list=[]
        #keys=list(self.dataloader.keys())
        keys=list(self.dataloader.keys())
        keys=list(self.dataloader.keys())+random.sample(keys,len(keys))[0:6]
        for i in keys:
            sample=next(self.dataloader[i])
            sample_list.append(sample)
        
     
       
        #sample_list = random.sample(sample_list,len(sample_list))[0:12]
        imgs_train=[]
        masks_train=[]
        tags_train=[]
        imgs_test = []
        masks_test = []
        tags_test=[]
        for i in sample_list[:len(sample_list)*3//4]:
            img,mask,tag = i['image'], i['label'], i['tag']
            imgs_train.append(img)
            masks_train.append(mask)
            tags_train.append(tag)

        for i in sample_list[len(sample_list)*3//4:]:
            img,mask,tag = i['image'], i['label'], i['tag']
            imgs_test.append(img)
            masks_test.append(mask)
            tags_test.append(tag)

        imgs_train=torch.cat(imgs_train,dim=0)
     
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
