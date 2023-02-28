import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import torch.nn.functional as F
class compactloss(nn.Module):
    def __init__(self,n_classes):
        super(compactloss,self).__init__()
        self.n_classes=n_classes
    def _get_compactness_cost(self,y_pred): 
     
     

        x = y_pred[:,1:,:] - y_pred[:,:-1,:] # horizontal and vertical directions 
        y = y_pred[:,:,1:] - y_pred[:,:,:-1]

        delta_x = x[:,:,1:]**2
        delta_y = y[:,1:,:]**2

        delta_u = torch.abs(delta_x + delta_y) 

        epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
        w = 0.01
        length = w * torch.sum(torch.sqrt(delta_u + epsilon), [1, 2])

        area = torch.sum(torch.abs(y_pred), [1,2])

        #compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926+epsilon))

        return length,area
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    def forward(self,y_pred,y_tar):
        total=0
        y_pred = torch.softmax(y_pred, dim=1)
        y_tar = self._one_hot_encoder(y_tar)
        for i in range(self.n_classes):
            yc_tar_l,yc_tar_a=self._get_compactness_cost(y_tar[:,i,:,:])
            yc_pred_l,yc_pred_a=self._get_compactness_cost(y_pred[:,i,:,:])
            total+=F.l1_loss(yc_pred_l,yc_tar_l)+F.l1_loss(yc_pred_a,yc_tar_a)
        return total
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
from scipy import ndimage
def _connectivity_region_analysis(mask):
    s = [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    label_im, nb_labels = ndimage.label(mask)  # , structure=s)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im
def test_single_volume(image, label, net, classes,ib, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    #image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image, label = image.squeeze(0).cpu().detach().numpy(), label.cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        G=torch.zeros([1,1,224,224]).cuda()
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            
            #outputs,c,fuse_attention= net(input)
            '''
            B, n_patch, hidden = fuse_attention.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            f = fuse_attention.permute(0, 2, 1)
            f = f.contiguous().view(B, hidden, h, w)
            '''
            #FF=torch.sum(fuse_attention[:,:,7,7])/(512)
            #G_xy=F.relu(torch.autograd.grad(FF,input)[0])
            #G=torch.add(G,G_xy)
          
            with torch.no_grad():
                outputs,c = net(input)
                '''
                out = torch.softmax(outputs, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
              
                out = out[1,:,:]
                out = (out > 0.5).astype(float)
                out = _connectivity_region_analysis(out)
                '''
                
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                lab=label[ind,:,:]
                lab=lab.astype(np.uint8)
                
                
                metric_l = []
                for ii in range(1, 2):
                    metric_l.append(calculate_metric_percase(pred == ii, lab == ii))
                dice=np.mean(metric_l, axis=0)[0]
                l1=lab.copy()
                l2=lab.copy()
                l3=lab.copy()
                l4=lab.copy()
                l5=lab.copy()
                l6=lab.copy()
                l7=lab.copy()
                l8=lab.copy()
                l1[l1!=1]=0
                l2[l2!=2]=0
                l3[l3!=3]=0
                l4[l4!=4]=0
                l5[l5!=5]=0
                l6[l6!=6]=0
                l7[l7!=7]=0
                l8[l8!=8]=0
               
                slice = image[ind, :, :]
                
                slice=cv2.cvtColor(slice,cv2.COLOR_GRAY2BGR)
                slice=slice.astype(np.uint8)
                gt = slice.copy()
                pred=pred.astype(np.uint8)
                m1=pred.copy()
                m2=pred.copy()
                m3=pred.copy()
                m4=pred.copy()
                m5=pred.copy()
                m6=pred.copy()
                m7=pred.copy()
                m8=pred.copy()
                m1[m1!=1]=0
                m2[m2!=2]=0
                m3[m3!=3]=0
                m4[m4!=4]=0
                m5[m5!=5]=0
                m6[m6!=6]=0
                m7[m7!=7]=0
                m8[m8!=8]=0
                
                slice[..., 2] = np.where(m1 == 1, 0, slice[..., 2])
                slice[..., 1] = np.where(m1 == 1, 0, slice[..., 1])
                slice[..., 0] = np.where(m1 == 1, 255, slice[..., 0])
               

        
                slice[..., 2] = np.where(m2 == 2, 0, slice[..., 2])
                slice[..., 1] = np.where(m2 == 2, 255, slice[..., 1])
                slice[..., 0] = np.where(m2 == 2, 0, slice[..., 0])
              

                slice[..., 2] = np.where(m3 == 3, 255, slice[..., 2])
                slice[..., 1] = np.where(m3 == 3, 0, slice[..., 1])
                slice[..., 0] = np.where(m3 == 3, 0, slice[..., 0])
                
                slice[..., 2] = np.where(m4 == 4, 255, slice[..., 2])
                slice[..., 1] = np.where(m4 == 4, 0, slice[..., 1])
                slice[..., 0] = np.where(m4 == 4, 255, slice[..., 0])
         
              
                slice[..., 2] = np.where(m5 == 5, 0, slice[..., 2])
                slice[..., 1] = np.where(m5 == 5, 255, slice[..., 1])
                slice[..., 0] = np.where(m5 == 5, 255, slice[..., 0])
               
                slice[..., 2] = np.where(m6 == 6, 255, slice[..., 2])
                slice[..., 1] = np.where(m6 == 6, 255, slice[..., 1])
                slice[..., 0] = np.where(m6 == 6, 0, slice[..., 0])
               
                slice[..., 2] = np.where(m7 == 7, 100, slice[..., 2])
                slice[..., 1] = np.where(m7 == 7, 100, slice[..., 1])
                slice[..., 0] = np.where(m7 == 7, 255, slice[..., 0])
               
               
                slice[..., 2] = np.where(m8 == 8, 255, slice[..., 2])
                slice[..., 1] = np.where(m8 == 8, 100, slice[..., 1])
                slice[..., 0] = np.where(m8 == 8, 100, slice[..., 0])
              
                name='acdcvision/{}'.format(case)+'_'+str(ib)+'_'+str(dice)+'.jpg'
                cv2.imwrite(name,slice)
                '''
                gt[..., 2] = np.where(l1 == 1, 0, gt[..., 2])
                gt[..., 1] = np.where(l1 == 1, 0, gt[..., 1])
                gt[..., 0] = np.where(l1 == 1, 255, gt[..., 0])
               

        
                gt[..., 2] = np.where(l2 == 2, 0, gt[..., 2])
                gt[..., 1] = np.where(l2 == 2, 255, gt[..., 1])
                gt[..., 0] = np.where(l2 == 2, 0, gt[..., 0])
              

                gt[..., 2] = np.where(l3 == 3, 255, gt[..., 2])
                gt[..., 1] = np.where(l3 == 3, 0, gt[..., 1])
                gt[..., 0] = np.where(l3 == 3, 0, gt[..., 0])
                
                gt[..., 2] = np.where(l4 == 4, 255, gt[..., 2])
                gt[..., 1] = np.where(l4 == 4, 0, gt[..., 1])
                gt[..., 0] = np.where(l4 == 4, 255, gt[..., 0])
         
              
                gt[..., 2] = np.where(l5 == 5, 0, gt[..., 2])
                gt[..., 1] = np.where(l5 == 5, 255, gt[..., 1])
                gt[..., 0] = np.where(l5 == 5, 255, gt[..., 0])
               
                gt[..., 2] = np.where(l6 == 6, 255, gt[..., 2])
                gt[..., 1] = np.where(l6 == 6, 255, gt[..., 1])
                gt[..., 0] = np.where(l6 == 6, 0, gt[..., 0])
               
                gt[..., 2] = np.where(l7 == 7, 100, gt[..., 2])
                gt[..., 1] = np.where(l7 == 7, 100, gt[..., 1])
                gt[..., 0] = np.where(l7 == 7, 255, gt[..., 0])
               
               
                gt[..., 2] = np.where(l8 == 8, 255, gt[..., 2])
                gt[..., 1] = np.where(l8 == 8, 100, gt[..., 1])
                gt[..., 0] = np.where(l8 == 8, 100, gt[..., 0])
                name='groundtrue/{}'.format(case)+'_'+str(ind)+'.jpg'
                cv2.imwrite(name,gt)
                '''
                

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    #G=G/len(image)
    return metric_list