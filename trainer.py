import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
from torch.nn.functional import alpha_dropout
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss,compactloss
from torchvision import transforms
from utils import calculate_metric_percase,test_single_volume
from collections import OrderedDict
def trainer_synapse(args, model,snapshot_path):
    from datasets.dataset_chaos  import Chaos_dataset, RandomGenerator,Datafor_metalearning
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    
    batch_size = args.batch_size 
    # max_iterations = args.max_iterations
    
    ##print("The length of train set is: {}".format(len(db_train)))
    db_test =Chaos_dataset(base_dir='/data/ssd1/typ/TransUNet/Train_Sets/dataset', list_dir='/data/ssd1/typ/TransUNet/Train_Sets/lists/', split="test")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    
    valloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    compact_loss=compactloss(num_classes)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    #optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * 16650#(2160.0/batch_size)  # max_epoch = max_iterations // len(trainloader) + 1
  
    best_performance = 0.0
    iterator = tqdm(range(300), ncols=70)
    lr_=base_lr
    beta=0.0065
    best_dice=0
    meta_lr=0.006
    lam=0
    best_hd=100
    for epoch_num in iterator:
        all_loss=0
        mean_dice=0
        mean_ce=0
        mean_loss_inner=0
        mean_loss_outer=0
        mean_cl=0
        all_com=0
        loss_compact=torch.tensor(0)
        loss_compact1=torch.tensor(0)
        db_train = Datafor_metalearning(base_dir=args.root_path, list_dir=args.list_dir, split="train",batchsize=batch_size,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        model.train()
      
        for i_batch, sampled_batch in enumerate(db_train):
            image_batch, label_batch ,tag_batch ,image_batch_test, label_batch_test ,tag_batch_test= sampled_batch[0], sampled_batch[1], sampled_batch[2],sampled_batch[3], sampled_batch[4], sampled_batch[5]
            image_batch, label_batch, tag_batch ,image_batch_test, label_batch_test ,tag_batch_test = image_batch.cuda(), label_batch.cuda(), tag_batch.cuda() ,image_batch_test.cuda(), label_batch_test.cuda() ,tag_batch_test.cuda()
            meta_train_outputs,c1 = model(image_batch)
         
           
            loss_classify1 = ce_loss(c1,tag_batch)
            loss_ce = ce_loss(meta_train_outputs, label_batch[:].long())
            loss_dice = dice_loss(meta_train_outputs, label_batch, softmax=True)
         
            loss_inner = 0.5 * loss_ce + 0.5 * loss_dice+0.01*loss_classify1#+lam*loss_compact
            
            grad = torch.autograd.grad(loss_inner, list(model.parameters()),retain_graph=True)
            fast_weights = OrderedDict(
                (name, param - torch.mul(meta_lr, grad)) for
                ((name, param), grad) in
                zip(model.named_parameters(), grad))
            
            meta_test_outputs,c2=model(image_batch_test,fast_weights)
            
            loss_classify2 = ce_loss(c2,tag_batch_test)
          
            loss_outer=0.5*ce_loss(meta_test_outputs, label_batch_test[:].long())+0.5*dice_loss(meta_test_outputs, label_batch_test, softmax=True)+0.01*loss_classify2#+lam*loss_compact1
            loss=loss_inner+beta*loss_outer
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            meta_lr = meta_lr*(1.0-iter_num/max_iterations)**0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            all_loss+=0#loss.item()
            mean_loss_inner+=loss_inner.item()
            mean_loss_outer+=0#loss_outer.item()
            #mean_cl+=loss_classify1.item()+loss_classify2.item()
            all_com+=loss_compact1.item()

        model.eval()
        metric_list = 0.0
    
        for i_batch1, val_batch in enumerate(valloader):
            
            image, label, case_name = val_batch["image"], val_batch["label"], val_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,ib=i_batch1, patch_size=[args.img_size, args.img_size],
                                        test_save_path=None, case=case_name, z_spacing=1)
            metric_list += np.array(metric_i)
            #logging.info('idx test case %s mean_dice %f mean_hd95 %f' % (i_batch1, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        performance = np.mean(metric_list, axis=0)[0]/len(db_test)
        mean_hd95 = np.mean(metric_list, axis=0)[1]/len(db_test)
        metric_list = metric_list / len(db_test)
    
        for i in range(1, args.num_classes):
            print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        
        print("val_dice:{},val_hd:{}".format(performance,mean_hd95))
        save_interval = 50  # int(max_epoch/6)


        if performance>best_dice:
            best_dice=performance
            best_hd = mean_hd95
            save_mode_path = os.path.join('modelchaos', 'beforefusebest_meta' + '.pth')
            torch.save(model.state_dict(), save_mode_path)
        logging.info('epoch %d : loss : %f,loss_inner: %f loss_outer: %f,loss_cliaaify: %f ,loss_compact: %f ,bestdice:%f ,besthd:%f' % (epoch_num, all_loss,mean_loss_inner, 0,0,all_com,best_dice,best_hd))
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join('modelchaos', 'beforefuseepoch_meta' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join('modelchaos', 'beforefuseepoch_meta' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
