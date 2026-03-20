# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import  calMetric_iou
import numpy as np
import random
import torchvision

from sklearn.model_selection import train_test_split
from return_models import return_models
import itertools
from loss.losses import cross_entropy
from argparse import ArgumentParser


import argparse
from dataset_from_index import DA_DatasetFromFolder

from predict import test_model
parser = argparse.ArgumentParser(description='Training Change Detection Network')

# training parameters   
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=4, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=6, type=int, help='num of workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')


parser.add_argument('--dataset_name', required=False,default='WSISEG-Database-master', type=str, help='model save path')

#parser.add_argument('--dataset_type', required=False,default='test', type=str, help='model save path')
parser.add_argument('--model_name', required=False,default='SwinUnet', type=str, help='model save path')
parser.add_argument('--para', required=False, default=320, type=int, help='model save path')
#UNext_S_official
args = parser.parse_args()
#L8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCS
parser.add_argument('--hr1_train', default='/root/autodl-tmp/cloud_detection_datasets/data0/cloud_mask/'+args.dataset_name+'/input', type=str, help='image at t1 in training set')
parser.add_argument('--lab_train', default='/root/autodl-tmp/cloud_detection_datasets/data0/cloud_mask/'+args.dataset_name+'/output', type=str, help='label image in training set')


parser.add_argument('--hr1_val', default='/root/autodl-tmp/cloud_detection_datasets/data0/cloud_mask/'+args.dataset_name+'/val_input', type=str, help='image at t1 in validation set')
parser.add_argument('--lab_val', default='/root/autodl-tmp/cloud_detection_datasets/data0/cloud_mask/'+args.dataset_name+'/val_output', type=str, help='label image in validation set')
parser.add_argument('--model_dir', default='./epochs/'+args.dataset_name+'/'+args.model_name+'/', type=str)

#para = [320,160,80,40,160,160]


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# 因为前面已经隔离了显卡，所以这里永远只用逻辑上的 0 号卡
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

# set seeds
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)

if __name__ == '__main__':
    mloss = 0

    # load data
    train_set = DA_DatasetFromFolder(dataset_name=args.dataset_name, dataset_type='train')

    val_set = DA_DatasetFromFolder(dataset_name=args.dataset_name, dataset_type='val')
    print("train_set:len, val_set:len",len(train_set),len(val_set))
    
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)
    #val_loader = train_loader

    # define model
    #CDNet = UNet(3,2)
    print(args.para,"----")
    CDNet = return_models(args.model_name)
    CDNet = CDNet.to(device, dtype=torch.float)
    
    if torch.cuda.device_count() > 2:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))

    # set optimization
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr= args.lr, betas=(0.9, 0.999))
    
    #
    #CDcriterionCD = torchvision.ops.sigmoid_focal_loss#.to(device, dtype=torch.float)
    #
    CDcriterionCD = cross_entropy().to(device, dtype=torch.float)
    
    loss_list_all = []

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'edge_loss':0, 'CD_loss':0, 'loss': 0 }

        CDNet.train()
        for hr_img1, label,_ in train_bar:
            running_results['batch_sizes'] += args.batchsize

            hr_img1 = hr_img1.to(device, dtype=torch.float)

            label = label.to(device, dtype=torch.float)

            # result1,result2,mutual = CDNet(hr_img1, hr_img2)
            if args.model_name == "MCDNet":
                result1 = CDNet(hr_img1,hr_img1)
            else:
                result1 = CDNet(hr_img1)#[0]
            

            label = torch.argmax(label, 1).unsqueeze(1).float()
            #print(label.shape,"label")

            #loss = CDcriterionCD(result1, label,reduction="mean")  # + args.mutual_para*(mutual128+mutual64+mutual32)
            
            loss = CDcriterionCD(result1, label)
            
            
            CDNet.zero_grad()
            loss.backward()
            optimizer.step()

            running_results['CD_loss'] += loss.item() * args.batchsize

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, args.num_epochs,
                    running_results['CD_loss'] / running_results['batch_sizes'],
                    ))

        # eval
        CDNet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            valing_results = {'batch_sizes': 0, 'IoU': 0}

            for hr_img1, label,_ in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                hr_img1 = hr_img1.to(device, dtype=torch.float)

                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()

                #cd_map = CDNet(hr_img1)#[0]
                if args.model_name == "MCDNet":
                    cd_map = CDNet(hr_img1,hr_img1)
                else:
                    cd_map = CDNet(hr_img1)#[0]

                cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()
                
                gt_value = (label > 0).float()
                prob = (cd_map > 0).float()
                prob = prob.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                intr, unn = calMetric_iou(gt_value, result)
                #print(result.shape,"res")
                inter = inter + intr
                unin = unin + unn

                valing_results['IoU'] = (inter * 1.0 / unin)

                val_bar.set_description(
                    desc='IoU: %.4f' % (  valing_results['IoU'],))

        # save model parameters 
        val_loss = valing_results['IoU']
        loss_list_all.append(val_loss)
        print("loss_list_all",loss_list_all)

        if val_loss > mloss or epoch==1:
            mloss = val_loss
            torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_best.pth')
    #do testing 
    if not os.path.exists("./csv/"):
        os.makedirs("./csv/")
    
    test_results = test_model(args.model_name, args.dataset_name, args.model_dir+'netCD_epoch_best.pth',\
     gpu_id=args.gpu_id,\
     save_cd='./result/{}/{}/'.format(args.model_name,args.dataset_name),\
     csv_dir="./csv/")

