import numpy as np
import torch
import os
from tqdm import tqdm 
import argparse     # The argparse module makes it easy to write user-friendly command-line interfaces.
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from utils import decode_seg_map_sequence
from utils import decode_segmap
from loss import SegmentationLoss
from metrics import Evaluator
from data_loader import make_data_loader
from enet import ENet
from visualize import visualize_image
from lr_scheduler import LR_Scheduler


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args)

        # define network
        model = ENet(num_classes=self.nclass)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

        self.criterion = SegmentationLoss(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.evaluator = Evaluator(self.nclass)
        self.best_pred = 0.0

        self.writer = SummaryWriter('/home/wan/Segmentation/tensorboard_1000images')

        self.trainloss_history = []
        self.valloss_history = []

        self.train_plot = []
        self.val_plot = []
        # every 10 epochs the lr will multiply 0.1
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.base_lr, args.epochs, len(self.train_loader), lr_step=20)

        if args.cuda:
            self.model = self.model.cuda()

    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        #tbar = tqdm(self.train_loader)    # Instantly make your loops show a smart progress meter

        for i, sample in enumerate(self.train_loader):
            
            if i >= 1250:
                break
            else:
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.long().cuda()
                
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.optimizer.zero_grad()
                output = self.model(image)
                
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.trainloss_history.append(loss.data.cpu().numpy())
                

                if epoch == 3 and i == 0:
                    print("after 3 epochs, the result of training data:")
                    visualize_image("coco", image, target, output)

                if epoch == 9 and i == 0:
                    print("after 10 epochs, the result of training data:")
                    visualize_image("coco", image, target, output)
                  
                if epoch == 29 and i == 0:
                    print("after 20 epochs, the result of training data:")
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 1:
             
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 2:
                  
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 3:
                  
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 4:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 1:
                    print("after 30 epochs, the result of training data:")
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 3:
             
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 5:
                  
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 7:
                  
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 9:
                   
                    visualize_image("coco", image, target, output)
                
                pred = output.data.cpu().numpy()
                
                pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(target.cpu().numpy(), pred)

        acc = self.evaluator.Pixel_Accuracy()
        #acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        
        last_loss = self.trainloss_history[-i:]
        train_loss = np.mean(last_loss) 
        self.train_plot.append(train_loss)
        
        self.writer.add_scalar('train/epoch', train_loss, epoch)
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size ))
        print('Loss: %.4f' % train_loss)
        print("Train_Acc:{}".format(acc), "Train_mIoU:{}".format(mIoU))

        if epoch == 20:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_20')
        if epoch == 30:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_30')
        if epoch == 40:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_40')
        if epoch == 50:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_50')
        if epoch == 60:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_60')
        if epoch == 80:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_80')
        if epoch == 99:
            torch.save(self.model.state_dict(), '/home/wan/Segmentation/model_100')

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        #tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        for i, sample in enumerate(self.val_loader):
            
            if i >= 150:
                break
            else:
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.long().cuda()
                with torch.no_grad():
                    output = self.model(image)

                loss = self.criterion(output, target)
                self.valloss_history.append(loss.data.cpu().numpy())

                pred = output.data.cpu().numpy()
                
                pred = np.argmax(pred, axis=1)
                # visualize the prediction and target
                if epoch == 3 and i == 0:
                    print("after 3 epochs, the result of val data:")
                    visualize_image("coco", image, target, output)
                if epoch ==9 and i == 0:
                    print("after 10 epochs, the result of val data:")
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 0:
                    print("after 20 epochs, the result of val data:")
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 1:
                    
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 2:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 3:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 29 and i == 4:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 0:
                    print("after 30 epochs, the result of val data:")
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 1:
                    
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 2:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 3:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 49 and i == 4:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 69 and i == 0:
                    print("after 30 epochs, the result of val data:")
                    visualize_image("coco", image, target, output)
                if epoch == 69 and i == 1:
                    
                    visualize_image("coco", image, target, output)
                if epoch == 69 and i == 2:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 69 and i == 3:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 69 and i == 4:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 99 and i == 0:
                    print("after 30 epochs, the result of val data:")
                    visualize_image("coco", image, target, output)
                if epoch == 99 and i == 1:
                    
                    visualize_image("coco", image, target, output)
                if epoch == 99 and i == 2:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 99 and i == 3:
                   
                    visualize_image("coco", image, target, output)
                if epoch == 99 and i == 4:
                   
                    visualize_image("coco", image, target, output)

                self.evaluator.add_batch(target.cpu().numpy(), pred)
            
        acc = self.evaluator.Pixel_Accuracy()
        #acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()

        last_loss = self.valloss_history[-i:]
        val_loss = np.mean(last_loss)
        self.val_plot.append(val_loss)

        self.writer.add_scalar('val/epoch', val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        print('Validation: ')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size ))
        print("Acc:{}".format(acc), "mIoU:{}".format(mIoU))
        print('Loss: %.4f' % val_loss)
     



def main():
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    # define all the parameters
    args.cuda = torch.cuda.is_available()
    args.epochs = 200
    args.batch_size = 8
    
    args.lr_scheduler = 'step'
    args.base_lr = 0.001

    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.loss_type = 'focal'
    args.out_stride = 16
    args.base_size = 500
    args.crop_size = 300
    print(args)

    trainer = Trainer(args)
    for epoch in range(0, trainer.args.epochs):
        
        trainer.training(epoch)
        
        trainer.validation(epoch)
    
    plt.subplot(2,1,2)
    plt.title('Training and Validation loss')
    plt.plot(trainer.train_plot, '-o', label='train')
    plt.plot(trainer.val_plot, '-o', label='val')
    plt.legend(['train','val'], loc='upper right')
    plt.xlabel('Epoch')
    plt.gcf().set_size_inches(15, 12)
    plt.show()

    print('best pred(mIoU): ', trainer.best_pred)
        
# lr, loss-function, 

if __name__ == '__main__':
    main()
    




            




