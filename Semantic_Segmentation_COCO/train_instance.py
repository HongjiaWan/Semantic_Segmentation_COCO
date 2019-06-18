import numpy as np
import torch
import os
from tqdm import tqdm 
import argparse     # The argparse module makes it easy to write user-friendly command-line interfaces.
import matplotlib.pyplot as plt

from utils import decode_seg_map_sequence
from loss import SegmentationLoss
from data_loader import make_data_loader
from deeplab import InstanceBranch


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args)

        # define network
        model = InstanceBranch()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.criterion = SegmentationLoss(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.trainloss_history = []
        self.valloss_history = []

        self.train_plot = []
        self.val_plot = []

        if args.cuda:
            self.model = self.model.cuda()

    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        #tbar = tqdm(self.train_loader)    # Instantly make your loops show a smart progress meter

        for i, sample in enumerate(self.train_loader):
            if i >= 200:
                break
            else:
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
            
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.trainloss_history.append(loss.data.cpu().numpy())

        
        last_loss = self.trainloss_history[-i:]
        train_loss = np.mean(last_loss) 
        self.train_plot.append(train_loss)
        
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size ))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        #tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        for i, sample in enumerate(self.val_loader):
            if i >= 30:
                break
            else:
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)

                loss = self.criterion(output, target)
                self.valloss_history.append(loss.data.cpu().numpy())

                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
           

        last_loss = self.valloss_history[-i:]
        val_loss = np.mean(last_loss)
        self.val_plot.append(val_loss)

        print('Validation: ')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size ))


def main():
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    # define all the parameters
    args.cuda = torch.cuda.is_available()
    args.epochs = 30
    args.batch_size = 2
    args.lr = 0.01
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.loss_type = 'ce'
    args.out_stride = 16
    args.base_size = 500
    args.crop_size = 300
    print(args)

    trainer = Trainer(args)
    for epoch in range(0, trainer.args.epochs):
        if epoch > 10:
            args.lr = 0.001
        if epoch > 20:
            args.lr = 0.0001

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


if __name__ == '__main__':
    main()
    