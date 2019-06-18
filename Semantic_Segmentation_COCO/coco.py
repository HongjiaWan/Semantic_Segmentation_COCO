import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pycocotools.coco import COCO
from pycocotools import mask    
from tqdm import trange
from PIL import Image, ImageFile
from torchvision import transforms
import custom_transforms as tr
ImageFile.LOAD_TRUNCATED_IMAGES = True   # prevent PIL from reading images that have not been uploaded

class COCOSegmentation(Dataset):  #inherit from Dataset of torch
   
    
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    
    def __init__(self,
                 args,
                 base_dir='/home/wan/',
                 split='train',
                 year='2014'):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split,year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'MSCOCO{}/{}'.format(year,split))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask     #Interface for manipulating masks stored in RLE format.
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)       # load model from given path
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args   
    
    # By defining __getitem__ method, the instance object of class(example: p) can use p[index] to get value
    def __getitem__(self, index):    
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image':_img, 'label':_target} 
        
        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        
    

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    # decode masks from RLE, save them in ids_file
    def _preprocess(self, ids, ids_file):
        print('Preprocessing mask, it only run once for each split')
        time_bar = trange(len(ids))   # a progress bar
        new_ids = []
        for i in time_bar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))  #load annotations from given id
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
            # only use pictures that have segmentations more than 1000 piexls 
            if(mask > 0).sum() > 1000:
                new_ids.append(img_id)
            time_bar.set_description('Doing: {}/{}, got {} qualified images'.format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids


    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'],h,w)   # frPyObjects - Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.
            m = coco_mask.decode(rle)               #Decode binary masks encoded via RLE.
            cat = instance['category_id']             # number for category
            if cat in self.CAT_LIST:                    # only use CAT_LIST classes
                c = self.CAT_LIST.index(cat)      # return the index of cat
            else:
                continue

            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)      # every segmentation's category number add on mask
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),

            # to make the image in the same batch has same shape 
            tr.FixScaleCrop(crop_size=self.args.crop_size),

            #tr.RandomGaussianBlur(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    def __len__(self):
        return len(self.ids)

