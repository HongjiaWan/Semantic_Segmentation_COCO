from torch.utils.data import DataLoader
import coco
import cocoInstance

def make_data_loader(args, **kwargs):
    train_set = coco.COCOSegmentation(args, split='train')
    val_set = coco.COCOSegmentation(args, split='val')
    num_class = 21

    # Dataloader: every time return a set of data whose length is batch_size
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader=None

    return train_loader, val_loader, test_loader, num_class

def make_data_loader_instance(args, **kwargs):
    train_set = cocoInstance.COCOInstanceSeg(args, split='train')
    val_set = cocoInstance.COCOInstanceSeg(args, split='val')
    num_class = train_set.NUM_CLASSES

    # Dataloader: every time return a set of data whose length is batch_size
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader=None

    return train_loader, val_loader, test_loader, num_class
