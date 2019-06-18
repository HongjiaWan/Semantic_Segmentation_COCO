import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils import decode_seg_map_sequence



def visualize_image(dataset, image, target, output):
    grid_image = make_grid(image[:4].clone().cpu().data / 255, padding=4, normalize=False, range=(0, 255))
    plt.imshow(np.transpose(grid_image,(1,2,0)))
    #plt.imshow(grid_image)
    plt.show() 
    print("Predicted label")
    # torch.max(a, 1)[1]: a has 21 channels, so the function returns the index of the max num in each channel for the whole image
    grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:4], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
    plt.imshow(np.transpose(grid_image,(1,2,0)))
    plt.show() 
    print("Groundtruth label")
    grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:4], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
    plt.imshow(np.transpose(grid_image,(1,2,0)))
    plt.show() 
