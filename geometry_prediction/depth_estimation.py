# metric depth estimation by Depth Anything V2 for color editing task
# run the command 'python depth_estimation.py --dir_path xxx' 

import cv2
import torch
import os
import datetime
import numpy as np
from Depth-Anything-V2.depth_anything_v2.dpt import DepthAnythingV2
from Depth-Anything-V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2_metric
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('-dir', '--dir_path', type=str, default=None)

args = argparser.parse_args()

paths = []
with open(args.dir_path + '/image_paths.txt') as f:
    for line in f:
       paths.append(line[:-1])

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#DEVICE = 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


# metric depth
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model_metric = DepthAnythingV2_metric(**{**model_configs[encoder], 'max_depth': max_depth})
model_metric.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model_metric = model_metric.to(DEVICE).eval()

for i, path in enumerate(paths):
    raw_img = cv2.imread(path)
    print(path)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    depth_metric = model_metric.infer_image(raw_img)
    depth = 1 - (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    d_max = np.max(depth_metric)
    d_min = np.min(depth_metric)
    depth = ((d_min)+depth*(d_max - d_min))/d_max
    np.save(args.dir_path+'/depth_'+str(i)+'.npy', depth)
    print('save No.'+str(i)+' depth npy')
