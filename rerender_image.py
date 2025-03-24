# rerender images with optimized area lights
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import math
import numpy as np
import Imath, array
import OpenEXR
import os
import my_util_func
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import my_util_func
import math
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser


mi.set_variant('cuda_ad_rgb')

argparser = ArgumentParser()
argparser.add_argument('-res', '--result_path', help='directory path for optimization results')
argparser.add_argument('-im', '--image_path', default=None)
argparser.add_argument('-me_1', '--mesh_path_1', default='example1/livingroom.ply')
argparser.add_argument('-me_2', '--mesh_path_2', help='albedo changed mesh file path', default=None)
argparser.add_argument('-fov', '--fov',type=float, default=50)
argparser.add_argument('-h', '--height',type=int, default=360)
argparser.add_argument('-w', '--width',type=int, default=640)


args = argparser.parse_args()

img1 = my_util_func.render_with_arealight(args.mesh_path_1, args.result_path, fov=args.fov, res=[args.height,args.width])
mi.util.write_bitmap(args.result_path+'/reconstructed.png', img1)

if args.image_path is not None and args.mesh_path_2 is not None:
    if args.image_path[-3:] == 'png' or args.image_path[-3:] == 'jpg' or args.image_path[-4:] == 'jpeg': 
        img_input = (np.array(Image.open(args.image_path)).astype(np.float32)/255)**(2.2)
    elif args.image_path[-3:] == 'exr':
        img_input = my_util_func.exr2np(args.image_path, args.height)
    else:
        print('invalid --image_path')
        exit()
    img2 = my_util_func.render_with_arealight(args.mesh_path_2, args.result_path, fov=args.fov, res=[args.height,args.width])
    mi.util.write_bitmap(args.result_path+'/edited.png', img2)
    img_diff = (((np.clip(np.abs(img2.numpy() - img1.numpy()),0,1))**(1/2.2))*255).astype(np.uint8)
    Image.fromarray(img_diff).save(args.result_path+'/diff.png')
    img_final = (((np.clip(img_input + (img2.numpy() - img1.numpy()),0,1))**(1/2.2))*255).astype(np.uint8)
    Image.fromarray(img_final).save(args.result_path+'/finalimage.png')


 