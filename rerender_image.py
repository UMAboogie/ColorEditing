# rerender images with optimized area lights
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import math
import numpy as np
import Imath, array
import OpenEXR
import datetime
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import my_util_func
import math
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser


mi.set_variant('cuda_ad_rgb')

argparser = ArgumentParser()
argparser.add_argument('-n', '--light_num', type=int, default=6)
argparser.add_argument('-dir', '--dir_path', help='directory path of optimization results')
argparser.add_argument('-im', '--image_path', default='example1/livingroom.png')
argparser.add_argument('-pl', '--ply_path', default='example1/livingroom.ply')
argparser.add_argument('-fov', '--fov',type=float, default=50)
argparser.add_argument('-h', '--height',type=int, default=360)
argparser.add_argument('-w', '--width',type=int, default=640)


args = argparser.parse_args()

img = render_with_arealight(args.ply_path, args.dir_path, fov=args.fov, res=[args.height,args.width]):
 