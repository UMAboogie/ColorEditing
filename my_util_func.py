import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import math
import numpy as np
import Imath, array
import OpenEXR
import csv
import os
from PIL import Image
import torch
import torch.nn.functional as F

mi.set_variant('cuda_ad_rgb')


def exr2np(exr_path, height=480):
    img= OpenEXR.InputFile(exr_path)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str, g_str, b_str = img.channels('RGB', pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))

    return np.stack([red.reshape([height,-1]),green.reshape([height,-1]),blue.reshape([height,-1])], axis=-1)


def load_img(path, to_linear=False):
    img_np = np.array(Image.open(path)).astype(np.float32) / 255
    if to_linear:
        img_np = img_np**(2.2)
    return img_np


def render_with_arealight(mesh_path, dir_path, twosided=False, spp=1000, max_depth=8, fov=50, res=[360,640]):
    if twosided:
        mesh_loaded = mi.load_dict({
            'type': 'ply',
            'filename': mesh_path,
            'bsdf': {
                'type':'twosided',
                'material':{        
                    'type': 'principled',
                    'base_color': {
                        'type': 'mesh_attribute',
                        'name': 'vertex_albedo',
                    },
                    'metallic': {
                        'type': 'mesh_attribute',
                        'name': 'vertex_metallic',
                    },
                    'roughness': {
                        'type': 'mesh_attribute',
                        'name': 'vertex_roughness',
                    },
                }
            },
        })
    else:    
        mesh_loaded = mi.load_dict({
            'type': 'ply',
            'filename': mesh_path,
            'bsdf': {
                'type': 'principled',
                'base_color': {
                    'type': 'mesh_attribute',
                    'name': 'vertex_albedo',
                },
                'metallic': {
                    'type': 'mesh_attribute',
                    'name': 'vertex_metallic',
                },
                'roughness': {
                    'type': 'mesh_attribute',
                    'name': 'vertex_roughness',
                },
            },
        })
    

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path', 
            'max_depth': max_depth, 
            'hide_emitters':True
            },
        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'near_clip': 0.01,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
            ),
            'film_id': {
            'type': 'hdrfilm',
            'width': res[1],
            'height': res[0],
            'pixel_format': 'rgb',
            'component_format': 'float32',
            'filter': { 'type': 'tent' }
            },
        },
        'wholescene': mesh_loaded,
    }

    centers = np.load(dir_path+'/disk_centers.npy')
    light_num = centers.shape[0]
    eulers = np.load(dir_path+'/disk_eulers.npy')
    radiances = np.load(dir_path+'/disk_radiances.npy')
    scales = np.load(dir_path+'/disk_scales.npy')

    for i in range(light_num):
        name = 'light_'+str(i+1)
        scene_dict[name] = {
            'type': 'disk',
            'to_world': mi.ScalarTransform4f().translate(centers[i]).rotate(axis=[0,0,1], angle=eulers[i][2]).rotate(axis=[0,1,0], angle=eulers[i][1]).rotate(axis=[1,0,0], angle=eulers[i][0]).scale(float(scales[i])),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': radiances[i] 
                },
            },
        }
    scene = mi.load_dict(scene_dict)

    return  mi.render(scene, spp=spp)


def render_with_envmap(mesh_path, dir_path, twosided=False, spp=1000, max_depth=8, fov=50, res=[360,640]):
    if twosided:
        mesh_loaded = mi.load_dict({
            'type': 'ply',
            'filename': mesh_path,
            'bsdf': {
                'type':'twosided',
                'material':{        
                    'type': 'principled',
                    'base_color': {
                        'type': 'mesh_attribute',
                        'name': 'vertex_albedo',
                    },
                    'metallic': {
                        'type': 'mesh_attribute',
                        'name': 'vertex_metallic',
                    },
                    'roughness': {
                        'type': 'mesh_attribute',
                        'name': 'vertex_roughness',
                    },
                }
            },
        })
    else:    
        mesh_loaded = mi.load_dict({
            'type': 'ply',
            'filename': mesh_path,
            'bsdf': {
                'type': 'principled',
                'base_color': {
                    'type': 'mesh_attribute',
                    'name': 'vertex_albedo',
                },
                'metallic': {
                    'type': 'mesh_attribute',
                    'name': 'vertex_metallic',
                },
                'roughness': {
                    'type': 'mesh_attribute',
                    'name': 'vertex_roughness',
                },
            },
        })
    

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path', 
            'max_depth': max_depth, 
            'hide_emitters':True
            },
        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'near_clip': 0.01,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
            ),
            'film_id': {
            'type': 'hdrfilm',
            'width': res[1],
            'height': res[0],
            'pixel_format': 'rgb',
            'component_format': 'float32',
            'filter': { 'type': 'tent' }
            },
        },
        'wholescene': mesh_loaded,
    }

    centers = np.load(dir_path+'/disk_centers.npy')
    light_num = centers.shape[0]
    eulers = np.load(dir_path+'/disk_eulers.npy')
    radiances = np.load(dir_path+'/disk_radiances.npy')
    scales = np.load(dir_path+'/disk_scales.npy')

    scene_dict['light'] = {
        'type': 'envmap',
        'filename': dir_path+'/best_env.exr'
    }
    
    scene = mi.load_dict(scene_dict)

    return  mi.render(scene, spp=spp)



def change_albedo(albedo_filename, mask_filename, new_color=[0,0,0], height=360):
    if len(albedo_filename) >= 4 and albedo_filename[-3:] == 'png':
        albedo_np = np.array(Image.open(albedo_filename))
        if albedo_np.shape[2] == 4:
            albedo_np = albedo_np[:,:,:3]
    elif albedo_filename[-3:] == 'npy':
        albedo_np = np.load(albedo_filename)
        if albedo_np.ndim == 4:
            #cv2 format
            albedo_np = albedo_np.squeeze().transpose([1,2,0])
    elif albedo_filename[-3:] == 'exr':
        albedo_np = exr2np(albedo_filename, height=height)
    if not albedo_np.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint32]:
        albedo_np = (albedo_np*255).astype(np.uint8)
    
    if type(mask_filename) is list:
        mask_filenames = mask_filename
        new_colors = new_color
        if not type(new_color[0]) is list or len(mask_filename) != len(new_color):
            print('error: mask number and color number are not same.')
            return
    else:
        mask_filenames = [mask_filename]
        new_colors = [new_color]
    
    for i, mp in enumerate(mask_filenames):  
        mask_np = np.array(Image.open(mp).convert('L')).astype(np.float32)/255
        h_albedo = albedo_np.shape[0]
        h_mask = mask_np.shape[0]
        nc = new_colors[i]
        if h_albedo == h_mask:
            # just change the color
            mask_np = mask_np[:,:,np.newaxis]
            new_albedo_np = mask_np*np.array([*nc]*(h_albedo)*(albedo_np.shape[1])).reshape(h_albedo,albedo_np.shape[1],3) + (1-mask_np)*(albedo_np)
            new_albedo_np = new_albedo_np.astype(np.uint8) 
        else:
            # down/up sample the size of mask
            mask_torch = torch.from_numpy((mask_np[:,:,np.newaxis,np.newaxis]).transpose(2,3,0,1))
            mask_np = F.adaptive_avg_pool2d(mask_torch, (h_albedo, albedo_np.shape[1])).numpy().squeeze(0).transpose(1,2,0)
            new_albedo_np = mask_np*np.array([*nc]*(h_albedo)*(albedo_np.shape[1])).reshape(h_albedo,albedo_np.shape[1],3) + (1-mask_np)*(albedo_np)
            new_albedo_np = new_albedo_np.astype(np.uint8)
        albedo_np = new_albedo_np
    return new_albedo_np


def depth_from_plane(depth_2d, fov):
    h = depth_2d.shape[0]
    w = depth_2d.shape[1]
    refined_depth = np.zeros([h,w])
    c_x = w/2
    c_y = h/2
    for i in range(h):
        for j in range(w):
            tan_theta = (((i-c_y)**2+(j-c_x)**2)**0.5)*math.tan(math.radians(fov/2))/(w/2)
            refined_depth[i,j] = depth_2d[i,j] / np.sqrt(1+(tan_theta)**2)
    return refined_depth
