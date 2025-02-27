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

def build_mesh_default(z_depth_pos, name, albedo_1d_array, material_array, res=[256,256], fov=90, write_ply=True, threshold=0.01, delete_square=False):
    # mesh : H * W * 3

    if np.max(z_depth_pos) > 1:
        z_depth_pos /= np.max(z_depth_pos)

    if albedo_1d_array.ndim != 1:
        albedo_1d_array = albedo_1d_array.reshape([-1])

    y_dim = res[0]; x_dim = res[1] 
    N = y_dim*x_dim

    y,x = np.mgrid[(y_dim-1)/(x_dim-1):-(y_dim-1)/(x_dim-1):y_dim*1j, -1:1:x_dim*1j]
    y = y.reshape([-1])
    x = x.reshape([-1])
    z = (-1*z_depth_pos).reshape([-1])

    y_modified = y*np.abs(z)*math.tan(math.radians(fov/2))
    x_modified = x*np.abs(z)*math.tan(math.radians(fov/2))

    vertex_pos = mi.Point3f(x_modified, y_modified, z)

    # face list : 
    # q--p           r
    # | /  x res,  / | x res
    # r           p--q

    p_tmp_1 = np.arange(x_dim*(y_dim-1))
    p_1 = p_tmp_1[p_tmp_1 % x_dim != (x_dim-1)]

    p = np.concatenate([p_1+1, p_1+x_dim])
    q = np.concatenate([p_1, p_1+x_dim+1])
    r = np.concatenate([p_1+x_dim, p_1+1])

    delete_idx = []
    
    if not delete_square:
        for i in range(2*(x_dim-1)*(y_dim-1)):
            dist1 = ((x_modified[p[i]] - x_modified[q[i]])**2 + (y_modified[p[i]] - y_modified[q[i]])**2 + (z[p[i]] - z[q[i]])**2)**0.5
            dist2 = ((x_modified[q[i]] - x_modified[r[i]])**2 + (y_modified[q[i]] - y_modified[r[i]])**2 + (z[q[i]] - z[r[i]])**2)**0.5
        
            #dist = np.max([z[p[i]], z[q[i]], z[r[i]]]) - np.min([z[p[i]], z[q[i]], z[r[i]]])
            dist = max(dist1,dist2)
            if dist > threshold: #delete mesh
                delete_idx.append(i)
    else:
        for i in range((x_dim-1)*(y_dim-1)):
            dist1 = ((x_modified[p[i]] - x_modified[q[i]])**2 + (y_modified[p[i]] - y_modified[q[i]])**2 + (z[p[i]] - z[q[i]])**2)**0.5
            dist2 = ((x_modified[q[i]] - x_modified[r[i]])**2 + (y_modified[q[i]] - y_modified[r[i]])**2 + (z[q[i]] - z[r[i]])**2)**0.5
            dist3 = ((x_modified[p[i+(x_dim-1)*(y_dim-1)]] - x_modified[q[i+(x_dim-1)*(y_dim-1)]])**2 + (y_modified[p[i+(x_dim-1)*(y_dim-1)]] - y_modified[q[i+(x_dim-1)*(y_dim-1)]])**2 + (z[p[i+(x_dim-1)*(y_dim-1)]] - z[q[i+(x_dim-1)*(y_dim-1)]])**2)**0.5
            dist4 = ((x_modified[q[i+(x_dim-1)*(y_dim-1)]] - x_modified[r[i+(x_dim-1)*(y_dim-1)]])**2 + (y_modified[q[i+(x_dim-1)*(y_dim-1)]] - y_modified[r[i+(x_dim-1)*(y_dim-1)]])**2 + (z[q[i+(x_dim-1)*(y_dim-1)]] - z[r[i+(x_dim-1)*(y_dim-1)]])**2)**0.5

            dist = np.max([dist1,dist2,dist3,dist4])
            if dist > threshold: #delete mesh
                # p[i] q[i] r[i] 
                delete_idx.append(i)
                delete_idx.append(i+(x_dim-1)*(y_dim-1))
    p = np.delete(p,delete_idx)
    q = np.delete(q,delete_idx)
    r = np.delete(r,delete_idx)

    face_indices = mi.Vector3u(p, q, r)
    mesh = mi.Mesh(
        name,
        vertex_count=N,
        face_count=2*(x_dim-1)*(y_dim-1)-len(delete_idx),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertex_pos)
    mesh_params['faces'] = dr.ravel(face_indices)
    if normal_array is not None:
        mesh_params['vertex_normals'] = dr.ravel(normal_array.reshape(-1))

    mesh.add_attribute('vertex_albedo', 3, albedo_1d_array)
    mesh.add_attribute('vertex_roughness', 1, material_array[:,:,0].reshape(-1))
    mesh.add_attribute('vertex_metallic', 1, material_array[:,:,1].reshape(-1))
    mesh_params.update()
    if write_ply:
        mesh.write_ply(name+'.ply')
    return mesh


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

def calculate_angle(a,b,c):
    ab = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    ac = np.array([c[0] - a[0], c[1] - a[1], c[2] - a[2]])
    dot_product = ab @ ac
    norm_ab = np.linalg.norm(ab, ord=2)
    norm_ac = np.linalg.norm(ac, ord=2)
    cos = dot_product / (norm_ab*norm_ac)
    angle_rad = math.acos(max(min(cos, 1.0), -1.0))
    return math.degrees(angle_rad)

def min_triangle_angle(p1,p2,p3):
    return min(calculate_angle(p1,p2,p3),calculate_angle(p2,p3,p1),calculate_angle(p3,p1,p2))

def build_mesh_pixel_face_cut(z_depth_pos, name, albedo_1d_array, material_array=None, normal_array=None, res=[256,256], fov=90, write_ply=True, threshold_angle=5):
    # mesh : (4*H*W) * 3
    # modified xy

    if np.max(z_depth_pos) > 1:
        z_depth_pos /= np.max(z_depth_pos)

    if albedo_1d_array.ndim != 1:
        albedo_1d_array = albedo_1d_array.reshape([-1])
    
    if normal_array is not None and normal_array.ndim > 2:
        normal_array = normal_array.reshape([-1, 3])

    y_dim = res[0]; x_dim = res[1] 
    N = y_dim*x_dim
    pixel_length = 2 / x_dim # one side length of pixel face

    y_screen,x_screen = np.mgrid[y_dim/x_dim-pixel_length/2 :-y_dim/x_dim+pixel_length/2 :y_dim*1j, -1+pixel_length/2 :1-pixel_length/2:x_dim*1j]
    y_screen = y_screen.reshape([-1])
    x_screen = x_screen.reshape([-1])
    z_screen = -1/math.tan(math.radians(fov/2))*np.ones([N])

    
    y_modified = y_screen * np.abs(z_depth_pos.reshape([-1]))*math.tan(math.radians(fov/2))
    x_modified = x_screen * np.abs(z_depth_pos.reshape([-1]))*math.tan(math.radians(fov/2))

    center_pos_screen = np.stack([x_screen, y_screen, z_screen], axis=1)
    center_pos = np.stack([x_modified, y_modified, (-1*z_depth_pos).reshape([-1])], axis=1)
    vertex_pos = np.zeros([4*N, 3])
    error_num = 0
    angle_error_num = 0
    exception_index = []

    for i in range(y_dim): # from top to bottom
        for j in range(x_dim): # from left to right
            # p --- q
            # |  c  |
            # s --- r
            c_screen = center_pos_screen[i*x_dim+j,:]
            p_screen = np.array([c_screen[0]-pixel_length/2, c_screen[1]-pixel_length/2, c_screen[2]])
            q_screen = np.array([c_screen[0]-pixel_length/2, c_screen[1]+pixel_length/2, c_screen[2]])
            r_screen = np.array([c_screen[0]+pixel_length/2, c_screen[1]+pixel_length/2, c_screen[2]])
            s_screen = np.array([c_screen[0]+pixel_length/2, c_screen[1]-pixel_length/2, c_screen[2]])
            n = normal_array[i*x_dim+j,:]
            c = center_pos[i*x_dim+j,:]
            # if np.dot(n,p_screen) < 0.0000001 or np.dot(n,q_screen) < 0.0000001 or np.dot(n,r_screen) < 0.0000001 or np.dot(n,s_screen) < 0.0000001:
            #     print('Normal vector and view vector are nearly orthogonal. (i,j) = ('+str(i)+','+str(j)+')')
            #     return 
            k_p = np.dot(n,c) / np.dot(n,p_screen)
            k_q = np.dot(n,c) / np.dot(n,q_screen)
            k_r = np.dot(n,c) / np.dot(n,r_screen)
            k_s = np.dot(n,c) / np.dot(n,s_screen)

            if k_p < 0 or k_q < 0 or k_r < 0 or k_s < 0:
                error_num += 1
                p = 0.2
                q = 0.2
                r = 0.2
                s = 0.2
                exception_index.append(x_dim*i+j)
            else:
                p = k_p * p_screen
                q = k_q * q_screen
                r = k_r * r_screen
                s = k_s * s_screen
                if min_triangle_angle(p,q,s) < threshold_angle or min_triangle_angle(q,r,s) < threshold_angle:
                    angle_error_num += 1
                    p = 0.2
                    q = 0.2
                    r = 0.2
                    s = 0.2
                    exception_index.append(x_dim*i+j)
            
            vertex_pos[4*(i*x_dim+j),:] = p
            vertex_pos[4*(i*x_dim+j)+1,:] = q
            vertex_pos[4*(i*x_dim+j)+2,:] = r
            vertex_pos[4*(i*x_dim+j)+3,:] = s
            #print('p: ', end='')
            #print(vertex_pos[4*(i*x_dim+j),:])
    
    delete_vertex_index = np.array(list(np.arange(4*x,4*x+4) for x in exception_index)).reshape(-1)
    al = np.repeat(albedo_1d_array.reshape([-1,3]), 4, axis=0) # 4N x 3
    rough = np.repeat(material_array[:,:,0].reshape(-1),4) # 4N
    metal = np.repeat(material_array[:,:,1].reshape(-1),4)

    vertex_pos = np.delete(vertex_pos, delete_vertex_index, axis=0)
    al = np.delete(al, delete_vertex_index, axis=0)
    rough = np.delete(rough, delete_vertex_index)
    metal = np.delete(metal, delete_vertex_index)


    # face list : 
    # m--m+1           m+1
    # | /  x N,      /  |   x N
    # m+3          m+3-m+2
    face_ind = np.array(range(N-len(exception_index)))
    faces_1 = np.concatenate([4*face_ind+1, 4*face_ind+2])
    faces_2 = np.concatenate([4*face_ind, 4*face_ind+1])
    faces_3 = np.concatenate([4*face_ind+3, 4*face_ind+3])
    face_indices = mi.Vector3u(faces_1, faces_2, faces_3)
    
    #print(vertex_pos)
    #print(face_indices)
    
    mesh = mi.Mesh(
        name,
        vertex_count=4*(N-len(exception_index)),
        face_count=2*(N-len(exception_index)),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(mi.Point3f(vertex_pos.transpose(1,0)))
    mesh_params['faces'] = dr.ravel(face_indices)
    print('error num:'+str(error_num))
    print('angle error num:'+str(angle_error_num))
    
    # albedo_1d_array : N*3 -> 4*N*3
    mesh.add_attribute('vertex_albedo', 3, al.reshape([-1]))
    #mesh.add_attribute('vertex_albedo', 3, albedo_1d_array)
    if material_array is not None:
        mesh.add_attribute('vertex_roughness', 1, rough)
        mesh.add_attribute('vertex_metallic', 1, metal)
    mesh_params.update()
    if write_ply:
        mesh.write_ply(name+'.ply')
    return mesh

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