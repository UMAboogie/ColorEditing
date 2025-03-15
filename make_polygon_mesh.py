import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys
from argparse import ArgumentParser
from PIL import Image

def build_mesh_default(name, depth_array, albedo_array, material_array, res=[256,256], fov=90, write_ply=True, threshold=0.01, delete_square=False):
    # mesh : H * W * 3 vertices and (H-1) * (W-1) * 2 faces
    # build polygon mesh with corresponding pixels and vertices

    if np.max(depth_array) > 1:
        depth_array /= np.max(depth_array)

    if albedo_array.ndim != 1:
        albedo_array = albedo_array.reshape([-1])

    y_dim = res[0]; x_dim = res[1] 
    N = y_dim*x_dim

    y,x = np.mgrid[(y_dim-1)/(x_dim-1):-(y_dim-1)/(x_dim-1):y_dim*1j, -1:1:x_dim*1j]
    y = y.reshape([-1])
    x = x.reshape([-1])
    z = (-1*depth_array).reshape([-1])

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
    
    mesh.add_attribute('vertex_albedo', 3, albedo_array)
    mesh.add_attribute('vertex_roughness', 1, material_array[:,:,0].reshape(-1))
    mesh.add_attribute('vertex_metallic', 1, material_array[:,:,1].reshape(-1))
    mesh_params.update()
    if write_ply:
        mesh.write_ply(name+'.ply')
    return mesh




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

def build_mesh_pixel_face(name, depth_array, albedo_array, material_array, normal_array, res=[256,256], fov=90, write_ply=True, threshold_angle=1):
    # mesh : (4*H*W) * 3 vertices and H * W * 2 faces
    # build polygon mesh with corresponding pixels and faces
    # yuma proposed it

    if np.max(depth_array) > 1:
        depth_array /= np.max(depth_array)

    if albedo_array.ndim != 1:
        albedo_array = albedo_array.reshape([-1])
    
    if normal_array.ndim > 2:
        normal_array = normal_array.reshape([-1, 3])

    y_dim = res[0]; x_dim = res[1] 
    N = y_dim*x_dim
    pixel_length = 2 / x_dim # one side length of pixel face

    y_screen,x_screen = np.mgrid[y_dim/x_dim-pixel_length/2 :-y_dim/x_dim+pixel_length/2 :y_dim*1j, -1+pixel_length/2 :1-pixel_length/2:x_dim*1j]
    y_screen = y_screen.reshape([-1])
    x_screen = x_screen.reshape([-1])
    z_screen = -1/math.tan(math.radians(fov/2))*np.ones([N])

    
    y_modified = y_screen * np.abs(depth_array.reshape([-1]))*math.tan(math.radians(fov/2))
    x_modified = x_screen * np.abs(depth_array.reshape([-1]))*math.tan(math.radians(fov/2))

    center_pos_screen = np.stack([x_screen, y_screen, z_screen], axis=1)
    center_pos = np.stack([x_modified, y_modified, (-1*depth_array).reshape([-1])], axis=1)
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
   
    delete_vertex_index = np.array(list(np.arange(4*x,4*x+4) for x in exception_index)).reshape(-1)
    al = np.repeat(albedo_array.reshape([-1,3]), 4, axis=0) # 4N x 3
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
    
    # albedo_1d_array : N*3 -> 4*N*3
    mesh.add_attribute('vertex_albedo', 3, al.reshape([-1]))
    if material_array is not None:
        mesh.add_attribute('vertex_roughness', 1, rough)
        mesh.add_attribute('vertex_metallic', 1, metal)
    mesh_params.update()
    if write_ply:
        mesh.write_ply(name+'.ply')
    return mesh


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-p', '--dir_path', default=None)
    # argparser.add_argument('-t', '--target', default=None)
    argparser.add_argument('-dp', '--depth_path', default=None) # npy file
    argparser.add_argument('-ap', '--albedo_path', default=None) # exr, png, jpeg or jpg file
    
    argparser.add_argument('-amp', '--albedo_mask_path', default=None)  
    argparser.add_argument('-new_al', '--new_albedo_value', nargs=3, type=float)  

    argparser.add_argument('-mp', '--material_path', default=None)
    argparser.add_argument('-np', '--normal_path', default=None)
    argparser.add_argument('--not_use_StableNormal', action='store_true') 
    argparser.add_argument('-fov', '--fov',type=float, default=50)
    argparser.add_argument('-name', '--name', default='400x640_0')
    argparser.add_argument('-he', '--height',type=int, default=360)
    argparser.add_argument('-wi', '--width',type=int, default=640)
    argparser.add_argument('-tr', '--threshold',type=float, default=1.0)
    argparser.add_argument('--use_default', action='store_false')



    args = argparser.parse_args()



    # load depth
    if args.depth_path is not None:
        depth_path = args.depth_path
    else:
        depth_path = args.dir_path+'/depth.npy'
    depth_array = np.load(depth_path)

    # load albedo
    if args.albedo_path is not None:
        albedo_path = args.albedo_path
    else:
        albedo_path = args.dir_path+'dense_v1/albedo/000.exr'



    if amp is not None:
        albedo_array = (change_albedo(albedo_path, args.albedo_mask_path, new_color=list(args.new_albedo_value), height=args.height)).astype(np.float32) / 255
    elif albedo_path[-3:] == 'png' or albedo_path[-3:] == 'jpg' or albedo_path[-4:] == 'jpeg': 
        albedo_array = np.array(Image.open(albedo_path)).astype(np.float32)/255
    elif albedo_path[-3:] == 'exr':
        albedo_array = my_util_func.exr2np(albedo_path, height=args.height)

    # load material
    if args.material_path is not None:
        material_path = args.material_path
    else:
        material_path = args.dir_path+'dense_v1/material/000.exr'
    material_array = my_util_func.exr2np(material_path, height=args.height)


    if args.use_default:
        build_mesh_default(name=args.name, depth_array, albedo_array, material_array, res=[args.height,args.width], fov=args.fov, threshold=args.threshold)
    else:
        # load normal  
        if args.normal_path is not None:
            normal_path = args.normal_path
        else:
            normal_path = args.dir_path+'/normal.png'

        if not args.not_use_StableNormal:
            # In StableNormal's normal image, the x-axis is the opposite of that of the our method
            normal = Image.open(normal_path).resize((res[1],res[0]), Image.BICUBIC) # (W,H)
            normal_new = np.zeros([res[0], res[1], 3])
            normal_new[:,:,0] = -1*(((np.array(normal).astype(np.float32) / 255)[:,:,0])*2-1)
            normal_new[:,:,1] = ((np.array(normal).astype(np.float32) / 255)[:,:,1])*2-1
            normal_new[:,:,2] = (((np.array(normal).astype(np.float32) / 255)[:,:,2])*2-1)
            normal_array = normal_new
        else:
            if normal_path[-3:] == 'png' or normal_path[-3:] == 'jpg' or normal_path[-4:] == 'jpeg': 
                normal_array = Image.open(normal_path).resize((res[1],res[0]), Image.BICUBIC) # (W,H)
            elif normal_path[-3:] == 'npy':
                normal_array = np.load(normal_path)
        build_mesh_pixel_face(name=args.name, depth_array, albedo_array, material_array, normal_array, res=[args.height,args.width], fov=args.fov, threshold_angle=args.threshold)