# optimise area disk lights of a scene
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
import logging
import copy
import my_util_func
import math
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser


mi.set_variant('cuda_ad_rgb')

argparser = ArgumentParser()
argparser.add_argument('-n', '--num', type=int, default=1)
# argparser.add_argument('-t', '--target', default=None)
argparser.add_argument('-im', '--image_path', default='real_images_for_exp_400x640_ours/0.png')
argparser.add_argument('-pl', '--ply_path', default='real_images_for_exp_400x640_ours/0.ply')
argparser.add_argument('-fov', '--fov',type=float, default=50)
argparser.add_argument('-name', '--name', default='400x640_0')
argparser.add_argument('-he', '--height',type=int, default=360)
argparser.add_argument('-spp', '--spp',type=int, default=40)
argparser.add_argument('-so', '--size_opt',type=bool, default=False)
argparser.add_argument('-cp', '--checkpoint_path',type=str, default=None)
argparser.add_argument('-it', '--iteration_count',type=int, default=600)
argparser.add_argument('-lr', '--learning_rate',type=float, default=1.0)

args = argparser.parse_args()

def vertexpos2index(vertex_postions, res, fov):
    vertex_postions = vertex_postions.reshape([-1,3])
    vertex_postions = vertex_postions / (vertex_postions[:,2] * math.tan(math.radians(fov/2)))
    vert_num = vertex_postions.shape(0) // 4
    pixel_pos = np.zeros([vert_num,3])
    for i in range(pixel_pos):
        pixel_pos[i,:] = (vertex_postions[4*i,:] + vertex_postions[4*i+1,:] + vertex_postions[4*i+2,:] + vertex_postions[4*i+3,:])/4
        pixel_pos = res[0]*pixel_pos
        pixel_pos[:, 0] = pixel_pos[:, 0]+1-(1/res[0])
    return pixel_pos



# setting
#fov = 19.5
#fov = 50
fov = args.fov
iteration_count = args.iteration_count
iteration_count_second = 0
output_interval = 10
learning_rate = args.learning_rate
lambda_l1 = 0.0001

#name = 'colorful_obj1'
#name_target = 'livingroom'
#name_target = 'scene2'
#name_target = 'cornell'
#name_target = 'livingroom_default'
#name_target = 'bed'
name_target = args.name

#gt_image_full = mi.render(mi.load_file('scenes/cbox_depth_0220.xml'), spp=1024)
#gt_image = my_util_func.exr2np('living-room-3/living-room-3/livingroom_nograss.exr', height=360)
#gt_image = my_util_func.exr2np('scene_2/scene2_2_360.exr', height=360)
#gt_image = (cv2.imread('colorful_obj1/colorful_obj1.png')[:,:,::-1]/ 255.).astype(np.float32)[::2,::2,:] 
#gt_image = my_util_func.exr2np('cornell-box/cornellbox.exr', height=512)
if args.image_path[-3:] == 'png' or args.image_path[-3:] == 'jpg' or args.image_path[-4:] == 'jpeg':
    gt_image = ((np.array(Image.open(args.image_path)).astype(np.float32)/255)**(2.2))[:,:,:3]
else:
    gt_image = my_util_func.exr2np(args.image_path, height=args.height)

env_h = 30
env_w = 60
res = [gt_image.shape[0],gt_image.shape[1]]
#res = [512,512]
radiance_init = 0.1
radius_init = 0.05
angles_init = [180, 0, 0]
center_init = [0, 0, 2]
spp = args.spp
#mesh_path = 'livingroom_pixelface_cut1_depthanyv2_raw.ply'
mesh_path = args.ply_path
#mesh_path = 'scene2_2_01101708.ply'
#mesh_path = 'cornell_01112135.ply'

centers_init = []
#light_num = 16
# for i in range(4):
#     for j in range(4):
#         centers_init.append([(i-1.5)*0.4,(j-1.5)*0.4,0.01]) 



half_w = math.tan(math.radians(fov/2))
half_h = half_w * res[0]/res[1]


light_num = args.num
for i in range(light_num):
    if i == 0:
        centers_init.append([0,0,2])
    else:
        centers_init.append([0.2*math.cos(2*math.pi/(light_num-1)*(i-1)), 0.2*math.sin(2*math.pi/(light_num-1)*(i-1)), 2])

# for i in range(light_num):
#     if light_num % 2 == 0: centers_init.append([(i)*0.2, 0,0.5])
#     else: centers_init.append([-1*(i)*0.2,0,0.5]) 


size_opt = args.size_opt
position_opt = True
use_env = False
write_disk_ply = True

dt = datetime.datetime.now()
dt_str = dt.strftime('%m%d_%H:%M:%S')
os.makedirs(name_target+'_arealight_opt_disk_results/', exist_ok=True)
os.mkdir(name_target+'_arealight_opt_disk_results/'+dt_str)

f = open(name_target+'_arealight_opt_disk_results/'+dt_str+'/setting_and_result.log', 'w')
f.close()

logging.basicConfig(filename=name_target+'_arealight_opt_disk_results/'+dt_str+'/setting_and_result.log', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.debug('iteration count: %f', iteration_count)
logging.debug('2nd iteration count: %f', iteration_count_second)
logging.debug('learning rate: %f', learning_rate)
logging.debug('light number: %d', light_num)
logging.debug('env height: %d', env_h)
logging.debug('env width: %d', env_w)
logging.debug('initial radiance: %f', radiance_init)
logging.debug('initial radius: %f', radius_init)
logging.debug('initial euler angles(z-y-x) (theta_x, theta_y, theta_z): (%d, %d, %d)', angles_init[0], angles_init[1], angles_init[2])
logging.debug('initial light position: (%d, %d, %d)', center_init[0], center_init[1], center_init[2])
logging.debug('image path: '+args.image_path)
logging.debug('ply path: '+args.ply_path)
logging.debug('image path: '+args.image_path)
if not args.checkpoint_path is None:
    logging.debug('checkpoint path: '+args.checkpoint_path)

origin_linear = my_util_func.load_img('../OmniSR/data/input/origin/input_0.png', to_linear=True)
noshadow_linear = my_util_func.load_img('../OmniSR/data/result/input_0.png', to_linear=True)
shadow_weight = np.sqrt(np.sum((origin_linear - noshadow_linear)**2, axis=2))
shadow_weight = shadow_weight + 1.0
shadow_weight = np.repeat(shadow_weight[:,:,None], 3, axis=2)


if size_opt: logging.debug('optimize radius')
if position_opt: logging.debug('optimize position')

if use_env:
    scene_env = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "light": {'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': 0,
        }},
        #"light" : {"type" :"constant"},
        "sensor": {
            "type": "perspective",
            'fov': 90,
            'near_clip': 0.01,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
            ),
            'film_id': {
            'type': 'hdrfilm',
            'width': env_w,
            'height': env_h,
            'pixel_format': 'rgb',
            'component_format': 'float32',
            'filter': { 'type': 'tent' }
            },
        },
    })

    env_bitmap = mi.util.convert_to_bitmap(mi.render(scene_env, spp=10))



mesh_loaded = mi.load_dict({
    "type": "ply",
    "filename": mesh_path,
    "bsdf": {
        'type': 'principled',
        'base_color': {
            "type": "mesh_attribute",
            "name": "vertex_albedo",
        },
        'metallic': {
            "type": "mesh_attribute",
            "name": "vertex_metallic",
        },
        'roughness': {
            "type": "mesh_attribute",
            "name": "vertex_roughness",
        },
    },
})

scene_dict = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 6, "hide_emitters":True},
    #"light": {'type': 'point', 'position': [0, 0, 0.1], 'intensity': {'type': 'spectrum', 'value': 1.0,}},
    #"light" : {"type" :"constant"},
    "sensor": {
        "type": "perspective",
        'fov': fov,
        'near_clip': 0.01,
        "to_world": mi.ScalarTransform4f().look_at(
            origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
        ),
        'film_id': {
        'type': 'hdrfilm',
        'width': res[1],
        'height': res[0],
        'pixel_format': 'rgb',
        'component_format': 'float32',
        'filter': { 'type': 'tent' },
        'sample_border': True,
        },
    },
    "wholescene": mesh_loaded,
}

if use_env:
    scene_dict['light'] = {"type":"envmap", "bitmap": env_bitmap}

param_list = []
cnt = 0

if not args.checkpoint_path is None:
    centers_init = np.load(args.checkpoint_path+'/disk_centers.npy')
    rads = np.load(args.checkpoint_path+'/disk_radiances.npy')
    eulers = np.load(args.checkpoint_path+'/disk_eulers.npy')
    scales = np.load(args.checkpoint_path+'/disk_scales.npy')

    for i in range(light_num):
        name = 'light_'+str(cnt+1)
        scene_dict[name] = {
            'type': 'disk',
            #'flip_normals': True,
            'to_world': mi.ScalarTransform4f().translate(centers_init[i]).rotate(axis=[0,0,1], angle=eulers[i,2]).rotate(axis=[0,1,0], angle=eulers[i,1]).rotate(axis=[1,0,0], angle=eulers[i,0]).scale(float(scales[i])),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [rads[i,0], rads[i,1], rads[i,2]] 
                },
            },
        }
        param_list.append(centers_init[i])
        cnt += 1
else:
    # back disk pattern
    for i in range(light_num):
        name = 'light_'+str(cnt+1)
        scene_dict[name] = {
            'type': 'disk',
            #'flip_normals': True,
            'to_world': mi.ScalarTransform4f().translate(centers_init[i]).rotate(axis=[0,0,1], angle=angles_init[2]).rotate(axis=[0,1,0], angle=angles_init[1]).rotate(axis=[1,0,0], angle=angles_init[0]).scale(radius_init),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [radiance_init, radiance_init, radiance_init] 
                },
            },
        }
        param_list.append(centers_init[i])
        cnt += 1




scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)

#print(params)

img = mi.render(scene, params, spp=256)
mi.util.write_bitmap(name_target+'_arealight_opt_disk_results/'+ dt_str +'/light_opt_'+str(0)+'.png', img)

opt = mi.ad.Adam(lr=learning_rate)
for i in range(cnt):
    name = 'light_'+str(i+1)
    if position_opt: 
        opt[name+'.center'] = mi.Point3f(float(param_list[i][0]), float(param_list[i][1]), float(param_list[i][2]))
    opt[name+'.euler_angles'] = mi.Point3f(angles_init)
    if size_opt: 
        opt[name+'.scale'] = mi.Float(radius_init)
    opt[name+'.emitter.radiance.value'] = params[name+'.emitter.radiance.value']

if use_env:
    opt['light.scale'] = params['light.scale']
    opt['light.data'] = params['light.data']
    
params.update(opt)


def apply_transformation(params, opt, it):
    for i in range(cnt):
        name = 'light_'+str(i+1)
        #if size_opt and it >= iteration_count: 
        if size_opt: 
            opt[name+'.scale'] = dr.clip(opt[name+'.scale'], 0, 1000000)
        if position_opt: 
            # z-y-x
            #opt[name+'.euler_angles'].x = dr.clip(opt[name+'.euler_angles'].x, -180, 180)
            #opt[name+'.euler_angles'].z = dr.clip(opt[name+'.euler_angles'].z, -180, 180)
            #opt[name+'.euler_angles'].y = dr.clip(opt[name+'.euler_angles'].y, -89.9, 89.9)
            opt[name+'.center'].z = dr.clip(opt[name+'.center'].z, -1, 100)
        opt[name+'.emitter.radiance.value'] = dr.clip(opt[name+'.emitter.radiance.value'], 0.0, 100000000)
        if position_opt:
            # if size_opt and it >= iteration_count:
            if size_opt:    
                trafo = mi.Transform4f().translate([opt[name+'.center'].x, opt[name+'.center'].y, opt[name+'.center'].z]).rotate(axis=[0,0,1], angle=opt[name+'.euler_angles'].z).rotate(axis=[0,1,0], angle=opt[name+'.euler_angles'].y).rotate(axis=[1,0,0], angle=opt[name+'.euler_angles'].x).scale(opt[name+'.scale'])
            else:
                trafo = mi.Transform4f().translate([opt[name+'.center'].x, opt[name+'.center'].y, opt[name+'.center'].z]).rotate(axis=[0,0,1], angle=opt[name+'.euler_angles'].z).rotate(axis=[0,1,0], angle=opt[name+'.euler_angles'].y).rotate(axis=[1,0,0], angle=opt[name+'.euler_angles'].x).scale(radius_init)
        else: trafo = mi.Transform4f().translate([param_list[i][0], param_list[i][1], param_list[i][2]]).rotate(axis=[0,0,1], angle=opt[name+'.euler_angles'].z).rotate(axis=[0,1,0], angle=opt[name+'.euler_angles'].y).rotate(axis=[1,0,0], angle=opt[name+'.euler_angles'].x).scale(radius_init)    
        params[name+'.to_world'] = trafo
    params.update(opt)


# sensor_2 = mi.load_dict({
#         'type': 'perspective',
#         'fov': fov,
#         'near_clip': 0.001,
#         'to_world': mi.ScalarTransform4f.look_at(
#             origin=[0, 0, 5], target=[0, 0, -1/3], up=[0, 1, 0]
#         ),
#         'film_id': {
#         'type': 'hdrfilm',
#         'width': res[1],
#         'height': res[0],
#         'pixel_format': 'rgb',
#         'component_format': 'float32',
#         'filter': { 'type': 'tent' }
#         },
#     })
# mi.util.write_bitmap(name_target+'_arealight_opt_disk_results/'+ dt_str +'/from_back.png', mi.render(scene, sensor=sensor_2, spp=128)[:,:,:3])

def mse(img_tmp):
    #return dr.mean(dr.square(img_tmp - gt_image[::2,::2,:]))
    return dr.mean(dr.square(img_tmp - gt_image))

def weighted_mse(img_tmp):
    #return dr.mean(dr.square((img_tmp - gt_image[::2,::2,:])*shadow_weight))
    return dr.mean(dr.square((img_tmp - gt_image)*shadow_weight))

def l1_loss(img_tmp):
    return dr.mean(dr.abs(img_tmp - gt_image))

def l1_light(params):
    l1 = 0
    for i in range(cnt):
        name = 'light_'+str(i+1)
        l1 += dr.abs(dr.mean(dr.square(params[name+'.emitter.radiance.value'])))
    return l1

# def loss_depth(depth_tmp):
#     return dr.mean(dr.square(depth_tmp - depth_init))

errors = []
mses = []
loss_min = float('inf')
arg_min = 0
for it in range(iteration_count+iteration_count_second):
    # Perform a (noisy) differentiable rendering of the scene
    img = mi.render(scene, params, spp=spp)
    batch_index = it%4
    batch_part = [[0,0],[0,256],[256,0],[256,256]]
    
    if it % output_interval == 0:
        mi.util.write_bitmap(name_target+'_arealight_opt_disk_results/'+ dt_str +'/light_opt_'+str(it)+'.png', img[:,:,:3])
    # Evaluate the objective function from the current rendered image
    # if it % 25 == 0:
    #     mi.util.write_bitmap(name_target+'_arealight_opt_disk_results/'+ dt_str +'/from_back_'+str(it)+'.png', mi.render(scene, sensor=sensor_2, spp=128)[:,:,:3])
    
    #loss = mse(img[:,:,:3])+lambda_l1 * l1_light(params)
    #loss = loss_depth(img[:,:,3])
    #loss = l1_loss(img[:,:,:3])
    #loss = weighted_mse(img[:,:,:3])
    loss = mse(img[:,:,:3])

    mse_value = np.mean((img[:,:,:3].numpy() - gt_image)**2)

    # Backpropagate through the rendering process
    dr.backward(loss)
    # Optimizer: take a gradient descent step
    
    opt.step()
    if use_env: opt['light.data'] = dr.clip(opt['light.data'], 0.0, 1000000)
    
    # Update the scene state to the new optimized values
    
    #params.update(opt)
    apply_transformation(params, opt, it)
    #print(loss.shape)

    # Track the difference between the current color and the true value
    print(f'Iteration {it:02d}: loss = {loss.array[0]:6f}', end='\r')
    logging.info(f'Iteration {it:02d}: loss = {loss.array[0]:6f}')
    #print(params['sphere_1.to_world'], end='\r')
    errors.append(loss.array[0])
    mses.append(mse_value)
    
    #if it >=  1 and errors[-2] <  errors[-1]:
    #    params = params_tmp
    #    break
    if loss.array[0] < loss_min:
        radiances_best = []
        centers_best = []
        scales_best = []
        eulers_best = []

        #print(opt.keys())
        for i in range(cnt):
            name = 'light_'+str(i+1)
            if position_opt: centers_best.append(opt[name+'.center'].numpy())
            else: centers_best.append(np.array(param_list[i]).reshape([3,1]))
            #if size_opt: scale_best = opt[name+'.scale'].numpy()
            if size_opt: scales_best.append(opt[name+'.scale'].numpy())
            else: scales_best.append(radius_init)
            #euler_best = opt[name+'.euler_angles'].numpy()
            eulers_best.append(opt[name+'.euler_angles'].numpy())
            tmp = ((opt[name+'.emitter.radiance.value']).numpy())
            radiances_best.append([tmp[0,0], tmp[1,0], tmp[2,0]])         
        arg_min = it
        loss_min = loss.array[0]
        # if use_env: scale_best = np.array(params['light.scale'])
        # if use_env: data_best = np.array(params['light.data'])
    # if it in (int(0.7 * (iteration_count+iteration_count_second)), int(0.9 * (iteration_count+iteration_count_second))):
    #     spp *= 2
    #     opt.set_learning_rate(0.5 * opt.lr['data'])



print('\nOptimization complete. best params: iteration '+str(arg_min)+', loss: '+str(loss_min))
logging.info('Optimization complete. best params: iteration '+str(arg_min)+', loss: '+str(loss_min))

#params = mi.traverse(scene)

plt.plot(errors, label='Loss')
#plt.plot(mses, label='MSE')
plt.legend()
plt.savefig(name_target+'_arealight_opt_disk_results/'+ dt_str +'/errors.png')

# plt.plot(errors)
# plt.savefig(name_target+'_arealight_opt_disk_results/'+ dt_str +'/errors.png')

#mi.util.write_bitmap(name_target+'_arealight_opt_disk_results/'+ dt_str +'/light_opt_best.png', mi.render(scene, params_best, spp=1024)[:,:,:3])

#mi.util.write_bitmap(name_target+'_arealight_opt_disk_results/'+ dt_str +'/from_back_last.png', mi.render(scene, sensor=sensor_2, spp=128)[:,:,:3])


logging.info('best params at iteration '+str(arg_min))
for i in range(cnt):
    #print(centers_best[i].shape)
    if position_opt: logging.info('center position: ('+str(centers_best[i][0][0])+', '+str(centers_best[i][1][0])+', '+str(centers_best[i][2][0])+')')
    if size_opt: logging.info('disk radius: '+str(scales_best[i][0]))
    logging.info('euler angle: ('+str(eulers_best[i][0][0])+', '+str(eulers_best[i][1][0])+', '+str(eulers_best[i][2][0])+')')
    logging.info('radiances: ('+str(radiances_best[i][0])+', '+str(radiances_best[i][1])+', '+str(radiances_best[0][2])+')')



if use_env: print('best scale is ' + str(scale_best) )
if use_env: logging.info('best scale is ' + str(scale_best))
if use_env: np.save(name_target+'_arealight_opt_disk_results/'+dt_str+'/best_data.npy', data_best)
if use_env: cv2.imwrite(name_target+'_arealight_opt_disk_results/'+dt_str+'/best_env.exr', data_best[:,:,::-1])
#np.save(name_target+'_arealight_opt_disk_results/'+dt_str+'/best_sphere_radiance.npy', radiance_np_best)

if write_disk_ply:
    for i in range(cnt):
        vert_num=12
        # vert_num(outer vertecies) + 1(center vertex)
        vertex_pos_init = mi.Point3f([0]+[math.cos(math.radians(30*i)) for i in range(vert_num)], [0]+[math.sin(math.radians(30*i)) for i in range(vert_num)], [0]*(vert_num+1))
        if size_opt: 
            trafo = mi.Transform4f().translate(centers_best[i][:,0]).rotate(axis=[0,0,1], angle=float(eulers_best[i][2][0])).rotate(axis=[0,1,0], angle=float(eulers_best[i][1][0])).rotate(axis=[1,0,0], angle=float(eulers_best[i][0][0])).scale(float(scales_best[i][0]))
        else:
            trafo = mi.Transform4f().translate(centers_best[i][:,0]).rotate(axis=[0,0,1], angle=float(eulers_best[i][2][0])).rotate(axis=[0,1,0], angle=float(eulers_best[i][1][0])).rotate(axis=[1,0,0], angle=float(eulers_best[i][0][0])).scale(radius_init)
        vertex_pos = trafo @ vertex_pos_init

        #         4         y
        #     5       3     ^
        #    6         2    |
        #   7     0     1     -->x
        #    8        12
        #     9     11
        #        10

        p = np.zeros([vert_num])
        q = np.arange(1,vert_num+1)
        r = np.append(np.arange(2,vert_num+1), 1)

        face_indices = mi.Vector3u(p, q, r)
        
        mesh = mi.Mesh(
            'disk'+str(cnt+1),
            vertex_count=vert_num+1,
            face_count=vert_num,
            has_vertex_normals=False,
            has_vertex_texcoords=False,
        )      
        mesh_params = mi.traverse(mesh)
        mesh_params['vertex_positions'] = dr.ravel(vertex_pos)
        mesh_params['faces'] = dr.ravel(face_indices)
        mesh_params.update()
        
        mesh.write_ply(name_target+'_arealight_opt_disk_results/'+dt_str+'/disk_'+dt_str+'_'+str(i+1)+'.ply')

np.save(name_target+'_arealight_opt_disk_results/'+dt_str+'/disk_radiances.npy', np.array(radiances_best))
np.save(name_target+'_arealight_opt_disk_results/'+dt_str+'/disk_centers.npy', np.array(centers_best).reshape(cnt,3))
np.save(name_target+'_arealight_opt_disk_results/'+dt_str+'/disk_eulers.npy', np.array(eulers_best).reshape(cnt,3))
np.save(name_target+'_arealight_opt_disk_results/'+dt_str+'/disk_scales.npy', np.array(scales_best).reshape(cnt))
