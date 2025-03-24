# Interreflection-aware color editing of images using inverse rendering

## Requirement & Installation
Please check updates of used packages especially mitsuba and dr.jit.

I strongly recommend the use of a virtual environment such as conda.
Python version I used is 3.11.5.
In virtual environment you created, please run following commands to build package:
```
pip install -r requirements.txt
```
If you use [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [StableNormal](https://github.com/Stable-X/StableNormal) in this project, you need clone them in geometry_prediction.




## Color Editing
1. Prepare input data.
    * Three formats are supported: png, jpg and exr.
    
2. Depth and normal prediction. 
    * I used [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [StableNormal](https://github.com/Stable-X/StableNormal) for geometry prediction in the project.
    * If you can use more accurate methods, use them.
3. Material Prediction.
    * I used mgnet in [IndoorInverseRendering](https://github.com/jingsenzhu/IndoorInverseRendering) in the project.
    * If you can use more accurate methods, use them.
4. Make polygon mesh file for Mitsuba 3.
```
python make_polygon_mesh.py --dir_path (path for input data directory) --fov (field of view) --name (mesh file name to be given) -height (height of the input image) -width (width of the input image)
```
    * If the input directory has the following structure, a polygon mesh file can be created by specifying --dir_path.
```
.
└── example1(any name)
    ├── depth.npy
    ├── normal.png
    └── dense_v1
        ├── albedo -- 0000.exr
        └── material -- 0000.exr
```
    * Otherwise, please add --depth_path, --albedo_path, --material_path, or --normal_path.
    * If you use normal maps estimated by methods other than StableNormal, add --not_use_StableNormal.
    * If you use option '--threshold x', you can change threshold value to x for deleting mesh faces.
5. Light source optimization.

```
python arealight_optimization.py --image_path (path for the input image) --mesh_path (path for the mesh file)  --fov (field of view) --name (scene name) --height (height of the input image)
```
    * You have to name the scene of input image.
    * Other optional arguments are listed below. If you want to change some condition, plase use them.
        * --spp
        * --light_num
        * --fix_position
        * --size_opt
        * --use_envmap
        * --iteration_count
        * --learning_rate
    * If you got a memory error, please reduce --spp value.
    * After optimization, the log and results (as npy files) are in 'results/'+(--name)+'_arealight_opt_disk_results/'+(month_day_hour:minute:second)

6. Edit albedo with mask.
    * Please prepare a mask image which value are 0 (don't change albedo) or 255 (change albedo). GIMP is a useful tool.
    * You have to create a different polygon mesh file for every albedo change.
```
python make_polygon_mesh.py --dir_path xxx --fov n --name yyy --height H --width W --albedo_mask_path (path for the mask image) --new_albedo_value r g b 
```
    * Please assign new albedo values between 0 and 1.

7. Re-render the image.
```
python rerender_image.py --result_path (directory path for optimization results) --image_path yyy --mesh_path_1 (path for original mesh file) --mesh_path_2 (path for albedo changed mesh file) --fov n --height H --width W 
```
    * The result images are output to --result_path. 

# Related work
* [IndoorLightEditing](https://github.com/ViLab-UCSD/IndoorLightEditing)
* [rgbx](https://github.com/zheng95z/rgbx)
* [IntrinsicImageDiffusion](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion)