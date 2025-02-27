# Interreflection-aware color editing of images using inverse rendering

## Structure

## Requirement & Installation
Please check updates of used packages especially mitsuba and dr.jit.

I strongly recommend the use of a virtual environment such as conda.
In virtual environment you created, please run following commands to build package:
```
pip install -r requirements.txt
```
If you use [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [StableNormal](https://github.com/Stable-X/StableNormal) as this project, you need clone them in geometry_prediction.




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
5. Light source optimization.
6. Edit albedo with mask.
7. Re-render the image.



# Related work
* [IndoorLightEditing](https://github.com/ViLab-UCSD/IndoorLightEditing)
* [rgbx](https://github.com/zheng95z/rgbx)
* [IntrinsicImageDiffusion](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion)