import numpy as np
import math

im_input


Image.fromarray(((np.clip(np.abs(im), 0, 1)**(1/2.2))*255).astype(np.uint8)).save(dir_path+'/rendered.png')
Image.fromarray(((np.clip(np.abs(im_g), 0, 1)**(1/2.2))*255).astype(np.uint8)).save(dir_path+'/rendered_changed.png')
Image.fromarray(((np.clip(np.abs(im_g-im), 0, 1)**(1/2.2))*255).astype(np.uint8)).save(dir_path+'/diff.png')
Image.fromarray(((np.clip(im_input+(im_g-im), 0, 1)**(1/2.2))*255).astype(np.uint8)).save(dir_path+'/finalimage.png')