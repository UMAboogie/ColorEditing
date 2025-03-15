import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from glob import glob
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
from tqdm.contrib import tenumerate, tzip
import yaml
import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import torchvision.utils as vutils


class MGNetTest(pl.LightningModule):
    def __init__(self, cfg, opt, exp_name, test_dir, imList):
        super().__init__()
        self.models = nn.ModuleDict(models.get_model(cfg.model))
        self.opt = opt
        self.exp_name = exp_name
        self.test_dir = test_dir
        if isinstance(imList, str):
            with open(imList) as f:
                self.imList = f.readlines()
        elif isinstance(imList, list):
            self.imList = imList
        elif imList is None:
            self.imList = glob(os.path.join(self.test_dir, "*.png"))
            self.imList += glob(os.path.join(self.test_dir, "*.exr"))
        else:
            raise ValueError("Image list not identified")
        self.imList = [os.path.basename(x.strip()) for x in self.imList]
        os.makedirs(os.path.join(self.test_dir, self.exp_name), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, self.exp_name, 'albedo'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, self.exp_name, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, self.exp_name, 'normal'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, self.exp_name, 'material'), exist_ok=True)
    
    def forward(self):
        pass
    
    def test_dataloader(self):
        return DataLoader(torch.arange(len(self.imList), dtype=torch.int), batch_size=1, shuffle=False)
    
    def test_step(self, batch, batch_nb):
        imName = os.path.join(self.test_dir, self.imList[batch_nb])
        is_hdr = imName.endswith('.exr')
        print(is_hdr)
        #im = cv2.imread(imName)[:,:,::-1] if not is_hdr else cv2.imread(imName, -1)[:,:,::-1]
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        im = cv2.imread(imName)[:,:,::-1] if not is_hdr else cv2.imread(imName, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,::-1]
        nh, nw = im.shape[0], im.shape[1]
        im_orig = im
        if nh < nw:
            newW = opt.width
            newH = int(float(opt.width) / float(nw) * nh)
        else:
            newH = opt.height
            newW = int(float(opt.height) / float(nh) * nw)
        
        if nh < newH:
            im = cv2.resize(im, (newW, newH), interpolation = cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (newW, newH), interpolation = cv2.INTER_LINEAR)
        if not is_hdr:
            im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
            im = torch.from_numpy(im**2.2)
            im_orig = (im_orig.astype(np.float32)**2.2 / 255.0)
        else:
            im = np.transpose(im, [2, 0, 1] )[np.newaxis, :, :, :]
            im = torch.from_numpy(im)
        im = im.cuda()
        features = self.models['encoder'](im)
        albedoPred = self.models['albedo'](im, *features)
        normalPred = self.models['normal'](im, *features)
        matPred = self.models['material'](im, *features)
        depthPred = self.models['depth'](im, *features)
        albedoPred = torch.clamp(albedoPred, 0, 1)
        depthPred = torch.clamp(depthPred, min=0)
        depthPred /= torch.max(depthPred)
        #normalPred = (normalPred + 1) * 0.5
        '''
        if not is_hdr:
            im = im ** (1/2.2)
        '''
        im = im ** (1/2.2)
        #utils.plot_images_wo_gt(albedoPred, im, os.path.join(self.test_dir, self.exp_name, f'albedo/{batch_nb:04d}.png'))
        #utils.plot_images_wo_gt(normalPred, im, os.path.join(self.test_dir, self.exp_name, f'normal/{batch_nb:04d}.png'))
        #utils.plot_images_wo_gt(depthPred[:,0,:,:], im, os.path.join(self.test_dir, self.exp_name, f'depth/{batch_nb:04d}.png'), colormap='magma')
        #utils.plot_images_wo_gt(matPred[:,0,:,:], im, os.path.join(self.test_dir, self.exp_name, f'roughness/{batch_nb:04d}.png'), colormap='jet')
        #utils.plot_images_wo_gt(matPred[:,1,:,:], im, os.path.join(self.test_dir, self.exp_name, f'metallic/{batch_nb:04d}.png'), colormap='jet')
        
        cv2.imwrite(os.path.join(self.test_dir, self.exp_name, f'albedo/{batch_nb:04d}.exr'), albedoPred.cpu().numpy().squeeze(0).transpose(1,2,0)[:,:,::-1])
        cv2.imwrite(os.path.join(self.test_dir, self.exp_name, f'normal/{batch_nb:04d}.exr'), normalPred.cpu().numpy().squeeze(0).transpose(1,2,0)[:,:,::-1])
        cv2.imwrite(os.path.join(self.test_dir, self.exp_name, f'depth/{batch_nb:04d}.exr'), depthPred.cpu().numpy().squeeze(0).transpose(1,2,0)[:,:,::-1])
        mat = matPred.cpu().numpy().squeeze(0).transpose(1,2,0)
        cv2.imwrite(os.path.join(self.test_dir, self.exp_name, f'material/{batch_nb:04d}.exr'), np.concatenate([mat, np.zeros([mat.shape[0], mat.shape[1], 1],dtype=np.float32)], axis=2)[:,:,::-1])
             
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--path', '-p', required=True, help='path to input images')
    parser.add_argument('--ckpt', default='last.ckpt', help='name of checkpoint file')
    parser.add_argument('--height', '-H', type=int, default=480, help='height of image')
    parser.add_argument('--width', '-W', type=int, default=640, help='width of image')
    parser.add_argument('--gpuId', '-g', type=int, default=0)
    parser.add_argument("--version", '-v', type=int, required=True)
    parser.add_argument("--list", '-l', nargs='+', default=None, help='list of input images')
    parser.add_argument("--listfile", default=None, help='file name of input image list')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    print("Start...")
    opt = parse_args()
    torch.cuda.set_device(opt.gpuId)

    with open(opt.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = utils.CfgNode(cfg_dict)

    if not opt.ckpt.endswith('.ckpt'):
        opt.ckpt += '.ckpt'
    ckpt_path = os.path.join(cfg.experiment.path_logs, cfg.experiment.id, f'version_{opt.version}', f'checkpoint/{opt.ckpt}')
    system = MGNetTest.load_from_checkpoint(
        ckpt_path,
        map_location=f'cuda:{opt.gpuId}',
        strict=False,
        cfg=cfg, opt=opt, exp_name=f"{cfg.experiment.id}_v{opt.version}", test_dir=opt.path, imList=opt.list if opt.list is not None else opt.listfile
    )
    progbar_callback = RichProgressBar(leave=False)
    '''
    trainer = pl.Trainer(
        logger=False,
        gpus=[opt.gpuId],
        callbacks=[progbar_callback]
    )
    '''
    trainer = pl.Trainer(
        logger=False,
        accelerator='gpu',
        devices=[opt.gpuId],
        callbacks=[progbar_callback]
    )
    torch.set_float32_matmul_precision('high')
    trainer.test(system)
