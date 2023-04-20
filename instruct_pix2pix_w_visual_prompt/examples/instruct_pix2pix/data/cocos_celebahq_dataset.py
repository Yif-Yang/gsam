# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
import numpy as np
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
instruct_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(instruct_root)
from data.cocos_pix2pix_dataset import Pix2pixDataset
from data.cocos_base_dataset import get_params, get_transform
from utils.cocos_utils import colormap


class CelebAHQDataset(Pix2pixDataset):
    # hair, skin, l_brow, r_blow, l_eye, r_eye, l_ear, r_ear, nose, u_lip, mouth, l_lip, neck,
    # cloth, hat, eye_g, ear_r, neck_l
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=19)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        if opt.phase == 'train':
            fd = open(os.path.join(opt.dataroot, 'train.txt'))
            lines = fd.readlines()
            fd.close()
        elif opt.phase == 'test':
            fd = open(os.path.join(opt.dataroot, 'val.txt'))
            lines = fd.readlines()
            fd.close()

        image_paths = []
        label_paths = []
        for i in range(len(lines)):
            image_paths.append(os.path.join(opt.dataroot, 'CelebA-HQ-img', lines[i].strip() + '.jpg'))
            label_paths.append(os.path.join(opt.dataroot, 'CelebAMask-HQ-mask-anno', 'all_parts_except_glasses',
                                            lines[i].strip().zfill(5) + '.png'))

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = ''
        if opt.phase == 'test':
            extra = '_test'
        with open(os.path.join(instruct_root,'./data/celebahq_ref{}.txt'.format(extra))) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_label_tensor(self, path):
        # parts = ['skin', 'hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'neck',
        #             'cloth', 'hat', 'eye_g', 'ear_r', 'neck_l']
        label_except_glasses = Image.open(path).convert('L')
        root, name = path.replace('\\', '/').split('all_parts_except_glasses/')
        idx = name.split('.')[0]
        subfolder = str(int(idx) // 2000)
        if os.path.exists(os.path.join(root, subfolder, idx + '_eye_g.png')):
            glasses = Image.open(os.path.join(root, subfolder, idx + '_eye_g.png')).convert('L')
        else:
            glasses = Image.fromarray(np.zeros(label_except_glasses.size, dtype=np.uint8))

        params = get_params(self.opt, label_except_glasses.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_except_glasses_tensor = transform_label(label_except_glasses) * 255.0
        glasses_tensor = transform_label(glasses)

        label_tensor = torch.zeros((3,256,256)).to(glasses_tensor)
        colormap_tensor = torch.from_numpy(colormap / 255.).float()
        for i in range(19):
            mask_i = label_except_glasses_tensor[0] == i
            label_tensor[:,mask_i] = colormap_tensor[i, :].unsqueeze(-1)
        # import pdb; pdb.set_trace()
        # label_tensor = torch.cat((label_except_glasses_tensor, glasses_tensor, torch.zeros_like(glasses_tensor)), dim=0)
        return label_tensor, params

    def imgpath_to_labelpath(self, path):
        root, name = path.split('CelebA-HQ-img/')
        label_path = os.path.join(root, 'CelebAMask-HQ-mask-anno', 'all_parts_except_glasses',
                                  name.split('.')[0].zfill(5) + '.png')
        return label_path


if __name__ == '__main__':
    from omegaconf import OmegaConf
    opt = OmegaConf.load(os.path.join(instruct_root,'configs/celebahq.yaml'))
    celebhq_dataset = CelebAHQDataset()
    opt.isTrain = False
    opt.phase = 'test'
    celebhq_dataset.initialize(opt, tokenizer=None, args=None)
    data_i = celebhq_dataset[0]
    print('dataset length: ', len(celebhq_dataset))
    import pdb; pdb.set_trace()
