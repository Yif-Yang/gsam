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
from data.cocos_image_folder import make_dataset
from utils.cocos_utils import create_ade20k_label_colormap
from data.cocos_base_dataset import get_params, get_transform

ade20k_colormap = create_ade20k_label_colormap()

class ADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        vis_label_tensor = torch.zeros((3,256,256)).to(label_tensor)
        colormap_tensor = torch.from_numpy(ade20k_colormap / 255.).float()
        # import pdb; pdb.set_trace()
        # colormap_tensor = torch.flip(colormap_tensor, dims=[1,])
        for i in range(len(colormap_tensor)):
            mask_i = label_tensor[0] == i
            vis_label_tensor[:, mask_i] = colormap_tensor[i, :].unsqueeze(-1)

        return vis_label_tensor, params1

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'
        subfolder = 'validation' if opt.phase == 'test' else 'training'
        cache = False if opt.phase == 'test' else True
        all_images = sorted(make_dataset(root + '/' + subfolder, recursive=True, read_cache=cache, write_cache=False))
        image_paths = []
        label_paths = []
        for p in all_images:
            if '_%s_' % phase not in p:
                continue
            if p.endswith('.jpg'):
                image_paths.append(p)
            elif p.endswith('.png'):
                label_paths.append(p)

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = '_test' if opt.phase == 'test' else ''
        with open(os.path.join(instruct_root, './data/ade20k_ref{}.txt'.format(extra))) as fd:
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
        train_test_folder = ('training', 'validation')
        return ref_dict, train_test_folder


if __name__ == '__main__':
    from omegaconf import OmegaConf
    opt = OmegaConf.load('configs/ade20k.yaml')
    ade_dataset = ADE20KDataset()
    opt.isTrain = False
    opt.phase = 'test'
    # opt.isTrain = True
    # opt.phase = 'train'
    ade_dataset.initialize(opt, tokenizer=None, args=None)
    data_i = ade_dataset[0]
    print('dataset length: ', len(ade_dataset))
    import pdb; pdb.set_trace()