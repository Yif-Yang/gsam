import os
import sys
import cv2
import torch
import glob
import numpy as np
import math
import random
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
instruct_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(instruct_root)
from data.cocos_pix2pix_dataset import Pix2pixDataset
from data.cocos_base_dataset import get_params, get_transform


class StyleClipDataset(Pix2pixDataset):
    def initialize(self, opt, tokenizer=None, args=None):
        self.initialize_common_utils(opt, tokenizer=tokenizer, args=args)
        incontext_tasks = os.listdir(os.path.join(opt.dataroot, opt.phase))
        incontext_task_roots = [os.path.join(opt.dataroot, opt.phase, task_name) for task_name in incontext_tasks]
        self.paths = []
        self.incontext_paths_dict = {}
        for incontext_dir, task_name in zip(incontext_task_roots, incontext_tasks):
            incontext_paths = sorted(glob.glob(os.path.join(incontext_dir, '*.jpg')))
            self.paths.extend(incontext_paths)
            self.incontext_paths_dict[task_name] = incontext_paths

        transform_mean, transform_std = [0.5], [0.5]
        self.dataset_size = len(self.paths)

        ## TODO: add approriate data augmentation
        self.transform_train = transforms.Compose([
            transforms.Resize(size=(self.opt.load_size, self.opt.load_size*2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std)
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize(size=(self.opt.load_size, self.opt.load_size*2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std)
        ])

    def get_ref_path(self, qa_path):
        task_name = os.path.basename(os.path.dirname(qa_path))
        task_instance_num = len(self.incontext_paths_dict[task_name])
        qa_idx = int(os.path.basename(qa_path).split('.')[0])
        offset = 888 if self.opt.phase == 'test' else random.randint(0, task_instance_num)
        ref_idx = (qa_idx + offset) % task_instance_num
        templ = '{:05d}.jpg'.format(ref_idx)
        ref_path = os.path.join(os.path.dirname(qa_path), templ.format(ref_idx))
        return ref_path

    def __getitem__(self, index):
        qa_path = self.paths[index]
        example_path = self.get_ref_path(qa_path)
        qa_pil = Image.open(qa_path)
        example_pil = Image.open(example_path)

        transform_image = self.transform_train if self.opt.phase == 'train' else self.transform_val
        qa_tensor = transform_image(qa_pil)
        example_tensor = transform_image(example_pil)
        # import pdb; pdb.set_trace()
        w = qa_tensor.shape[2]//2
        image_tensor = example_tensor[:, :, w:]
        label_tensor = example_tensor[:, :, :w]
        ref_tensor = qa_tensor[:, :, w:]
        label_ref_tensor = qa_tensor[:, :, :w]

        image_path = 'dummy_path'
        self_ref_flag = torch.zeros_like(ref_tensor)

        input_dict = {'label': (label_tensor+1)*0.5,
                      'image': (image_tensor+1)*0.5,
                      'path': image_path,
                      'self_ref': self_ref_flag,
                      'ref': (ref_tensor+1)*0.5,
                      'label_ref': (label_ref_tensor+1)*0.5
                      }
        # import pdb; pdb.set_trace()
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict


if __name__ == '__main__':
    from omegaconf import OmegaConf
    opt = OmegaConf.load('configs/styleclip.yaml')
    styleclip_dataset = StyleClipDataset()
    # opt.isTrain = False
    # opt.phase = 'test'
    opt.isTrain = True
    opt.phase = 'train'
    styleclip_dataset.initialize(opt, tokenizer=None, args=None)
    data_i = styleclip_dataset[0]
    print('dataset length: ', len(styleclip_dataset))

    import pdb; pdb.set_trace()
