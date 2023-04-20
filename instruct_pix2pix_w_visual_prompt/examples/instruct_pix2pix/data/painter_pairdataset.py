# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random
import sys

from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
instruct_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(instruct_root)
import data.painter_pair_transforms as pair_transforms
from utils.dataset_utils import imagenet_templates_small, imagenet_style_templates_small
from data.cocos_pix2pix_dataset import Pix2pixDataset


class PairDataset(VisionDataset, Pix2pixDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            json_path_list: Optional[list] = None,
            transform: Optional[Callable] = None,
            transform2: Optional[Callable] = None,
            transform3: Optional[Callable] = None,
            transform_seccrop: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            masked_position_generator: Optional[Callable] = None,
            use_two_pairs: bool = True,
            half_mask_ratio: float = 0.,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.pairs = []
        self.weights = []
        self.use_two_pairs = use_two_pairs
        self.root = root

    def initialize(self, opt, tokenizer=None, args=None):
        root = self.root
        val_json_path_list = [os.path.join(root, 'denoise/denoise_ssid_val.json'),
                              os.path.join(root, 'derain/derain_test_rain100h.json'),
                              os.path.join(root, 'light_enhance/enhance_lol_val.json')]
        train_json_path_list = [os.path.join(root, 'denoise/denoise_ssid_train.json'),
                              os.path.join(root, 'derain/derain_train.json'),
                              os.path.join(root, 'light_enhance/enhance_lol_train.json')]
        json_path_list = val_json_path_list if opt.phase == 'test' else train_json_path_list
        use_two_pairs = self.use_two_pairs
        # type_weight_list = [0.1, 0.2, 0.15, 0.25, 0.2, 0.15, 0.05, 0.05]
        type_weight_list = [0.15, 0.05, 0.05]
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            # print('cur_name: ', cur_num, 'json_path: ', json_path)
            self.weights.extend([type_weight_list[idx] * 1. / cur_num] * cur_num)
            # print(json_path, type_weight_list[idx])
        self.use_two_pairs = use_two_pairs
        if self.use_two_pairs:
            self.pair_type_dict = {}
            for idx, pair in enumerate(self.pairs):
                if "type" in pair:
                    if pair["type"] not in self.pair_type_dict:
                        self.pair_type_dict[pair["type"]] = [idx]
                    else:
                        self.pair_type_dict[pair["type"]].append(idx)
            for t in self.pair_type_dict:
                print(t, len(self.pair_type_dict[t]))

        self.opt = opt
        self.tokenizer = tokenizer
        if args is not None:
            placeholder_token, stochastic_attribute, learnable_property = args.placeholder_token, args.stochastic_attribute, args.learnable_property
        else:
            placeholder_token, stochastic_attribute, learnable_property = None, None, None
        self.task_description = opt.task_description

        self.placeholder_token = placeholder_token
        self.stochastic_attribute = (
            stochastic_attribute.split(",") if stochastic_attribute else []
        )
        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )

        self.clip_transforms = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.pil_transforms = transforms.ToPILImage()

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        transform_mean, transform_std = [0.5], [0.5]

        transform_train = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(opt.input_size[1], scale=(opt.min_random_scale, 1.0), interpolation=3),
            # 3 is bicubic
            # pair_transforms.RandomApply([
            #     pair_transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            # ], p=0.8),
            pair_transforms.RandomHorizontalFlip(),
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=transform_mean, std=transform_std)])
        transform_val = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(opt.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=transform_mean, std=transform_std)])
        transform_train2 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(opt.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=transform_mean, std=transform_std)])
        transform_train3 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(opt.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=transform_mean, std=transform_std)])
        transform_train_seccrop = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop((opt.input_size[0], opt.input_size[1]),
                                              scale=(opt.min_random_scale, 1.0), ratio=(0.3, 0.7), interpolation=3),
            # 3 is bicubic
        ])
        # self.transform_train, self.transform_val, self.transform_train2, self.transform_train3, self.transform_train_seccrop = \
        #     transform_train, transform_val, transform_train2, transform_train3, transform_train_seccrop
        target_transform = None
        if opt.phase == 'test':
            transform = transform_val
            transform2, transform3 = None, None
            transform_seccrop = None
        else:
            transform = transform_train
            transform2, transform3 = transform_train2, transform_train3
            transform_seccrop = transform_train_seccrop
        self.transforms = PairStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.transforms_seccrop = PairStandardTransform(transform_seccrop,
                                                        target_transform) if transform_seccrop is not None else None

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def _combine_images(self, image, image2, interpolation='bicubic'):
        # image under image2
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])

        # decide mode for interpolation
        pair_type = pair['type']
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'

        # no aug for instance segmentation
        if "inst" in pair['type'] and self.transforms2 is not None:
            cur_transforms = self.transforms2
        elif "pose" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms

        # import pdb; pdb.set_trace()
        image, target = cur_transforms(image, target, interpolation1, interpolation2)

        if self.use_two_pairs:
            pair_type = pair['type']
            # sample the second pair belonging to the same type
            pair2_index = random.choice(self.pair_type_dict[pair_type])
            pair2 = self.pairs[pair2_index]
            image2 = self._load_image(pair2['image_path'])
            target2 = self._load_image(pair2['target_path'])
            assert pair2['type'] == pair_type
            image2, target2 = cur_transforms(image2, target2, interpolation1, interpolation2)
            image = self._combine_images(image, image2, interpolation1)
            target = self._combine_images(target, target2, interpolation2)

        # use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
        if (self.transforms_seccrop is None) or ("inst" in pair['type']) or ("pose" in pair['type']):
            pass
        else:
            image, target = self.transforms_seccrop(image, target, interpolation1, interpolation2)

        # import pdb; pdb.set_trace()
        input_dict = self.pair2input_dict(image*0.5+0.5, target*0.5+0.5)
        res_dict = self.postprocess(input_dict)
        return res_dict

    def pair2input_dict(self, example, qa):
        input_dict = {}
        h = qa.shape[1]
        input_dict['label_ref'] = example[:, :h//2]
        input_dict['ref'] = qa[:, :h//2]

        input_dict['label'] = example[:, h//2:]
        input_dict['image'] = qa[:, h//2:]
        return input_dict

    def __len__(self) -> int:
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target


if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = OmegaConf.load(os.path.join(instruct_root, 'configs/lowlevel_pair.yaml'))
    # import pdb; pdb.set_trace()
    # simple augmentation

    root = '/data/yashengsun/local_storage/vision_in_context_caoyue/'
    val_json_path_list = [os.path.join(root, 'denoise/denoise_ssid_val.json'),
                      os.path.join(root, 'derain/derain_test_rain100h.json'),
                      os.path.join(root, 'light_enhance/enhance_lol_val.json')]
    val_pair_dataset = PairDataset(root, json_path_list=val_json_path_list,
                                   transform=transform_val)

    # train_json_path_list = [os.path.join(root, 'denoise/denoise_ssid_train.json'),
    #                       os.path.join(root, 'derain/derain_train.json'),
    #                       os.path.join(root, 'light_enhance/enhance_lol_train.json')]
    # val_pair_dataset = PairDataset(root, json_path_list=train_json_path_list,
    #                                      transform=transform_train,
    #                                      transform2=transform_train2,
    #                                      transform3=transform_train3,
    #                                      transform_seccrop=transform_train_seccrop,
    #                                      )


    batch = val_pair_dataset[0]
    import pdb; pdb.set_trace()
