# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data.cocos_base_dataset import BaseDataset, get_params, get_transform
import torch
import torchvision.transforms as transforms
from PIL import Image
import utils.cocos_utils as util
import os
import random
import numpy as np
from utils.dataset_utils import _generate_random_mask, _get_cutout_holes, _generate_inference_mask, _randomset, _shuffle
from utils.dataset_utils import imagenet_templates_small, imagenet_style_templates_small


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize_common_utils(self, opt, tokenizer, args):
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

    def initialize(self, opt, tokenizer=None, args=None):
        self.initialize_common_utils(opt, tokenizer=tokenizer, args=args)

        label_paths, image_paths = self.get_paths(opt)
        # import pdb; pdb.set_trace()
        if opt.dataset_mode != 'celebahq' and opt.dataset_mode != 'deepfashion':
            util.natural_sort(label_paths)
            util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (
                    path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

        self.real_reference_probability = 1 if opt.phase == 'test' else opt.real_reference_probability
        self.hard_reference_probability = 0 if opt.phase == 'test' else opt.hard_reference_probability
        self.ref_dict, self.train_test_folder = self.get_ref(opt)

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return label_tensor, params1

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label_tensor, params1 = self.get_label_tensor(label_path)

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)

        ref_tensor = 0
        label_ref_tensor = 0

        random_p = random.random()
        if random_p < self.real_reference_probability or self.opt.phase == 'test':
            key = image_path.replace('\\', '/').split('img_highres_rgb/')[
                -1] if self.opt.dataset_mode == 'deepfashion' else os.path.basename(image_path)
            val = self.ref_dict[key]
            if random_p < self.hard_reference_probability:
                path_ref = val[1]  # hard reference
            else:
                path_ref = val[0]  # easy reference
            if self.opt.dataset_mode == 'deepfashion':
                path_ref = os.path.join(self.opt.dataroot, path_ref)
            else:
                path_ref = os.path.dirname(image_path).replace(self.train_test_folder[1],
                                                               self.train_test_folder[0]) + '/' + path_ref

            image_ref = Image.open(path_ref).convert('RGB')
            if self.opt.dataset_mode != 'deepfashion':
                path_ref_label = path_ref.replace('.jpg', '.png')
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            else:
                path_ref_label = self.imgpath_to_labelpath(path_ref)

            label_ref_tensor, params = self.get_label_tensor(path_ref_label)
            transform_image = get_transform(self.opt, params)
            ref_tensor = transform_image(image_ref)
            # ref_tensor = self.reference_transform(image_ref)
            self_ref_flag = torch.zeros_like(ref_tensor)
        else:
            pair = False
            if self.opt.dataset_mode == 'deepfashion' and self.opt.video_like:
                # if self.opt.hdfs:
                #     key = image_path.split('DeepFashion.zip@/')[-1]
                # else:
                #     key = image_path.split('DeepFashion/')[-1]
                key = image_path.replace('\\', '/').split('img_highres_rgb/')[-1]
                val = self.ref_dict[key]
                ref_name = val[0]
                key_name = key
                if os.path.dirname(ref_name) == os.path.dirname(key_name) and os.path.basename(ref_name).split('_')[
                    0] == os.path.basename(key_name).split('_')[0]:
                    path_ref = os.path.join(self.opt.dataroot, ref_name)
                    image_ref = Image.open(path_ref).convert('RGB')
                    label_ref_path = self.imgpath_to_labelpath(path_ref)
                    label_ref_tensor, params = self.get_label_tensor(label_ref_path)
                    transform_image = get_transform(self.opt, params)
                    ref_tensor = transform_image(image_ref)
                    pair = True
            if not pair:
                label_ref_tensor, params = self.get_label_tensor(label_path)
                transform_image = get_transform(self.opt, params)
                ref_tensor = transform_image(image)
            # ref_tensor = self.reference_transform(image)
            self_ref_flag = torch.ones_like(ref_tensor)

        input_dict = {'label': label_tensor,
                      'image': (image_tensor+1)*0.5,
                      'path': image_path,
                      'self_ref': self_ref_flag,
                      'ref': (ref_tensor+1)*0.5,
                      'label_ref': label_ref_tensor
                      }
        # import pdb; pdb.set_trace()
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def tensor2pils(self, input_dict, is_reverse_casual):
        # note that the reverse casual logic need to comply with instruct dataset
        if is_reverse_casual is False:
            A, A_, B, B_ = self.pil_transforms(input_dict['ref']), \
                           self.pil_transforms(input_dict['label_ref']), \
                           self.pil_transforms(input_dict['image']), \
                           self.pil_transforms(input_dict['label'])
        if is_reverse_casual is True:
            A_, A, B_, B = self.pil_transforms(input_dict['ref']), \
                           self.pil_transforms(input_dict['label_ref']), \
                           self.pil_transforms(input_dict['image']), \
                           self.pil_transforms(input_dict['label'])

        pil_all = [A, A_, B, B_]
        pil_winput_concat = Image.new("RGB", (512, 512))
        pil_winput_concat.paste(A, (0, 0))
        pil_winput_concat.paste(A_, (256, 0))
        pil_winput_concat.paste(B_, (0, 256))
        pil_winput_concat.paste(B_, (256, 256))
        return pil_all, pil_winput_concat

    def put_together(self, input_dict, reverse_casual_prob=0.0):
        is_reverse_casual = True if random.random() < reverse_casual_prob else False
        if is_reverse_casual is False:
            example = torch.cat([input_dict['label_ref'], input_dict['ref']],dim=2)
            qa = torch.cat([input_dict['label'], input_dict['image']],dim=2)
        else:
            example = torch.cat([input_dict['ref'], input_dict['label_ref']], dim=2)
            qa = torch.cat([input_dict['image'], input_dict['label']], dim=2)
        square_grid = torch.cat([example, qa], dim=1)
        square_grid_norm = transforms.Normalize([0.5], [0.5])(square_grid)
        self.task_description = self.opt.inverse_task_description if is_reverse_casual else self.task_description
        pil_all, pil_winput_concat = self.tensor2pils(input_dict, is_reverse_casual)
        return square_grid, square_grid_norm, pil_winput_concat, pil_all


    def postprocess(self, input_dict):
        # again, make it consistent with instruction dataset implementation
        square_grid, square_grid_norm, pil_winput_concat, pil_all = self.put_together(input_dict)
        input_dict["instance_images"] = square_grid_norm
        input_dict["instance_masks"], input_dict["instance_masked_images"] = _generate_random_mask(
            input_dict["instance_images"])
        transform_image = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
        input_dict["instance_images_winput"] = transform_image(pil_winput_concat)
        input_dict["pil_instance_images_winput"] = pil_winput_concat
        instance_image = self.pil_transforms(square_grid)
        input_dict["pil_instance_images"] = instance_image
        input_dict["pil_instance_masks"] = Image.fromarray(input_dict["instance_masks"].squeeze(0).cpu().numpy().astype(np.uint8)*255)

        input_dict["dummy_masks"] = torch.zeros_like(square_grid[:1])
        input_dict['pil_dummy_masks'] = Image.fromarray(
            input_dict['dummy_masks'].squeeze(0).cpu().numpy().astype(np.uint8) * 255)
        input_dict['clip_tensor_all'] = [self.clip_transforms(cur_i) for cur_i in pil_all]
        input_dict['null_clip_tensor_all'] = [self.clip_transforms(Image.new('RGB', (256, 256), color=(0, 0, 0))) for _ in pil_all]

        text = random.choice(self.templates).format(
            ", ".join(
                [self.placeholder_token]
                + _shuffle(_randomset(self.stochastic_attribute))
            )
        )
        input_dict["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        prompt_dict = {'edit': self.task_description}
        input_dict['prompt_dict'] = prompt_dict
        input_dict['edit_ids'] = self.tokenizer(
            prompt_dict['edit'], padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length,
        ).input_ids
        input_dict['null_edit_ids'] = self.tokenizer(
            "", padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length,
        ).input_ids

        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path
