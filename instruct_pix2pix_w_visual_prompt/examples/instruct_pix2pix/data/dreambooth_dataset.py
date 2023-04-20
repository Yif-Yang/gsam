import json
import sys
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import glob
import random
# from lavis.models import load_model_and_preprocess
import torch
import copy
from PIL import Image, ImageDraw
import numpy as np
from utils.dataset_utils import _generate_random_mask, _get_cutout_holes, _generate_inference_mask, _randomset, _shuffle
from utils.dataset_utils import imagenet_templates_small, imagenet_style_templates_small


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        learnable_property,
        placeholder_token,
        stochastic_attribute,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        resize=False,
        h_flip=False, # might result in ambiguity
        train_inpainting=False,
        every_num_instance=1,
        freeze_sample_order=False
    ):
        self.train_inpainting = train_inpainting
        self.every_num_instance = every_num_instance
        self.freeze_sample_order = freeze_sample_order
        self.size = size
        assert self.size %2 == 0, '{} size must be divided by 2'.format(self.size)
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        # import pdb; pdb.set_trace()
        if os.path.isdir(self.instance_images_path[0]):
            dirs = self.instance_images_path
            self.instance_images_path = [dir for dir in dirs if len(glob.glob(os.path.join(dir, '*.jpg')))>=4]

        self.num_instance_images = len(self.instance_images_path)

        self.placeholder_token = placeholder_token
        self.stochastic_attribute = (
            stochastic_attribute.split(",") if stochastic_attribute else []
        )

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )

        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        if resize:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.Lambda(lambda x: x),
                    transforms.ColorJitter(0.2, 0.1)
                    if color_jitter
                    else transforms.Lambda(lambda x: x),
                    transforms.RandomHorizontalFlip()
                    if h_flip
                    else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.Lambda(lambda x: x),
                    transforms.ColorJitter(0.2, 0.1)
                    if color_jitter
                    else transforms.Lambda(lambda x: x),
                    transforms.RandomHorizontalFlip()
                    if h_flip
                    else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        self.clip_transforms = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        # import pdb; pdb.set_trace()
        # _, vis_processors, _ = load_model_and_preprocess("blip2", "pretrain", device='cpu', is_eval=True)
        # self.vis_processors = vis_processors


    def __len__(self):
        return self._length // self.every_num_instance

    def put_together(self, dir_root):
        paths = sorted(glob.glob(os.path.join(dir_root, '*.jpg')))
        num_item = len(paths) // 2

        idxes = random.sample(list(range(num_item)), 2) if self.freeze_sample_order is False else [0, 1]
        sample_paths = list(reversed(paths[idxes[0]*2: idxes[0]*2+2])) +\
                       list(reversed(paths[idxes[1]*2: idxes[1]*2+2]))

        base_size = self.size // 2
        pil_concat = Image.new("RGB", (base_size*2, base_size*2))
        pil_all = []
        for i, p in enumerate(sample_paths):
            pil_i = Image.open(p)
            pil_i = pil_i.resize((base_size, base_size))
            pil_concat.paste(pil_i, ((i+1)%2*base_size, i//2*base_size))
            pil_all.append(pil_i)

        pil_winput_concat = copy.deepcopy(pil_concat)
        pil_winput_concat.paste(pil_all[-1], (base_size, base_size))

        return pil_concat, pil_winput_concat, pil_all


    def __getitem__(self, idx):
        index = (idx * self.every_num_instance) % self.num_instance_images
                # self._length
        example = {}
        pil_all = None
        if os.path.isfile(self.instance_images_path[index % self.num_instance_images]):
            instance_image = Image.open(
                self.instance_images_path[index % self.num_instance_images]
            )
            instance_image = instance_image.resize((self.size,self.size), Image.NEAREST)
        else:
            instance_image, instance_image_winput, pil_all = self.put_together(
                self.instance_images_path[index % self.num_instance_images])

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        transform_seed = random.randint(0, 999999999)
        torch.random.manual_seed(transform_seed)
        random.seed(transform_seed)

        example["instance_images"] = self.image_transforms(instance_image)
        if pil_all is not None:
            ## imitate the original instruct-pix2pix settings, input reference image
            example["instance_images_winput"] = self.image_transforms(instance_image_winput)
            example["dummy_masks"] = torch.zeros_like(example["instance_images_winput"][:1])

        if self.train_inpainting:
            (
                example["instance_masks"],
                example["instance_masked_images"]
            ) = _generate_random_mask(example["instance_images"])

        example['pil_instance_images'] = instance_image
        example['pil_instance_masks'] = Image.fromarray(example['instance_masks'].squeeze(0).cpu().numpy().astype(np.uint8)*255)

        if pil_all is not None:
            example['pil_instance_images_winput'] = instance_image_winput
            example['pil_dummy_masks'] = Image.fromarray(example['dummy_masks'].squeeze(0).cpu().numpy().astype(np.uint8)*255)
            example['clip_tensor_all'] = [self.clip_transforms(cur_i) for cur_i in pil_all]
            example['null_clip_tensor_all'] = [self.clip_transforms(Image.new('RGB',(256,256),color=(0,0,0))) for cur_i in pil_all]

        text = random.choice(self.templates).format(
            ", ".join(
                [self.placeholder_token]
                + _shuffle(_randomset(self.stochastic_attribute))
            )
        )
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids


        # if pil_all is not None:
        #     img0 = self.vis_processors["eval"](pil_all[0]).unsqueeze(0)
        #     img1 = self.vis_processors["eval"](pil_all[1]).unsqueeze(0)
        #     example["examples"] = torch.cat([img0, img1], dim=0)

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        meta_fpath = os.path.join(self.instance_images_path[index % self.num_instance_images], 'prompt.json')
        with open(meta_fpath, 'r') as f:
            prompt_dict = json.load(f)
        example['prompt_dict'] = prompt_dict
        example['edit_ids'] = self.tokenizer(
            prompt_dict['edit'], padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length,
        ).input_ids
        example['null_edit_ids'] = self.tokenizer(
            "", padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example

class DreamBoothCollateFunc:
    def __init__(self, train_inpainting, tokenizer):
        super(DreamBoothCollateFunc, self).__init__()
        self.train_inpainting = train_inpainting
        self.tokenizer = tokenizer

    def __call__(self, examples):
        train_inpainting = self.train_inpainting
        tokenizer = self.tokenizer

        input_ids = [example["instance_prompt_ids"] for example in examples]
        edit_ids = [example["edit_ids"] for example in examples]
        null_edit_ids = [example["null_edit_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        if "examples" in examples[0]:
            example_values = [example["examples"] for example in examples]
            example_values = torch.stack(example_values)

        if "clip_tensor_all" in examples[0]:
            example_clipnorm_values = []
            for jj in range(len(examples[0]['clip_tensor_all'])):
                example_clipnorm_value = torch.stack([example['clip_tensor_all'][jj] for example in examples])
                example_clipnorm_values.append(example_clipnorm_value)
            example_null_clipnorm_values = []
            for jj in range(len(examples[0]['null_clip_tensor_all'])):
                example_null_clipnorm_value = torch.stack([example['null_clip_tensor_all'][jj] for example in examples])
                example_null_clipnorm_values.append(example_null_clipnorm_value)
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        # if args.with_prior_preservation:
        #     input_ids += [example["class_prompt_ids"] for example in examples]
        #     pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        dummy_mask_values = [example["dummy_masks"] for example in examples]
        winput_image_values = [
            example["instance_images_winput"] for example in examples]

        dummy_mask_values = (
            torch.stack(dummy_mask_values).to(memory_format=torch.contiguous_format).float()
        )
        winput_image_values = (
            torch.stack(winput_image_values).to(memory_format=torch.contiguous_format).float()
        )

        if train_inpainting:
            mask_values = [example["instance_masks"] for example in examples]
            masked_image_values = [
                example["instance_masked_images"] for example in examples]

            if examples[0].get("class_prompt_ids", None) is not None:
                mask_values += [example["class_masks"] for example in examples]
                masked_image_values += [
                    example["class_masked_images"] for example in examples
                ]
            mask_values = (
                torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
            )
            masked_image_values = (
                torch.stack(masked_image_values)
                .to(memory_format=torch.contiguous_format)
                .float()
            )

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        edit_ids = tokenizer.pad(
            {"input_ids": edit_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        null_edit_ids = tokenizer.pad(
            {"input_ids": null_edit_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        base_image = [example['pil_instance_images'] for example in examples]
        mask_image = [example['pil_instance_masks'] for example in examples]
        edit_prompt = [example['prompt_dict']['edit'] for example in examples]

        base_image_winput = [example['pil_instance_images_winput'] for example in examples]
        dummy_mask_image = [example['pil_dummy_masks'] for example in examples]


        batch = {
            "input_ids": input_ids,
            "edit_ids": edit_ids,
            "null_edit_ids": null_edit_ids,
            "pixel_values": pixel_values,
            "base_image": base_image,
            "mask_image": mask_image,
            "base_image_winput": base_image_winput,
            "dummy_mask_image": dummy_mask_image,
            "edit_prompt": edit_prompt,
            "dummy_mask_values": dummy_mask_values,
            "winput_image_values": winput_image_values,
        }

        if "clip_tensor_all" in examples[0]:
            batch["clipnorm_values"] = example_clipnorm_values

        if 'null_clip_tensor_all' in examples[0]:
            batch['null_clipnorm_values'] = example_null_clipnorm_values

        if 'examples' in examples[0]:
            batch["examples"] = example_values

        if train_inpainting:
            batch['mask_values'] = mask_values
            batch['masked_image_values'] = masked_image_values
            if examples[0].get("mask", None) is not None:
                batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch
