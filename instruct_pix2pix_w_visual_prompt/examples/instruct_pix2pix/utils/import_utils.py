import os
import sys
import torch
from torch.utils.data import ConcatDataset


diffuser_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
src_root = os.path.join(diffuser_root, 'src')
print('src_root: ', src_root)
sys.path.insert(0, src_root)
example_root = os.path.join(diffuser_root, 'examples')
print('example_root: ', example_root)
sys.path.insert(0, example_root)
instruction_root = os.path.join(example_root, 'instruct_pix2pix')
print('intruction_root: ', instruction_root)
sys.path.insert(0, instruction_root)

from data.dreambooth_dataset import DreamBoothDataset, DreamBoothCollateFunc
from omegaconf import OmegaConf
from data.cocos_celebahq_dataset import CelebAHQDataset
from data.cocos_celebahqedge_dataset import CelebAHQEdgeDataset
from data.cocos_deepfashion_dataset import DeepFashionDataset
from data.cocos_ade20k_dataset import ADE20KDataset
from data.painter_pairdataset import PairDataset as LowLevelPairDataset
from data.synthesized_styleclip_dataset import StyleClipDataset


def get_task(args, logger):
   if args.task_name == 'instruct_pix2pix':
      from tasks.instruct_pix2pix_task import InstructPix2PixTask
      task = InstructPix2PixTask(args, logger)
   elif args.task_name == 'instruct_pix2pix_inpainting':
      from tasks.instruct_pix2pix_inpainting_task import InstructPix2PixInpaintingTask
      task = InstructPix2PixInpaintingTask(args, logger)
   else:
      raise ValueError
   return task


def get_dataloader(args, tokenizer, split_names='train,test'):
   splits = split_names.split(',')
   # DataLoaders creation:
   collate_fn = DreamBoothCollateFunc(args.train_inpainting, tokenizer)
   if args.dataset_names is None:
      train_dataset = None
      if 'train' in splits:
         train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            placeholder_token=args.placeholder_token,
            stochastic_attribute=args.stochastic_attribute,
            learnable_property=args.learnable_property,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            color_jitter=args.color_jitter,
            resize=args.resize,
            train_inpainting=args.train_inpainting
         )
         train_dataset._length = int(5e8)

      val_dataset = None
      if 'test' in splits:
         val_instance_data_dir = args.instance_data_dir.replace('train', 'val')
         val_dataset = DreamBoothDataset(
            instance_data_root=val_instance_data_dir,
            placeholder_token=args.placeholder_token,
            stochastic_attribute=args.stochastic_attribute,
            learnable_property=args.learnable_property,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=False,
            color_jitter=False,
            resize=args.resize,
            train_inpainting=args.train_inpainting,
            every_num_instance=args.val_every_num_instance
         )
   else:
      train_datasets, val_datasets = [], []
      if 'instruct' in args.dataset_names:
         train_instruct_dataset = None
         if 'train' in splits:
            train_instruct_dataset = DreamBoothDataset(
               instance_data_root=args.instance_data_dir,
               placeholder_token=args.placeholder_token,
               stochastic_attribute=args.stochastic_attribute,
               learnable_property=args.learnable_property,
               class_data_root=args.class_data_dir if args.with_prior_preservation else None,
               class_prompt=args.class_prompt,
               tokenizer=tokenizer,
               size=args.resolution,
               center_crop=args.center_crop,
               color_jitter=args.color_jitter,
               resize=args.resize,
               train_inpainting=args.train_inpainting
            )
            train_datasets.append(train_instruct_dataset)

         val_instruct_dataset = None
         if 'test' in splits:
            val_instance_data_dir = args.instance_data_dir.replace('train', 'val')
            val_instruct_dataset = DreamBoothDataset(
               instance_data_root=val_instance_data_dir,
               placeholder_token=args.placeholder_token,
               stochastic_attribute=args.stochastic_attribute,
               learnable_property=args.learnable_property,
               class_data_root=args.class_data_dir if args.with_prior_preservation else None,
               class_prompt=args.class_prompt,
               tokenizer=tokenizer,
               size=args.resolution,
               center_crop=False,
               color_jitter=False,
               resize=args.resize,
               train_inpainting=args.train_inpainting,
               every_num_instance=args.val_every_num_instance
            )
            val_datasets.append(val_instruct_dataset)

      if 'celebaedge' in args.dataset_names:
         train_celebahqedge_dataset = None
         if 'train' in splits:
            train_opt_celebahqedge = OmegaConf.load(os.path.join(instruction_root, 'configs/celebahqedge.yaml'))
            train_celebahqedge_dataset = CelebAHQEdgeDataset()
            train_celebahqedge_dataset.initialize(train_opt_celebahqedge, tokenizer, args)
            train_datasets.append(train_celebahqedge_dataset)

         val_celebahqedge_dataset = None
         if 'test' in splits:
            val_opt_celebahqedge = OmegaConf.load(os.path.join(instruction_root,'configs/celebahqedge.yaml'))
            val_opt_celebahqedge.phase, val_opt_celebahqedge.isTrain = 'test', False
            val_celebahqedge_dataset = CelebAHQEdgeDataset()
            val_celebahqedge_dataset.initialize(val_opt_celebahqedge, tokenizer, args)
            val_datasets.append(val_celebahqedge_dataset)

      if 'ade20k' in args.dataset_names:
         train_ade20k_dataset = None
         if 'train' in splits:
            train_opt_ade20k = OmegaConf.load(os.path.join(instruction_root, 'configs/ade20k.yaml'))
            train_ade20k_dataset = ADE20KDataset()
            train_ade20k_dataset.initialize(train_opt_ade20k, tokenizer, args)
            train_datasets.append(train_ade20k_dataset)

         val_ade20k_dataset = None
         if 'test' in splits:
            val_opt_ade20k = OmegaConf.load(os.path.join(instruction_root,'configs/ade20k.yaml'))
            val_opt_ade20k.phase, val_opt_ade20k.isTrain = 'test', False
            val_ade20k_dataset = ADE20KDataset()
            val_ade20k_dataset.initialize(val_opt_ade20k, tokenizer, args)
            val_datasets.append(val_ade20k_dataset)

      if 'celebahq' in args.dataset_names:
         train_celebahqedge_dataset = None
         if 'train' in splits:
            train_opt_celebahq = OmegaConf.load(os.path.join(instruction_root,'configs/celebahq.yaml'))
            train_celebahq_dataset = CelebAHQDataset()
            train_celebahq_dataset.initialize(train_opt_celebahq, tokenizer, args)
            train_datasets.append(train_celebahq_dataset)

         val_celebahqedge_dataset = None
         if 'test' in splits:
            val_opt_celebahq = OmegaConf.load(os.path.join(instruction_root,'configs/celebahq.yaml'))
            val_opt_celebahq.phase, val_opt_celebahq.isTrain = 'test', False
            val_celebahq_dataset = CelebAHQDataset()
            val_celebahq_dataset.initialize(val_opt_celebahq, tokenizer, args)
            val_datasets.append(val_celebahq_dataset)

      if 'deepfashion' in args.dataset_names:
         train_deepfashion_dataset = None
         if 'train' in splits:
            train_opt_deepfashion = OmegaConf.load(os.path.join(instruction_root,'configs/deepfashion.yaml'))
            train_deepfashion_dataset = DeepFashionDataset()
            train_deepfashion_dataset.initialize(train_opt_deepfashion, tokenizer, args)
            train_datasets.append(train_deepfashion_dataset)

         val_deepfashion_dataset = None
         if 'test' in splits:
            val_opt_deepfashion = OmegaConf.load(os.path.join(instruction_root,'configs/deepfashion.yaml'))
            val_opt_deepfashion.phase, val_opt_deepfashion.isTrain = 'test', False
            val_deepfashion_dataset = DeepFashionDataset()
            val_deepfashion_dataset.initialize(val_opt_deepfashion, tokenizer, args)
            val_datasets.append(val_deepfashion_dataset)

      if 'lowlevelpair' in args.dataset_names:
         train_lowlevelpair_dataset = None
         if 'train' in splits:
            train_opt_lowlevelpair = OmegaConf.load(os.path.join(instruction_root, 'configs/lowlevel_pair.yaml'))
            train_lowlevelpair_dataset = LowLevelPairDataset(train_opt_lowlevelpair.dataroot)
            train_lowlevelpair_dataset.initialize(train_opt_lowlevelpair, tokenizer, args)
            train_datasets.append(train_lowlevelpair_dataset)

         val_lowlevelpair_dataset = None
         if 'test' in splits:
            val_opt_lowlevelpair = OmegaConf.load(os.path.join(instruction_root, 'configs/lowlevel_pair.yaml'))
            val_opt_lowlevelpair.phase, val_opt_lowlevelpair.isTrain = 'test', False
            val_lowlevelpair_dataset = LowLevelPairDataset(val_opt_lowlevelpair.dataroot)
            val_lowlevelpair_dataset.initialize(val_opt_lowlevelpair, tokenizer, args)
            val_datasets.append(val_lowlevelpair_dataset)

      if 'styleclip' in args.dataset_names:
         train_styleclip_dataset = None
         if 'train' in splits:
            train_opt_styleclip = OmegaConf.load(os.path.join(instruction_root, 'configs/styleclip.yaml'))
            train_styleclip_dataset = StyleClipDataset()
            train_styleclip_dataset.initialize(train_opt_styleclip, tokenizer, args)
            train_datasets.append(train_styleclip_dataset)

         val_styleclip_dataset = None
         if 'test' in splits:
            val_opt_styleclip = OmegaConf.load(os.path.join(instruction_root, 'configs/styleclip.yaml'))
            val_opt_styleclip.phase, val_opt_styleclip.isTrain = 'test', False
            val_styleclip_dataset = StyleClipDataset()
            val_styleclip_dataset.initialize(val_opt_styleclip, tokenizer, args)
            val_datasets.append(val_styleclip_dataset)

      train_dataset = ConcatDataset(train_datasets) if len(train_datasets) else None
      val_dataset = ConcatDataset(val_datasets) if len(val_datasets) else None

   train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
   ) if train_dataset is not None else None

   is_shuffle = True if 'test' in splits and 'train' in splits else False
   val_dataloader = torch.utils.data.DataLoader(
      val_dataset, batch_size=1, shuffle=is_shuffle, collate_fn=collate_fn
   ) if val_dataset is not None else None
   # import pdb; pdb.set_trace()
   return train_dataloader, val_dataloader
