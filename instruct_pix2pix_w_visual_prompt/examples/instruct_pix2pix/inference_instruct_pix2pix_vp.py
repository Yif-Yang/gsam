import utils.import_utils

import os
import torch
import logging

import transformers
import diffusers
from data.dreambooth_dataset import DreamBoothDataset, DreamBoothCollateFunc

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger

from utils.import_utils import get_task, get_dataloader
from utils.torch_utils import load_state_dict
from options.arguments_parser import parse_args

logger = get_logger(__name__, log_level="INFO")


def main():
   args = parse_args()
   logging_dir = os.path.join(args.output_dir, args.logging_dir)
   accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
   accelerator = Accelerator(
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      mixed_precision=args.mixed_precision,
      log_with=args.report_to,
      logging_dir=logging_dir,
      project_config=accelerator_project_config,
   )

   generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

   logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
   )
   logger.info(accelerator.state, main_process_only=False)
   if accelerator.is_local_main_process:
      transformers.utils.logging.set_verbosity_warning()
      diffusers.utils.logging.set_verbosity_info()
   else:
      transformers.utils.logging.set_verbosity_error()
      diffusers.utils.logging.set_verbosity_error()

   task = get_task(args, logger)
   noise_scheduler, tokenizer, text_encoder, vae, unet, vision_encoder = \
      task.noise_scheduler, task.tokenizer, task.text_encoder, task.vae, task.unet, task.vision_encoder

   train_dataloader, test_dataloader = get_dataloader(args, tokenizer=tokenizer, split_names='test')
   import pdb; pdb.set_trace()
   weight_dtype = torch.float32
   if accelerator.mixed_precision == "fp16":
      weight_dtype = torch.float16
   elif accelerator.mixed_precision == "bf16":
      weight_dtype = torch.bfloat16

   accelerator.trackers = []

   load_state_dict(unet, args.unet_checkpoint_path)
   unet = accelerator.prepare(unet)
   if vision_encoder is not None:
      load_state_dict(vision_encoder, args.vision_enc_checkpoint_path)
      vision_encoder = accelerator.prepare(vision_encoder)

   epoch = 0
   unet.eval()
   pipeline = None

   for step, batch in enumerate(test_dataloader):
      # import pdb; pdb.set_trace()
      pipeline = task.inference(vae, unet, text_encoder, generator, accelerator, weight_dtype, batch, logger,
                           epoch, global_step=step, args=args, vision_encoder=vision_encoder, pipeline=pipeline)


if __name__ == '__main__':
    main()
