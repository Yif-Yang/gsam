"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""
import utils.import_utils
from utils.import_utils import get_task, get_dataloader
from utils.torch_utils import resume_accelerator, load_state_dict
from utils.save_utils import save2data_drive

import logging
import math
import os

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available

from options.arguments_parser import parse_args
from utils.torch_utils import download_image, convert_to_np, save_concat_image, get_full_repo_name
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    # ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # gradient_accumulation_steps=1, # inner accumulation sucks
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_handler]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if accelerator.is_main_process:
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb
            wandb.init(project="vision_incontext_learning", name=args.wandb_jobname)

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.

    task = get_task(args, logger)
    noise_scheduler, tokenizer, text_encoder, vae, unet, vision_encoder = \
        task.noise_scheduler, task.tokenizer, task.text_encoder, task.vae, task.unet, task.vision_encoder

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            save2data_drive(os.path.join(output_dir, "unet"))

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * np.sqrt(args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        )
        args.vision_enc_learning_rate = (
            args.vision_enc_learning_rate * np.sqrt(args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        )
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = [
        {
            "params": unet.parameters(),
            "lr": args.learning_rate,
        }
    ]
    if vision_encoder is not None:
        if isinstance(vision_encoder, torch.nn.parallel.DistributedDataParallel):
            vision_encoder_params = list(vision_encoder.module.mapper.parameters())+ \
                                    list(vision_encoder.module.final_ln.parameters())
        else:
            vision_encoder_params = list(vision_encoder.mapper.parameters()) + \
                                    list(vision_encoder.final_ln.parameters())
        params_to_optimize.extend([
            {
                "params": vision_encoder_params,
                "lr": args.vision_enc_learning_rate
            }
        ])

    optimizer = optimizer_cls(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader, val_dataloader = get_dataloader(args, tokenizer)
    iter_val_dataloader = iter(val_dataloader)

    # import pdb; pdb.set_trace()
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # print(unet.device)
    # always load the pretrained weights before move the module to cuda to save memory
    if accelerator.is_main_process:
        if args.unet_checkpoint_path:
            load_state_dict(unet, args.unet_checkpoint_path)

    # Prepare everything with our `accelerator`.
    # import pdb; pdb.set_trace()
    optimizer, train_dataloader, lr_scheduler, unet = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, unet
    )

    if vision_encoder is not None:
        vision_encoder = accelerator.prepare(vision_encoder)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=torch.float32)

    if vision_encoder is not None:
        vision_encoder.to(accelerator.device, dtype=torch.float32)
        if isinstance(vision_encoder, torch.nn.parallel.DistributedDataParallel):
            vision_encoder.module.mapper.to(accelerator.device, dtype=torch.float32)
            vision_encoder.module.final_ln.to(accelerator.device, dtype=torch.float32)
        else:
            vision_encoder.mapper.to(accelerator.device, dtype=torch.float32)
            vision_encoder.final_ln.to(accelerator.device, dtype=torch.float32)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("vision_incontext_learning", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)*args.train_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator, global_step, resume_global_step, first_epoch, resume_step =\
                    resume_accelerator(accelerator, num_update_steps_per_epoch, args)


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    unet.train()
    train_loss = 0.0
    # cyclic_dataloader = itertools.cycle(train_dataloader)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet) if vision_encoder is None else accelerator.accumulate(unet), accelerator.accumulate(vision_encoder):
            # with accelerator.accumulate(unet):
                # Forward data and compute the backward losses
                model_pred, loss = task(vae, unet, text_encoder, noise_scheduler, accelerator, generator, weight_dtype,
                                        batch, args, global_step, vision_encoder=vision_encoder)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:  # after you run 4 times.
                # if (global_step+1) % args.gradient_accumulation_steps == 0:
                    # print('global_step: ', global_step, 'step: ', step)
                    params_to_clip = unet.parameters()
                    if vision_encoder is not None:
                        if isinstance(vision_encoder, torch.nn.parallel.DistributedDataParallel):
                            vision_enc_params = list(vision_encoder.module.mapper.parameters()) + \
                                                list(vision_encoder.module.final_ln.parameters())
                        else:
                            vision_enc_params = list(vision_encoder.mapper.parameters()) + \
                                                list(vision_encoder.final_ln.parameters())
                        params_to_clip = list(unet.parameters()) + vision_enc_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients: #after you run 4 times.
            # global_step += 1
            # if (global_step + 1) % args.gradient_accumulation_steps == 0:
            #     print('global step, step:  ', global_step, step)
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)

                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        # import pdb; pdb.set_trace()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # if global_step >= args.max_train_steps:
            #     break

            if accelerator.is_main_process:
                try:
                    val_batch = next(iter_val_dataloader)
                except Exception as ex:
                    print(str(ex))
                    iter_val_dataloader = iter(val_dataloader)
                    val_batch = next(iter_val_dataloader)

                if global_step % args.validation_steps == 0:
                    unet.eval()
                    task.inference(vae, unet, text_encoder, generator, accelerator, weight_dtype, val_batch, logger,
                                   epoch, global_step, args, vision_encoder=vision_encoder)
                    unet.train()


if __name__ == "__main__":
    main()
