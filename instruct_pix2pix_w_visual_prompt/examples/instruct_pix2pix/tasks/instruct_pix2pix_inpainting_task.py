import logging

import torch
import torch.nn.functional as F
from einops import rearrange
import os
import wandb
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.import_utils
from utils.torch_utils import save_concat_image
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionInpaintWVisualPromptPipeline, DPMSolverMultistepScheduler

from .base_task import BaseTask

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


class InstructPix2PixInpaintingTask(BaseTask):
    def __init__(self, args, logger):
        super(InstructPix2PixInpaintingTask, self).__init__(args, logger)


    def __call__(self,  vae, unet, text_encoder, noise_scheduler, accelerator, generator, weight_dtype, batch, args, step, vision_encoder=None, **kwargs):
        def get_loss(noise, model_pred):
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            return loss

        # import pdb;pdb.set_trace()
        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        masked_latents = vae.encode(
            batch["masked_image_values"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
        ).latent_dist.sample()
        masked_latents = masked_latents * vae.config.scaling_factor

        mask = batch["mask_values"]
        mask = torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
        mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # concatenate the noised latents with the mask and the masked latents
        c_concat = torch.cat([mask, masked_latents], dim=1)

        uncond = args.conditioning_dropout_prob

        if uncond > 0:
            random = torch.rand(c_concat.size(0), device=c_concat.device)

            input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
            latent_model_input = torch.cat([noisy_latents, c_concat * input_mask], dim=1)

            if args.use_language_instruction is True:
                text_prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
                edit_hidden_states = text_encoder(batch["edit_ids"])[0]
                null_edit_hidden_states = text_encoder(batch["null_edit_ids"])[0]
                encoder_hidden_states = torch.where(text_prompt_mask, null_edit_hidden_states.detach(),
                                                    edit_hidden_states.detach())

            if args.use_vision_instruction is True:
                vision_prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
                vision_edit_hidden_states = vision_encoder(batch["clipnorm_values"]) # encode the context information
                vision_null_edit_hidden_states = vision_encoder(batch["null_clipnorm_values"])
                vision_encoder_hidden_states = torch.where(vision_prompt_mask, vision_null_edit_hidden_states.detach(),
                                                    vision_edit_hidden_states.detach())

        else:
            # latent_model_input = torch.cat([noisy_latents, c_concat], dim=1)
            # encoder_hidden_states = text_encoder(batch["edit_ids"])[0]
            # We believe the classifier-free guidance is necessary
            raise ValueError

        switch_flag = False if args.use_vision_instruction and args.use_language_instruction else True

        model_pred = None
        if args.use_vision_instruction is True and (switch_flag or step%2==0):
            v_cross_attention_kwargs = {'context_mode': 'vision_instruction'}
            model_pred_v = unet(latent_model_input, timesteps, vision_encoder_hidden_states, cross_attention_kwargs=v_cross_attention_kwargs).sample
            loss_vision = get_loss(noise, model_pred_v)
        else:
            loss_vision = 0
        # if False:
        if args.use_language_instruction is True and (switch_flag or step%2==1):
            l_cross_attention_kwargs = {'context_mode': 'language_instruction'}
            model_pred_l = unet(latent_model_input, timesteps, encoder_hidden_states, cross_attention_kwargs=l_cross_attention_kwargs).sample
            loss_language = get_loss(noise, model_pred_l)
        else:
            loss_language = 0

        # if accelerator.is_main_process:
        #     print('language: ', loss_language, ' vision: ', loss_vision)
        loss = 0.5 * (loss_language + loss_vision)

        return model_pred, loss

    def inference(self, vae, unet, text_encoder, generator, accelerator, weight_dtype, batch, logger, epoch, \
                  global_step, args, vision_encoder=None, ema_unet=None, pipeline=None):
        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        if args.use_ema:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())

        if pipeline is None:
            pipeline = StableDiffusionInpaintWVisualPromptPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                vae=accelerator.unwrap_model(vae),
                revision=args.revision,
                text_encoder=accelerator.unwrap_model(text_encoder),
                torch_dtype=weight_dtype,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

        # run inference
        # original_image = download_image(args.val_image_url)
        base_image = batch['base_image'][0]
        mask_image = batch['mask_image'][0]

        base_image = base_image.resize((args.resolution, args.resolution))
        mask_image = mask_image.resize((args.resolution, args.resolution))

        args.validation_prompt = batch['edit_prompt'][0]
        # edited_images = []

        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )

        original_image = base_image
        # import pdb; pdb.set_trace()
        cross_attention_kwargs = None
        if args.use_vision_instruction:
            batch["clipnorm_values"] = [x.to(accelerator.device) for x in batch["clipnorm_values"]]
            batch["null_clipnorm_values"] = [x.to(accelerator.device) for x in batch["null_clipnorm_values"]]
            prompt_embeds = accelerator.unwrap_model(vision_encoder)(batch["clipnorm_values"])
            negative_prompt_embeds = accelerator.unwrap_model(vision_encoder)(batch["null_clipnorm_values"]).float()
            cross_attention_kwargs = {'context_mode': 'vision_instruction'}
        else:
            prompt_embeds = accelerator.unwrap_model(text_encoder)(batch["edit_ids"])[0]
            negative_prompt_embeds = None

        edited_image = pipeline(
            prompt_embeds=prompt_embeds.float(),
            negative_prompt_embeds=negative_prompt_embeds,
            # args.validation_prompt,
            image=original_image,
            mask_image=mask_image,
            num_inference_steps=25,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
            guidance_scale=args.text_guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs
        ).images[0]
        # )
        concat_path = os.path.join(args.output_dir, 'images', 'ts{}_is{}'.format(args.text_guidance_scale, args.image_guidance_scale),
                   args.validation_prompt + '_ep{}_step{}'.format(epoch, global_step) + '.jpg')
        save_concat_image(concat_path, batch['base_image'][0], edited_image)

        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    # for edited_image in edited_images:
                    wandb_table.add_data(
                        wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt
                    )
                    tracker.log({"validation": wandb_table})

        if args.use_ema:
            # Switch back to the original UNet parameters.
            ema_unet.restore(unet.parameters())

        # del pipeline
        torch.cuda.empty_cache()
        return pipeline