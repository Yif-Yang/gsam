import torch
import torch.nn.functional as F
import os
import wandb
import torch.nn as nn
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.import_utils
from utils.torch_utils import save_concat_image
from diffusers import StableDiffusionInstructPix2PixPipeline


from .base_task import BaseTask

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


class InstructPix2PixTask(BaseTask):
    def __init__(self, args, logger):
        super(InstructPix2PixTask, self).__init__(args, logger)
        unet = self.unet
        logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")

        in_channels = 8
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    def __call__(self, vae, unet, text_encoder, noise_scheduler, accelerator, generator, weight_dtype, batch, args, **kwargs):
        latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning.
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        # import pdb; pdb.set_trace()
        original_image_embeds = vae.encode(batch["winput_image_values"].to(weight_dtype)).latent_dist.mode()

        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        if args.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device, generator=generator)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * args.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            null_conditioning = text_encoder(batch["null_edit_ids"].to(accelerator.device))[0]
            encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

            # Sample masks for the original images.
            image_mask_dtype = original_image_embeds.dtype
            image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            original_image_embeds = image_mask * original_image_embeds

        # Concatenate the `original_image_embeds` with the `noisy_latents`.
        concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1).to(torch.float32)

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return model_pred, loss

    def inference(self, vae, unet, text_encoder, generator, accelerator, weight_dtype, batch, logger, epoch, global_step, args, ema_unet=None):
        if args.use_ema:
            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            revision=args.revision,
            text_encoder=accelerator.unwrap_model(text_encoder),
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        # original_image = download_image(args.val_image_url)
        original_image = batch['base_image_winput'][0]
        args.validation_prompt = batch['edit_prompt'][0]
        # edited_images = []

        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        # with torch.autocast(str(accelerator.device), enabled=accelerator.mixed_precision == "fp16"):
        # for _ in range(args.num_validation_images):
        #     edited_images.append(
        edited_image = pipeline(
            args.validation_prompt,
            image=original_image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
        ).images[0]
        # )
        concat_path = os.path.join(args.output_dir, 'images',
                                   args.validation_prompt + '_ep_{}step_{}'.format(epoch, global_step) + '.jpg')
        save_concat_image(concat_path, batch['base_image'][0], edited_image)

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

        del pipeline
        torch.cuda.empty_cache()
