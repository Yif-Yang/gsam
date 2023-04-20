import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from diffusers.models.attention_processor import AttnRPEProcessor, AttnRoutineProcessor, AttnProcessor
from networks.vision_network import FrozenCLIPImageEmbedder
# from networks.eva import EVAV2ImageEmbedder
from networks.eva2_clip import EVAV2CLIPEmbedder


class BaseTask:
    def __init__(self, args, logger):
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        assert not (args.use_vision_instruction and args.is_rpe), 'rpe and use_vision_instruction could not be both'

        if args.use_vision_instruction:
            if args.use_vision_instruction is True and args.use_language_instruction is False:
                args.del_origin_cross_attn = True
            else:
                args.del_origin_cross_attn = False
            # import pdb; pdb.set_trace()
            attn_procs = {}
            for idx, name in enumerate(unet.attn_processors.keys()):
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

                if cross_attention_dim is None:
                    attn_procs[name] = AttnProcessor()
                else:
                    # selectively choose the appropriate layer for context cross-attn
                    if args.cross_or_self is None:
                        attn_procs[name] = AttnRoutineProcessor(del_origin_cross_attn=args.del_origin_cross_attn, cross_attention_dim1=args.cross_attention_dim1)
                    else:
                        if name.startswith('mid_block'):  # 1 for cross
                            attn_procs[name] = AttnRoutineProcessor(del_origin_cross_attn=args.del_origin_cross_attn,
                                                                    cross_attention_dim1=args.cross_attention_dim1,
                                                                    is_cross=True)
                        else:
                            attn_procs[name] = AttnRoutineProcessor(del_origin_cross_attn=args.del_origin_cross_attn,
                                                                    cross_attention_dim1=args.cross_attention_dim1,
                                                                    is_cross=False)
            # import pdb; pdb.set_trace()
            unet.set_attn_processor(attn_procs)
            logger.info("we added vision to the unet cross attention.")

        if args.is_rpe:
            attn_procs = {}
            for name in unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    spatial_size = 8
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                    spatial_size = 16 * (1280 // hidden_size)
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                    spatial_size = 16 * (1280 // hidden_size)
                else:
                    raise ValueError

                # with or without PE
                if cross_attention_dim is None:
                    attn_procs[name] = AttnRPEProcessor(spatial_size=spatial_size)
                else:
                    attn_procs[name] = AttnRPEProcessor(spatial_size=None)
            unet.set_attn_processor(attn_procs)
            logger.info("we added relative position embedding to the unet self attention.")

        if args.use_vision_instruction is True:
            # vision_encoder = EVAV2ImageEmbedder(args) if args.use_eva2 else FrozenCLIPImageEmbedder()
            vision_encoder = EVAV2CLIPEmbedder(args) if args.use_eva2 else FrozenCLIPImageEmbedder()
        else:
            vision_encoder = None

        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.vision_encoder = vision_encoder

    def __call__(self, *args, **kwargs):
        pass

    def inference(self, *args, **kwargs):
        pass
