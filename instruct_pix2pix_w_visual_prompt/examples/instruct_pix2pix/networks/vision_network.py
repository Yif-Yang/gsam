import torch
import torch.nn as nn
from functools import partial
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel,CLIPModel
from .xf import LayerNorm, Transformer

# from utils.save_utils import save2data_drive

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, save_dir, save_fn):
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_fn)
        torch.save(self.state_dict(), save_path)

        # save2data_drive(os.path.join(save_dir, save_fn), save_dir)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                1,
                1024,
                5,
                1,
            )

        self.freeze()

    def forward(self, images):
        with torch.no_grad():
            images_select = images[:2] + images[3:]
            images_concat = torch.concat(images_select).to(self.transformer.dtype)
            outputs_concat = self.transformer(pixel_values=images_concat)

            cls_token = outputs_concat.last_hidden_state[:, 0]
            pool_token = outputs_concat.pooler_output # cls + layer_norm

            cls_tokens = list(torch.chunk(cls_token, 3))
            pool_tokens = list(torch.chunk(pool_token, 3))

            dif_token = [cls_tokens[0] - cls_token[1]]
            z = torch.stack(cls_tokens+pool_tokens+dif_token, dim=1).float()

        z = z.detach().requires_grad_(True)
        # z = z.to(self.mapper.dtype)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def save_pretrained(self, save_dir, save_fn='clip_vision_prompt_encoder.pth'):
        super().save_pretrained(save_dir, save_fn)

    # def encode(self, image):
    #     return self(image)

