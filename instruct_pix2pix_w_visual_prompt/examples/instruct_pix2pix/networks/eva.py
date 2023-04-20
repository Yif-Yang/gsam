import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
eva2_path = os.path.join(root_path, 'src', 'third_party', 'EVA', 'EVA-02')
asuka_path = os.path.join(eva2_path, 'asuka')
sys.path.append(eva2_path)
sys.path.append(asuka_path)

print(root_path)
print(eva2_path)
print(asuka_path)

from asuka.modeling_pretrain import eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init
import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
import asuka.utils as utils

from networks.vision_network import AbstractEncoder
from .xf import LayerNorm, Transformer


class EVAV2ImageEmbedder(AbstractEncoder):
    def __init__(self, args):
        super(EVAV2ImageEmbedder, self).__init__()
        self.args = args
        path = huggingface_hub.hf_hub_download('Yuxin-CV/EVA-02', 'eva02_L_pt_m38m_p14.pt', subfolder='eva02/pt')
        # print(path)
        model = eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init()
        checkpoint = torch.load(path, map_location="cpu")
        # model.load_state_dict(checkpoint["module"])
        utils.load_state_dict(model, checkpoint["module"])
        # print(model)
        self.transformer = model
        self.transformer.eval()
        # we hard coded the datatype of vision understanding
        self.transformer.half()
        self.transformer.dtype = torch.float16
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
            1,
            1024,
            5,
            1,
        )
        # self.mapper.dtype = torch.float32
        self.freeze()

    def forward(self, images):
        top_k = self.args.top_k
        with torch.no_grad():
            images_select = images[:2] + images[3:]
            images_concat = torch.concat(images_select).to(self.transformer.dtype)
            outputs_concat = self.transformer.forward_features(images_concat,bool_masked_pos=torch.ones(1, 256).bool().to(images_concat.device))

            if top_k:
                ## TODO: check this implementation, 1) which feature should be leveraged for distance calculation (lm before or after),
                ## TODO: 2) which feature should be kept for learning (lm before or after)
                outputs_concat_lm = self.transformer.lm_head(outputs_concat)

                cls_token = outputs_concat_lm[:, :1]
                cls_tokens = list(torch.chunk(cls_token, 3))
                dif_token = [cls_tokens[0] - cls_token[1]]

                spatial_token = outputs_concat_lm[:, 1:]
                spatial_tokens = list(torch.chunk(spatial_token, 3))
                before_edit, after_edit = spatial_tokens[1], spatial_tokens[0]
                patch_sim = F.cosine_similarity(before_edit, after_edit, dim=-1)
                top_k_vals, top_k_indices = torch.topk(patch_sim, k=top_k, dim=-1)

                pool_token = outputs_concat[:, :1]
                pool_token_spatial = outputs_concat[:, 1:]
                pool_tokens = list(torch.chunk(pool_token, 3))
                pool_token_spatials = list(torch.chunk(pool_token_spatial, 3))
                import pdb; pdb.set_trace()
                pool_tokens_spatial_top_ks = [torch.gather(si, dim=1, index=top_k_indices.unsqueeze(-1)) for si in pool_token_spatials]

                ## further add topk patch features for guidance
                z = torch.concat(cls_tokens + dif_token + pool_tokens + pool_tokens_spatial_top_ks, dim=1).float()
            else:
                cls_token = self.transformer.lm_head(outputs_concat[:, 0])
                pool_token = outputs_concat[:, 0]

                cls_tokens = list(torch.chunk(cls_token, 3))
                pool_tokens = list(torch.chunk(pool_token, 3))

                dif_token = [cls_tokens[0] - cls_token[1]]
                z = torch.stack(cls_tokens + pool_tokens + dif_token, dim=1).float()

        z = z.detach().requires_grad_(True)
        # z = z.to(self.mapper.dtype)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def save_pretrained(self, save_dir, save_fn='eva2_vision_prompt_encoder.pth'):
        super().save_pretrained(save_dir, save_fn)
