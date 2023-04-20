import os
import sys
import torch

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
eva_clip_path = os.path.join(root_path, 'src', 'third_party', 'EVA', 'EVA-CLIP')
asuka_path = os.path.join(eva_clip_path, 'rei')
sys.path.append(eva_clip_path)
sys.path.append(asuka_path)

print(root_path)
print(eva_clip_path)
print(asuka_path)


from eva_clip import create_model_and_transforms, get_tokenizer
from networks.vision_network import AbstractEncoder
from .xf import LayerNorm, Transformer


from PIL import Image
import torch.nn.functional as F

# this one should have better semantic information
class EVAV2CLIPEmbedder(AbstractEncoder):
    def __init__(self, args):
        super(EVAV2CLIPEmbedder, self).__init__()
        self.args = args
        model_name = "EVA02-CLIP-B-16"
        pretrained = "eva_clip"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
        model, _, _ = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        # import pdb; pdb.set_trace()

        self.transformer = model.visual
        self.transformer.eval()

        self.final_ln = LayerNorm(768)
        self.mapper = Transformer(
            1,
            768,
            5,
            1,
        )
        # self.mapper.dtype = torch.float32
        self.freeze()

    def forward(self, images):
        top_k = self.args.top_k
        with torch.no_grad():
            images_select = images[:2] + images[3:]
            # import pdb; pdb.set_trace()
            images_concat = torch.concat(images_select)#.to(torch.float16)
            outputs_concat = self.transformer.forward_features(images_concat, return_all_features=True)

            if top_k:
                ## TODO: check this implementation, 1) which feature should be leveraged for distance calculation (lm before or after),
                ## TODO: 2) which feature should be kept for learning (lm before or after)
                # import pdb; pdb.set_trace()
                # outputs_concat_lm = self.transformer.head(outputs_concat)
                outputs_concat_lm = outputs_concat

                cls_token = outputs_concat_lm[:, :1]
                cls_tokens = list(torch.chunk(cls_token, 3))
                dif_token = [cls_tokens[0] - cls_token[1]]

                spatial_token = outputs_concat_lm[:, 1:]
                spatial_tokens = list(torch.chunk(spatial_token, 3))
                before_edit, after_edit = spatial_tokens[1], spatial_tokens[0]
                patch_sim = F.cosine_similarity(before_edit, after_edit, dim=-1)
                # import pdb; pdb.set_trace()
                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(patch_sim, k=top_k, dim=-1)
                else:
                    top_k_vals, top_k_indices = torch.topk(-patch_sim, k=-top_k, dim=-1)
                pool_token = outputs_concat[:, :1]
                pool_token_spatial = outputs_concat[:, 1:]
                pool_tokens = list(torch.chunk(pool_token, 3))
                pool_token_spatials = list(torch.chunk(pool_token_spatial, 3))
                # import pdb; pdb.set_trace()

                pool_tokens_spatial_top_ks = [torch.stack([si[i,top_k_indices[i]] for i in range(si.shape[0])]) for si in pool_token_spatials]
                # import pdb; pdb.set_trace()

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

# ====================== official test code ======================

if __name__ == '__main__':
    model_name = "EVA02-CLIP-B-16"
    pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

    image_path = "CLIP.png"
    caption = ["a diagram", "a dog", "a cat"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
    # import pdb; pdb.set_trace()
    # model.visual.image_size 224
    # getattr(model.visual, 'image_mean', None) (0.48145466, 0.4578275, 0.40821073),
    # getattr(model.visual, 'image_std', None), (0.26862954, 0.26130258, 0.27577711)
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device).half()
    text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]

