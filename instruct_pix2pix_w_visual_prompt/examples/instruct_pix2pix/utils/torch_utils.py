import os
import cv2
import torch
import numpy as np
from typing import Optional
from huggingface_hub import HfFolder, Repository, create_repo, whoami
import PIL
import requests


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True



def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def save_concat_image(concat_path, base_image, image):
    os.makedirs(os.path.dirname(concat_path), exist_ok=True)
    import cv2
    base_image_np = cv2.cvtColor(np.array(base_image), cv2.COLOR_BGR2RGB)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    concat = np.hstack([base_image_np, image_np])
    cv2.imwrite(concat_path, concat)
    print('save to ', concat_path)


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def copy_state_dict(state_dict, model, strip=None, replace=None):
    # import pdb; pdb.set_trace()
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and replace is None and name.startswith(strip):
            name = name[len(strip):]
        if strip is not None and replace is not None:
            name = name.replace(strip, replace)
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    del state_dict


def load_state_dict(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print('{} does not exist.'.format(ckpt_path))
    else:
        print('load form {}'.format(ckpt_path))
        src_state_dict = torch.load(ckpt_path)
        copy_state_dict(src_state_dict, model)
        del src_state_dict


def resume_accelerator(accelerator, num_update_steps_per_epoch, args):
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        return accelerator, None, None, None, None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

        return accelerator, global_step, resume_global_step, first_epoch, resume_step