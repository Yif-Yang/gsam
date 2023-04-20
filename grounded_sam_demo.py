import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from .instruct_pix2pix_w_visual_prompt.utils.data import get_task, get_dataloader


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


# def show_box(box, ax, label):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
#     ax.text(x0, y0, label)


def save_mask_data(output_dir, box_list, label_list, image_name):
    value = 0  # 0 for background
    # import pickle
    # with open(os.path.join(output_dir, f'{image_name}_mask.pkl'), 'wb') as f:
    #     pickle.dump(mask_list.cpu(), f)
    # mask_img = torch.zeros(mask_list.shape[-2:])
    # # import ipdb; ipdb.set_trace()
    # for idx, mask in enumerate(mask_list):
    #     mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_img.numpy())
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f'{image_name}_mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.cpu().numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'{image_name}.json'), 'w') as f:
        json.dump(json_data, f)
    
from pathlib import Path
import glob
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--gpu_appendix", type=int, default=0, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path_dir = args.input_image
    text_prompt = args.text_prompt
    # output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device
    instance_images_path = list(Path(image_path_dir).iterdir())
    # import ipdb; ipdb.set_trace()
    if os.path.isdir(instance_images_path[0]):
        dirs = instance_images_path
        instance_images_path = [dir for dir in dirs if len(glob.glob(os.path.join(dir, '*.jpg'))) >= 4]

    # make dir
    # os.makedirs(output_dir, exist_ok=True)
    # load image
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    # visualize raw image
    # import ipdb; ipdb.set_trace()
    image_extension = ['*.jpg', '*.jpeg', '*.png', '*.gif']


    # run grounding dino model
    def run_image(image_path):
        start_time = time.time()

        with open(os.path.join(os.path.dirname(image_path), 'prompt.json'), 'r') as f:
            data = json.load(f)
            edit_str = data['edit']
        image_pil, image = load_image(image_path)
        output_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path).split('.')[0]
        load_image_time = time.time()
        # print(f"load_image time: {load_image_time - start_time}")

        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, edit_str, box_threshold, text_threshold, device=device
        )
        get_grounding_output_time = time.time()

        # initialize SAM
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

        # print(f"get_grounding_output time: {get_grounding_output_time - load_image_time}")
        # masks, _, _ = predictor.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes,
        #     multimask_output = False,
        # )
        # predictor_predict_torch_time = time.time()
        # print(f"predictor.predict_torch time: {predictor_predict_torch_time - get_grounding_output_time}")

        # # draw output image
        # plt.figure(figsize=(10, 10))
        # # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, f"grounded_sam_output_{image_name}.jpg"),
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )

        save_mask_data(output_dir, boxes_filt, pred_phrases, image_name)
        # save_mask_data_time = time.time()
        # print(f"save_mask_data time: {save_mask_data_time - predictor_predict_torch_time}")
        # end_time = time.time()
        # print(f"total time: {end_time - start_time}")
        # load_image_percent = (load_image_time - start_time) / (end_time - start_time) * 100
        # get_grounding_output_percent = (get_grounding_output_time - load_image_time) / (end_time - start_time) * 100
        # predictor_predict_torch_percent = (predictor_predict_torch_time - get_grounding_output_time) / (
        #             end_time - start_time) * 100
        # save_mask_data_percent = (save_mask_data_time - predictor_predict_torch_time) / (end_time - start_time) * 100
        #
        # print(f"load_image percent: {load_image_percent}")
        # print(f"get_grounding_output percent: {get_grounding_output_percent}")
        # print(f"predictor.predict_torch percent: {predictor_predict_torch_percent}")
        # print(f"save_mask_data percent: {save_mask_data_percent}")


    failed_dirs = []  # 创建一个空列表，以记录所有失败的文件夹
    print('going into for loop')
    for image_path in instance_images_path:
        if int(os.path.basename(image_path)[-1]) != args.gpu_appendix:
            # print('skip', image_path, 'gpu_appendix', args.gpu_appendix)
            continue
        image_files = []
        for root, dirs, files in os.walk(image_path):
            for ext in image_extension:
                image_files.extend(glob.glob(os.path.join(root, ext)))
        print(image_path)
        for _image_path in image_files:
            if "grounded_sam_output" in _image_path or "mask" in _image_path:  # 如果文件名中包含 "grounded_sam_output"，则跳过当前图片
                continue
            try:
                run_image(_image_path)
            except Exception:  # 如果出现异常，则将失败的文件夹记录到 failed_dirs 列表中
                if image_path not in failed_dirs:
                    failed_dirs.append(image_path)
            # run_image(_image_path)

        print(f"finished {image_path}")
    # 将 failed_dirs 中的所有文件夹名称写入 failed_dir.txt 文件中
    with open(f"failed_dir_{args.gpu_appendix}.txt", "w") as f:
        for failed_dir in failed_dirs:
            f.write(failed_dir + "\n")