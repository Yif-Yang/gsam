from __future__ import annotations

import glob
import sys
import os
import numpy as np
from argparse import ArgumentParser

import torch
import cv2
from tqdm.auto import tqdm

import json
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path


from clip_similarity import ClipSimilarity


def compute_metrics(output_path,
                    scales_img,
                    scales_txt,
                    args):
    clip_similarity = ClipSimilarity().cuda()

    for scale_txt in scales_txt:
        for scale_img in scales_img:
            image_paths = glob.glob(os.path.join(output_path, 'images', 'ts{}_is{}'.format(scale_txt, scale_img), '*.jpg'))
            # import pdb; pdb.set_trace()
            # print(len(image_paths))
            num_samples = len(image_paths)
            if num_samples <= 0: continue

            outpath = Path(output_path, f"n={num_samples}.jsonl")
            Path(output_path).mkdir(parents=True, exist_ok=True)

            print(f'Processing t={scale_txt}, i={scale_img}')
            torch.manual_seed(args.seed)
            count = 0
            i = 0

            sim_0_avg = 0
            sim_1_avg = 0
            sim_direction_avg = 0
            sim_image_avg = 0
            count = 0

            pbar = tqdm(total=num_samples)
            while count < num_samples:
                image = cv2.imread(image_paths[count])[:, 512:]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                i += 1
                # import pdb; pdb.set_trace()
                h = image.shape[1]//2
                image_0, image_1, ref_image_0, ref_image_1 = image[h:,:h], image[h:,h:], image[:h, :h], image[:h, h:]
                np2tensor = lambda x: torch.from_numpy(np.transpose(x,(2,0,1))).unsqueeze(0).cuda()
                image_0, image_1, ref_image_0, ref_image_1 = list(map(np2tensor, [image_0, image_1, ref_image_0, ref_image_1]))
                sim_0, sim_1, sim_direction, sim_image = clip_similarity(
                    image_0, image_1, ref_image_0, ref_image_1
                )
                sim_0_avg += sim_0.item()
                sim_1_avg += sim_1.item()
                sim_direction_avg += sim_direction.item()
                sim_image_avg += sim_image.item()
                count += 1
                pbar.update(count)

            pbar.close()

            sim_0_avg /= count
            sim_1_avg /= count
            sim_direction_avg /= count
            sim_image_avg /= count

            # with open(outpath, "a") as f:
            #     f.write(
            #         f"{json.dumps(dict(sim_0=sim_0_avg, sim_1=sim_1_avg, sim_direction=sim_direction_avg, sim_image=sim_image_avg, num_samples=num_samples, split=split, scale_txt=scale_txt, scale_img=scale_img, steps=steps, res=res, seed=seed))}\n")
            result = [sim_0_avg, sim_1_avg, sim_direction_avg, sim_image_avg]
            print(result)


def plot_metrics(metrics_file, output_path):
    with open(metrics_file, 'r') as f:
        data = [json.loads(line) for line in f]

    plt.rcParams.update({'font.size': 11.5})
    seaborn.set_style("darkgrid")
    plt.figure(figsize=(20.5 * 0.7, 10.8 * 0.7), dpi=200)

    x = [d["sim_direction"] for d in data]
    y = [d["sim_image"] for d in data]

    plt.plot(x, y, marker='o', linewidth=2, markersize=4)

    plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
    plt.ylabel("CLIP Image Similarity", labelpad=10)

    plt.savefig(Path(output_path) / Path("plot.pdf"), bbox_inches="tight")


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_path", default="analysis/", type=str)
    parser.add_argument("--seed", default=888, type=int)
    args = parser.parse_args()

    # scales_img = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    scales_img = [1.5] # local visual concat
    scales_txt = [7.5] # global visual feature injection

    metrics_file = compute_metrics(
        args.output_path,
        scales_img,
        scales_txt,
        args=args
    )

    # plot_metrics(metrics_file, args.output_path)


if __name__ == "__main__":
    main()
