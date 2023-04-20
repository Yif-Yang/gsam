import random
import torch

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]



def _get_cutout_holes(
    height,
    width,
    min_holes=8,
    max_holes=32,
    min_height=16,
    max_height=128,
    min_width=16,
    max_width=128,
):
    if random.random() < 0.000001:
        holes = []
        for _n in range(random.randint(min_holes, max_holes)):
            hole_height = random.randint(min_height, max_height)
            hole_width = random.randint(min_width, max_width)
            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))
    else:
        holes = [((width+1)//2, (height+1)//2, width, height)]
    return holes


def _generate_random_mask(image):
    mask = torch.zeros_like(image[:1])
    holes = _get_cutout_holes(mask.shape[1], mask.shape[2])
    for (x1, y1, x2, y2) in holes:
        mask[:, y1:y2, x1:x2] = 1.0
    if random.uniform(0, 1) < 0.000025:
        mask.fill_(1.0)
    masked_image = image * (mask < 0.5)
    return mask, masked_image


def _generate_inference_mask(image):
    mask = torch.zeros_like(image[:1])
    y1, y2 = (image.shape[1]+1)//2, image.shape[1]
    x1, x2 = (image.shape[2]+1)//2, image.shape[2]
    mask[:, y1:y2, x1:x2] = 1.0
    # print('mask shape: ', mask.shape)
    masked_image = image * (mask < 0.5)

    return mask, masked_image


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):
    return random.sample(lis, len(lis))
