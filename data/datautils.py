import os
from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


def _sample_radius(image_size, radius_mean, radius_std):
    """Sample a positive radius (in pixels) from a low-variance Gaussian."""
    sampled = np.random.normal(loc=radius_mean, scale=radius_std)
    min_radius = max(1.0, 0.01 * image_size)
    max_radius = max(min_radius, 0.5 * image_size)
    return float(np.clip(sampled, min_radius, max_radius))


def apply_structured_occlusion(image, num_spots=3, radius_mean_ratio=0.08,
                               radius_std_ratio=0.02, shape='circle', fill='mean'):
    """Apply random structured masks to simulate partial visibility."""
    occluded = image.copy()
    width, height = occluded.size
    min_side = float(min(width, height))

    radius_mean = radius_mean_ratio * min_side
    radius_std = radius_std_ratio * min_side

    if fill == 'mean':
        image_np = np.array(occluded, dtype=np.float32)
        fill_color = tuple(np.mean(image_np.reshape(-1, image_np.shape[-1]), axis=0).astype(np.uint8).tolist())
    else:
        fill_color = (0, 0, 0)

    draw = ImageDraw.Draw(occluded)
    for _ in range(max(0, int(num_spots))):
        center_x = np.random.uniform(0, width)
        center_y = np.random.uniform(0, height)
        radius = _sample_radius(min_side, radius_mean, radius_std)

        if shape == 'ellipse':
            # Slight anisotropy keeps masks realistic while preserving object-level structure.
            radius_y = radius * np.random.uniform(0.75, 1.25)
            bbox = [
                center_x - radius,
                center_y - radius_y,
                center_x + radius,
                center_y + radius_y,
            ]
            draw.ellipse(bbox, fill=fill_color)
        elif shape == 'square':
            bbox = [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ]
            draw.rectangle(bbox, fill=fill_color)
        else:
            bbox = [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ]
            draw.ellipse(bbox, fill=fill_color)

    return occluded


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, occlusion_views=0, occlusion_spots=3,
                    occlusion_radius_mean=0.08, occlusion_radius_std=0.02,
                    occlusion_shape='circle', occlusion_fill='mean'):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.occlusion_views = occlusion_views
        self.occlusion_spots = occlusion_spots
        self.occlusion_radius_mean = occlusion_radius_mean
        self.occlusion_radius_std = occlusion_radius_std
        self.occlusion_shape = occlusion_shape
        self.occlusion_fill = occlusion_fill
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        for _ in range(self.occlusion_views):
            # Keep RandomResizedCrop as the base view generator, then add structured masking.
            base_view = get_preaugment()(x)
            occluded_view = apply_structured_occlusion(
                base_view,
                num_spots=self.occlusion_spots,
                radius_mean_ratio=self.occlusion_radius_mean,
                radius_std_ratio=self.occlusion_radius_std,
                shape=self.occlusion_shape,
                fill=self.occlusion_fill,
            )
            views.append(self.preprocess(occluded_view))
        return [image] + views



