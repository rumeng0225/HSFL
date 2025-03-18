"""
https://github.com/Manikvsin/TinyImagenet-pytorch/blob/master/tiny_image_net_torch.py

The Tiny ImageNet 200 Classification dataset.

Tiny ImageNet contains 100000 images of 200 classes (500 for each class)
downsized to 64×64 colored images.
Each class has 500 training images, 50 validation images and 50 test images.
"""
import os
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):

    def __init__(self, data_dir, train, transform=None):
        url = "https://drive.usercontent.google.com/download?id=1rYW2eOg-kERCTfgp5CiybvFtlAxyHSHF"
        # url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

        if not os.path.exists(data_dir):
            raise ValueError(f"TinyImageNet is not exist, please download from {url}")

        if train:
            self.data_dir = data_dir + "/train"
        else:
            self.data_dir = data_dir + "/val"

        self.transform = transform

        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.image_path = []
        self.labels = []
        self.imgs: Any = []

        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if train:
                cls_dir = os.path.join(cls_dir, "images")

            for img_name in os.listdir(cls_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")):
                    img_path = os.path.join(cls_dir, img_name)
                    self.image_path.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

        for path in self.image_path:
            with open(path, "rb") as f:
                img = Image.open(path).convert("RGB")
                img = np.array(img)
                self.imgs.append(img)
                f.close()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
