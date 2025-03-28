"""
AutoAugment policies

Reference:
    https://github.com/DeepVoltaire/AutoAugment
"""

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        policy = CIFAR10Policy()
        transformed = policy(image)

        Example as a PyTorch Transform:
        transform=transforms.Compose([
        transforms.Resize(256),
        CIFAR10Policy(),
        transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            # "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        self.fillcolor = fillcolor

        self.p1 = p1
        # self.operation1 = func[operation1]
        self.operation1 = operation1
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        # self.operation2 = func[operation2]
        self.operation2 = operation2
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            # img = self.operation1(img, self.magnitude1)
            img = self._image_ops(self.operation1, img, self.magnitude1, self.fillcolor)
        if random.random() < self.p2:
            # img = self.operation2(img, self.magnitude2)
            img = self._image_ops(self.operation2, img, self.magnitude2, self.fillcolor)
        return img

    def _image_ops(self, operation, img, magnitude, fillcolor):
        if operation == 'shearX':
            img = img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                                Image.BICUBIC, fillcolor=fillcolor)
        elif operation == "shearY":
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                                Image.BICUBIC, fillcolor=fillcolor)
        elif operation == "translateX":
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                                fillcolor=fillcolor)
        elif operation == "translateY":
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                                fillcolor=fillcolor)
        elif operation == "rotate":
            img = rotate_with_fill(img, magnitude)
        elif operation == "color":
            img = ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))
        elif operation == "posterize":
            img = ImageOps.posterize(img, magnitude)
        elif operation == "solarize":
            img = ImageOps.solarize(img, magnitude)
        elif operation == "contrast":
            img = ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))
        elif operation == "sharpness":
            img = ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))
        elif operation == "brightness":
            img = ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))
        elif operation == "autocontrast":
            img = ImageOps.autocontrast(img)
        elif operation == "equalize":
            img = ImageOps.equalize(img)
        elif operation == "invert":
            img = ImageOps.invert(img)
        else:
            return img
        return img


# from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)
