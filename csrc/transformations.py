import io
import random
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import numpy as np
 
class JPEGCompression:
    def __init__(self, quality_range=(10, 50)):
        self.quality_range = quality_range
    
    def __call__(self, img):
        """ takes in a PIL image and returns the image which has been compressed. """
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

class GaussianBlur:
    def __init__(self, blur_range=(0.5, 3.0)):
        self.blur_range = blur_range
    
    def __call__(self, img):
        """ takes in a PIL image and returns the image which has gaussian blur applied to it."""
        blur = random.uniform(*self.blur_range)
        img = img.filter(ImageFilter.GaussianBlur(radius = blur))
        return img

class GaussianNoise:
    def __init__(self, std_range=(5, 30)):
        self.std_range = std_range

    def __call__(self, img):
        std = random.uniform(*self.std_range)
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class ReUpload:
    def __init__(self, scale_range=(0.25, 0.6)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = random.uniform(*self.scale_range)
        W, H = img.size
        small = img.resize((int(W * scale), int(H * scale)), Image.BILINEAR)
        return small.resize((W, H), Image.BILINEAR)

class ScreenShot:
    def __init__(self, sharpness_range=(1.1, 2.0), color_range=(0.9, 1.1)):
        self.sharpness_range = sharpness_range
        self.color_range = color_range

    def __call__(self, img):
        sharpness = random.uniform(*self.sharpness_range)
        color = random.uniform(*self.color_range)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        img = ImageEnhance.Color(img).enhance(color)
        compressor = JPEGCompression(quality_range=(60, 85))
        return compressor(img)


# here is the pipeline that randomly selects the transformations and applies them to an image.

# class DegradationPipeline:
#     def __init__(self):
#         self.transforms = [
#             JPEGCompression(),
#             GaussianBlur(),
#             GaussianNoise(),
#             ReUpload(),
#             ScreenShot(),
#         ]

#         self.finalize = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#         ])

#     def __call__(self, img):
#         # sometimes apply one, sometimes a combination
#         if random.random() < 0.3:  # 30% chance of combination
#             k = random.randint(2, 3)
#             chosen = random.sample(self.transforms, k)
#             for t in chosen:
#                 img = t(img)
#         else:
#             t = random.choice(self.transforms)
#             img = t(img)

#         return self.finalize(img)
    
class DegradationPipeline:
    def __init__(self, disabled=None):
        if disabled is None:
            disabled = []

        all_transforms = {
            "jpeg": JPEGCompression(),
            "blur": GaussianBlur(),
            "noise": GaussianNoise(),
            "reupload": ReUpload(),
            "screenshot": ScreenShot(),
        }

        self.transforms = [
            transform for name, transform in all_transforms.items()
            if name not in disabled
        ]

        self.finalize = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        if len(self.transforms) == 0:
            return self.finalize(img)

        if random.random() < 0.3:
            k = min(random.randint(2, 3), len(self.transforms))
            chosen = random.sample(self.transforms, k)
            for t in chosen:
                img = t(img)
        else:
            t = random.choice(self.transforms)
            img = t(img)

        return self.finalize(img)