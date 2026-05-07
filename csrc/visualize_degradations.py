import argparse
import random
from pathlib import Path
 
import matplotlib.pyplot as plt
from PIL import Image
  
from transformations import (
    JPEGCompression,
    GaussianBlur,
    GaussianNoise,
    ReUpload,
    ScreenShot,
    DegradationPipeline,
)
 
ROOT = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help='path to a specific image to degrade')
    parser.add_argument('--random', type=str, default=None,
                        help='generator class name to pick a random image from '
                             '(e.g. "Midjourney", "Dall-E 3")')
    parser.add_argument('--out', type=str,
                        default='degradations.png',
                        help='where to save the comparison figure')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
 
    random.seed(args.seed)
 
    # pick the image
    if args.image:
        img_path = Path(args.image)
    elif args.random:
        class_dir = ROOT / args.random
        candidates = sorted(class_dir.glob('*.png'))
        img_path = random.choice(candidates)
        print(f'Picked: {img_path.name}')
    else:
        raise ValueError('Pass either --image <path> or --random <class_name>')
 
    img = Image.open(img_path).convert('RGB')
 
    # individual degradations — fix the random params per transform so the
    # figure is reproducible. you can tweak these for the slide.
    transforms = [
        ('Clean (original)', None),
        ('JPEG compression',  JPEGCompression(quality_range=(15, 15))),
        ('Gaussian blur',     GaussianBlur(blur_range=(2.5, 2.5))),
        ('Gaussian noise',    GaussianNoise(std_range=(25, 25))),
        ('Re-upload (downscale + upscale)', ReUpload(scale_range=(0.3, 0.3))),
        ('Screenshot simulation', ScreenShot()),
        ('Random combination', DegradationPipeline()),
    ]
 
    n = len(transforms)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()
 
    for ax, (title, t) in zip(axes, transforms):
        if t is None:
            view = img
        elif isinstance(t, DegradationPipeline):
            # pipeline returns a normalized tensor — undo it for display
            tensor = t(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            denorm = tensor.clone()
            for c in range(3):
                denorm[c] = denorm[c] * std[c] + mean[c]
            denorm = denorm.clamp(0, 1).permute(1, 2, 0).numpy()
            view = denorm
        else:
            view = t(img)
 
        ax.imshow(view)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
 
    # hide any unused subplots
    for ax in axes[n:]:
        ax.axis('off')
 
    plt.suptitle(f'Source: {img_path.parent.name}  —  {img_path.name}',
                 fontsize=10, y=1.02)
    plt.tight_layout()
 
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
 
 
if __name__ == '__main__':
    main()