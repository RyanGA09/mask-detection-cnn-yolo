import os
import shutil
import random

def split_dataset(image_dir, label_dir, output_img_dir, output_lbl_dir, split_ratio=0.8):
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split, files in zip(['train', 'val'], [train_imgs, val_imgs]):
        for f in files:
            shutil.copy(os.path.join(image_dir, f), os.path.join(output_img_dir, split, f))
            label_file = os.path.splitext(f)[0] + ".txt"
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_lbl_dir, split, label_file))

# Example usage:
# split_dataset("images", "labels", "data/images", "data/labels")
