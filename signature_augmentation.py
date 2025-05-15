import os
import cv2
import numpy as np
from albumentations import (
    Compose, Rotate, RandomBrightnessContrast, GaussianBlur,
    ShiftScaleRotate, GaussNoise, Perspective, Resize
)
from albumentations.augmentations.transforms import ToGray
from tqdm import tqdm
import math

# Define the augmentation pipeline
def get_augmentations():
    return Compose([
        ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.9, border_mode=0),
        Perspective(scale=(0.02, 0.05), p=0.4),
        RandomBrightnessContrast(p=0.5),
        GaussianBlur(blur_limit=(3, 5), p=0.3),
        GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        Rotate(limit=5, border_mode=0, p=0.5),
        ToGray(p=1.0),
        Resize(300, 150)  # Resize to standard signature image dimensions
    ])

# Perform augmentation on signature images
def augment_images(input_folder, output_folder, total_augmented_images=1000):
    os.makedirs(output_folder, exist_ok=True)
    aug = get_augmentations()

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_originals = len(image_files)

    if num_originals == 0:
        print("No original images found.")
        return

    augmentations_per_image = math.ceil(total_augmented_images / num_originals)
    print(f"Approximately {augmentations_per_image} augmentations will be generated per image.")

    saved_count = 0
    for img_name in tqdm(image_files, desc="Augmenting images"):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Couldn't read image: {img_path}")
            continue

        for _ in range(augmentations_per_image):
            if saved_count >= total_augmented_images:
                break
            augmented = aug(image=image)['image']
            out_name = f"aug_{saved_count}.png"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, augmented)
            saved_count += 1

    print(f"Successfully created {saved_count} augmented signature images.")

# Run the script
if __name__ == "__main__":
    # Set your input and output folders here
    input_folder = "signatures/original"       # Example: place original images here
    output_folder = "signatures/augmented"     # Augmented images will be saved here

    print("Starting signature image augmentation...")
    augment_images(input_folder, output_folder, total_augmented_images=1000)
    print("Augmentation completed.")
