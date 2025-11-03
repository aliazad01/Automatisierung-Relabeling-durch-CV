from pyspark.sql import functions as F
import os
import random
import cv2
import numpy as np
import pandas as pd
from transforms.api import transform, Input, Output
from myproject.datasets.libs.utils import (
    read_dir_recursive,
)
from myproject.datasets.libs.lib_foundry_filesystem import (
    save_foundry_files_in_dir,
    write_files_to_dataset
)

import tempfile

@transform(
    rotated_images=Output("/Trans-o-flex/[Use Case] MS-Daten/data/09_retraining/create_train_data_4_rotation/output/rotated_images"),
    source_df=Input("ri.foundry.main.dataset.64c12a63-bf47-40b0-87ab-6733b867312b"),
)
def compute(source_df, rotated_images):

    temp_dir = tempfile.mkdtemp()

    # Get a list of the images which should be analyzed and save them as temporary files.
    _, train_temp_dir = save_foundry_files_in_dir(source_df, temp_dir, sub_dir_name="data")
    # train_temp_dir = train_temp_dir + "/data"

    def rotate_image_same_size(image, angle):
        """
        Rotates the image while keeping the same dimensions as the original.
        This will crop parts that go outside the original boundaries.
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation with original dimensions
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated

    def rotate_image_fit_content(image, angle):
        """
        Alternative approach: Rotates and then resizes to fit original dimensions.
        This preserves more content but may introduce scaling artifacts.
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation with expanded canvas
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # Resize back to original dimensions
        resized = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_CUBIC)
        return resized

    def generate_rotated_dataset(input_dir, output_dir, num_rotations, angle_range, near_180_ratio, preserve_size_method="crop"):
        """
        Generates a dataset of rotated images with consistent dimensions.
        preserve_size_method: 
        - "crop": Keeps original dimensions by cropping (may lose some content)
        - "resize": Expands canvas then resizes back (preserves content but may introduce scaling)
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset = []

        near_180_count = int(num_rotations * near_180_ratio)
        uniform_count = num_rotations - near_180_count

        # Choose rotation function based on method
        rotate_func = rotate_image_same_size if preserve_size_method == "crop" else rotate_image_fit_content

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: Unable to load image {filename}. Skipping...")
                    continue

                basename = os.path.splitext(filename)[0]
                original_shape = image.shape[:2]  # Store original dimensions

                # --- Near ±180° rotations ---
                for i in range(near_180_count):
                    base_angle = 180 if random.random() < 0.5 else -180
                    jitter = random.uniform(-20, 20)
                    angle = base_angle + jitter

                    # Normalize to [-180, 180]
                    if angle <= -180:
                        angle += 360
                    elif angle >= 180:
                        angle -= 360

                    rotated_image = rotate_func(image, angle)

                    # Verify dimensions match original
                    assert rotated_image.shape[:2] == original_shape, f"Dimension mismatch for {filename}"

                    output_filename = f"{basename}_near180_{i}_{int(angle)}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, rotated_image)
                    dataset.append((output_filename, round(angle, 2)))

                # --- Uniform rotations ---
                for i in range(uniform_count):
                    while True:
                        angle = random.uniform(*angle_range)
                        if not (160 <= abs(angle) <= 180):  # avoid ±160–180° overlap
                            break

                    rotated_image = rotate_func(image, angle)

                    # Verify dimensions match original
                    assert rotated_image.shape[:2] == original_shape, f"Dimension mismatch for {filename}"

                    output_filename = f"{basename}_uniform_{i}_{int(angle)}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, rotated_image)
                    dataset.append((output_filename, round(angle, 2)))

        return dataset

    input_dir = train_temp_dir
    output_dir = tempfile.mkdtemp()
    num_rotations_per_image = 150
    angle_range = (-180, 180)
    near_180_ratio = 0.2  # 20% near ±180°

    rotated_dataset = generate_rotated_dataset(input_dir, output_dir, num_rotations_per_image, angle_range, near_180_ratio)

    # save CSV labels
    df = pd.DataFrame(rotated_dataset, columns=["filename", "angle"])
    df.to_csv(os.path.join(output_dir, "rotated_dataset_labels.csv"), index=False)
    # ────────────────────────────────
    # Save checkpoint and logs to checkpoint dataset.
    # ────────────────────────────────

    output_filepaths = read_dir_recursive(output_dir)

    # Save the new checkpoints
    write_files_to_dataset(
        dest_dataset=rotated_images,
        files_to_write=output_filepaths,
        #output_func=output_func
    )
