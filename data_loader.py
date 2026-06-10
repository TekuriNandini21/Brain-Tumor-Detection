import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(dataset_path):

    images = []
    masks = []

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            if file.endswith(".tif") and "_mask" not in file:

                image_path = os.path.join(root, file)

                mask_path = os.path.join(
                    root,
                    file.replace(".tif", "_mask.tif")
                )

                if os.path.exists(mask_path):

                    image = cv2.imread(
                        image_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    mask = cv2.imread(
                        mask_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if image is None or mask is None:
                        continue

                    image = cv2.resize(
                        image,
                        (IMG_SIZE, IMG_SIZE)
                    )

                    mask = cv2.resize(
                        mask,
                        (IMG_SIZE, IMG_SIZE)
                    )

                    image = image.astype(np.float32) / 255.0

                    mask = mask.astype(np.float32) / 255.0

                    mask = (mask > 0.5).astype(np.float32)

                    images.append(image)
                    masks.append(mask)

    return np.array(images), np.array(masks)