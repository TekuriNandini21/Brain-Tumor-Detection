import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("tumor_model.h5")

def calculate_percentage(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (128,128)
    )

    image_norm = image.astype(np.float32) / 255.0

    prediction = model.predict(
        image_norm.reshape(1,128,128,1),
        verbose=0
    )

    print("Prediction Max:", prediction.max())
    print("Prediction Min:", prediction.min())

    mask = (prediction > 0.3).astype(np.uint8)

    tumor_pixels = np.count_nonzero(mask)

    total_pixels = mask.shape[1] * mask.shape[2]

    percentage = (
        tumor_pixels / total_pixels
    ) * 100

    return percentage, mask