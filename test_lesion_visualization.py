import os

import numpy as np
import torch
from PIL import Image

from fundus_lesions_toolkit.models.segmentation import segment

from utils.visualization import (
    probabilities_to_masks,
    mask_statistics,
    create_lesion_overlay,
    create_mask_image,
)


INPUT_PATH = "test_data/test_fundus.jpg"
OUTPUT_DIR = "outputs"

MODEL_SIZE = (1024, 1024)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading image...")
    original = Image.open(INPUT_PATH).convert("RGB")

    print("Original size:", original.size)

    image = original.resize(MODEL_SIZE)
    image_array = np.array(image)

    print("Model input shape:", image_array.shape)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Running lesion segmentation...")

    prediction = segment(
        image_array,
        encoder="seresnext50_32x4d",
        device=device,
        autofit_resolution=False,
        reverse_autofit=False,
    )

    print("Prediction shape:", tuple(prediction.shape))

    print("\nMaximum probability by class:")

    class_names = [
        "Background",
        "Cotton Wool Spots",
        "Exudates",
        "Hemorrhages",
        "Microaneurysms",
    ]

    for index, name in enumerate(class_names):
        maximum = float(prediction[index].max())
        mean = float(prediction[index].mean())

        print(
            f"{name}: "
            f"max={maximum:.8f}, "
            f"mean={mean:.8f}"
        )

    masks = probabilities_to_masks(
        prediction,
        threshold=0.5,
    )

    statistics = mask_statistics(masks)

    print("\nLesion statistics at threshold 0.5:")

    for code, values in statistics.items():
        print(
            f"{values['name']}: "
            f"{values['pixels']} pixels "
            f"({values['percentage']:.6f}%)"
        )

    overlay = create_lesion_overlay(
        image,
        masks,
        alpha=0.45,
    )

    combined_mask = create_mask_image(masks)

    original.save(
        os.path.join(OUTPUT_DIR, "original_fundus.jpg")
    )

    image.save(
        os.path.join(OUTPUT_DIR, "fundus_1024.jpg")
    )

    combined_mask.save(
        os.path.join(OUTPUT_DIR, "lesion_mask.png")
    )

    overlay.save(
        os.path.join(OUTPUT_DIR, "lesion_overlay.png")
    )

    print("\nLesion visualization test: SUCCESS")


if __name__ == "__main__":
    main()