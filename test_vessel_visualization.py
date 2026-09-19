import numpy as np
import torch
from PIL import Image

from fundus_image_toolbox.vessel_segmentation import (
    load_segmentation_ensemble,
    ensemble_predict_segmentation,
)


INPUT_IMAGE = "test_data/test_fundus.jpg"
MASK_OUTPUT = "outputs/vessel_mask.png"
OVERLAY_OUTPUT = "outputs/vessel_overlay.png"


def main():
    print("Loading fundus image...")

    image = Image.open(INPUT_IMAGE).convert("RGB")
    image_array = np.asarray(image)

    print("Original image size:", image.size)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    print("Loading FR-U-Net ensemble...")
    models = load_segmentation_ensemble(device=device)

    print("Models loaded:", len(models))

    print("Running vessel segmentation...")

    vessel_mask = ensemble_predict_segmentation(
        models,
        image,
        device=device,
        size=(512, 512),
        threshold=0.5,
    )

    vessel_mask = np.asarray(vessel_mask)

    print("Mask shape:", vessel_mask.shape)
    print("Mask dtype:", vessel_mask.dtype)
    print("Mask min:", float(vessel_mask.min()))
    print("Mask max:", float(vessel_mask.max()))

    vessel_pixels = int((vessel_mask > 0).sum())

    print("Vessel pixels:", vessel_pixels)

    if vessel_pixels == 0:
        print("WARNING: No vessel pixels detected.")

    # Save vessel mask at the original image resolution.
    mask_uint8 = (vessel_mask > 0).astype(np.uint8) * 255

    mask_image = Image.fromarray(mask_uint8, mode="L")
    mask_image.save(MASK_OUTPUT)

    # Use the original image because the returned mask
    # has the original image dimensions.
    overlay = image_array.copy()

    # Highlight detected vessels.
    overlay[vessel_mask > 0] = [255, 255, 255]

    # Blend original image and vessel-highlighted image.
    overlay = (
        0.65 * image_array + 0.35 * overlay
    ).clip(0, 255).astype(np.uint8)

    overlay_image = Image.fromarray(overlay, mode="RGB")
    overlay_image.save(OVERLAY_OUTPUT)

    print()
    print("Saved vessel mask:", MASK_OUTPUT)
    print("Saved vessel overlay:", OVERLAY_OUTPUT)
    print()
    print("Vessel visualization test: SUCCESS")


if __name__ == "__main__":
    main()