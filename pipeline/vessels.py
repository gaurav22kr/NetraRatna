from pathlib import Path

import numpy as np
from PIL import Image

from fundus_image_toolbox.vessel_segmentation import (
    load_segmentation_ensemble,
    ensemble_predict_segmentation,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

VESSEL_SIZE = (512, 512)
VESSEL_THRESHOLD = 0.5


def get_device():
    """
    Select the best available device.

    Apple Silicon:
        MPS

    Other systems:
        CUDA if available, otherwise CPU.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda:0"

    except Exception:
        pass

    return "cpu"


def load_vessel_models(device=None):
    """
    Load the pretrained FR-U-Net vessel segmentation ensemble.

    The Fundus Image Toolbox downloads/caches the five pretrained
    FR-U-Net models automatically.
    """
    if device is None:
        device = get_device()

    models = load_segmentation_ensemble(device=device)

    return models


def save_vessel_mask(mask):
    """
    Save the binary vessel mask as a PNG image.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    mask_array = np.asarray(mask)

    if mask_array.dtype != np.uint8:
        mask_array = (mask_array > 0.5).astype(np.uint8) * 255

    image = Image.fromarray(mask_array, mode="L")
    output_path = OUTPUTS_DIR / "vessel_mask.png"
    image.save(output_path)

    return output_path


def save_vessel_overlay(image, mask):
    """
    Save a visualization of the vessel mask over the fundus image.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    original = image.convert("RGB")

    original_array = np.asarray(original).copy()

    mask_array = np.asarray(mask) > 0.5

    if mask_array.shape != original_array.shape[:2]:
        raise ValueError(
            "Vessel mask dimensions do not match the original image: "
            f"mask={mask_array.shape}, "
            f"image={original_array.shape[:2]}"
        )

    # Highlight vessels without changing the original image geometry.
    overlay = original_array.copy()

    vessel_pixels = mask_array

    # Add a bright vessel highlight.
    overlay[vessel_pixels] = np.array([255, 255, 0], dtype=np.uint8)

    # Blend original and highlighted image.
    blended = (
        original_array.astype(np.float32) * 0.55
        + overlay.astype(np.float32) * 0.45
    )

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    output_path = OUTPUTS_DIR / "vessel_overlay.png"
    Image.fromarray(blended, mode="RGB").save(output_path)

    return output_path


def segment_vessels(
    image,
    models=None,
    device=None,
    threshold=VESSEL_THRESHOLD,
):
    """
    Run pretrained FR-U-Net vessel segmentation.

    Parameters
    ----------
    image:
        PIL.Image, NumPy array, or compatible image input.

    models:
        Optional already-loaded FR-U-Net ensemble.

    device:
        Optional device such as "mps", "cuda:0", or "cpu".

    threshold:
        Binary vessel threshold.

    Returns
    -------
    dict
        Contains:
            mask
            vessel_pixels
            vessel_fraction
            device
            threshold
            mask_path
            overlay_path
    """
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        array = np.asarray(image)

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(array).convert("RGB")

    if device is None:
        device = get_device()

    if models is None:
        models = load_vessel_models(device=device)

    result = ensemble_predict_segmentation(
        models,
        pil_image,
        device=device,
        size=VESSEL_SIZE,
        threshold=threshold,
    )

    mask = np.asarray(result)

    # The toolkit reverse-resizes the result to the original image size.
    expected_shape = (pil_image.height, pil_image.width)

    if mask.shape != expected_shape:
        raise ValueError(
            "Unexpected vessel mask dimensions: "
            f"received {mask.shape}, expected {expected_shape}"
        )

    binary_mask = mask > threshold

    vessel_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)

    vessel_fraction = (
        float(vessel_pixels) / float(total_pixels)
        if total_pixels > 0
        else 0.0
    )

    mask_path = save_vessel_mask(binary_mask)
    overlay_path = save_vessel_overlay(pil_image, binary_mask)

    return {
        "mask": binary_mask,
        "vessel_pixels": vessel_pixels,
        "vessel_fraction": vessel_fraction,
        "device": device,
        "threshold": float(threshold),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "image_size": pil_image.size,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DR-XPLAIN — VESSEL SEGMENTATION TEST")
    print("=" * 60)

    test_image_path = PROJECT_ROOT / "test_data" / "test_fundus.jpg"

    if not test_image_path.exists():
        print()
        print("ERROR: Test image not found:")
        print(test_image_path)
        raise SystemExit(1)

    print()
    print(f"Loading image: {test_image_path}")

    image = Image.open(test_image_path).convert("RGB")

    print(f"Image size: {image.size}")

    device = get_device()

    print(f"Device: {device}")
    print()
    print("Loading FR-U-Net ensemble...")

    models = load_vessel_models(device=device)

    print(f"Loaded {len(models)} vessel models.")
    print()
    print("Running vessel segmentation...")

    result = segment_vessels(
        image,
        models=models,
        device=device,
    )

    print()
    print(f"Mask shape: {result['mask'].shape}")
    print(f"Vessel pixels: {result['vessel_pixels']}")
    print(f"Vessel fraction: {result['vessel_fraction']:.4f}")
    print(f"Mask saved: {result['mask_path']}")
    print(f"Overlay saved: {result['overlay_path']}")

    print()
    print("Vessel segmentation test: SUCCESS")
    print("=" * 60)