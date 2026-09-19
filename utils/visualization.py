import numpy as np
from PIL import Image

from fundus_lesions_toolkit.constants import lesions2names


LESION_CHANNELS = {
    1: "CTW",
    2: "EX",
    3: "HE",
    4: "MA",
}


def probabilities_to_masks(
    prediction,
    threshold=0.5,
):
    """
    Convert a 5-channel lesion probability tensor into binary masks.

    Channel mapping:
        0 = Background
        1 = Cotton Wool Spots
        2 = Exudates
        3 = Hemorrhages
        4 = Microaneurysms

    Returns:
        Dictionary containing one binary NumPy mask per lesion class.
    """

    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError(
            f"Expected prediction with shape (5, H, W), "
            f"but received {prediction.shape}"
        )

    if prediction.shape[0] != 5:
        raise ValueError(
            f"Expected 5 segmentation channels, "
            f"but received {prediction.shape[0]}"
        )

    masks = {}

    for channel, code in LESION_CHANNELS.items():
        masks[code] = prediction[channel] >= threshold

    return masks


def mask_statistics(masks):
    """
    Calculate basic pixel statistics for each lesion mask.

    Returns:
        Dictionary containing pixel count and percentage of image area.
    """

    statistics = {}

    for code, mask in masks.items():
        pixel_count = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)

        percentage = (
            (pixel_count / total_pixels) * 100
            if total_pixels > 0
            else 0.0
        )

        statistics[code] = {
            "name": lesions2names.get(code, code),
            "pixels": pixel_count,
            "percentage": percentage,
        }

    return statistics


def create_lesion_overlay(
    image,
    masks,
    alpha=0.45,
):
    """
    Create a visualization overlay from the original RGB image
    and lesion masks.

    Different lesion classes are represented using different
    RGB colors.

    Returns:
        PIL.Image
    """

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), "
            f"but received {image.shape}"
        )

    base = image.astype(np.float32)

    colors = {
        "CTW": np.array([255, 255, 0], dtype=np.float32),
        "EX": np.array([0, 255, 0], dtype=np.float32),
        "HE": np.array([255, 0, 0], dtype=np.float32),
        "MA": np.array([255, 0, 255], dtype=np.float32),
    }

    overlay = base.copy()

    for code, mask in masks.items():
        if code not in colors:
            continue

        mask = np.asarray(mask, dtype=bool)

        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask for {code} has shape {mask.shape}, "
                f"but image has shape {image.shape[:2]}"
            )

        color = colors[code]

        overlay[mask] = (
            (1.0 - alpha) * overlay[mask]
            + alpha * color
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def create_mask_image(masks):
    """
    Create a combined RGB visualization of all lesion masks.

    Returns:
        PIL.Image
    """

    if not masks:
        raise ValueError("No lesion masks were provided.")

    first_mask = next(iter(masks.values()))
    height, width = first_mask.shape

    output = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    colors = {
        "CTW": np.array([255, 255, 0], dtype=np.uint8),
        "EX": np.array([0, 255, 0], dtype=np.uint8),
        "HE": np.array([255, 0, 0], dtype=np.uint8),
        "MA": np.array([255, 0, 255], dtype=np.uint8),
    }

    for code, mask in masks.items():
        if code not in colors:
            continue

        output[np.asarray(mask, dtype=bool)] = colors[code]

    return Image.fromarray(output)