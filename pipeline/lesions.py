from pathlib import Path

import numpy as np
from PIL import Image

from fundus_lesions_toolkit.models.segmentation import segment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_ARCH = "unet"
MODEL_ENCODER = "seresnext50_32x4d"
IMAGE_SIZE = 1024
THRESHOLD = 0.5


CLASS_NAMES = [
    "background",
    "cotton_wool_spots",
    "exudates",
    "hemorrhages",
    "microaneurysms",
]


def get_device():
    """
    Select the best available PyTorch device.

    Apple Silicon:
        MPS

    Other systems:
        CUDA if available, otherwise CPU.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def prepare_image(image):
    """Convert an image to RGB and resize it to 1024x1024."""
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        array = np.asarray(image)

        if array.dtype != np.uint8:
            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        image = Image.fromarray(array).convert("RGB")

    return image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )


def probabilities_to_masks(
    prediction,
    threshold=THRESHOLD,
):
    """Convert model probabilities into binary class masks."""
    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError(
            "Expected lesion prediction with shape (5, H, W), "
            f"received {prediction.shape}"
        )

    if prediction.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} lesion classes, "
            f"received {prediction.shape[0]}"
        )

    masks = prediction >= threshold

    # Background is not a lesion.
    masks[0] = False

    return masks


def create_mask_image(masks):
    """
    Create a class-index mask.

    0 = background
    1 = cotton wool spots
    2 = exudates
    3 = hemorrhages
    4 = microaneurysms
    """
    height, width = masks.shape[1:]

    mask_image = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for class_index in range(
        1,
        len(CLASS_NAMES),
    ):
        mask_image[masks[class_index]] = class_index

    return Image.fromarray(
        mask_image,
        mode="L",
    )


def create_lesion_overlay(image, masks):
    """Create a visualization of lesion evidence."""
    image_array = np.asarray(
        image.convert("RGB")
    ).copy()

    if masks.shape[1:] != image_array.shape[:2]:
        raise ValueError(
            "Lesion mask dimensions do not match image dimensions: "
            f"mask={masks.shape[1:]}, "
            f"image={image_array.shape[:2]}"
        )

    overlay = image_array.astype(
        np.float32
    )

    # Visualization colors only.
    class_colors = {
        1: np.array(
            [255, 255, 0],
            dtype=np.float32,
        ),
        2: np.array(
            [255, 0, 255],
            dtype=np.float32,
        ),
        3: np.array(
            [255, 0, 0],
            dtype=np.float32,
        ),
        4: np.array(
            [0, 255, 255],
            dtype=np.float32,
        ),
    }

    for class_index, color in class_colors.items():
        class_mask = masks[class_index]

        if np.any(class_mask):
            overlay[class_mask] = (
                overlay[class_mask] * 0.45
                + color * 0.55
            )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        overlay,
        mode="RGB",
    )


def calculate_statistics(masks):
    """Calculate pixel counts and fractions for each lesion class."""
    total_pixels = int(
        masks.shape[1] * masks.shape[2]
    )

    statistics = {}

    for class_index in range(
        1,
        len(CLASS_NAMES),
    ):
        pixel_count = int(
            masks[class_index].sum()
        )

        fraction = (
            float(pixel_count)
            / float(total_pixels)
            if total_pixels > 0
            else 0.0
        )

        statistics[
            CLASS_NAMES[class_index]
        ] = {
            "class_index": class_index,
            "pixels": pixel_count,
            "fraction": fraction,
            "detected": pixel_count > 0,
        }

    return statistics


def tensor_to_numpy(prediction):
    """
    Safely convert a PyTorch tensor from MPS/CUDA/CPU
    into a NumPy array on the CPU.
    """
    if hasattr(prediction, "detach"):
        prediction = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(prediction)


def segment_lesions(
    image,
    threshold=THRESHOLD,
    device=None,
):
    """
    Run the pretrained U-Net + SEResNeXt50_32x4d
    retinal lesion segmentation model.

    The toolkit expects a NumPy HWC image and returns
    a PyTorch tensor. The tensor is explicitly moved to
    CPU before NumPy conversion.
    """
    if isinstance(image, Image.Image):
        original_image = image.convert("RGB")
    else:
        array = np.asarray(image)

        if array.dtype != np.uint8:
            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        original_image = Image.fromarray(
            array
        ).convert("RGB")

    if device is None:
        device = get_device()

    model_image = prepare_image(
        original_image
    )

    # Toolkit expects NumPy HWC input.
    model_array = np.asarray(
        model_image,
        dtype=np.uint8,
    )

    print(
        f"Lesion model device: {device}"
    )

    prediction = segment(
        model_array,
        arch=MODEL_ARCH,
        encoder=MODEL_ENCODER,
        image_resolution=IMAGE_SIZE,
        autofit_resolution=False,
        reverse_autofit=False,
        device=device,
    )

    # The model returns a PyTorch tensor on MPS.
    # Move it to CPU before converting to NumPy.
    prediction = tensor_to_numpy(
        prediction
    )

    expected_shape = (
        len(CLASS_NAMES),
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if prediction.shape != expected_shape:
        raise ValueError(
            "Unexpected lesion model output shape: "
            f"received {prediction.shape}, "
            f"expected {expected_shape}"
        )

    masks = probabilities_to_masks(
        prediction,
        threshold=threshold,
    )

    statistics = calculate_statistics(
        masks
    )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1024x1024 model-space mask.
    mask_image = create_mask_image(
        masks
    )

    mask_path = (
        OUTPUTS_DIR
        / "lesion_mask.png"
    )

    mask_image.save(
        mask_path
    )

    # 1024x1024 model-space overlay.
    model_overlay = create_lesion_overlay(
        model_image,
        masks,
    )

    model_overlay_path = (
        OUTPUTS_DIR
        / "lesion_overlay.png"
    )

    model_overlay.save(
        model_overlay_path
    )

    # Resize masks back to original image dimensions.
    original_masks = np.zeros(
        (
            len(CLASS_NAMES),
            original_image.height,
            original_image.width,
        ),
        dtype=bool,
    )

    for class_index in range(
        1,
        len(CLASS_NAMES),
    ):
        class_mask = Image.fromarray(
            (
                masks[class_index]
                .astype(np.uint8)
                * 255
            ),
            mode="L",
        )

        class_mask = class_mask.resize(
            original_image.size,
            Image.Resampling.NEAREST,
        )

        original_masks[class_index] = (
            np.asarray(class_mask) > 127
        )

    original_overlay = create_lesion_overlay(
        original_image,
        original_masks,
    )

    original_overlay_path = (
        OUTPUTS_DIR
        / "lesion_overlay_original.png"
    )

    original_overlay.save(
        original_overlay_path
    )

    return {
        "prediction": prediction,
        "masks": masks,
        "statistics": statistics,
        "threshold": float(threshold),
        "device": device,
        "model": {
            "architecture": MODEL_ARCH,
            "encoder": MODEL_ENCODER,
            "input_size": IMAGE_SIZE,
        },
        "image_size": original_image.size,
        "mask_path": str(mask_path),
        "overlay_path": str(
            model_overlay_path
        ),
        "original_overlay_path": str(
            original_overlay_path
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DR-XPLAIN — LESION SEGMENTATION TEST")
    print("=" * 60)

    test_image_path = (
        PROJECT_ROOT
        / "test_data"
        / "test_fundus.jpg"
    )

    if not test_image_path.exists():
        print()
        print("ERROR: Test image not found:")
        print(test_image_path)
        raise SystemExit(1)

    print()
    print(
        f"Loading image: {test_image_path}"
    )

    image = Image.open(
        test_image_path
    ).convert("RGB")

    print(
        f"Original image size: {image.size}"
    )

    device = get_device()

    print(
        f"Selected device: {device}"
    )

    print()
    print(
        "Running U-Net + "
        "SEResNeXt50_32x4d..."
    )

    result = segment_lesions(
        image,
        device=device,
    )

    print()
    print(
        f"Prediction shape: "
        f"{result['prediction'].shape}"
    )

    print()
    print("Lesion statistics:")

    for class_name, stats in (
        result["statistics"].items()
    ):
        print(
            f"  {class_name}: "
            f"{stats['pixels']} pixels "
            f"({stats['fraction']:.8f})"
        )

    print()
    print(
        f"Mask saved: "
        f"{result['mask_path']}"
    )

    print(
        f"Overlay saved: "
        f"{result['overlay_path']}"
    )

    print(
        "Original-size overlay saved: "
        f"{result['original_overlay_path']}"
    )

    print()
    print(
        "Lesion segmentation test: SUCCESS"
    )

    print("=" * 60)