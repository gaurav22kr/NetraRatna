from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def assess_quality(image):
    """
    Perform a lightweight fundus image quality assessment.

    This is a preprocessing quality check, not a clinically
    validated quality classifier.

    Returns:
        quality_score
        quality_label
        brightness
        contrast
        sharpness
    """

    if not isinstance(image, Image.Image):
        image = Image.fromarray(
            np.asarray(image).astype(np.uint8)
        )

    image = image.convert("RGB")

    image_array = np.asarray(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    sharpness = float(
        laplacian.var()
    )

    # Normalize individual indicators.
    brightness_score = min(
        brightness / 128.0,
        1.0,
    )

    contrast_score = min(
        contrast / 64.0,
        1.0,
    )

    sharpness_score = min(
        sharpness / 500.0,
        1.0,
    )

    quality_score = (
        brightness_score * 0.25
        + contrast_score * 0.35
        + sharpness_score * 0.40
    )

    if quality_score >= 0.70:
        quality_label = "Good"

    elif quality_score >= 0.45:
        quality_label = "Acceptable"

    else:
        quality_label = "Low"

    return {
        "quality_score": float(
            quality_score
        ),
        "quality_label": quality_label,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def enhance_fundus(image):
    """
    Enhance a fundus image using CLAHE.

    CLAHE improves local contrast while limiting
    excessive amplification of noise.
    """

    if not isinstance(image, Image.Image):
        image = Image.fromarray(
            np.asarray(image).astype(np.uint8)
        )

    image = image.convert("RGB")

    image_array = np.asarray(image)

    lab = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        [
            enhanced_l,
            a_channel,
            b_channel,
        ]
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB,
    )

    return Image.fromarray(
        enhanced_rgb
    )


def save_enhanced_image(
    image,
    output_path=None,
):
    """
    Enhance and save the fundus image.
    """

    if output_path is None:
        output_path = (
            OUTPUT_DIR
            / "enhanced_fundus.jpg"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    enhanced = enhance_fundus(
        image
    )

    enhanced.save(
        output_path,
        quality=95,
    )

    return enhanced, output_path


def main():
    test_image_path = (
        PROJECT_ROOT
        / "test_data"
        / "test_fundus.jpg"
    )

    print(
        "DR-XPLAIN — QUALITY + ENHANCEMENT TEST"
    )

    print(
        f"Loading image: {test_image_path}"
    )

    image = Image.open(
        test_image_path
    ).convert("RGB")

    print(
        f"Image size: {image.size}"
    )

    print(
        "\nRunning quality assessment..."
    )

    quality = assess_quality(
        image
    )

    print(
        f"Quality score: "
        f"{quality['quality_score']:.4f}"
    )

    print(
        f"Quality label: "
        f"{quality['quality_label']}"
    )

    print(
        f"Brightness: "
        f"{quality['brightness']:.2f}"
    )

    print(
        f"Contrast: "
        f"{quality['contrast']:.2f}"
    )

    print(
        f"Sharpness: "
        f"{quality['sharpness']:.2f}"
    )

    print(
        "\nRunning CLAHE enhancement..."
    )

    enhanced, output_path = (
        save_enhanced_image(
            image
        )
    )

    print(
        f"Enhanced image saved: "
        f"{output_path}"
    )

    print(
        f"Enhanced image size: "
        f"{enhanced.size}"
    )

    print(
        "\nQuality + enhancement test: SUCCESS"
    )


if __name__ == "__main__":
    main()