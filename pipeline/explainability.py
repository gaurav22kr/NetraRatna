from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from pipeline.grading import (
    CLASS_NAMES,
    preprocess_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_connected_gradcam_model(model):
    """
    Build a connected Grad-CAM model.

    The DR model structure is:

        0 -> InputLayer
        1 -> EfficientNetB0 backbone
        2 -> GlobalAveragePooling2D
        3 -> Dropout
        4 -> Dense classifier

    We expose:
        1. EfficientNet top_conv feature maps
        2. EfficientNet output before the classifier head
    """

    # The EfficientNetB0 backbone is layer 1.
    efficientnet = model.layers[1]

    target_layer = efficientnet.get_layer(
        "top_conv"
    )

    feature_and_output_model = keras.Model(
        inputs=efficientnet.input,
        outputs=[
            target_layer.output,
            efficientnet.output,
        ],
    )

    # Layers after EfficientNet form the classifier head.
    classifier_layers = model.layers[2:]

    return (
        efficientnet,
        target_layer,
        feature_and_output_model,
        classifier_layers,
    )


def apply_classifier_head(
    efficientnet_output,
    classifier_layers,
):
    """
    Apply the original DR classification head
    to the EfficientNet output.
    """

    x = efficientnet_output

    for layer in classifier_layers:
        x = layer(x)

    return x


def save_gradcam_heatmap(
    heatmap,
    output_path,
):
    """
    Save the Grad-CAM heatmap as a PNG image.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    plt.axis("off")

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150,
    )

    plt.close()


def create_gradcam_overlay(
    image,
    heatmap,
    output_path,
    alpha=0.45,
):
    """
    Create and save a Grad-CAM overlay
    on the original fundus image.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not isinstance(
        image,
        Image.Image,
    ):
        image = Image.fromarray(
            np.asarray(image).astype(
                np.uint8
            )
        )

    image = image.convert("RGB")

    image_array = np.asarray(
        image
    ).astype(
        np.uint8
    )

    height = image_array.shape[0]
    width = image_array.shape[1]

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    heatmap_array = np.asarray(
        heatmap_image
    ).astype(
        np.float32
    ) / 255.0

    colormap = plt.colormaps["jet"]

    colored_heatmap = colormap(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(
        np.uint8
    )

    overlay = (
        image_array.astype(
            np.float32
        ) * (1.0 - alpha)
        +
        colored_heatmap.astype(
            np.float32
        ) * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )


def generate_gradcam(
    model,
    image,
    class_index=None,
):
    """
    Generate standard Grad-CAM for the DR classifier.

    Returns:
        heatmap
        probabilities
        class index
        class name
        confidence
        output paths
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_tensor = preprocess_image(
        image
    )

    (
        efficientnet,
        target_layer,
        feature_and_output_model,
        classifier_layers,
    ) = build_connected_gradcam_model(
        model
    )

    print(
        f"Grad-CAM target layer: "
        f"{target_layer.name}"
    )

    with tf.GradientTape() as tape:

        feature_maps, efficientnet_output = (
            feature_and_output_model(
                input_tensor,
                training=False,
            )
        )

        predictions = apply_classifier_head(
            efficientnet_output,
            classifier_layers,
        )

        predictions = tf.convert_to_tensor(
            predictions
        )

        if class_index is None:
            class_index = int(
                tf.argmax(
                    predictions[0]
                ).numpy()
            )

        class_score = predictions[
            :,
            class_index,
        ]

    gradients = tape.gradient(
        class_score,
        feature_maps,
    )

    if gradients is None:
        raise RuntimeError(
            "TensorFlow returned None for the "
            "Grad-CAM gradients."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    feature_maps = feature_maps[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        feature_maps
        * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if float(
        maximum.numpy()
    ) > 0:

        heatmap = (
            heatmap
            / maximum
        )

    else:

        heatmap = tf.zeros_like(
            heatmap
        )

    heatmap = heatmap.numpy()

    probabilities = tf.nn.softmax(
        predictions,
        axis=-1,
    ).numpy()[0]

    heatmap_path = (
        OUTPUT_DIR
        / "gradcam_heatmap.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / "gradcam_overlay.png"
    )

    save_gradcam_heatmap(
        heatmap,
        heatmap_path,
    )

    create_gradcam_overlay(
        image,
        heatmap,
        overlay_path,
    )

    print(
        f"Grad-CAM heatmap saved: "
        f"{heatmap_path}"
    )

    print(
        f"Grad-CAM overlay saved: "
        f"{overlay_path}"
    )

    return {
        "heatmap": heatmap,
        "probabilities": probabilities,
        "class_index": class_index,
        "class_name": CLASS_NAMES[
            class_index
        ],
        "confidence": float(
            probabilities[
                class_index
            ]
        ),
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
    }


if __name__ == "__main__":
    print(
        "Grad-CAM module loaded successfully."
    )