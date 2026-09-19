import numpy as np
from PIL import Image

import tensorflow as tf
import keras
from huggingface_hub import hf_hub_download


MODEL_REPO = "Aldahmashi/DR-EfficientNetB0"
MODEL_FILE = "final_model.keras"

IMAGE_PATH = "test_data/test_fundus.jpg"

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
]


def load_model():
    print("Loading model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = keras.saving.load_model(
        model_path
    )

    print("Model loaded.")
    return model


def prepare_image(image):
    """
    Prepare the test image exactly as the DR classifier expects.
    """
    image = image.convert("RGB")

    original = np.array(image)

    resized = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    array = np.array(
        resized
    ).astype(np.float32)

    batch = np.expand_dims(
        array,
        axis=0,
    )

    return original, batch


def print_prediction(
    name,
    predictions,
):
    probabilities = tf.nn.softmax(
        predictions,
        axis=-1,
    ).numpy()[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for index, probability in enumerate(
        probabilities
    ):
        print(
            f"Class {index} - "
            f"{CLASS_NAMES[index]}: "
            f"{probability * 100:.4f}%"
        )

    print()
    print(
        f"Predicted class: {predicted_class}"
    )

    print(
        f"Predicted label: "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence: "
        f"{probabilities[predicted_class] * 100:.4f}%"
    )

    return probabilities


def main():

    print("Loading test image...")

    image = Image.open(
        IMAGE_PATH
    )

    print(
        f"Original image size: "
        f"{image.size}"
    )

    model = load_model()

    original, batch = prepare_image(
        image
    )

    print()
    print(
        f"Prepared tensor shape: "
        f"{batch.shape}"
    )

    print(
        f"Prepared tensor dtype: "
        f"{batch.dtype}"
    )

    print(
        f"Prepared tensor min: "
        f"{batch.min():.4f}"
    )

    print(
        f"Prepared tensor max: "
        f"{batch.max():.4f}"
    )

    # ---------------------------------------------------------
    # PATH 1
    # Direct call to the COMPLETE trained model.
    # ---------------------------------------------------------

    direct_output = model(
        batch,
        training=False,
    )

    direct_probabilities = print_prediction(
        "PATH 1 — COMPLETE MODEL",
        direct_output,
    )

    # ---------------------------------------------------------
    # PATH 2
    # Manually reproduce the nested EfficientNet + classifier
    # path used by the Grad-CAM implementation.
    # ---------------------------------------------------------

    efficientnet = model.get_layer(
        "efficientnetb0"
    )

    efficientnet_output = efficientnet(
        batch,
        training=False,
    )

    print()
    print(
        "EfficientNet output shape:",
        efficientnet_output.shape,
    )

    x = efficientnet_output

    classifier_layers = []

    found = False

    for layer in model.layers:

        if layer.name == efficientnet.name:
            found = True
            continue

        if found:
            classifier_layers.append(
                layer
            )

    print(
        "Classifier layers:"
    )

    for layer in classifier_layers:
        print(
            f"  {layer.name} "
            f"({layer.__class__.__name__})"
        )

    for layer in classifier_layers:
        x = layer(
            x,
            training=False,
        )

    manual_output = x

    manual_probabilities = print_prediction(
        "PATH 2 — MANUAL EFFICIENTNET + CLASSIFIER",
        manual_output,
    )

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    difference = np.abs(
        direct_probabilities
        - manual_probabilities
    )

    for index, diff in enumerate(
        difference
    ):
        print(
            f"Class {index}: "
            f"difference = "
            f"{diff:.10f}"
        )

    print()
    print(
        "Maximum probability difference:",
        f"{difference.max():.10f}",
    )

    if np.allclose(
        direct_probabilities,
        manual_probabilities,
        atol=1e-6,
    ):
        print()
        print(
            "RESULT: PATHS MATCH"
        )
        print(
            "The model and classifier head "
            "are producing consistent predictions."
        )
    else:
        print()
        print(
            "RESULT: PATHS DO NOT MATCH"
        )
        print(
            "There is a difference between "
            "the complete model and our manual path."
        )


if __name__ == "__main__":
    main()