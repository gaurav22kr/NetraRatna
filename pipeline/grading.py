import numpy as np
from PIL import Image

import keras
from huggingface_hub import hf_hub_download


MODEL_REPO = "Aldahmashi/DR-EfficientNetB0"
MODEL_FILE = "final_model.keras"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
]

IMAGE_SIZE = (224, 224)


def load_grading_model():
    """
    Download/load the pretrained DR grading model.
    """

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = keras.saving.load_model(model_path)

    return model


def preprocess_image(image):
    """
    Prepare a PIL/numpy image for the DR classifier.

    The model card specifies:
    - RGB input
    - 224 x 224 resolution
    - raw image array passed to the Keras model
    """

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(
            np.asarray(image).astype(np.uint8)
        ).convert("RGB")

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    )

    batch = np.expand_dims(array, axis=0)

    return batch


def predict_dr(image, model=None):
    """
    Predict diabetic retinopathy grade.

    Returns:
        grade
        class_name
        confidence
        probabilities
    """

    if model is None:
        model = load_grading_model()

    batch = preprocess_image(image)

    # The model already outputs a 5-class softmax.
    output = model.predict(
        batch,
        verbose=0,
    )[0]

    probabilities = np.asarray(
        output,
        dtype=np.float32,
    )

    # Safety normalization in case a compatible model returns
    # values that do not sum exactly to 1.
    probability_sum = float(probabilities.sum())

    if probability_sum <= 0:
        raise RuntimeError(
            "DR grading model returned invalid probabilities."
        )

    probabilities = probabilities / probability_sum

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    result = {
        "grade": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[index]: float(probabilities[index])
            for index in range(len(CLASS_NAMES))
        },
    }

    return result


if __name__ == "__main__":

    import sys
    from pathlib import Path

    print("DR-XPLAIN — DR GRADING TEST")
    print("=" * 60)

    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            "python pipeline/grading.py "
            "path/to/fundus_image.jpg"
        )
        raise SystemExit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print(f"Loading image: {image_path}")

    image = Image.open(image_path).convert("RGB")

    print(f"Image size: {image.size}")

    print("\nLoading grading model...")

    model = load_grading_model()

    print("Model loaded.")

    print("\nRunning DR grading...")

    result = predict_dr(
        image,
        model=model,
    )

    print("\nDR probabilities:")

    for class_name, probability in result[
        "probabilities"
    ].items():
        print(
            f"  {class_name}: "
            f"{probability * 100:.2f}%"
        )

    print("\nPrediction:")
    print(
        f"  Grade: {result['grade']}"
    )
    print(
        f"  Class: {result['class_name']}"
    )
    print(
        f"  Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nDR grading test: SUCCESS")