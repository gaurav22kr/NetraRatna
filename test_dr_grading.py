import numpy as np
import keras
from PIL import Image
from huggingface_hub import hf_hub_download


INPUT_PATH = "test_data/test_fundus.jpg"

MODEL_REPO = "Aldahmashi/DR-EfficientNetB0"
MODEL_FILE = "final_model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
]


def main():
    print("Downloading/loading DR grading model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = keras.saving.load_model(model_path)

    print("Model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    print("\nLoading fundus image...")

    image = Image.open(INPUT_PATH).convert("RGB")

    print("Original size:", image.size)

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    print("Model input:", image_array.shape)

    print("\nRunning DR grading...")

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    print("\nClass probabilities:")

    for index, probability in enumerate(probabilities):
        print(
            f"Class {index} - "
            f"{CLASS_NAMES[index]}: "
            f"{probability * 100:.2f}%"
        )

    print("\nPrediction:")
    print("DR Grade:", predicted_index)
    print("Interpretation:", CLASS_NAMES[predicted_index])
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nDR grading test: SUCCESS")


if __name__ == "__main__":
    main()