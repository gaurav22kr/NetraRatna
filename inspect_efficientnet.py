from huggingface_hub import hf_hub_download
import keras


MODEL_REPO = "Aldahmashi/DR-EfficientNetB0"
MODEL_FILE = "final_model.keras"


def main():
    print("Loading DR model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = keras.saving.load_model(model_path)

    efficientnet = model.get_layer("efficientnetb0")

    print("\nEfficientNet-B0 layers:")
    print("-" * 90)

    for index, layer in enumerate(efficientnet.layers):
        print(
            f"{index:3d} | "
            f"{layer.name:45s} | "
            f"{layer.__class__.__name__}"
        )

    print("-" * 90)
    print("\nTotal EfficientNet layers:", len(efficientnet.layers))


if __name__ == "__main__":
    main()