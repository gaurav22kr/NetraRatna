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

    print("\nModel layers:")
    print("-" * 80)

    for index, layer in enumerate(model.layers):
        print(
            f"{index:3d} | "
            f"{layer.name:40s} | "
            f"{layer.__class__.__name__}"
        )

    print("-" * 80)

    print("\nTotal layers:", len(model.layers))
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)


if __name__ == "__main__":
    main()