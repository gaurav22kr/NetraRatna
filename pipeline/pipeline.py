from pathlib import Path
import shutil

from PIL import Image

from pipeline.quality import (
    assess_quality,
    save_enhanced_image,
)

from pipeline.vessels import (
    segment_vessels,
)

from pipeline.lesions import (
    segment_lesions,
)

from pipeline.grading import (
    load_grading_model,
    predict_dr,
)

from pipeline.explainability import (
    generate_gradcam,
)

from pipeline.report import (
    generate_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TEST_IMAGE = (
    PROJECT_ROOT
    / "test_data"
    / "test_fundus.jpg"
)


def clear_outputs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in OUTPUT_DIR.iterdir():

        if item.is_file():
            item.unlink()

        elif item.is_dir():
            shutil.rmtree(item)


def save_original(image):

    output_path = (
        OUTPUT_DIR
        / "original_fundus.jpg"
    )

    image.save(
        output_path
    )

    return output_path


def run_pipeline(image):

    clear_outputs()

    results = {
        "original": None,
        "quality": None,
        "enhancement": None,
        "vessels": None,
        "lesions": None,
        "grading": None,
        "gradcam": None,
        "report": None,
    }

    errors = {}

    # ---------------------------------------------------------
    # STAGE 01 — ORIGINAL IMAGE
    # ---------------------------------------------------------

    try:

        results["original"] = (
            save_original(image)
        )

    except Exception as error:

        errors["original"] = str(error)

    # ---------------------------------------------------------
    # STAGE 02 — IMAGE QUALITY
    # ---------------------------------------------------------

    try:

        results["quality"] = (
            assess_quality(image)
        )

    except Exception as error:

        errors["quality"] = str(error)

    # ---------------------------------------------------------
    # STAGE 03 — ENHANCEMENT
    # ---------------------------------------------------------

    try:

        enhanced_image, enhanced_path = (
            save_enhanced_image(
                image
            )
        )

        results["enhancement"] = {
            "image": enhanced_image,
            "path": enhanced_path,
        }

    except Exception as error:

        errors["enhancement"] = str(error)

    # ---------------------------------------------------------
    # STAGE 04 — VESSEL SEGMENTATION
    # ---------------------------------------------------------

    try:

        results["vessels"] = (
            segment_vessels(
                image
            )
        )

    except Exception as error:

        errors["vessels"] = str(error)

    # ---------------------------------------------------------
    # STAGE 05 — LESION SEGMENTATION
    # ---------------------------------------------------------

    try:

        results["lesions"] = (
            segment_lesions(
                image
            )
        )

    except Exception as error:

        errors["lesions"] = str(error)

    # ---------------------------------------------------------
    # STAGE 06 — DR GRADING
    # ---------------------------------------------------------

    grading_model = None

    try:

        grading_model = (
            load_grading_model()
        )

        results["grading"] = (
            predict_dr(
                image,
                model=grading_model,
            )
        )

    except Exception as error:

        errors["grading"] = str(error)

    # ---------------------------------------------------------
    # STAGE 07 — GRAD-CAM
    # ---------------------------------------------------------

    try:

        if grading_model is None:

            grading_model = (
                load_grading_model()
            )

        results["gradcam"] = (
            generate_gradcam(
                grading_model,
                image,
            )
        )

    except Exception as error:

        errors["gradcam"] = str(error)

    # ---------------------------------------------------------
    # STAGE 08 — REPORT
    # ---------------------------------------------------------

    try:

        if (
            results["quality"] is not None
            and results["grading"] is not None
        ):

            results["report"] = (
                generate_report(
                    quality_result=results[
                        "quality"
                    ],
                    grading_result=results[
                        "grading"
                    ],
                    output_dir=OUTPUT_DIR,
                )
            )

        else:

            raise RuntimeError(
                "Report requires successful "
                "quality and grading stages."
            )

    except Exception as error:

        errors["report"] = str(error)

    return results, errors


def print_results(
    results,
    errors,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PIPELINE RESULTS"
    )

    print(
        "=" * 60
    )

    for stage in results:

        if stage in errors:

            print(
                f"[FAILED]  {stage}"
            )

        else:

            print(
                f"[SUCCESS] {stage}"
            )

    # ---------------------------------------------------------
    # PRINT DR RESULT
    # ---------------------------------------------------------

    if results.get("grading"):

        grading = results["grading"]

        print(
            "\nDR GRADING RESULT:"
        )

        print(
            f"Grade: "
            f"{grading['grade']}"
        )

        print(
            f"Class: "
            f"{grading['class_name']}"
        )

        print(
            f"Confidence: "
            f"{grading['confidence'] * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # PRINT ERRORS
    # ---------------------------------------------------------

    if errors:

        print(
            "\nErrors:"
        )

        for stage, error in errors.items():

            print(
                f"\nStage: {stage}"
            )

            print(
                f"Error: {error}"
            )

    # ---------------------------------------------------------
    # GENERATED OUTPUTS
    # ---------------------------------------------------------

    print(
        "\nGenerated outputs:"
    )

    if OUTPUT_DIR.exists():

        files = sorted(
            item.name
            for item in OUTPUT_DIR.iterdir()
            if item.is_file()
        )

        if files:

            for filename in files:

                print(
                    f"- {filename}"
                )

        else:

            print(
                "- None"
            )


def main():

    print(
        f"Loading test image: "
        f"{TEST_IMAGE}"
    )

    if not TEST_IMAGE.exists():

        raise FileNotFoundError(
            f"Test image not found: "
            f"{TEST_IMAGE}"
        )

    image = (
        Image.open(
            TEST_IMAGE
        )
        .convert("RGB")
    )

    print(
        f"Image size: {image.size}"
    )

    print(
        "\nRunning pipeline..."
    )

    results, errors = (
        run_pipeline(
            image
        )
    )

    print_results(
        results,
        errors,
    )


if __name__ == "__main__":

    main()