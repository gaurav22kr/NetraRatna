from pathlib import Path

import streamlit as st
from PIL import Image

from pipeline.pipeline import run_pipeline


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="NetraRatna",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 12% 8%,
                rgba(75, 105, 255, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 88% 15%,
                rgba(0, 220, 190, 0.10),
                transparent 26%
            ),
            linear-gradient(
                145deg,
                #07101f 0%,
                #0a1324 50%,
                #050b15 100%
            );
        color: #f5f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ================= HERO ================= */

    .nr-hero {
        position: relative;
        overflow: hidden;
        padding: 42px 46px;
        margin-bottom: 26px;
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.10);
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.105),
                rgba(255,255,255,0.035)
            );
        box-shadow:
            0 25px 80px rgba(0,0,0,0.30),
            inset 0 1px rgba(255,255,255,0.10);
        backdrop-filter: blur(25px);
    }

    .nr-hero-glow {
        position: absolute;
        width: 300px;
        height: 300px;
        right: -90px;
        top: -140px;
        border-radius: 50%;
        background: rgba(70,105,255,0.20);
        filter: blur(40px);
    }

    .nr-badge {
        display: inline-block;
        padding: 8px 14px;
        margin-bottom: 16px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        color: #cbd7e9;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .nr-title {
        margin: 0;
        font-size: 64px;
        line-height: 0.95;
        font-weight: 850;
        letter-spacing: -0.045em;
        background: linear-gradient(
            110deg,
            #ffffff,
            #a9c7ff 48%,
            #9ff6e8
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nr-subtitle {
        max-width: 800px;
        margin-top: 20px;
        color: #aab7ca;
        font-size: 17px;
        line-height: 1.65;
    }

    /* ================= SECTION ================= */

    .nr-section {
        margin-top: 32px;
        margin-bottom: 14px;
    }

    .nr-section-title {
        color: #f4f7fb;
        font-size: 22px;
        font-weight: 780;
        letter-spacing: -0.02em;
    }

    .nr-section-subtitle {
        margin-top: 5px;
        color: #8997ac;
        font-size: 14px;
    }

    /* ================= GLASS ================= */

    .nr-glass {
        padding: 22px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.045);
        box-shadow:
            0 15px 45px rgba(0,0,0,0.20),
            inset 0 1px rgba(255,255,255,0.055);
        backdrop-filter: blur(20px);
    }

    /* ================= METRICS ================= */

    .nr-label {
        color: #8f9db1;
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.10em;
    }

    .nr-value {
        margin-top: 7px;
        color: #f7f9fc;
        font-size: 27px;
        font-weight: 820;
    }

    .nr-note {
        margin-top: 4px;
        color: #8190a6;
        font-size: 12px;
    }

    /* ================= RESULT ================= */

    .nr-result {
        min-height: 215px;
        padding: 28px;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,0.11);
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.095),
                rgba(255,255,255,0.035)
            );
        box-shadow:
            0 20px 65px rgba(0,0,0,0.28),
            inset 0 1px rgba(255,255,255,0.08);
        backdrop-filter: blur(25px);
    }

    .nr-result-label {
        color: #92a0b5;
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .nr-grade {
        margin-top: 9px;
        font-size: 58px;
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.04em;
        background: linear-gradient(
            110deg,
            #ffffff,
            #a9c7ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nr-class {
        margin-top: 12px;
        color: #f5f7fb;
        font-size: 19px;
        font-weight: 720;
    }

    .nr-confidence {
        margin-top: 7px;
        color: #9eacc0;
        font-size: 14px;
    }

    /* ================= PROBABILITY ================= */

    .nr-prob {
        min-height: 112px;
        padding: 19px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.045);
        box-shadow: inset 0 1px rgba(255,255,255,0.05);
    }

    .nr-prob-name {
        color: #8f9db1;
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .nr-prob-value {
        margin-top: 10px;
        color: #f7f9fc;
        font-size: 25px;
        font-weight: 800;
    }

    /* ================= QUALITY ================= */

    .nr-quality {
        min-height: 112px;
        padding: 19px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
    }

    .nr-quality-status {
        margin-top: 10px;
        color: #f7f9fc;
        font-size: 23px;
        font-weight: 800;
    }

    /* ================= DISCLAIMER ================= */

    .nr-disclaimer {
        margin-top: 35px;
        padding: 20px 22px;
        border-radius: 18px;
        border: 1px solid rgba(255,190,90,0.20);
        background: rgba(255,190,90,0.055);
        color: #c7c0ae;
        font-size: 13px;
        line-height: 1.65;
    }

    .nr-footer {
        margin-top: 48px;
        text-align: center;
        color: #65748a;
        font-size: 12px;
    }

    </style>
    """
)


# ============================================================
# HELPERS
# ============================================================

def get_quality_value(quality, *keys, default=None):

    if not isinstance(quality, dict):
        return default

    for key in keys:

        if key in quality and quality[key] is not None:
            return quality[key]

    return default


def format_percent(value):

    if value is None:
        return "—"

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def format_number(value, decimals=2):

    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def get_output(filename):

    path = OUTPUT_DIR / filename

    if path.exists():
        return path

    return None


def section(title, subtitle=None):

    subtitle_html = ""

    if subtitle:
        subtitle_html = f"""
        <div class="nr-section-subtitle">
            {subtitle}
        </div>
        """

    st.html(
        f"""
        <div class="nr-section">

            <div class="nr-section-title">
                {title}
            </div>

            {subtitle_html}

        </div>
        """
    )


def metric_card(label, value, note=""):

    st.html(
        f"""
        <div class="nr-glass">

            <div class="nr-label">
                {label}
            </div>

            <div class="nr-value">
                {value}
            </div>

            <div class="nr-note">
                {note}
            </div>

        </div>
        """
    )


def image_card(title, filename):

    path = get_output(filename)

    if path is None:

        st.warning(
            f"{title} output was not generated."
        )

        return

    st.html(
        f"""
        <div
            style="
                color:#a8b5c8;
                font-size:13px;
                font-weight:700;
                margin-bottom:8px;
            "
        >
            {title}
        </div>
        """
    )

    st.image(
        str(path),
        width="stretch",
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="nr-hero">

        <div class="nr-hero-glow"></div>

        <div class="nr-badge">
            👁️ AI-ASSISTED RETINAL SCREENING PROTOTYPE
        </div>

        <div class="nr-title">
            NetraRatna
        </div>

        <div class="nr-subtitle">
            Explainable AI for diabetic retinopathy screening.
            Upload a fundus image and explore image quality,
            retinal vessels, lesion evidence, DR grading,
            and Grad-CAM-based visual explanation in one workflow.
        </div>

    </div>
    """
)


# ============================================================
# UPLOAD
# ============================================================

section(
    "Retinal Image",
    "Upload a fundus photograph to begin the analysis."
)

uploaded_file = st.file_uploader(
    "Upload fundus image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


if uploaded_file is None:

    st.html(
        """
        <div class="nr-glass">

            <div
                style="
                    color:#f5f7fb;
                    font-size:17px;
                    font-weight:700;
                "
            >
                Start your analysis
            </div>

            <div
                style="
                    color:#8997ac;
                    margin-top:7px;
                    font-size:14px;
                "
            >
                Upload a JPG, JPEG, or PNG fundus image above.
            </div>

        </div>
        """
    )

    analyze = False

else:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(
        [1.25, 0.75],
        gap="large",
    )

    with left:

        st.html(
            """
            <div
                style="
                    color:#a8b5c8;
                    font-size:13px;
                    font-weight:700;
                    margin-bottom:8px;
                "
            >
                Uploaded fundus
            </div>
            """
        )

        st.image(
            image,
            width="stretch",
        )

    with right:

        metric_card(
            "Input image",
            "Ready",
            "Fundus image loaded successfully.",
        )

        st.write("")

        metric_card(
            "Resolution",
            f"{image.width} × {image.height}",
            "RGB retinal photograph",
        )

        st.write("")

        analyze = st.button(
            "Analyze with NetraRatna",
            type="primary",
            width="stretch",
        )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    section("Analysis")

    with st.spinner(
        "Running NetraRatna analysis — first run may take a little while..."
    ):

        results, errors = run_pipeline(image)

    if errors:

        with st.expander(
            "Pipeline messages",
            expanded=False,
        ):

            for error in errors:
                st.warning(str(error))

    grading = results.get("grading")
    quality = results.get("quality")

    if grading is None:

        st.error(
            "The DR grading stage did not return a result. "
            "Please check the terminal for the pipeline error."
        )

        st.stop()


    # ========================================================
    # SCREENING RESULT
    # ========================================================

    section("Screening Result")

    grade = grading.get(
        "grade",
        "—",
    )

    class_name = grading.get(
        "class_name",
        "Unknown",
    )

    confidence = grading.get(
        "confidence"
    )

    result_left, result_right = st.columns(
        [1.55, 1],
        gap="large",
    )

    with result_left:

        st.html(
            f"""
            <div class="nr-result">

                <div class="nr-result-label">
                    Predicted diabetic retinopathy grade
                </div>

                <div class="nr-grade">
                    Grade {grade}
                </div>

                <div class="nr-class">
                    {class_name}
                </div>

                <div class="nr-confidence">
                    Model confidence: {format_percent(confidence)}
                </div>

            </div>
            """
        )

    with result_right:

        st.html(
            f"""
            <div class="nr-result">

                <div class="nr-result-label">
                    Confidence
                </div>

                <div class="nr-grade">
                    {format_percent(confidence)}
                </div>

                <div class="nr-confidence">
                    Probability assigned to the predicted class.
                </div>

            </div>
            """
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    section(
        "DR Classification Probabilities",
        "Model probability distribution across the five DR grades."
    )

    probabilities = grading.get(
        "probabilities",
        {},
    )

    probability_names = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative",
    ]

    probability_columns = st.columns(
        5,
        gap="small",
    )

    for column, name in zip(
        probability_columns,
        probability_names,
    ):

        value = probabilities.get(
            name
        )

        with column:

            st.html(
                f"""
                <div class="nr-prob">

                    <div class="nr-prob-name">
                        {name}
                    </div>

                    <div class="nr-prob-value">
                        {format_percent(value)}
                    </div>

                </div>
                """
            )


    # ========================================================
    # QUALITY
    # ========================================================

    section(
        "Image Quality",
        "Input-image quality assessment before interpretation."
    )

    quality_status = get_quality_value(
        quality,
        "label",
        "quality",
        "status",
        "quality_label",
        default="Unknown",
    )

    quality_score = get_quality_value(
        quality,
        "score",
        "quality_score",
        "overall_score",
        default=None,
    )

    brightness = get_quality_value(
        quality,
        "brightness",
        "brightness_score",
        default=None,
    )

    contrast = get_quality_value(
        quality,
        "contrast",
        "contrast_score",
        default=None,
    )

    sharpness = get_quality_value(
        quality,
        "sharpness",
        "sharpness_score",
        default=None,
    )

    quality_columns = st.columns(
        4,
        gap="small",
    )

    quality_items = [
        (
            "Overall",
            format_number(
                quality_score,
                3,
            ),
        ),
        (
            "Brightness",
            format_number(
                brightness,
                2,
            ),
        ),
        (
            "Contrast",
            format_number(
                contrast,
                2,
            ),
        ),
        (
            "Sharpness",
            format_number(
                sharpness,
                2,
            ),
        ),
    ]

    for column, (label, value) in zip(
        quality_columns,
        quality_items,
    ):

        with column:

            st.html(
                f"""
                <div class="nr-quality">

                    <div class="nr-label">
                        {label}
                    </div>

                    <div class="nr-quality-status">
                        {value}
                    </div>

                </div>
                """
            )

    st.html(
        f"""
        <div
            class="nr-glass"
            style="margin-top:12px;"
        >

            <div class="nr-label">
                Quality assessment
            </div>

            <div class="nr-value">
                {quality_status}
            </div>

        </div>
        """
    )


    # ========================================================
    # VISUAL EVIDENCE
    # ========================================================

    section(
        "Visual Evidence",
        "Visual outputs generated by the NetraRatna analysis pipeline."
    )

    visual_left, visual_right = st.columns(
        2,
        gap="large",
    )

    with visual_left:

        image_card(
            "Original Fundus",
            "original_fundus.jpg",
        )

    with visual_right:

        image_card(
            "Enhanced Fundus",
            "enhanced_fundus.jpg",
        )

    visual_left, visual_right = st.columns(
        2,
        gap="large",
    )

    with visual_left:

        image_card(
            "Vessel Segmentation",
            "vessel_overlay.png",
        )

    with visual_right:

        image_card(
            "Lesion Evidence",
            "lesion_overlay_original.png",
        )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    section(
        "Explainability",
        "Grad-CAM highlights regions that contributed to the model's classification output."
    )

    image_card(
        "Grad-CAM Explanation",
        "gradcam_overlay.png",
    )


    # ========================================================
    # REPORT
    # ========================================================

    report_path = results.get(
        "report"
    )

    if report_path:

        report_file = Path(
            report_path
        )

        if report_file.exists():

            section(
                "Report",
                "Complete NetraRatna analysis report."
            )

            st.download_button(
                "Download Analysis Report",
                data=report_file.read_bytes(),
                file_name="netraratna_report.html",
                mime="text/html",
                width="stretch",
            )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.html(
        """
        <div class="nr-disclaimer">

            <strong>
                Hackathon Prototype
            </strong>

            <br>

            This prototype integrates models
            for demonstration and explainability. 
            Model outputs should not be treated as a medical
            diagnosis and should be reviewed by a qualified
            healthcare professional.

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="nr-footer">
        NetraRatna · Explainable AI for Diabetic Retinopathy Screening
    </div>
    """
)