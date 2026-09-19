from pathlib import Path
from html import escape
import base64


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _image_data_uri(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        return ""

    extension = image_path.suffix.lower()

    if extension in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif extension == ".png":
        mime_type = "image/png"
    else:
        return ""

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def _image_card(title, image_path, description):
    data_uri = _image_data_uri(image_path)

    if not data_uri:
        return ""

    return f"""
    <article class="visual-card">

        <div class="visual-heading">
            <div>
                <h3>{escape(title)}</h3>
                <p>{escape(description)}</p>
            </div>
        </div>

        <div class="image-frame">
            <img
                src="{data_uri}"
                alt="{escape(title)}"
            >
        </div>

    </article>
    """


def generate_report(
    quality_result,
    grading_result,
    output_dir=None,
):
    """
    Generate a self-contained NetraRatna HTML report.
    """

    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir / "netraratna_report.html"
    )

    # =========================================================
    # QUALITY DATA
    # =========================================================

    quality_score = float(
        quality_result.get("score", 0.0)
    )

    quality_label = quality_result.get(
        "label",
        "Unknown",
    )

    brightness = float(
        quality_result.get(
            "brightness",
            0.0,
        )
    )

    contrast = float(
        quality_result.get(
            "contrast",
            0.0,
        )
    )

    sharpness = float(
        quality_result.get(
            "sharpness",
            0.0,
        )
    )

    # =========================================================
    # GRADING DATA
    # =========================================================

    grade = int(
        grading_result.get(
            "grade",
            0,
        )
    )

    class_name = grading_result.get(
        "class_name",
        "Unknown",
    )

    confidence = float(
        grading_result.get(
            "confidence",
            0.0,
        )
    )

    probabilities = grading_result.get(
        "probabilities",
        {},
    )

    probability_rows = ""

    for name, probability in probabilities.items():

        probability_value = (
            float(probability) * 100
        )

        probability_rows += f"""
        <div class="probability-row">

            <div class="probability-label">
                <span>{escape(str(name))}</span>
                <strong>
                    {probability_value:.2f}%
                </strong>
            </div>

            <div class="probability-track">
                <div
                    class="probability-fill"
                    style="width: {probability_value:.2f}%"
                ></div>
            </div>

        </div>
        """

    # =========================================================
    # IMAGE OUTPUTS
    # =========================================================

    original_path = (
        output_dir /
        "original_fundus.jpg"
    )

    enhanced_path = (
        output_dir /
        "enhanced_fundus.jpg"
    )

    vessel_path = (
        output_dir /
        "vessel_overlay.png"
    )

    lesion_path = (
        output_dir /
        "lesion_overlay_original.png"
    )

    gradcam_path = (
        output_dir /
        "gradcam_overlay.png"
    )

    # =========================================================
    # VISUAL CARDS
    # =========================================================

    original_card = _image_card(
        "Original Fundus",
        original_path,
        "Original retinal image supplied to NetraRatna.",
    )

    enhanced_card = _image_card(
        "Enhanced Fundus",
        enhanced_path,
        "Contrast-enhanced retinal image.",
    )

    vessel_card = _image_card(
        "Vessel Segmentation",
        vessel_path,
        "Retinal vessel evidence from the FR-U-Net ensemble.",
    )

    lesion_card = _image_card(
        "Lesion Evidence",
        lesion_path,
        "Predicted lesion evidence from the segmentation model.",
    )

    gradcam_card = _image_card(
        "Grad-CAM Explanation",
        gradcam_path,
        "Regions contributing to the predicted DR classification.",
    )

    # =========================================================
    # HTML
    # =========================================================

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>NetraRatna Report</title>

<style>

:root {{
    --bg: #eef2f7;
    --surface: rgba(255, 255, 255, 0.72);
    --surface-solid: #ffffff;
    --text: #172033;
    --muted: #697586;
    --line: rgba(255, 255, 255, 0.72);
    --accent: #4f46e5;
    --accent-soft: rgba(79, 70, 229, 0.10);
    --shadow:
        0 20px 60px rgba(31, 41, 55, 0.10);
    --small-shadow:
        0 10px 35px rgba(31, 41, 55, 0.08);
}}

* {{
    box-sizing: border-box;
}}

html {{
    scroll-behavior: smooth;
}}

body {{
    margin: 0;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI",
        Arial,
        sans-serif;

    color: var(--text);

    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(99, 102, 241, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(56, 189, 248, 0.14),
            transparent 25%
        ),
        linear-gradient(
            145deg,
            #f8fafc 0%,
            #eef2f7 50%,
            #e9eef5 100%
        );

    min-height: 100vh;
}}

body::before {{
    content: "";

    position: fixed;

    width: 340px;
    height: 340px;

    top: 35%;
    left: -170px;

    border-radius: 50%;

    background:
        rgba(129, 140, 248, 0.10);

    filter: blur(80px);

    pointer-events: none;
}}

body::after {{
    content: "";

    position: fixed;

    width: 300px;
    height: 300px;

    right: -130px;
    bottom: 10%;

    border-radius: 50%;

    background:
        rgba(14, 165, 233, 0.10);

    filter: blur(80px);

    pointer-events: none;
}}

/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.container {{
    width: min(
        1180px,
        calc(100% - 36px)
    );

    margin: 0 auto;

    padding:
        30px
        0
        70px;
}}

/* =========================================================
   TOP NAV / BRAND
   ========================================================= */

.topbar {{
    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 22px;
}}

.brand {{
    display: flex;

    align-items: center;

    gap: 12px;
}}

.brand-mark {{
    width: 42px;
    height: 42px;

    display: grid;

    place-items: center;

    border-radius: 14px;

    background:
        linear-gradient(
            145deg,
            #6366f1,
            #4338ca
        );

    color: white;

    font-size: 20px;

    font-weight: 700;

    box-shadow:
        0 10px 25px
        rgba(79, 70, 229, 0.25);
}}

.brand-name {{
    font-size: 18px;

    font-weight: 700;

    letter-spacing: -0.4px;
}}

.report-pill {{
    padding:
        8px
        14px;

    border-radius: 999px;

    background:
        rgba(255, 255, 255, 0.60);

    border:
        1px solid
        rgba(255, 255, 255, 0.85);

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);

    color: var(--muted);

    font-size: 12px;

    font-weight: 600;
}}

/* =========================================================
   HERO
   ========================================================= */

.hero {{
    position: relative;

    overflow: hidden;

    border-radius: 34px;

    padding:
        48px
        48px
        42px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.86),
            rgba(255,255,255,0.56)
        );

    border:
        1px solid
        rgba(255,255,255,0.88);

    box-shadow:
        var(--shadow);

    backdrop-filter:
        blur(30px);

    -webkit-backdrop-filter:
        blur(30px);
}}

.hero::before {{
    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    right: -90px;
    top: -120px;

    border-radius: 50%;

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.18),
            rgba(56,189,248,0.10)
        );

    filter: blur(8px);
}}

.hero::after {{
    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    right: 160px;
    bottom: -110px;

    border-radius: 50%;

    background:
        rgba(129,140,248,0.12);

    filter: blur(30px);
}}

.hero-content {{
    position: relative;

    z-index: 2;

    max-width: 760px;
}}

.eyebrow {{
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding:
        8px
        12px;

    border-radius: 999px;

    background:
        var(--accent-soft);

    color:
        var(--accent);

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.3px;

    margin-bottom: 20px;
}}

.eyebrow-dot {{
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background:
        #6366f1;

    box-shadow:
        0 0 0 5px
        rgba(99,102,241,0.10);
}}

.hero h1 {{
    margin: 0;

    font-size:
        clamp(42px, 7vw, 72px);

    line-height: 0.98;

    letter-spacing: -4px;

    font-weight: 800;

    background:
        linear-gradient(
            110deg,
            #111827,
            #4338ca 65%,
            #0284c7
        );

    -webkit-background-clip: text;
    background-clip: text;

    color: transparent;
}}

.hero-subtitle {{
    margin:
        18px
        0
        0;

    max-width: 650px;

    color: var(--muted);

    font-size: 17px;

    line-height: 1.7;
}}

/* =========================================================
   SECTION
   ========================================================= */

.section {{
    margin-top: 34px;
}}

.section-heading {{
    display: flex;

    align-items: flex-end;

    justify-content: space-between;

    gap: 20px;

    margin:
        0
        0
        15px
        4px;
}}

.section-heading h2 {{
    margin: 0;

    font-size: 23px;

    letter-spacing: -0.7px;
}}

.section-heading p {{
    margin: 4px 0 0;

    color: var(--muted);

    font-size: 13px;
}}

/* =========================================================
   SUMMARY
   ========================================================= */

.summary {{
    display: grid;

    grid-template-columns:
        1.35fr
        1fr;

    gap: 16px;
}}

.summary-card {{
    position: relative;

    overflow: hidden;

    min-height: 190px;

    padding: 26px;

    border-radius: 26px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.88),
            rgba(255,255,255,0.58)
        );

    border:
        1px solid
        rgba(255,255,255,0.90);

    box-shadow:
        var(--small-shadow);

    backdrop-filter:
        blur(24px);

    -webkit-backdrop-filter:
        blur(24px);
}}

.summary-card.primary {{
    background:
        linear-gradient(
            135deg,
            rgba(79,70,229,0.95),
            rgba(67,56,202,0.90)
        );

    color: white;

    box-shadow:
        0 20px 45px
        rgba(79,70,229,0.24);
}}

.summary-card.primary::after {{
    content: "";

    position: absolute;

    width: 230px;
    height: 230px;

    right: -70px;
    bottom: -120px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.13);

    filter: blur(2px);
}}

.summary-label {{
    color: var(--muted);

    font-size: 12px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}}

.primary .summary-label {{
    color:
        rgba(255,255,255,0.72);
}}

.grade-number {{
    position: relative;

    z-index: 2;

    margin-top: 15px;

    font-size: 54px;

    line-height: 1;

    font-weight: 800;

    letter-spacing: -3px;
}}

.grade-name {{
    position: relative;

    z-index: 2;

    margin-top: 10px;

    font-size: 17px;

    font-weight: 600;
}}

.confidence {{
    margin-top: 24px;

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding:
        8px
        12px;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.15);

    font-size: 12px;

    font-weight: 700;
}}

.confidence-dot {{
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background:
        #ffffff;

    box-shadow:
        0 0 0 4px
        rgba(255,255,255,0.12);
}}

.confidence-number {{
    margin-top: 13px;

    font-size: 38px;

    line-height: 1;

    font-weight: 800;

    letter-spacing: -1.5px;
}}

/* =========================================================
   PROBABILITIES
   ========================================================= */

.probabilities {{
    padding: 25px;

    border-radius: 26px;

    background:
        rgba(255,255,255,0.70);

    border:
        1px solid
        rgba(255,255,255,0.90);

    box-shadow:
        var(--small-shadow);

    backdrop-filter:
        blur(24px);

    -webkit-backdrop-filter:
        blur(24px);
}}

.probability-row {{
    margin-bottom: 17px;
}}

.probability-row:last-child {{
    margin-bottom: 0;
}}

.probability-label {{
    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 7px;

    font-size: 13px;
}}

.probability-label span {{
    color: var(--muted);
}}

.probability-label strong {{
    font-size: 12px;
}}

.probability-track {{
    height: 7px;

    overflow: hidden;

    border-radius: 999px;

    background:
        rgba(100,116,139,0.10);
}}

.probability-fill {{
    height: 100%;

    border-radius: inherit;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #06b6d4
        );

    box-shadow:
        0 0 12px
        rgba(99,102,241,0.22);
}}

/* =========================================================
   QUALITY
   ========================================================= */

.quality-grid {{
    display: grid;

    grid-template-columns:
        1.15fr
        repeat(3, 1fr);

    gap: 14px;
}}

.quality-card {{
    padding: 22px;

    min-height: 125px;

    border-radius: 22px;

    background:
        rgba(255,255,255,0.68);

    border:
        1px solid
        rgba(255,255,255,0.88);

    box-shadow:
        var(--small-shadow);

    backdrop-filter:
        blur(20px);

    -webkit-backdrop-filter:
        blur(20px);
}}

.quality-card.main {{
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.90),
            rgba(238,242,255,0.78)
        );
}}

.quality-title {{
    color: var(--muted);

    font-size: 12px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.7px;
}}

.quality-value {{
    margin-top: 10px;

    font-size: 27px;

    font-weight: 800;

    letter-spacing: -1px;
}}

.quality-label {{
    margin-top: 5px;

    color:
        var(--accent);

    font-weight: 700;

    font-size: 13px;
}}

/* =========================================================
   VISUAL EVIDENCE
   ========================================================= */

.visual-grid {{
    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 18px;
}}

.visual-card {{
    padding: 17px;

    border-radius: 26px;

    background:
        rgba(255,255,255,0.68);

    border:
        1px solid
        rgba(255,255,255,0.90);

    box-shadow:
        var(--small-shadow);

    backdrop-filter:
        blur(22px);

    -webkit-backdrop-filter:
        blur(22px);

    transition:
        transform 180ms ease,
        box-shadow 180ms ease;
}}

.visual-card:hover {{
    transform:
        translateY(-3px);

    box-shadow:
        0 18px 45px
        rgba(31,41,55,0.13);
}}

.visual-heading {{
    padding:
        5px
        5px
        14px;
}}

.visual-heading h3 {{
    margin: 0;

    font-size: 16px;

    letter-spacing: -0.3px;
}}

.visual-heading p {{
    margin:
        5px
        0
        0;

    color: var(--muted);

    font-size: 12px;

    line-height: 1.5;
}}

.image-frame {{
    overflow: hidden;

    border-radius: 18px;

    background:
        #111827;

    box-shadow:
        inset
        0 0 0 1px
        rgba(255,255,255,0.08);
}}

.image-frame img {{
    display: block;

    width: 100%;

    max-height: 450px;

    object-fit: contain;

    transition:
        transform 250ms ease;
}}

.visual-card:hover
.image-frame img {{
    transform:
        scale(1.012);
}}

/* =========================================================
   INTERPRETATION
   ========================================================= */

.interpretation {{
    padding: 26px;

    border-radius: 26px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.80),
            rgba(255,255,255,0.56)
        );

    border:
        1px solid
        rgba(255,255,255,0.90);

    box-shadow:
        var(--small-shadow);

    backdrop-filter:
        blur(22px);

    -webkit-backdrop-filter:
        blur(22px);

    line-height: 1.75;

    color: var(--muted);
}}

.interpretation strong {{
    color: var(--text);
}}

/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {{
    padding: 23px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(255,247,237,0.88),
            rgba(255,237,213,0.68)
        );

    border:
        1px solid
        rgba(251,146,60,0.18);

    color:
        #7c4a20;

    line-height: 1.7;

    box-shadow:
        var(--small-shadow);
}}

.disclaimer strong {{
    color:
        #9a3412;
}}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {{
    margin-top: 45px;

    text-align: center;

    color: var(--muted);

    font-size: 12px;

    padding: 20px;
}}

.footer-line {{
    width: 50px;

    height: 3px;

    margin:
        0
        auto
        13px;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #06b6d4
        );
}}

/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 800px) {{

    .container {{
        width:
            calc(100% - 22px);

        padding-top: 15px;
    }}

    .hero {{
        padding: 32px 25px;
        border-radius: 26px;
    }}

    .hero h1 {{
        letter-spacing: -2.5px;
    }}

    .summary {{
        grid-template-columns: 1fr;
    }}

    .quality-grid {{
        grid-template-columns:
            repeat(2, 1fr);
    }}

    .quality-card.main {{
        grid-column:
            1 / -1;
    }}

    .visual-grid {{
        grid-template-columns: 1fr;
    }}

    .topbar {{
        padding: 0 3px;
    }}
}}

@media (max-width: 500px) {{

    .quality-grid {{
        grid-template-columns: 1fr;
    }}

    .quality-card.main {{
        grid-column: auto;
    }}

    .hero h1 {{
        font-size: 44px;
    }}

    .brand-name {{
        font-size: 16px;
    }}

    .report-pill {{
        display: none;
    }}
}}

</style>

</head>


<body>


<div class="container">


<!-- =====================================================
     BRAND
     ===================================================== -->

<div class="topbar">

    <div class="brand">

        <div class="brand-mark">
            N
        </div>

        <div class="brand-name">
            NetraRatna
        </div>

    </div>

    <div class="report-pill">
        AI Screening Report
    </div>

</div>


<!-- =====================================================
     HERO
     ===================================================== -->

<section class="hero">

    <div class="hero-content">

        <div class="eyebrow">

            <span class="eyebrow-dot"></span>

            Explainable Retinal Analysis

        </div>

        <h1>
            NetraRatna
        </h1>

        <p class="hero-subtitle">

            Explainable AI for diabetic retinopathy
            screening, combining retinal image analysis
            with visual evidence and model explanations.

        </p>

    </div>

</section>


<!-- =====================================================
     ANALYSIS SUMMARY
     ===================================================== -->

<section class="section">

    <div class="section-heading">

        <div>

            <h2>
                Analysis Summary
            </h2>

            <p>
                Current model prediction
            </p>

        </div>

    </div>


    <div class="summary">


        <!-- DR GRADE -->

        <div class="summary-card primary">

            <div class="summary-label">
                Predicted DR Grade
            </div>

            <div class="grade-number">
                {grade}
            </div>

            <div class="grade-name">
                {escape(str(class_name))}
            </div>

            <div class="confidence">

                <span class="confidence-dot"></span>

                Model confidence

            </div>

        </div>


        <!-- CONFIDENCE -->

        <div class="summary-card">

            <div class="summary-label">
                Prediction Confidence
            </div>

            <div class="confidence-number">

                {confidence * 100:.2f}%

            </div>

            <p
                style="
                    color: var(--muted);
                    font-size: 13px;
                    line-height: 1.5;
                    margin-top: 13px;
                "
            >

                Confidence associated with
                the predicted DR class.

            </p>

        </div>


    </div>

</section>


<!-- =====================================================
     PROBABILITIES
     ===================================================== -->

<section class="section">

    <div class="section-heading">

        <div>

            <h2>
                Classification Probabilities
            </h2>

            <p>
                Distribution across the five DR grades
            </p>

        </div>

    </div>


    <div class="probabilities">

        {probability_rows}

    </div>

</section>


<!-- =====================================================
     IMAGE QUALITY
     ===================================================== -->

<section class="section">

    <div class="section-heading">

        <div>

            <h2>
                Image Quality
            </h2>

            <p>
                Input image quality assessment
            </p>

        </div>

    </div>


    <div class="quality-grid">


        <div class="quality-card main">

            <div class="quality-title">
                Overall Quality
            </div>

            <div class="quality-value">
                {quality_score:.3f}
            </div>

            <div class="quality-label">
                {escape(str(quality_label))}
            </div>

        </div>


        <div class="quality-card">

            <div class="quality-title">
                Brightness
            </div>

            <div class="quality-value">
                {brightness:.2f}
            </div>

        </div>


        <div class="quality-card">

            <div class="quality-title">
                Contrast
            </div>

            <div class="quality-value">
                {contrast:.2f}
            </div>

        </div>


        <div class="quality-card">

            <div class="quality-title">
                Sharpness
            </div>

            <div class="quality-value">
                {sharpness:.2f}
            </div>

        </div>


    </div>

</section>


<!-- =====================================================
     VISUAL EVIDENCE
     ===================================================== -->

<section class="section">

    <div class="section-heading">

        <div>

            <h2>
                Visual Evidence
            </h2>

            <p>
                Model outputs and explainability visualizations
            </p>

        </div>

    </div>


    <div class="visual-grid">

        {original_card}

        {enhanced_card}

        {vessel_card}

        {lesion_card}

        {gradcam_card}

    </div>

</section>


<!-- =====================================================
     INTERPRETATION
     ===================================================== -->

<section class="section">

    <div class="section-heading">

        <div>

            <h2>
                Interpretation
            </h2>

        </div>

    </div>


    <div class="interpretation">

        The model predicts

        <strong>
            Grade {grade} —
            {escape(str(class_name))}
        </strong>

        with a prediction confidence of

        <strong>
            {confidence * 100:.2f}%
        </strong>.

        The report provides supporting visual evidence
        through retinal enhancement, vessel segmentation,
        lesion evidence, and Grad-CAM explainability.

    </div>

</section>


<!-- =====================================================
     DISCLAIMER
     ===================================================== -->

<section class="section">

    <div class="section-heading">

        <div>

            <h2>
                Research / Hackathon Disclaimer
            </h2>

        </div>

    </div>


    <div class="disclaimer">

        <strong>
            Important:
        </strong>

        This prototype integrates pretrained research
        models for demonstration and explainability.

        Always do consult an Opthalmologist.

        The displayed predictions and visualizations should
        not be interpreted as a medical diagnosis.

        Model performance may vary depending on image quality,
        dataset characteristics, camera systems, and other
        factors.

    </div>

</section>


<!-- =====================================================
     FOOTER
     ===================================================== -->

<div class="footer">

    <div class="footer-line"></div>

    NetraRatna · Explainable AI for
    Diabetic Retinopathy Screening

</div>


</div>


</body>

</html>
"""

    report_path.write_text(
        html,
        encoding="utf-8",
    )

    return report_path