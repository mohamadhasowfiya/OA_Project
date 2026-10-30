from flask import Flask, render_template, request, send_file
from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape

from model_predict import predict_xray
from risk_predict import predict_risk

from database import (
    init_database,
    save_screening,
    get_dashboard_stats,
    get_all_screenings,
    get_connection
)


# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# BASE DIRECTORIES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


# ==========================================================
# INITIALIZE DATABASE
# ==========================================================

init_database()


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/")
def home():

    stats = get_dashboard_stats()

    return render_template(
        "index.html",
        stats=stats
    )


# ==========================================================
# SCREENING HISTORY
# ==========================================================

@app.route("/history")
def history():

    screenings = get_all_screenings()

    return render_template(
        "history.html",
        screenings=screenings
    )


# ==========================================================
# PATIENTS
# ==========================================================

@app.route("/patients")
def patients():

    screenings = get_all_screenings()

    patient_map = {}

    for row in screenings:

        patient_id = row["patient_id"]

        if patient_id not in patient_map:

            patient_map[patient_id] = {

                "report_id": row["id"],

                "patient_id": row["patient_id"],

                "patient_name": row["patient_name"],

                "age": row["age"],

                "gender": row["gender"],

                "location": row["location"],

                "last_risk": row["risk_level"],

                "last_risk_confidence":
                    row["risk_confidence"],

                "last_screening":
                    row["created_at"]
            }


    patients_list = list(patient_map.values())


    return render_template(
        "patients.html",
        patients=patients_list
    )


# ==========================================================
# SENSOR DEVICE
# ==========================================================

@app.route("/sensor")
def sensor():

    return render_template(
        "sensor.html"
    )


# ==========================================================
# REPORTS
# ==========================================================

@app.route("/reports")
def reports():

    stats = get_dashboard_stats()

    screenings = get_all_screenings()

    return render_template(
        "reports.html",
        stats=stats,
        screenings=screenings
    )


# ==========================================================
# INDIVIDUAL PATIENT REPORT
# ==========================================================

@app.route("/report/<int:report_id>")
def patient_report(report_id):

    connection = get_connection()

    cursor = connection.cursor()

    screening = cursor.execute(
        """
        SELECT *
        FROM screenings
        WHERE id = ?
        """,
        (report_id,)
    ).fetchone()

    connection.close()


    if screening is None:

        return "Report not found", 404


    return render_template(
        "patient_report.html",
        screening=screening
    )


# ==========================================================
# DIRECT PDF DOWNLOAD
# ==========================================================

@app.route("/download-report/<int:report_id>")
def download_report(report_id):

    # ------------------------------------------------------
    # Get screening from database
    # ------------------------------------------------------

    connection = get_connection()

    cursor = connection.cursor()

    screening = cursor.execute(
        """
        SELECT *
        FROM screenings
        WHERE id = ?
        """,
        (report_id,)
    ).fetchone()

    connection.close()


    # ------------------------------------------------------
    # Report not found
    # ------------------------------------------------------

    if screening is None:

        return "Report not found", 404


    # ------------------------------------------------------
    # Import ReportLab
    # ------------------------------------------------------

    try:

        from reportlab.lib import colors

        from reportlab.lib.enums import TA_CENTER

        from reportlab.lib.pagesizes import letter

        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )

        from reportlab.lib.units import inch

        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )

    except ImportError:

        return (
            "ReportLab is not installed. "
            "Please install it using: "
            ".\\venv\\Scripts\\python.exe -m pip install reportlab"
        ), 500


    # ------------------------------------------------------
    # Create PDF in memory
    # ------------------------------------------------------

    pdf_buffer = BytesIO()


    document = SimpleDocTemplate(

        pdf_buffer,

        pagesize=letter,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40
    )


    # ------------------------------------------------------
    # Styles
    # ------------------------------------------------------

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        fontSize=20,

        leading=24,

        alignment=TA_CENTER,

        spaceAfter=8
    )


    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        textColor=colors.grey,

        spaceAfter=20
    )


    heading_style = ParagraphStyle(

        "Heading",

        parent=styles["Heading2"],

        fontSize=14,

        leading=18,

        spaceBefore=10,

        spaceAfter=10
    )


    normal_style = ParagraphStyle(

        "NormalReport",

        parent=styles["Normal"],

        fontSize=10,

        leading=14
    )


    small_style = ParagraphStyle(

        "SmallReport",

        parent=styles["Normal"],

        fontSize=8.5,

        leading=12,

        textColor=colors.grey
    )


    # ------------------------------------------------------
    # Helper function
    # ------------------------------------------------------

    def safe(value):

        if value is None:

            return "Not Available"

        return escape(str(value))


    # ------------------------------------------------------
    # PDF Content
    # ------------------------------------------------------

    story = []


    # ------------------------------------------------------
    # Title
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "OA Screening Report",
            title_style
        )
    )


    story.append(
        Paragraph(
            "AI-assisted osteoarthritis screening and risk assessment",
            subtitle_style
        )
    )


    # ------------------------------------------------------
    # Patient Information
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )


    patient_data = [

        [
            Paragraph("<b>Patient ID</b>", normal_style),
            Paragraph(safe(screening["patient_id"]), normal_style)
        ],

        [
            Paragraph("<b>Patient Name</b>", normal_style),
            Paragraph(safe(screening["patient_name"]), normal_style)
        ],

        [
            Paragraph("<b>Age</b>", normal_style),
            Paragraph(safe(screening["age"]), normal_style)
        ],

        [
            Paragraph("<b>Gender</b>", normal_style),
            Paragraph(safe(screening["gender"]), normal_style)
        ],

        [
            Paragraph("<b>Location</b>", normal_style),
            Paragraph(safe(screening["location"]), normal_style)
        ],

        [
            Paragraph("<b>Screening Date</b>", normal_style),
            Paragraph(safe(screening["created_at"]), normal_style)
        ]
    ]


    patient_table = Table(

        patient_data,

        colWidths=[
            1.7 * inch,
            4.8 * inch
        ]
    )


    patient_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#f3f4f6")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d1d5db")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )


    story.append(patient_table)

    story.append(Spacer(1, 18))


    # ------------------------------------------------------
    # AI Screening Results
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "AI Screening Results",
            heading_style
        )
    )


    xray_confidence = (

        f"{screening['xray_confidence']}%"

        if screening["xray_confidence"] is not None

        else "Not Available"
    )


    risk_confidence = (

        f"{screening['risk_confidence']}%"

        if screening["risk_confidence"] is not None

        else "Not Available"
    )


    ai_data = [

        [
            Paragraph(
                "<b>X-ray Screening Result</b>",
                normal_style
            ),

            Paragraph(
                "<b>OA Risk Assessment</b>",
                normal_style
            )
        ],

        [
            Paragraph(
                safe(screening["xray_class_name"]),
                normal_style
            ),

            Paragraph(
                safe(screening["risk_level"]),
                normal_style
            )
        ],

        [
            Paragraph(
                f"Confidence: {safe(xray_confidence)}",
                small_style
            ),

            Paragraph(
                f"Confidence: {safe(risk_confidence)}",
                small_style
            )
        ]
    ]


    ai_table = Table(

        ai_data,

        colWidths=[
            3.25 * inch,
            3.25 * inch
        ]
    )


    ai_table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d1d5db")
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#f3f4f6")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )


    story.append(ai_table)

    story.append(Spacer(1, 18))


    # ------------------------------------------------------
    # Clinical Questionnaire
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Clinical Questionnaire Summary",
            heading_style
        )
    )


    questionnaire_data = [

        [
            Paragraph("<b>Assessment</b>", normal_style),

            Paragraph("<b>Result</b>", normal_style)
        ],

        [
            "Pain Score",
            safe(screening["pain_score"])
        ],

        [
            "Morning Stiffness",
            safe(screening["morning_stiffness"])
        ],

        [
            "Walking Difficulty",
            safe(screening["walking_difficulty"])
        ],

        [
            "Difficulty Using Stairs",
            safe(screening["stairs_difficulty"])
        ],

        [
            "Knee Swelling",
            safe(screening["swelling"])
        ],

        [
            "Previous Knee Injury",
            safe(screening["previous_injury"])
        ]
    ]


    questionnaire_table = Table(

        questionnaire_data,

        colWidths=[
            3.25 * inch,
            3.25 * inch
        ]
    )


    questionnaire_table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d1d5db")
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#f3f4f6")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )


    story.append(questionnaire_table)

    story.append(Spacer(1, 18))


    # ------------------------------------------------------
    # Preventive Guidance
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Preventive Guidance",
            heading_style
        )
    )


    risk_level = screening["risk_level"]


    if risk_level == "High":

        guidance_text = [

            "<b>High-risk screening result:</b> "
            "The screening indicates a higher OA risk pattern.",

            "Consider referral to a qualified healthcare professional "
            "for further clinical evaluation.",

            "Preventive measures such as appropriate physical activity, "
            "weight management where appropriate, and joint-protective "
            "practices may be discussed with a healthcare professional."
        ]


    elif risk_level == "Medium":

        guidance_text = [

            "<b>Medium-risk screening result:</b> "
            "The screening indicates some OA-related risk markers.",

            "Consider follow-up assessment and preventive lifestyle "
            "guidance from a qualified healthcare professional."
        ]


    else:

        guidance_text = [

            "<b>Low-risk screening result:</b> "
            "The screening indicates relatively lower OA risk markers.",

            "Continue healthy movement and joint-protective practices "
            "and seek professional evaluation if symptoms persist "
            "or worsen."
        ]


    for text in guidance_text:

        story.append(
            Paragraph(
                text,
                normal_style
            )
        )

        story.append(Spacer(1, 6))


    story.append(Spacer(1, 10))


    # ------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------

    disclaimer = (

        "<b>Important:</b> This report is an AI-assisted screening "
        "and risk-assessment support tool. It is not a final medical "
        "diagnosis. Clinical decisions should be made by qualified "
        "healthcare professionals using appropriate clinical evaluation."
    )


    story.append(
        Paragraph(
            disclaimer,
            small_style
        )
    )


    # ------------------------------------------------------
    # Build PDF
    # ------------------------------------------------------

    document.build(story)


    pdf_buffer.seek(0)


    # ------------------------------------------------------
    # Filename
    # ------------------------------------------------------

    patient_id = str(
        screening["patient_id"]
    ).replace(" ", "_")


    filename = (
        f"OA_Screening_Report_{patient_id}.pdf"
    )


    # ------------------------------------------------------
    # Send PDF to browser
    # ------------------------------------------------------

    return send_file(

        pdf_buffer,

        as_attachment=True,

        download_name=filename,

        mimetype="application/pdf"
    )


# ==========================================================
# OA SCREENING
# ==========================================================

@app.route("/screening", methods=["GET", "POST"])
def screening():

    result = None


    if request.method == "POST":

        # --------------------------------------------------
        # Patient Registration
        # --------------------------------------------------

        patient_id = request.form.get(
            "patient_id"
        )

        patient_name = request.form.get(
            "patient_name"
        )

        age = request.form.get(
            "age"
        )

        gender = request.form.get(
            "gender"
        )

        location = request.form.get(
            "location"
        )


        # --------------------------------------------------
        # OA Questionnaire
        # --------------------------------------------------

        pain_score = request.form.get(
            "pain_score"
        )

        morning_stiffness = request.form.get(
            "morning_stiffness"
        )

        walking_difficulty = request.form.get(
            "walking_difficulty"
        )

        stairs_difficulty = request.form.get(
            "stairs_difficulty"
        )

        swelling = request.form.get(
            "swelling"
        )

        previous_injury = request.form.get(
            "previous_injury"
        )


        # --------------------------------------------------
        # X-ray Upload
        # --------------------------------------------------

        if "xray" not in request.files:

            result = {
                "error":
                    "Please select an X-ray image."
            }

            return render_template(
                "screening.html",
                result=result
            )


        file = request.files["xray"]


        if file.filename == "":

            result = {
                "error":
                    "Please select an X-ray image."
            }

            return render_template(
                "screening.html",
                result=result
            )


        # --------------------------------------------------
        # Save Uploaded X-ray
        # --------------------------------------------------

        filename = "uploaded_xray.jpg"

        image_path = (
            UPLOAD_FOLDER / filename
        )

        file.save(image_path)


        # --------------------------------------------------
        # MobileNetV2 X-ray Prediction
        # --------------------------------------------------

        xray_prediction = predict_xray(
            str(image_path)
        )


        # ==================================================
        # LightGBM Risk Prediction
        # ==================================================
        #
        # Current prototype:
        # Questionnaire values are real user inputs.
        # Sensor values are temporary prototype values.
        #
        # Later these values will come from ESP32 sensors.
        # ==================================================

        risk_input = {

            "age":
                int(age),

            "pain_score":
                int(pain_score),

            "morning_stiffness":
                1 if morning_stiffness == "Yes" else 0,

            "walking_difficulty":
                1 if walking_difficulty == "Yes" else 0,

            "stairs_difficulty":
                1 if stairs_difficulty == "Yes" else 0,

            "knee_swelling":
                1 if swelling == "Yes" else 0,

            "previous_knee_injury":
                1 if previous_injury == "Yes" else 0,


            # ------------------------------------------
            # TEMPORARY SENSOR VALUES
            # ------------------------------------------

            "thigh_acceleration":
                1.5,

            "shank_acceleration":
                1.4,

            "thigh_angular_velocity":
                1.8,

            "shank_angular_velocity":
                1.7,

            "knee_range_of_motion":
                115,

            "gait_speed":
                1.0,

            "emg_activity":
                0.5,

            "pressure_peak":
                50,

            "pressure_asymmetry":
                10
        }


        # --------------------------------------------------
        # Run LightGBM
        # --------------------------------------------------

        risk_prediction = predict_risk(
            risk_input
        )


        # --------------------------------------------------
        # Save Screening to Database
        # --------------------------------------------------

        save_screening(

            patient_id=patient_id,

            patient_name=patient_name,

            age=int(age),

            gender=gender,

            location=location,

            pain_score=int(pain_score),

            morning_stiffness=morning_stiffness,

            walking_difficulty=walking_difficulty,

            stairs_difficulty=stairs_difficulty,

            swelling=swelling,

            previous_injury=previous_injury,

            xray_class_id=
                xray_prediction["class_id"],

            xray_class_name=
                xray_prediction["class_name"],

            xray_confidence=
                xray_prediction["confidence"],

            risk_level=
                risk_prediction["risk_level"],

            risk_confidence=
                risk_prediction["confidence"]
        )


        # --------------------------------------------------
        # Combine AI Results
        # --------------------------------------------------

        result = {

            "class_id":
                xray_prediction["class_id"],

            "class_name":
                xray_prediction["class_name"],

            "confidence":
                xray_prediction["confidence"],

            "risk_level":
                risk_prediction["risk_level"],

            "risk_confidence":
                risk_prediction["confidence"],

            "risk_probabilities":
                risk_prediction["probabilities"]
        }


    return render_template(

        "screening.html",

        result=result
    )


# ==========================================================
# ESP32 STATUS
# ==========================================================

@app.route("/esp32/status")
def esp32_status():

    return {

        "connected": False,

        "device": "ESP32",

        "message": "ESP32 not connected"
    }


# ==========================================================
# RUN FLASK APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )