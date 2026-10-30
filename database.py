import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "oa_screening.db"


def get_connection():

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------
    # Create screenings table if it does not exist
    # --------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screenings (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            location TEXT NOT NULL,

            pain_score INTEGER,
            morning_stiffness TEXT,
            walking_difficulty TEXT,
            stairs_difficulty TEXT,
            swelling TEXT,
            previous_injury TEXT,

            xray_class_id INTEGER,
            xray_class_name TEXT,
            xray_confidence REAL,

            risk_level TEXT,
            risk_confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------
    # Add risk columns to existing database
    # --------------------------------------------------
    # This is important because our existing database
    # may already have the screenings table without
    # these new columns.
    # --------------------------------------------------

    cursor.execute("PRAGMA table_info(screenings)")

    existing_columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "risk_level" not in existing_columns:

        cursor.execute("""
            ALTER TABLE screenings
            ADD COLUMN risk_level TEXT
        """)

    if "risk_confidence" not in existing_columns:

        cursor.execute("""
            ALTER TABLE screenings
            ADD COLUMN risk_confidence REAL
        """)

    connection.commit()
    connection.close()


def save_screening(
    patient_id,
    patient_name,
    age,
    gender,
    location,
    pain_score,
    morning_stiffness,
    walking_difficulty,
    stairs_difficulty,
    swelling,
    previous_injury,
    xray_class_id,
    xray_class_name,
    xray_confidence,
    risk_level,
    risk_confidence
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO screenings (

            patient_id,
            patient_name,
            age,
            gender,
            location,

            pain_score,
            morning_stiffness,
            walking_difficulty,
            stairs_difficulty,
            swelling,
            previous_injury,

            xray_class_id,
            xray_class_name,
            xray_confidence,

            risk_level,
            risk_confidence
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        patient_id,
        patient_name,
        age,
        gender,
        location,

        pain_score,
        morning_stiffness,
        walking_difficulty,
        stairs_difficulty,
        swelling,
        previous_injury,

        xray_class_id,
        xray_class_name,
        xray_confidence,

        risk_level,
        risk_confidence
    ))

    connection.commit()
    connection.close()


def get_dashboard_stats():

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------
    # Total screenings
    # --------------------------------------------------

    total_screenings = cursor.execute(
        "SELECT COUNT(*) FROM screenings"
    ).fetchone()[0]

    # --------------------------------------------------
    # Total unique patients
    # --------------------------------------------------

    total_patients = cursor.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM screenings"
    ).fetchone()[0]

    # --------------------------------------------------
    # High risk count
    # --------------------------------------------------

    high_risk = cursor.execute("""
        SELECT COUNT(*)
        FROM screenings
        WHERE risk_level = 'High'
    """).fetchone()[0]

    # --------------------------------------------------
    # Medium risk count
    # --------------------------------------------------

    medium_risk = cursor.execute("""
        SELECT COUNT(*)
        FROM screenings
        WHERE risk_level = 'Medium'
    """).fetchone()[0]

    # --------------------------------------------------
    # Low risk count
    # --------------------------------------------------

    low_risk = cursor.execute("""
        SELECT COUNT(*)
        FROM screenings
        WHERE risk_level = 'Low'
    """).fetchone()[0]

    # --------------------------------------------------
    # Recent screenings
    # --------------------------------------------------

    recent_screenings = cursor.execute("""
        SELECT

            patient_id,
            patient_name,
            age,
            gender,

            xray_class_name,
            xray_confidence,

            risk_level,
            risk_confidence,

            created_at

        FROM screenings

        ORDER BY id DESC

        LIMIT 5
    """).fetchall()

    connection.close()

    return {

        "total_screenings":
            total_screenings,

        "total_patients":
            total_patients,

        "high_risk":
            high_risk,

        "medium_risk":
            medium_risk,

        "low_risk":
            low_risk,

        "recent_screenings":
            recent_screenings
    }
def get_all_screenings():

    connection = get_connection()
    cursor = connection.cursor()

    screenings = cursor.execute("""
        SELECT
            id,
            patient_id,
            patient_name,
            age,
            gender,
            location,
            pain_score,
            morning_stiffness,
            walking_difficulty,
            stairs_difficulty,
            swelling,
            previous_injury,
            xray_class_id,
            xray_class_name,
            xray_confidence,
            risk_level,
            risk_confidence,
            created_at
        FROM screenings
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return screenings