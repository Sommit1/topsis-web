import os
import re
import uuid
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv

import smtplib
from email.message import EmailMessage

load_dotenv()

app = Flask(__name__)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


# -----------------------
# TOPSIS CORE (logic as a function for web use)
# -----------------------
def run_topsis(input_file: str, weights_str: str, impacts_str: str, output_file: str) -> None:
    # Parse weights & impacts
    weights = [w.strip() for w in weights_str.split(",")]
    impacts = [i.strip() for i in impacts_str.split(",")]

    # Read CSV
    try:
        df = pd.read_csv(input_file)
    except Exception:
        raise ValueError("File not found")

    # Validations
    if df.shape[1] < 3:
        raise ValueError("Input file must contain three or more columns")

    data = df.iloc[:, 1:]

    try:
        data = data.astype(float)
    except Exception:
        raise ValueError("From 2nd to last columns must contain numeric values only")

    if len(weights) != data.shape[1] or len(impacts) != data.shape[1]:
        raise ValueError("Number of weights, impacts and columns must be same")

    for imp in impacts:
        if imp not in ["+", "-"]:
            raise ValueError("Impacts must be either + or -")

    weights = np.array(weights, dtype=float)

    norm = np.sqrt((data ** 2).sum())
    normalized_data = data / norm
    weighted_data = normalized_data * weights

    ideal_best = []
    ideal_worst = []

    for idx, imp in enumerate(impacts):
        col = weighted_data.iloc[:, idx]
        if imp == "+":
            ideal_best.append(col.max())
            ideal_worst.append(col.min())
        else:
            ideal_best.append(col.min())
            ideal_worst.append(col.max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    distance_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    topsis_score = distance_worst / (distance_best + distance_worst)

    df["Topsis Score"] = topsis_score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    df.to_csv(output_file, index=False)


# -----------------------
# EMAIL SENDER (SMTP)
# -----------------------
def send_email_with_attachment(to_email: str, attachment_path: str) -> None:
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not smtp_user or not smtp_pass:
        raise RuntimeError("Email credentials missing. Set SMTP_USER and SMTP_PASS in .env")

    msg = EmailMessage()
    msg["Subject"] = "TOPSIS Result File"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content("Hi,\n\nAttached is your TOPSIS result file.\n\nRegards")

    filename = os.path.basename(attachment_path)
    with open(attachment_path, "rb") as f:
        file_data = f.read()

    # attach as CSV
    msg.add_attachment(file_data, maintype="text", subtype="csv", filename=filename)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


# -----------------------
# ROUTES
# -----------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.post("/submit")
def submit():
    try:
        file = request.files.get("file")
        weights = (request.form.get("weights") or "").strip()
        impacts = (request.form.get("impacts") or "").strip()
        email = (request.form.get("email") or "").strip()

        # Assignment validations
        if not file or file.filename.strip() == "":
            return render_template("index.html", error="Please upload a CSV file.")

        if "," not in weights or "," not in impacts:
            return render_template("index.html", error="Weights and impacts must be separated by comma (,).")

        weights_list = [w.strip() for w in weights.split(",")]
        impacts_list = [i.strip() for i in impacts.split(",")]

        if len(weights_list) != len(impacts_list):
            return render_template("index.html", error="Number of weights must be equal to number of impacts.")

        for i in impacts_list:
            if i not in ["+", "-"]:
                return render_template("index.html", error="Impacts must be either + or -.")

        if not EMAIL_REGEX.match(email):
            return render_template("index.html", error="Format of email id is not correct.")

        # Save uploaded file
        uid = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")
        file.save(input_path)

        # Run TOPSIS
        output_path = os.path.join(RESULT_DIR, f"topsis_result_{uid}.csv")
        run_topsis(input_path, weights, impacts, output_path)

        # Email result
        send_email_with_attachment(email, output_path)

        return render_template("index.html", message="Result generated and emailed successfully!")

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

@app.get("/download/<filename>")
def download(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
