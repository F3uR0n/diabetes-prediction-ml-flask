from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
import os
import traceback

app = Flask(__name__)
app.secret_key = "dev-secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

print("=" * 60)
print("LOADING MODEL AND ARTIFACTS")
print("=" * 60)

try:
    model = pickle.load(open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb"))
    print(f"[OK] Model loaded: {type(model).__name__}")

    scaler = pickle.load(open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "rb"))
    print("[OK] Scaler loaded")

    try:
        FEATURE_ORDER = pickle.load(
            open(os.path.join(ARTIFACTS_DIR, "feature_order.pkl"), "rb")
        )
        print(f"[OK] Feature order: {FEATURE_ORDER}")
    except FileNotFoundError:
        FEATURE_ORDER = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]
        print(f"[WARN] Using fallback feature order: {FEATURE_ORDER}")

    label_encoders = pickle.load(
        open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "rb")
    )
    print("[OK] Label encoders loaded")

    gender_encoder  = label_encoders['gender']
    smoking_encoder = label_encoders['smoking_history']

    gender_mapping  = dict(zip(gender_encoder.classes_,  gender_encoder.transform(gender_encoder.classes_)))
    smoking_mapping = dict(zip(smoking_encoder.classes_, smoking_encoder.transform(smoking_encoder.classes_)))
    print(f"   Gender mapping:  {gender_mapping}")
    print(f"   Smoking mapping: {smoking_mapping}")

    try:
        metadata = pickle.load(open(os.path.join(ARTIFACTS_DIR, "model_metadata.pkl"), "rb"))
        print(f"[OK] Metadata: {metadata.get('model_name')} | acc={metadata.get('accuracy', 0):.4f} | f1={metadata.get('f1_diabetic', '?')}")
    except FileNotFoundError:
        metadata = None
        print("[WARN] Metadata not found")

    # Youden-optimal threshold (saved during training); fallback to 0.5
    PRED_THRESHOLD = float(metadata.get('threshold', 0.5)) if metadata else 0.5
    print(f"[OK] Prediction threshold: {PRED_THRESHOLD}")

    print("=" * 60)
    print("SERVER READY")
    print("=" * 60)

except Exception as e:
    print(f"[ERROR] Loading model: {e}")
    print(traceback.format_exc())
    raise


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            print("\n" + "=" * 60)
            print("NEW PREDICTION REQUEST")
            print("=" * 60)

            age           = float(request.form["age"])
            bmi           = float(request.form["bmi"])
            hba1c         = float(request.form["hba1c"])
            glucose       = float(request.form["glucose"])
            hypertension  = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])

            gender_text  = request.form["gender"]
            smoking_text = request.form["smoking"]

            gender  = gender_encoder.transform([gender_text])[0]
            smoking = smoking_encoder.transform([smoking_text])[0]

            print(f"  Age={age} BMI={bmi} HbA1c={hba1c} Glucose={glucose}")
            print(f"  Hypertension={hypertension} HeartDisease={heart_disease}")
            print(f"  Gender: {gender_text} -> {gender}")
            print(f"  Smoking: {smoking_text} -> {smoking}")

            input_dict = {
                'gender':             gender,
                'age':                age,
                'hypertension':       hypertension,
                'heart_disease':      heart_disease,
                'smoking_history':    smoking,
                'bmi':                bmi,
                'HbA1c_level':        hba1c,
                'blood_glucose_level': glucose,
            }

            # Pass a named DataFrame so the scaler sees the same feature names it was fit on
            user_input = pd.DataFrame([input_dict], columns=FEATURE_ORDER)
            user_input_scaled = scaler.transform(user_input)
            print(f"Scaled input: {user_input_scaled}")

            prob = None
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(user_input_scaled)[0]
                prob = float(prob_array[1])
                print(f"Probability array: {prob_array}")

            # Use Youden-optimal threshold instead of default 0.5
            result = 1 if (prob is not None and prob >= PRED_THRESHOLD) else int(model.predict(user_input_scaled)[0])
            print(f"Prediction class: {result}  (prob={prob:.4f}, threshold={PRED_THRESHOLD})")

            prediction_text = "Diabetic" if result == 1 else "Non-Diabetic"
            risk_pct = round(prob * 100, 2) if prob is not None else None
            risk_level = (
                "high"   if risk_pct is not None and risk_pct >= 60 else
                "medium" if risk_pct is not None and risk_pct >= 30 else
                "low"
            )

            session["prediction"]    = prediction_text
            session["probability"]   = risk_pct
            session["risk_level"]    = risk_level
            session["input_summary"] = {
                "age":          int(age),
                "bmi":          round(bmi, 1),
                "hba1c":        round(hba1c, 1),
                "glucose":      int(glucose),
                "hypertension": bool(hypertension),
                "heart_disease": bool(heart_disease),
            }

            print(f"Result: {prediction_text} | {risk_pct}% | {risk_level}")
            print("=" * 60 + "\n")
            return redirect(url_for("index"))

        except Exception as e:
            print(f"\n[ERROR] Prediction: {e}")
            print(traceback.format_exc())
            session["prediction"] = f"Error: {str(e)}"
            session["probability"] = None
            return redirect(url_for("index"))

    prediction    = session.pop("prediction", None)
    probability   = session.pop("probability", None)
    risk_level    = session.pop("risk_level", None)
    input_summary = session.pop("input_summary", None)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        input_summary=input_summary,
        gender_options=sorted(gender_encoder.classes_),
        smoking_options=sorted(smoking_encoder.classes_),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
