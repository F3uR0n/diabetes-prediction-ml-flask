from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
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
    # Load trained model & scaler
    model = pickle.load(open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb"))
    print(f"✅ Model loaded: {type(model).__name__}")

    scaler = pickle.load(open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "rb"))
    print(f"✅ Scaler loaded")

    # Load feature order
    try:
        FEATURE_ORDER = pickle.load(
            open(os.path.join(ARTIFACTS_DIR, "feature_order.pkl"), "rb")
        )
        print(f"✅ Feature order loaded: {FEATURE_ORDER}")
    except FileNotFoundError:
        FEATURE_ORDER = [
            'gender',
            'age',
            'hypertension',
            'heart_disease',
            'smoking_history',
            'bmi',
            'HbA1c_level',
            'blood_glucose_level'
        ]
        print(f"⚠️ Using fallback feature order: {FEATURE_ORDER}")

    # Load label encoders
    label_encoders = pickle.load(
        open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "rb")
    )
    print(f"✅ Label encoders loaded")
    
    # 🔥 CREATE REVERSE MAPPINGS FOR THE FORM
    gender_encoder = label_encoders['gender']
    smoking_encoder = label_encoders['smoking_history']
    
    # Get the actual mappings
    gender_mapping = dict(zip(gender_encoder.classes_, gender_encoder.transform(gender_encoder.classes_)))
    smoking_mapping = dict(zip(smoking_encoder.classes_, smoking_encoder.transform(smoking_encoder.classes_)))
    
    print(f"   Gender mapping: {gender_mapping}")
    print(f"   Smoking mapping: {smoking_mapping}")

    # Load metadata
    try:
        metadata = pickle.load(
            open(os.path.join(ARTIFACTS_DIR, "model_metadata.pkl"), "rb")
        )
        print(f"✅ Metadata loaded")
        print(f"   Model: {metadata.get('model_name', 'Unknown')}")
        print(f"   Accuracy: {metadata.get('accuracy', 'Unknown')}")
    except FileNotFoundError:
        metadata = None
        print(f"⚠️ Metadata not found")

    print("=" * 60)
    print("SERVER READY")
    print("=" * 60)

except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    print(traceback.format_exc())
    raise

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            print("\n" + "=" * 60)
            print("NEW PREDICTION REQUEST")
            print("=" * 60)

            # Read form inputs AS STRINGS FIRST
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            hba1c = float(request.form["hba1c"])
            glucose = float(request.form["glucose"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            
            # 🔥 USE LABEL ENCODERS TO TRANSFORM TEXT VALUES
            gender_text = request.form["gender"]  # This will be like "Male" or "Female"
            smoking_text = request.form["smoking"]  # This will be like "never" or "current"
            
            # Transform using the actual encoders
            gender = gender_encoder.transform([gender_text])[0]
            smoking = smoking_encoder.transform([smoking_text])[0]

            print("\nRaw form inputs:")
            print(f"  Age: {age}")
            print(f"  BMI: {bmi}")
            print(f"  HbA1c: {hba1c}")
            print(f"  Glucose: {glucose}")
            print(f"  Hypertension: {hypertension}")
            print(f"  Heart Disease: {heart_disease}")
            print(f"  Gender (text): {gender_text} → Encoded: {gender}")
            print(f"  Smoking (text): {smoking_text} → Encoded: {smoking}")

            input_dict = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'smoking_history': smoking,
                'bmi': bmi,
                'HbA1c_level': hba1c,
                'blood_glucose_level': glucose
            }

            user_input = np.array([[input_dict[col] for col in FEATURE_ORDER]])
            print(f"\nOrdered input array: {user_input}")
            print(f"Feature order: {FEATURE_ORDER}")

            user_input_scaled = scaler.transform(user_input)
            print(f"Scaled input: {user_input_scaled}")

            result = model.predict(user_input_scaled)[0]
            print(f"\nPrediction: {result}")

            prob = None
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(user_input_scaled)[0]
                prob = prob_array[1]
                print(f"Probability array: {prob_array}")
                print(f"Diabetic probability: {prob}")

            prediction_text = "Diabetic ❌" if result == 1 else "Non-Diabetic ✅"
            session["prediction"] = prediction_text
            session["probability"] = round(prob * 100, 2) if prob is not None else None

            print(f"\nFinal Result: {prediction_text}")
            if prob is not None:
                print(f"Risk Probability: {round(prob * 100, 2)}%")
            print("=" * 60 + "\n")

            return redirect(url_for("index"))

        except Exception as e:
            print(f"\n❌ ERROR during prediction: {e}")
            print(traceback.format_exc())
            session["prediction"] = f"Error: {str(e)}"
            session["probability"] = None
            return redirect(url_for("index"))

    prediction = session.pop("prediction", None)
    probability = session.pop("probability", None)

    # 🔥 PASS THE MAPPINGS TO THE TEMPLATE
    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        gender_options=sorted(gender_encoder.classes_),
        smoking_options=sorted(smoking_encoder.classes_)
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    