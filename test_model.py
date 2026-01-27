import pickle
import numpy as np

print("="*60)
print("DIAGNOSTIC SCRIPT - Checking Model Files")
print("="*60)

# Load model
try:
    model = pickle.load(open("artifacts/model.pkl", "rb"))
    print(f"\n✅ Model loaded successfully")
    print(f"   Type: {type(model).__name__}")
    print(f"   Has predict_proba: {hasattr(model, 'predict_proba')}")
except Exception as e:
    print(f"\n❌ Failed to load model.pkl: {e}")
    exit(1)

# Load scaler
try:
    scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
    print(f"\n✅ Scaler loaded successfully")
    print(f"   Features: {scaler.n_features_in_}")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")
except Exception as e:
    print(f"\n❌ Failed to load scaler.pkl: {e}")
    exit(1)

# Test with sample data
print("\n" + "="*60)
print("TESTING WITH SAMPLE DATA")
print("="*60)

# Test case 1: High-risk diabetic profile
print("\n📊 Test Case 1: High-risk profile (should predict diabetic)")
test_high_risk = np.array([[
    1,      # gender (male)
    65,     # age
    1,      # hypertension (yes)
    1,      # heart_disease (yes)
    2,      # smoking_history (current smoker)
    35.0,   # bmi (obese)
    8.5,    # HbA1c_level (high)
    200     # blood_glucose_level (high)
]])

print(f"Raw input: {test_high_risk}")
test_high_risk_scaled = scaler.transform(test_high_risk)
print(f"Scaled input: {test_high_risk_scaled}")

prediction = model.predict(test_high_risk_scaled)[0]
print(f"Prediction: {prediction} ({'Diabetic' if prediction == 1 else 'Non-Diabetic'})")

if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(test_high_risk_scaled)[0]
    print(f"Probabilities: Non-Diabetic={proba[0]:.4f}, Diabetic={proba[1]:.4f}")
    print(f"Risk: {proba[1]*100:.2f}%")

# Test case 2: Low-risk profile
print("\n📊 Test Case 2: Low-risk profile (should predict non-diabetic)")
test_low_risk = np.array([[
    0,      # gender (female)
    25,     # age
    0,      # hypertension (no)
    0,      # heart_disease (no)
    0,      # smoking_history (never)
    22.0,   # bmi (normal)
    5.0,    # HbA1c_level (normal)
    90      # blood_glucose_level (normal)
]])

print(f"Raw input: {test_low_risk}")
test_low_risk_scaled = scaler.transform(test_low_risk)
print(f"Scaled input: {test_low_risk_scaled}")

prediction = model.predict(test_low_risk_scaled)[0]
print(f"Prediction: {prediction} ({'Diabetic' if prediction == 1 else 'Non-Diabetic'})")

if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(test_low_risk_scaled)[0]
    print(f"Probabilities: Non-Diabetic={proba[0]:.4f}, Diabetic={proba[1]:.4f}")
    print(f"Risk: {proba[1]*100:.2f}%")

# Test case 3: Medium-risk profile
print("\n📊 Test Case 3: Medium-risk profile")
test_medium_risk = np.array([[
    1,      # gender (male)
    45,     # age
    0,      # hypertension (no)
    0,      # heart_disease (no)
    1,      # smoking_history (former)
    28.5,   # bmi (overweight)
    6.2,    # HbA1c_level (pre-diabetic)
    110     # blood_glucose_level (slightly elevated)
]])

print(f"Raw input: {test_medium_risk}")
test_medium_risk_scaled = scaler.transform(test_medium_risk)
print(f"Scaled input: {test_medium_risk_scaled}")

prediction = model.predict(test_medium_risk_scaled)[0]
print(f"Prediction: {prediction} ({'Diabetic' if prediction == 1 else 'Non-Diabetic'})")

if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(test_medium_risk_scaled)[0]
    print(f"Probabilities: Non-Diabetic={proba[0]:.4f}, Diabetic={proba[1]:.4f}")
    print(f"Risk: {proba[1]*100:.2f}%")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nIf all tests show 0 (Non-Diabetic) with 0% risk, there's a problem.")
print("Expected: Test 1 should be diabetic, Test 2 should be non-diabetic")
print("="*60)