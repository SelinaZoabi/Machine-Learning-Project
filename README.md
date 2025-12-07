# Machine-Learning-Project
Here is a **clean, short, professional README** for your project.
Minimalist, clear, exactly what your teacher expects — **you can copy-paste it directly into GitHub**.

---

# **Chicago Domestic Violence – 90-Day Repeat Risk Prediction**

**Machine Learning Project – Sirine Saghiri**

---

## 📌 **Project Overview**

This project predicts **whether a household in Chicago will experience another domestic-violence incident within the next 90 days**, based on past police reports.

It is a **binary classification problem** (`repeat` vs `non-repeat`) trained on a cleaned and filtered version of the Chicago Crimes dataset.

The final goal is to provide:

* an accurate ML model (F1 ≈ **0.80**)
* interpretable insights on risk factors
* a simple Streamlit app to test predictions

---

## 🗂️ **Dataset**

**Source:** Chicago Data Portal (Crimes 2001–2023)
**Size used:** 200,000-row cleaned sample
**Filtering:** kept only incidents where `Domestic == True`.

### Main preprocessing steps:

* Removed rows with missing `Date`
* Converted `Date` into proper datetime format
* Extracted new time features: **year, month, day, hour, weekday**
* Built one household key per block using `Block`
* Engineered additional features:
  * Weekend / Night
  * Cyclical month encoding (sin/cos)
  * Latitude–Longitude interactions
  * District-level and hour-level risk rates

## **Models Used**
I trained and compared three supervised classification models :

| Model             | F1 Score (Test)      |
| ----------------- | -------------------- |
| **Random Forest** | ~0.75 (after tuning) |
| **XGBoost**       | ~0.7952                |
| **CatBoost**      | ~0.7988 (final model)  |

### Optimization attempts:
* For RandomForest : Manual parameter tuning 
* Light Optuna search for XGBoost & CatBoost
CatBoost was selected as the **final model**.

## **Model Interpretation**
* Geographic features (Latitude/Longitude) were consistently important
* District-level repeat risk was highly predictive
* Time-based patterns (hour, month) also contributed strongly
* Feature importance differs across models (tree structures differ)

The model mainly captures:
* **location patterns**,
* **district-level hotspot patterns**,
* **time-of-day and seasonal repeat tendencies**.

## **Author**
**Selina Zoabi**
BSc Data Science for Responsible Business – emlyon & Centrale Lyon
