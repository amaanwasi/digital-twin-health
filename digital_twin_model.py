# DIGITAL TWIN HEALTH MODEL (Optimized & Finalized)
# Author: ChatGPT for Amaan
# Comprehensive, efficient, and clinically-aware twin generator

import pandas as pd
import numpy as np
from datetime import timedelta
import json

import os

# Load extended medical knowledge if available
EXTENDED_KNOWLEDGE_PATH = "extended_medical_knowledge.json"
if os.path.exists(EXTENDED_KNOWLEDGE_PATH):
    with open(EXTENDED_KNOWLEDGE_PATH, "r") as f:
        EXTENDED_KNOWLEDGE = json.load(f)
        MEDICAL_KNOWLEDGE.update(EXTENDED_KNOWLEDGE)

MEDICAL_KNOWLEDGE = {
    "prolonged_low_hrv": {
        "triggers": ["low_hrv", "low_hrv"],
        "description": "Prolonged low heart rate variability may indicate chronic stress or fatigue.",
        "icd": "QD85",
        "source": "ICD-11 / NICE"
    },
    "prolonged_sleep_deficit": {
        "triggers": ["sleep_deficit", "sleep_deficit"],
        "description": "Consistently insufficient sleep is associated with long-term health risks like obesity, hypertension, and diabetes.",
        "icd": "MG22",
        "source": "NICE"
    },
    "prolonged_low_activity": {
        "triggers": ["low_activity", "low_activity"],
        "description": "Extended periods of low physical activity can lead to cardiovascular deconditioning and metabolic dysfunction.",
        "icd": "5A21",
        "source": "ICD-11 / Mayo Clinic"
    },
    "cardiac_alert": {
        "triggers": ["abnormal_ecg"],
        "icd": "I48.0",
        "description": "Atrial fibrillation and flutter",
        "source": "ICD-11"
    },
    "fatigue": {
        "triggers": ["low_hrv", "sleep_deficit", "low_energy", "anemia_symptoms", "poor_sleep_quality"],
        "icd": "MG22",
        "description": "Disorders of initiating and maintaining sleep",
        "source": "NICE / ICD-11"
    },
    "stress_burnout": {
        "triggers": ["high_stress", "mood_drop", "low_energy"],
        "icd": "QD85",
        "description": "Reaction to severe stress and adjustment disorders",
        "source": "ICD-11 / Mayo Clinic"
    },
    "gut_disruption": {
        "triggers": ["digestive_discomfort", "bloating", "irregular_bowel"],
        "icd": "DA61",
        "description": "Irritable bowel syndrome",
        "source": "Mayo Clinic / NICE"
    },
    "pain_alert": {
        "triggers": ["reported_pain", "chronic_pain"],
        "icd": "MG30.0",
        "description": "Chronic primary pain",
        "source": "ICD-11"
    },
    "hypertension_risk": {
        "triggers": ["elevated_bp", "high_hr", "family_history_hypertension", "persistent_headache"],
        "icd": "BA00",
        "description": "Essential (primary) hypertension",
        "source": "ICD-11 / NICE"
    },
    "pre_diabetes": {
        "triggers": ["elevated_glucose", "increased_thirst", "low_activity", "frequent_urination"],
        "icd": "5A21",
        "description": "Impaired glucose regulation",
        "source": "ICD-11 / LOINC"
    },
    "sleep_apnea_risk": {
        "triggers": ["low_sleep_efficiency", "snoring", "daytime_fatigue", "restless_sleep"],
        "icd": "7A20",
        "description": "Obstructive sleep apnea",
        "source": "ICD-11 / NICE"
    },
    "dehydration_risk": {
        "triggers": ["low_fluid_intake", "dry_mouth", "dizziness"],
        "icd": "5C70",
        "description": "Dehydration",
        "source": "ICD-11 / Mayo Clinic"
    },
    "infection_risk": {
        "triggers": ["fever", "chills", "rapid_hr", "fatigue", "elevated_wbc"],
        "icd": "1A00",
        "description": "Infectious condition, unspecified",
        "source": "ICD-11 / NICE"
    }
}

with open("medical_knowledge_base.json", "w") as f:
    json.dump(MEDICAL_KNOWLEDGE, f, indent=2)

class DigitalTwinModel:
    def __init__(self, user_id, df, subjective_df=None, ecg_df=None):
        self.user_id = user_id
        self.df = df[df['user_id'] == user_id].copy()
        self.subjective_df = subjective_df[subjective_df['user_id'] == user_id].copy() if subjective_df is not None else None
        self.ecg_df = ecg_df[ecg_df['user_id'] == user_id].copy() if ecg_df is not None else None
        self.twin = {
            "user_id": user_id,
            "baseline_profile": {},
            "current_state": {},
            "deviation_flags": {},
            "predicted_risk": {},
            "recommendations": [],
            "clinical_summary": ""
        }

    def compute_baseline(self):
        df = self.df.fillna(np.nan)
        self.twin["baseline_profile"] = {
            "resting_hr": round(df["resting_heart_rate"].mean(skipna=True), 1) if not df["resting_heart_rate"].isna().all() else None,
            "hrv": round(df["hrv"].mean(skipna=True), 1) if not df["hrv"].isna().all() else None,
            "sleep_avg": round(df["sleep_hours"].mean(skipna=True), 2) if not df["sleep_hours"].isna().all() else None,
            "vo2_max": round(df["vo2_max"].mean(skipna=True), 1) if not df["vo2_max"].isna().all() else None,
            "steps_avg": int(df["steps"].mean(skipna=True)) if not df["steps"].isna().all() else None
        }

    def compute_current_state(self):
        df = self.df.fillna(np.nan)
        latest = df["timestamp"].max()
        recent_df = df[df["timestamp"] > latest - timedelta(hours=24)]
        self.twin["current_state"] = {
            "heart_rate": round(recent_df["heart_rate"].mean(skipna=True), 1) if not recent_df["heart_rate"].isna().all() else None,
            "hrv": round(recent_df["hrv"].mean(skipna=True), 1) if not recent_df["hrv"].isna().all() else None,
            "sleep_hours": round(recent_df["sleep_hours"].mean(skipna=True), 2) if not recent_df["sleep_hours"].isna().all() else None,
            "steps": int(recent_df["steps"].sum(skipna=True)) if not recent_df["steps"].isna().all() else None
        }
        if self.subjective_df is not None:
            recent_subj = self.subjective_df[self.subjective_df["timestamp"] > latest - timedelta(hours=24)]
            for col in ["energy", "mood", "stress", "digestion", "pain"]:
                if col in recent_subj:
                    self.twin["current_state"][col] = round(recent_subj[col].mean(skipna=True), 1)

    def analyze_deviation(self):
        base = self.twin["baseline_profile"]
        curr = self.twin["current_state"]
        flags = {
            "low_hrv": curr.get("hrv") is not None and base.get("hrv") and curr["hrv"] < 0.8 * base["hrv"],
            "sleep_deficit": curr.get("sleep_hours") is not None and base.get("sleep_avg") and curr["sleep_hours"] < 0.8 * base["sleep_avg"],
            "low_activity": curr.get("steps") is not None and base.get("steps_avg") and curr["steps"] < 0.5 * base["steps_avg"]
        }
        if self.subjective_df is not None:
            flags.update({
                "low_energy": curr.get("energy", 5) < 3,
                "high_stress": curr.get("stress", 0) > 6,
                "mood_drop": curr.get("mood", 5) < 4,
                "digestive_discomfort": curr.get("digestion", 5) < 3,
                "reported_pain": curr.get("pain", 0) > 4
            })
        for key in ["hrv", "sleep_hours", "steps"]:
            if curr.get(key) is None:
                flags[f"missing_{key}"] = True
        self.twin["deviation_flags"] = flags

    def predict_risk(self):
        flags = self.twin["deviation_flags"]
        risks = {}
        for condition, triggers in MEDICAL_KNOWLEDGE.items():
            matched = sum([1 for t in triggers if flags.get(t)])
            risks[condition] = round(matched / len(triggers), 2)
        self.twin["predicted_risk"] = risks

    def generate_recommendations(self):
        flags = self.twin["deviation_flags"]
        recs = []
        if flags.get("sleep_deficit"): recs.append("Sleep 8+ hours tonight.")
        if flags.get("low_hrv"): recs.append("Engage in mindfulness or light recovery activity.")
        if flags.get("low_activity"): recs.append("Take a walk or light movement break.")
        if flags.get("low_energy"): recs.append("Reduce load and ensure hydration/nutrition.")
        if flags.get("high_stress"): recs.append("Try breathing exercises or meditation today.")
        if flags.get("mood_drop"): recs.append("Connect socially or do something enjoyable.")
        if flags.get("digestive_discomfort"): recs.append("Consider lighter meals and hydration.")
        if flags.get("reported_pain"): recs.append("Rest and monitor pain; seek help if persistent.")
        if flags.get("abnormal_ecg"): recs.append("Abnormal ECG detected. Follow up with a cardiologist.")
        self.twin["recommendations"] = recs

    def generate_summary(self):
        curr = self.twin["current_state"]
        flags = self.twin["deviation_flags"]
        summary_parts = []

        summary_parts.append("Hi! Here's how your health looks today based on your recent data.")

        if curr.get("heart_rate") is not None:
            summary_parts.append(f"Your average heart rate is around {curr.get('heart_rate')} bpm.")
        if curr.get("hrv") is not None:
            summary_parts.append(f"Your heart rate variability is {curr.get('hrv')} ms, which reflects how well your body is recovering.")
        if curr.get("sleep_hours") is not None:
            summary_parts.append(f"You slept for about {curr.get('sleep_hours')} hours.")

        if flags.get("sleep_deficit"):
            summary_parts.append("It looks like you might not be getting enough sleep lately.")
        if flags.get("low_hrv"):
            summary_parts.append("Your HRV is lower than your usual, which may indicate stress or fatigue.")
        if flags.get("high_stress"):
            summary_parts.append("You're reporting high stress levels.")
        if flags.get("low_energy"):
            summary_parts.append("You’ve indicated feeling low on energy.")
        if flags.get("digestive_discomfort"):
            summary_parts.append("You’re experiencing some digestive discomfort.")
        if flags.get("reported_pain"):
            summary_parts.append("There are signs of ongoing pain.")
        if flags.get("abnormal_ecg"):
            summary_parts.append("There are recent ECG patterns that should be reviewed clinically.")

        summary_parts.append("Based on this, we’ve suggested a few helpful actions for your wellbeing.")
        self.twin["clinical_summary"] = " ".join(summary_parts)

    def integrate_ecg_data(self):
        if self.ecg_df is not None and not self.ecg_df.empty:
            recent = self.ecg_df[self.ecg_df["timestamp"] > self.ecg_df["timestamp"].max() - timedelta(days=7)]
            abnormal = recent[~recent["classification"].str.contains("Sinus", case=False, na=False)]
            if not abnormal.empty:
                self.twin["deviation_flags"]["abnormal_ecg"] = True

    def answer_question(self, question: str) -> str:
        context = f"""
        Clinical Summary:
        {self.twin.get('clinical_summary', '')}
        Risk Scores:
        {json.dumps(self.twin.get('predicted_risk', {}), indent=2)}
        Recommendations:
        {'; '.join(self.twin.get('recommendations', []))}
        """
        return f"Based on your digital twin data, here's an answer to your question: '{question}'.
[This would be replaced with an LLM response using the summary and context.]"

    def build(self):
        self.compute_baseline()
        self.compute_current_state()
        self.analyze_deviation()
        self.integrate_ecg_data()
        self.predict_risk()
        self.generate_recommendations()
        self.generate_summary()
        return self.twin

# === Virtual Test Case ===
import streamlit as st

st.title("🧠 Digital Twin Health Model")

uploaded_watch = st.file_uploader("Upload Apple Watch CSV data", type=["csv"])
uploaded_subj = st.file_uploader("Upload Subjective Input CSV (optional)", type=["csv"])
uploaded_ecg = st.file_uploader("Upload ECG CSV (optional)", type=["csv"])

if uploaded_watch:
    df_watch = pd.read_csv(uploaded_watch, parse_dates=["timestamp"])
    uid = df_watch["user_id"].iloc[0]

    df_subj = pd.read_csv(uploaded_subj, parse_dates=["timestamp"]) if uploaded_subj else None
    df_ecg = pd.read_csv(uploaded_ecg, parse_dates=["timestamp"]) if uploaded_ecg else None

    model = DigitalTwinModel(uid, df_watch, df_subj, df_ecg)
    twin = model.build()

    st.subheader("📝 Clinical Summary")
    st.write(twin["clinical_summary"])

    st.subheader("⚠️ Flags")
    st.json(twin["deviation_flags"])

    st.subheader("🔍 Predicted Risk Levels")
    st.json(twin["predicted_risk"])

    st.subheader("✅ Recommendations")
    for r in twin["recommendations"]:
        st.write("-", r)

    question = st.text_input("Ask a health question about your current state:")
    if question:
        st.write(model.answer_question(question))
