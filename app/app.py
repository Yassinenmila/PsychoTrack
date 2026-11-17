
import streamlit as st
import pandas as pd
import joblib

# Titre
st.title("PsychoTrack")
st.write("Prédiction du risque de santé mentale des étudiants")

# Charger les données
df = pd.read_csv("data/data-students.csv")

# Menu
page = st.sidebar.selectbox(
    "Choisir une page",
    ["Accueil", "Exploration", "Prédiction"]
)

# Accueil
if page == "Accueil":

    st.header("Accueil")

    st.write("Bienvenue dans PsychoTrack !")

    st.write("Nombre d'étudiants :", len(df))

    st.write("Nombre de colonnes :", len(df.columns))

    st.write("Distribution des risques")

    st.bar_chart(df["Mental_Health_Risk"].value_counts())


# Exploration
elif page == "Exploration":

    st.header("Exploration des données")

    st.write("Aperçu des données")

    st.dataframe(df.head())

    st.write("Statistiques")

    st.write(df.describe().T)

    st.write("Valeurs manquantes")

    st.write(df.isnull().sum())


# Prédiction
elif page == "Prédiction":

    st.header("Prédiction")

    # Charger le modèle
    model = joblib.load("models/model/model.pkl")

    # Charger le préprocesseur
    preprocessor = joblib.load("models/data/preprocessor.pkl")

    st.write("Informations de l'étudiant")

    age = st.number_input("Age", 15, 100, 22)
    gender = st.selectbox("Genre", ["Male", "Female"])
    year = st.number_input("Année d'étude", 1, 10, 2)
    sleep = st.number_input("Heures de sommeil", 0.0, 24.0, 7.0)
    study = st.number_input("Heures d'étude", 0.0, 100.0, 20.0)
    gpa = st.number_input("GPA", 0.0, 4.0, 3.0)
    financial = st.number_input("Stress financier", 1, 10, 5)
    social = st.number_input("Réseaux sociaux", 0.0, 24.0, 3.0)
    screen = st.number_input("Temps d'écran", 0.0, 24.0, 6.0)
    family = st.number_input("Soutien familial", 1, 10, 5)
    attendance = st.number_input("Présence (%)", 0.0, 100.0, 80.0)

    physical = st.selectbox(
        "Activité physique",
        ["Low", "Moderate", "High"]
    )

    diet = st.selectbox(
        "Alimentation",
        ["Poor", "Average", "Good"]
    )

    job = st.selectbox("Travail", ["No", "Yes"])

    history = st.selectbox(
        "Antécédents de santé mentale",
        ["No", "Yes"]
    )

    # Bouton
    if st.button("Prédire"):

        # Préparer les données
        student = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study,
            "GPA": gpa,
            "Financial_Stress": financial,
            "Social_Media_Hours": social,
            "Screen_Time_Hours": screen,
            "Family_Support": family,
            "Attendance_Rate": attendance,
            "Physical_Activity": physical,
            "Diet_Quality": diet,
            "Part_Time_Job": job,
            "Previous_Mental_Health_History": history
        }])

        # Transformer les données
        student = preprocessor.transform(student)

        # Faire la prédiction
        result = model.predict(student)[0]

        # Afficher le résultat
        if result == 0:
            st.success("Risque faible")

        elif result == 1:
            st.warning("Risque moyen")

        else:
            st.error("Risque élevé")

        st.write("Résultat :", result)
