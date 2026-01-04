"""
MODULE
    notebooks.NB03_data_science_churn_svm

BRIEF DESCRIPTION:
    Breve descripción de qué hace este archivo y cuál es su propósito dentro
    del proyecto de data science.

PROBLEM CONTEXT:
    Describe el problema de negocio o analítico que se está resolviendo.
    Por ejemplo: predicción, clasificación, limpieza de datos, análisis exploratorio, etc.

DATASET:
    - Fuente:
    - Descripción:
    - Variables principales:
    - Tamaño aproximado:

PROCESS:
    - Limpieza de datos
    - Feature engineering
    - Entrenamiento de modelo
    - Evaluación
    (ajusta según aplique)

ASSUMPTIONS:
    - Supuestos importantes sobre los datos o el modelo

DEPENDENCIES:
    ¬ pandas        2.3.3
        - numpy     2.4.0
    ¬ plotly        6.5.0
        ¬ narwhals  2.14.0
    ¬ nbformat  5.10.4

    
    ¬ matplotlib    3.10.8

    - scikit-learn
    - matplotlib
    (etc.)

USE:
    Ejemplo rápido de cómo se usa este módulo.

DATE - CHANGE - AUTHOR (NEWEST ON TOP):
    2026-01-04  Accuracy with SVM compared between kernels      Felix Armenta
"""
import pandas as pd
import plotly.express as px

"""
SECTION 1 - EXPLORATORY ANALYSIS
"""
datos = pd.read_csv("../datasets/netflix_customer_churn.csv")
datos.head(5)
datos.info()

# EXPLORING DATA
# Categorical variables
px.histogram(datos, x="gender", text_auto=True, color="churned", barmode="group")
px.histogram(datos, x="subscription_type", text_auto=True, color="churned", barmode="group")
px.histogram(datos, x="region", text_auto=True, color="churned", barmode="group")
px.histogram(datos, x="device", text_auto=True, color="churned", barmode="group")
px.histogram(datos, x="payment_method", text_auto=True, color="churned", barmode="group")
px.histogram(datos, x="favorite_genre", text_auto=True, color="churned", barmode="group")

# Numeric variables
px.box(datos, x="age", color="churned")
px.box(datos, x="watch_hours", color="churned")
px.box(datos, x="last_login_days", color="churned")
px.box(datos, x="monthly_fee", color="churned")
px.box(datos, x="number_of_profiles", color="churned")
px.box(datos, x="avg_watch_time_per_day", color="churned")





