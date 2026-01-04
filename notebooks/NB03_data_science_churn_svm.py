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
# import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

"""
SECTION 1 - EXPLORATORY ANALYSIS
"""
datos = pd.read_csv("../datasets/netflix_customer_churn.csv")
datos.head(5)
datos.info()
datos.isnull().sum()
# datos = datos.replace(["", " ", "NA", "N/A", "NONE", "None",
#                        "none", "NULL", "Null" "null"], np.nan)
# datos.isnull().sum()
datos = datos.drop_duplicates()
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

"""
SECTION 2 - DATA TRANSFORMATION
"""
datos
X_w_drop = datos.drop(columns=["customer_id", "gender", 
                        "monthly_fee", "avg_watch_time_per_day", 
                        "churned"])
y_series = datos["churned"]

X_w_drop            # Pandas array
X_w_drop.info()
y_series            # Pandas series
y_series.info()

# TRANSFORMING THE EXPLANATORY VARIABLES
# One hot enconder for ategorical variables
columnas = X_w_drop.columns
one_hot_categorical = make_column_transformer((OneHotEncoder(drop="if_binary"), # Ingnor binary cols
                                               ["subscription_type",
                                                "region",
                                                "device",
                                                "payment_method",
                                                "favorite_genre"]),
                                              remainder="passthrough",          # Omitir columnas restantes
                                              sparse_threshold=0,               # No quitar información relevante
                                              force_int_remainder_cols=False)   # No cambiar el nombre de las cols

# Apply one hot encoder
X = one_hot_categorical.fit_transform(X_w_drop)
one_hot_categorical.get_feature_names_out(columnas)
X
# Visualize X as DF
X_one_hot_visual = pd.DataFrame(X, columns=one_hot_categorical.get_feature_names_out(columnas))
X_one_hot_visual
X_one_hot_visual.info()

# TRANSFORMING THE RESPONSE VARIABLE
y_series
y_series.info()
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y_series)
y

