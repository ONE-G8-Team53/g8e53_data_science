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
    ¬ pandas            2.3.3
        - numpy         2.4.0
    ¬ plotly            6.5.0
        ¬ narwhals      2.14.0
    ¬ nbformat          5.10.4
    ¬ scikit-learn      1.8.0
        ¬ joblib        1.5.3
        ¬ scipy         1.16.3
        ¬ threadpoolctl 3.6.0
    ¬ matplotlib        3.10.8

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
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

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
x_w_drop = datos.drop(columns=["customer_id", "gender", 
                        "monthly_fee", "avg_watch_time_per_day", 
                        "churned"])
y_series = datos["churned"]

x_w_drop            # Pandas array
x_w_drop.info()
y_series            # Pandas series
y_series.info()

# TRANSFORMING THE EXPLANATORY VARIABLES
# One hot enconder for ategorical variables
columnas = x_w_drop.columns
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
x = one_hot_categorical.fit_transform(x_w_drop)
one_hot_categorical.get_feature_names_out(columnas)
x
# Visualize X as DF
x_one_hot_visual = pd.DataFrame(x, columns=one_hot_categorical.get_feature_names_out(columnas))
x_one_hot_visual
x_one_hot_visual.info()

# TRANSFORMING THE RESPONSE VARIABLE
y_series
y_series.info()
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y_series)
y


"""
SECTION 3 - ADJUSTING MODELS
"""
# Dividing dataset between training and test
# Default test_size=0.25
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=4,
                                                    stratify=y)
"""
SECTION 3.1 - BASELINE
"""
# BASELINE MODEL (DUMMY)
dummy_model = DummyClassifier(random_state=4)
dummy_model.fit(x_train, y_train)
dummy_model.score(x_test, y_test)   # Accuracy 0.5032

"""
SECTION 3.2 - DECISION TREE
"""
# DECISION TREE MODEL
tree_model_overfitted = DecisionTreeClassifier(random_state=4)
tree_model_overfitted.fit(x_train, y_train)
tree_model_overfitted.score(x_test, y_test) # Accuracy 0.9728

valores_columnas = ["Basic", "Standard", "Premium",
                    "Africa", "Europe", "Asia", "Oceania", "South America", "North America",
                    "TV", "Mobile", "Tablet", "Laptop", "Desktop",
                    "Gift Card", "Crypto", "PayPal", "Debit Card", "Credit Card",
                    "Action", "Sci-Fi", "Drama", "Horror", "Romance", "Comedy", "Documentary",
                    "age",
                    "watch_hours",
                    "last_login_days",
                    "number_of_profiles"]
plt.figure(figsize=(80, 25))
plot_tree(tree_model_overfitted, filled=True,
          class_names=["no", "yes"],
          feature_names=valores_columnas)
# Overfitting test
tree_model_overfitted.score(x_train, y_train)

# RIGHT DECISION TREE MODEL
# for i in range(2, 13):
#     tree_model = DecisionTreeClassifier(max_depth=i,
#                                     random_state=4)
#     tree_model.fit(x_train, y_train)
#     tree_model.score(x_test, y_test)
#     # Overfitting test
#     tree_model.score(x_train, y_train)
#     print(f"max_depth:{i},\
#           score: {tree_model.score(x_test, y_test)},\
#           overfitted: {tree_model.score(x_train, y_train)}")
"""
    max_depth   random_state    accuracy    overfitted
    2           4               0.816       0.829333
    3           4               0.8936      0.910133
    4           4               0.8944      0.911733
    5           4               0.9256      0.941333
    6           4               0.9496      0.9688
    7           4               0.9704      0.992266
    8           4               0.9728      0.9952
    9           4               0.9736      0.9992
    10          4               0.972       0.999733
    11          4               0.9728      1.0
    12          4               0.9728      1.0
    None        4               0.9728      1.0 
"""
tree_model = DecisionTreeClassifier(max_depth=5,
                                    random_state=4)
tree_model.fit(x_train, y_train)
tree_model.score(x_test, y_test)
plt.figure(figsize=(15, 6))
plot_tree(tree_model, filled=True, class_names=["no", "yes"])

"""
SECTION 3.3 - KNN
"""
# MIN MAX SCALER (NORMALIZACIÓN)
min_max_scaler = MinMaxScaler()
# Train data
x_train_min_max_normalized = min_max_scaler.fit_transform(x_train)
# Test data
x_test_min_max_normalized = min_max_scaler.transform(x_test)
pd.DataFrame(x_train_min_max_normalized)

knn = KNeighborsClassifier()
knn.fit(x_train_min_max_normalized, y_train)
knn.score(x_test_min_max_normalized, y_test)    # 0.6816

"""
SECTION 3.4 - SVM
"""

