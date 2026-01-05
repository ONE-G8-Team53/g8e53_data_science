"""
MODULE
    notebooks.NB03_data_science_churn_svm

BRIEF DESCRIPTION:
    This script compares different algorithms/models for churn 
    prediction.

PROBLEM CONTEXT:
    The current problem is identifying customers at risk of churn, 
    subscription services we decided to focused specifically on
    video streaming platforms.

DATASET:
    ¬ Source:       kaggle/netflix-customer-churn-dataset
    ¬ Main fields:  age, subscrition_tye, watch_hours, last_login_days
                    region, device, payment_method, number_of_profiles
                    favorite_genre
    ¬ Size:         5000 records
    ¬ Description:  This dataset contains synthetic data simulating
                    customer behavior for a netflix-like video streaming
                    service.

PROCESS:
    ¬ Data cleaning (not necessary for the dataset)
    ¬ Exploratory Analysis
    ¬ Data transformation
    ¬ Adjusting models
        ¬ Model training
        ¬ Model testing
    ¬ Model selection
    ¬ Models serialization

ASSUMPTIONS:
    ¬ The dataset field to drop were:
        FIELD                       REASON
        ------------------------------------------------------------------------------------------------
        customer_id                 Affects the model
        ------------------------------------------------------------------------------------------------
        gender                      Irrelevant
        ------------------------------------------------------------------------------------------------
        monthly_fee                 It has collinearity with the subscription_type
                                    Its values ​​between regions affect the weight for certain algorithms
                                    And it is not compared with a per capita income value
        ------------------------------------------------------------------------------------------------
        avg_watch_time_per_day      It has collinearity with watch_hours
        ------------------------------------------------------------------------------------------------
        
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
    Consult the README.md file to use the models

DATE - CHANGE - AUTHOR (NEWEST ON TOP):
    2026-01-04  Felix Armenta
                Accuracy with SVM's algorithms compared between kernels
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
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle


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
one_hot_categorical = make_column_transformer((OneHotEncoder(drop="if_binary"), # Ingnore binary cols
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
# 3.2.1 DECISION TREE DEPTH 4
tree_model_d4 = DecisionTreeClassifier(max_depth=4,
                                    random_state=4)
tree_model_d4.fit(x_train, y_train)
tree_model_d4.score(x_test, y_test)
plt.figure(figsize=(15, 6))
plot_tree(tree_model_d4, filled=True, class_names=["no", "yes"])

# 3.2.2 DECISION TREE DEPTH 5
tree_model_d5 = DecisionTreeClassifier(max_depth=5,
                                    random_state=4)
tree_model_d5.fit(x_train, y_train)
tree_model_d5.score(x_test, y_test)
plt.figure(figsize=(15, 6))
plot_tree(tree_model_d5, filled=True, class_names=["no", "yes"])


"""
SECTION 3.3 - SVM (Support Vector Machines)
"""
# Change y to 1d array
y_train
print(type(y_train))
y_train_1d_array = y_train.ravel()
y_train_1d_array
print(type(y_train_1d_array))
y_test_1d_array = y_test.ravel()
y_test_1d_array

# 3.3.1 SVC WITH KERNEL RADIAL BASIS FUNCTION
svc_rbf_model = svm.SVC(kernel="rbf", C=1.0, gamma="scale")
svc_rbf_model.fit(x_train, y_train_1d_array)
svc_rbf_model.score(x_test, y_test_1d_array)        # 0.8576
# Overfitting test
svc_rbf_model.score(x_train, y_train_1d_array)      # OK 0.874133

# 3.3.2 SVC WITH KERNEL LINEAR
svc_linear_model = svm.SVC(kernel="linear", C=1.0, gamma="scale")
svc_linear_model.fit(x_train, y_train_1d_array)
svc_linear_model.score(x_test, y_test_1d_array)     # 0.8768
# Overfitting test
svc_linear_model.score(x_train, y_train_1d_array)   # OK 0.893866

# 3.3.3 SVC WITH KERNEL POLYNOMIAL
svc_poly_model = svm.SVC(kernel="poly", C=1.0, gamma="scale")
svc_poly_model.fit(x_train, y_train_1d_array)
svc_poly_model.score(x_test, y_test_1d_array)       # 0.8472
# Overfitting test
svc_poly_model.score(x_train, y_train_1d_array)     # OK 0.855733 

# 3.3.4 SVC WITH KERNEL SIGMOID
svc_sigmoid_model = svm.SVC(kernel="sigmoid", C=1.0, gamma="scale")
svc_sigmoid_model.fit(x_train, y_train_1d_array)
svc_sigmoid_model.score(x_test, y_test_1d_array)       # 0.6992
# Overfitting test
svc_sigmoid_model.score(x_train, y_train_1d_array)     # 0.741866

# SVR
# STD SCALER (NORMALIZACIÓN)
std_scaler = StandardScaler()
x_train_std_normalized = std_scaler.fit_transform(x_train)
x_test_std_normalized = std_scaler.transform(x_test)
pd.DataFrame(x_train_std_normalized)

# 3.3.5 SVR WITH KERNEL RADIAL BASIS FUNCTION
svr_rbf_model = svm.SVR(kernel="rbf", C=1.0, gamma="scale")
svr_rbf_model.fit(x_train_std_normalized, y_train_1d_array)
svr_rbf_model.score(x_test_std_normalized, y_test_1d_array)     # 0.557856
# Overfitting test
svr_rbf_model.score(x_train_std_normalized, y_train_1d_array)   # 0.814929

# 3.3.6 SVR WITH KERNEL LINEAR
svr_linear_model = svm.SVR(kernel="linear", C=1.0, gamma="scale")
svr_linear_model.fit(x_train_std_normalized, y_train_1d_array)
svr_linear_model.score(x_test_std_normalized, y_test_1d_array)     # 0.476840
# Overfitting test
svr_linear_model.score(x_train_std_normalized, y_train_1d_array)   # 0.514136

# 3.3.7 SVR WITH KERNEL POLYNOMIAL
svr_poly_model = svm.SVR(kernel="poly", C=1.0, gamma="scale")
svr_poly_model.fit(x_train_std_normalized, y_train_1d_array)
svr_poly_model.score(x_test_std_normalized, y_test_1d_array)     # 0.239075
# Overfitting test
svr_poly_model.score(x_train_std_normalized, y_train_1d_array)   # 0.773447

# 3.3.8 SVR WITH KERNEL SIGMOID
svr_sigmoid_model = svm.SVR(kernel="sigmoid", C=1.0, gamma="scale")
svr_sigmoid_model.fit(x_train_std_normalized, y_train_1d_array)
svr_sigmoid_model.score(x_test_std_normalized, y_test_1d_array)     # -12.689629
# Overfitting test
svr_sigmoid_model.score(x_train_std_normalized, y_train_1d_array)   # -15.327026

"""
SECTION 3.4 - KNN (K-Nearest Neighbors)
"""
# MIN MAX SCALER (NORMALIZATION)
min_max_scaler = MinMaxScaler()
# Train data
x_train_min_max_normalized = min_max_scaler.fit_transform(x_train)
# Test data
x_test_min_max_normalized = min_max_scaler.transform(x_test)
pd.DataFrame(x_train_min_max_normalized)

# 3.4.1 KNN Model with min max scaler
knn_min_max_model = KNeighborsClassifier()
knn_min_max_model.fit(x_train_min_max_normalized, y_train_1d_array)
knn_min_max_model.score(x_test_min_max_normalized, y_test_1d_array)   # 0.6816

# 3.4.2 KNN Model with std scaler
knn_std_model = KNeighborsClassifier()
knn_std_model.fit(x_train_std_normalized, y_train_1d_array)
knn_std_model.score(x_test_std_normalized, y_test_1d_array)           # 0.7176


"""
SECTION 4 - MODEL SELECTION
"""
model_list = [("Baseline", dummy_model, x_test, y_test, x_train, y_train),
              ("DTree D4", tree_model_d4, x_test, y_test, x_train, y_train),
              ("DTree D5", tree_model_d5, x_test, y_test, x_train, y_train),
              ("SVC linear", svc_linear_model, x_test, y_test_1d_array, x_train, y_train_1d_array),
              ("SVC RBF", svc_rbf_model, x_test, y_test_1d_array, x_train, y_train_1d_array),
              ("SVC poly", svc_poly_model, x_test, y_test_1d_array, x_train, y_train_1d_array),
              ("SVC sigmoid", svc_sigmoid_model, x_test, y_test_1d_array, x_train, y_train_1d_array),
              ("SVR linear", svr_linear_model, x_test_std_normalized, y_test_1d_array, x_train_std_normalized, y_train_1d_array),
              ("SVR RBF", svr_rbf_model, x_test_std_normalized, y_test_1d_array, x_train_std_normalized, y_train_1d_array),
              ("SVR poly", svr_poly_model, x_test_std_normalized, y_test_1d_array, x_train_std_normalized, y_train_1d_array),
              ("SVR sigmoid", svr_sigmoid_model, x_test_std_normalized, y_test_1d_array, x_train_std_normalized, y_train_1d_array),
              ("KNN MinMax", knn_min_max_model, x_test_min_max_normalized, y_test_1d_array, x_train_min_max_normalized, y_train_1d_array),
              ("KNN Std", knn_std_model, x_test_std_normalized, y_test_1d_array, x_train_std_normalized, y_train_1d_array)]

print("\t\tMODEL RESULTS")
print("model\t\tts_score\ttr_score")
for i in model_list:
    if i[0] in ("SVC RBF", "SVR RBF", "KNN Std"):
        print(f"{i[0]}\t\t{round(i[1].score(i[2], i[3]), 4)}\t\t{round(i[1].score(i[4], i[5]), 4)}")
    elif i[0] == "SVR sigmoid":
        print(f"{i[0]}\t{round(i[1].score(i[2], i[3]), 4)}\t{round(i[1].score(i[4], i[5]), 4)}")
    else:    
        print(f"{i[0]}\t{round(i[1].score(i[2], i[3]), 4)}\t\t{round(i[1].score(i[4], i[5]), 4)}")

"""
SECTION 5 - MODELS SERIALIZATION
"""
with open("../models/faa_one_hot_encoder_model.pkl", "wb") as arc:
    pickle.dump(one_hot_categorical, arc)

with open("../models/faa_d_tree_dep4_model.pkl", "wb") as arc:
    pickle.dump(tree_model_d4, arc)

with open("../models/faa_d_tree_dep5_model.pkl", "wb") as arc:
    pickle.dump(tree_model_d5, arc)

with open("../models/faa_svc_linear_model.pkl", "wb") as arc:
    pickle.dump(svc_linear_model, arc)
