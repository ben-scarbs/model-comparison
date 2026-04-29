import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as imb_Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.feature_selection import mutual_info_classif
import shap


file = Path(__file__).with_name("framingham.csv")
df = pd.read_csv(file)

x = df.drop(columns="TenYearCHD")
y = df["TenYearCHD"]

# Calculate class balance
def class_bal(labels):
    negative_class_num = (labels == 0).sum()
    print(f"Negative class balacne: {negative_class_num / labels.size:.4f}")

# cigsPerDay=0 when cuurentSmoker=0
x.loc[x['currentSmoker'] == 0, 'cigsPerDay'] = 0

# missings indicator feature
missings_feature = False
if missings_feature:
    x["Nissings"] = x.isna().sum(axis=1)

#----Dataseet analysis----
# ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
# contains missing: ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate','glucose']
nominal_features = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
nominal_features_idx = [x.columns.get_loc(col) for col in nominal_features]
ordinal_features = ["age", "education", "cigsPerDay"]
continuous_features = ["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

num_rows = df.shape[0]
nan_rows=df.isna().any(axis=1).sum()
#print(nan_rows/num_rows)
#print(df.isna().sum())
#print(df["education"].describe())
#print(df.columns)
#print(len(df.columns))
#print(df.loc[:,df.isna().any()].columns)
#print(y.head)


# train test split
x_tval, x_test, y_tval, y_test = train_test_split(x, y, test_size=0.15, random_state=0, stratify=y, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_tval, y_tval, test_size=0.1, random_state=0, stratify=y_tval)

class_imbalance_weight = (y_train == 0).sum() / (y_train == 1).sum()    # XGBoost class imbalance weighting uaed in tuning

#----Data Preprocessing----
nominal_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent"))
])

ordinal_continuous_pipeline = Pipeline([
    ("impute", IterativeImputer(max_iter=5, initial_strategy="median", random_state=0)),
    ("scale", StandardScaler())
])

preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ("nominal", nominal_pipeline, nominal_features),
        ("ordinal_continuous", ordinal_continuous_pipeline, ordinal_features+continuous_features),
    ]
)

preprocessing_pipeline.fit(x_train, y_train)
x_processed = preprocessing_pipeline.transform(x)
y_processed = y
x_train_processed = preprocessing_pipeline.transform(x_train)
y_train_processed = y_train
x_val_processed = preprocessing_pipeline.transform(x_val)
y_val_processed = y_val
x_tval_processed = preprocessing_pipeline.transform(x_tval)
y_tval_processed = y_tval
x_test_processed = preprocessing_pipeline.transform(x_test)
y_test_processed = y_test

XGB_impute = ColumnTransformer(
    transformers=[
        ("nominal", SimpleImputer(strategy="most_frequent"), nominal_features),
        ("ordinal", IterativeImputer(max_iter=5, initial_strategy="median", random_state=0), ordinal_features+continuous_features)
    ]
)

# Data Modes:
data_mode = "smo"
#   - "raw" = No augmentation/synthesis
#   - "smo" = Data synthesis using interpolated oversampling (SMOTE)
#   - "syn" = Advanced data synthesis (Copula moddeling or tVAE using SDV)

if data_mode == "raw":
    x_train_oversampled, y_train_oversampled = x_train_processed, y_train_processed
    x_tval_oversampled, y_tval_oversampled = x_tval_processed, y_tval_processed
    x_train_oversampled_XGB, y_train_oversampled_XGB = x_train, y_train
    x_tval_oversampled_XGB, y_tval_oversampled_XGB = x_tval, y_tval

elif data_mode == "smo":
    #--SMOTE data synthesis--
    smote = SMOTENC(categorical_features= nominal_features_idx, sampling_strategy="minority")

    x_train_oversampled, y_train_oversampled = smote.fit_resample(x_train_processed, y_train)
    x_tval_oversampled, y_tval_oversampled = smote.fit_resample(x_tval_processed, y_tval)

    x_train_imp_XGB = XGB_impute.fit_transform(x_train, y_train)
    x_train_imp_oversampled_XGB, y_train_imp_oversampled_XGB = smote.fit_resample(x_train_imp_XGB, y_train)
    x_tval_imp_XGB = XGB_impute.fit_transform(x_tval, y_tval)
    x_tval_imp_oversampled_XGB, y_tval_imp_oversampled_XGB = smote.fit_resample(x_tval_imp_XGB, y_tval)

    XGB_x_train_synthetic = pd.DataFrame(
        x_train_imp_oversampled_XGB[len(x_train):],
        columns=x_train.columns
    )
    XGB_y_train_synthetic = pd.Series(
        y_train_imp_oversampled_XGB[len(y_train):],
        name=y_train.name
    )
    x_train_oversampled_XGB = pd.concat([x_train, XGB_x_train_synthetic], ignore_index=True)
    y_train_oversampled_XGB = pd.concat([y_train, XGB_y_train_synthetic], ignore_index=True)

    XGB_x_tval_synthetic = pd.DataFrame(
        x_tval_imp_oversampled_XGB[len(x_tval):],
        columns=x_tval.columns
    )
    XGB_y_tval_synthetic = pd.Series(
        y_tval_imp_oversampled_XGB[len(y_tval):],
        name=y_tval.name
    )
    x_tval_oversampled_XGB = pd.concat([x_tval, XGB_x_tval_synthetic], ignore_index=True)
    y_tval_oversampled_XGB = pd.concat([y_tval, XGB_y_tval_synthetic], ignore_index=True)


#----Logisitc Regression----
#--Model Hyperparameters--
LR_params = {
    #LR Val: Accuracy = 0.8532 | F1 = 0.0702 | ROC AUC = 0.7843
    "raw": {
        "C": 0.04497912998619529,
        "l1_ratio": 0.06363016933620981,
        "class_weight": "balanced"
    },
    #LR Val:...
    "smo": {
        "C": 1,
        "l1_ratio": 0.0,
        "class_weight": None
    },
    #LR Val:...
    "syn": {
        "C": 1,
        "l1_ratio": 0.0,
        "class_weight": None
    }
}

# base (untuned) model
LR_base_model_pipeline = Pipeline([
        ("model", LogisticRegression())
    ])

# tuned model
LR_tuned_model_pipeline = Pipeline([
    ('model', LogisticRegression(
        penalty= "elasticnet",
        solver= "saga",
        max_iter= 5000,
        random_state= 0,
        n_jobs= -1,
        # tuned params:
        C= LR_params[data_mode]["C"],
        l1_ratio= LR_params[data_mode]["l1_ratio"],
        class_weight= LR_params[data_mode]["class_weight"]
    ))
])

# Hyperparameter Tuning
# LR_hp_tuning = True to run hyperparameter tuning
LR_hp_tuning = False

if LR_hp_tuning:
    # Logistic Regression Tuning
    LR_param_space = {
        "model__C": Real(1e-4, 1, prior="log-uniform"),
        "model__l1_ratio": Real(0, 1),
        "model__class_weight": Categorical([None, "balanced"])
    }

    LR_search = BayesSearchCV(
        estimator=LR_tuned_model_pipeline,
        search_spaces=LR_param_space,
        n_iter=150,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
        random_state=0,
        verbose=1,
        refit=True
    )

    # model comparison
    LR_base_model = LR_base_model_pipeline.fit(x_train_oversampled, y_train_oversampled)
    LR_base_score = f1_score(y_val_processed, LR_base_model.predict(x_val_processed))

    LR_pretuned_model = LR_tuned_model_pipeline.fit(x_train_oversampled, y_train_oversampled)
    LR_pretuned_score = f1_score(y_val_processed, LR_pretuned_model.predict(x_val_processed))

    LR_search.fit(x_train_oversampled, y_train_oversampled)
    LR_tuned_model = LR_search.best_estimator_
    LR_tuned_score = f1_score(y_val_processed, LR_tuned_model.predict(x_val_processed))

    print("Logistic Regression:")
    print(f"Best score: {LR_search.best_score_}")
    print(f"F1 | Base: {LR_base_score}; Untuned:{LR_pretuned_score}; Tuned: {LR_tuned_score}")
    print(LR_search.best_params_)


#----XGBoost----
# no preprocessing required:
# no scaling necessary for trees
# xgboost handles missing/nan

#--Model Hyperparameters--
XGB_params = {
    #XGB Val: Accuracy = 0.7036 | F1 = 0.4398 | ROC AUC = 0.7607
    "raw": {
        "n_estimators": 475,
        "learning_rate": 0.007948263801032772,
        "max_depth": 2,
        "min_child_weight": 6,
        "subsample": 0.6767872227893332,
        "colsample_bytree": 0.6,
        "gamma": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": class_imbalance_weight
    },
    #XGB Val:...
    "smo": {
        "n_estimators": 475,
        "learning_rate": 0.007948263801032772,
        "max_depth": 2,
        "min_child_weight": 6,
        "subsample": 0.6767872227893332,
        "colsample_bytree": 0.6,
        "gamma": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": None
    },
    #XGb Val:...
    "syn": {
        "n_estimators": 475,
        "learning_rate": 0.007948263801032772,
        "max_depth": 2,
        "min_child_weight": 6,
        "subsample": 0.6767872227893332,
        "colsample_bytree": 0.6,
        "gamma": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": None
    }
}

# base (untuned) model
XGB_base_model_pipeline = XGBClassifier()

# tuned model
XGB_tuned_model_pipeline = Pipeline(steps=[
    ("model", XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=0,
        n_jobs=-1,
        #tuned params:
        n_estimators= XGB_params[data_mode]["n_estimators"],
        learning_rate= XGB_params[data_mode]["learning_rate"],
        max_depth= XGB_params[data_mode]["max_depth"],
        min_child_weight= XGB_params[data_mode]["min_child_weight"],
        subsample= XGB_params[data_mode]["subsample"],
        colsample_bytree= XGB_params[data_mode]["colsample_bytree"],
        gamma= XGB_params[data_mode]["gamma"],
        reg_alpha= XGB_params[data_mode]["reg_alpha"],
        reg_lambda= XGB_params[data_mode]["reg_lambda"],
        scale_pos_weight= XGB_params[data_mode]["scale_pos_weight"]
    ))
])

# Hyperparameter Tuning
# XGB_hp_tuning = True to run hyperparameter tuning
XGB_hp_tuning = False

if XGB_hp_tuning:
    # XGBoost Tuning
    XGB_param_space = {
        "model__n_estimators": Integer(300, 600),
        "model__learning_rate": Real(0.005, 0.1, prior="log-uniform"),
        "model__max_depth": Integer(2, 6),
        "model__min_child_weight": Integer(2, 6),
        "model__subsample": Real(0.6, 1.0),
        "model__colsample_bytree": Real(0.6, 1.0),
        "model__gamma": Real(0.0, 1.0),
        "model__reg_alpha": Real(0.0, 1.0),
        "model__reg_lambda": Real(0.0, 5.0),
        "model__scale_pos_weight": Real(1.0, class_imbalance_weight)
    }

    XGB_search = BayesSearchCV(
        estimator=XGB_tuned_model_pipeline,
        search_spaces=XGB_param_space,
        n_iter=200,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
        random_state=0,
        verbose=1,
        refit=True
    )

    # model comparison
    XGB_base_model = XGB_base_model_pipeline.fit(x_train_oversampled_XGB, y_train_oversampled_XGB)
    XGB_base_score = f1_score(y_val, XGB_base_model.predict(x_val))

    XGB_pretuned_model = XGB_tuned_model_pipeline.fit(x_train_oversampled_XGB, y_train_oversampled_XGB)
    XGB_pretuned_score = f1_score(y_val, XGB_pretuned_model.predict(x_val))

    XGB_search.fit(x_train_oversampled_XGB, y_train_oversampled_XGB)
    XGB_tuned_model = XGB_search.best_estimator_
    XGB_tuned_score = f1_score(y_val, XGB_tuned_model.predict(x_val))

    print("XGBoost:")
    print(f"Best score: {XGB_search.best_score_}")
    print(f"F1 | Base: {XGB_base_score}; Untuned:{XGB_pretuned_score}; Tuned: {XGB_tuned_score}")
    print(XGB_search.best_params_)

#----Support Vevtor Machine (SVM)----
#--Model Hyperparameters--
SVM_params = {
    #SVM Val: Accuracy = 0.8504 | F1 = 0.0690 | ROC AUC = 0.7871
    "raw": {
        "C": 8470189.161873985,
        "gamma": 1.0494718319326697e-05,
        "class_weight": None
    },
    #SVM Val:...
    "smo": {
        "C": 8470189.161873985,
        "gamma": 1.0494718319326697e-05,
        "class_weight": None
    },
    #SVM Val:...
    "syn": {
        "C": 8470189.161873985,
        "gamma": 1.0494718319326697e-05,
        "class_weight": None
    }
}

# base (untuned) model
SVM_base_model_pipeline = Pipeline([
        ("model", SVC(kernel= "rbf"))
    ])

# tuned model
SVM_tuned_model_pipeline = Pipeline([
    ('model', SVC(
        kernel = "rbf",
        C = SVM_params[data_mode]["C"],
        gamma = SVM_params[data_mode]["gamma"],
        class_weight = SVM_params[data_mode]["class_weight"]
    ))
])

# Hyperparameter Tuning
# SVM_hp_tuning = True to run hyperparameter tuning
SVM_hp_tuning = False

if SVM_hp_tuning:
    # SVM Hyperparameter Tuning
    SVM_param_space = {
        "model__C": Real(1, 1e9, prior="log-uniform"),
        "model__gamma": Real(1e-9, 1, prior="log-uniform"),
        "model__class_weight": Categorical([None, "balanced"])
    }

    SVM_search = BayesSearchCV(
        estimator=SVM_tuned_model_pipeline,
        search_spaces=SVM_param_space,
        n_iter=150,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
        random_state=0,
        verbose=1,
        refit=True
    )

    # model comparison
    SVM_base_model = SVM_base_model_pipeline.fit(x_train_oversampled, y_train_oversampled)
    SVM_base_score = f1_score(y_val_processed, SVM_base_model.predict(x_val_processed))

    SVM_pretuned_model = SVM_tuned_model_pipeline.fit(x_train_oversampled, y_train_oversampled)
    SVM_pretuned_score = f1_score(y_val_processed, SVM_pretuned_model.predict(x_val_processed))

    SVM_search.fit(x_train_oversampled, y_train_oversampled)
    SVM_tuned_model = SVM_search.best_estimator_
    SVM_tuned_score = f1_score(y_val_processed, SVM_tuned_model.predict(x_val_processed))

    print("SVM:")
    print(f"Best score: {SVM_search.best_score_}")
    print(f"F1 | Base: {SVM_base_score}; Untuned:{SVM_pretuned_score}; Tuned: {SVM_tuned_score}")
    print(SVM_search.best_params_)

#----Model evaluation----
model_eval = False
if model_eval:
    LR_val_model = LR_tuned_model_pipeline.fit(x_train_oversampled, y_train_oversampled)
    LR_train_accuracy = accuracy_score(y_train_processed, LR_val_model.predict(x_train_processed))
    LR_train_balanced_accuracy = balanced_accuracy_score(y_train_processed, LR_val_model.predict(x_train_processed))
    LR_train_f1 = f1_score(y_train_processed, LR_val_model.predict(x_train_processed))
    LR_train_ROC = roc_auc_score(y_train_processed, LR_val_model.predict_proba(x_train_processed)[:,1])
    LR_train_oversampled_accuracy = accuracy_score(y_train_oversampled, LR_val_model.predict(x_train_oversampled))
    LR_train_oversampled_balanced_accuracy = balanced_accuracy_score(y_train_oversampled, LR_val_model.predict(x_train_oversampled))
    LR_train_oversampled_f1 = f1_score(y_train_oversampled, LR_val_model.predict(x_train_oversampled))
    LR_train_oversampled_ROC = roc_auc_score(y_train_oversampled, LR_val_model.predict_proba(x_train_oversampled)[:,1])
    LR_val_accuracy = accuracy_score(y_val_processed, LR_val_model.predict(x_val_processed))
    LR_val_balanced_accuracy = balanced_accuracy_score(y_val_processed, LR_val_model.predict(x_val_processed))
    LR_val_f1 = f1_score(y_val_processed, LR_val_model.predict(x_val_processed))
    LR_val_ROC = roc_auc_score(y_val_processed, LR_val_model.predict_proba(x_val_processed)[:,1])
    print(f"LR Train: Accuracy = {LR_train_accuracy:.4f} | Balanced = {LR_train_balanced_accuracy:.4f} | F1 = {LR_train_f1:.4f} | ROC AUC = {LR_train_ROC:.4f}")
    print(f"LR Oversampled Train: Accuracy = {LR_train_oversampled_accuracy:.4f} | Balanced = {LR_train_oversampled_balanced_accuracy:.4f} | F1 = {LR_train_oversampled_f1:.4f} | ROC AUC = {LR_train_oversampled_ROC:.4f}")
    print(f"LR Val: Accuracy = {LR_val_accuracy:.4f} | Balanced = {LR_val_balanced_accuracy:.4f} | F1 = {LR_val_f1:.4f} | ROC AUC = {LR_val_ROC:.4f}")

    XGB_val_model = XGB_tuned_model_pipeline.fit(x_train_oversampled_XGB, y_train_oversampled_XGB)
    XGB_train_accuracy = accuracy_score(y_train, XGB_val_model.predict(x_train))
    XGB_train_balanced_accuracy = balanced_accuracy_score(y_train, XGB_val_model.predict(x_train))
    XGB_train_f1 = f1_score(y_train, XGB_val_model.predict(x_train))
    XGB_train_ROC = roc_auc_score(y_train, XGB_val_model.predict_proba(x_train)[:,1])
    XGB_train_oversampled_accuracy = accuracy_score(y_train_oversampled_XGB, XGB_val_model.predict(x_train_oversampled_XGB))
    XGB_train_oversampled_balanced_accuracy = balanced_accuracy_score(y_train_oversampled_XGB, XGB_val_model.predict(x_train_oversampled_XGB))
    XGB_train_oversampled_f1 = f1_score(y_train_oversampled_XGB, XGB_val_model.predict(x_train_oversampled_XGB))
    XGB_train_oversampled_ROC = roc_auc_score(y_train_oversampled_XGB, XGB_val_model.predict_proba(x_train_oversampled_XGB)[:,1])
    XGB_val_accuracy = accuracy_score(y_val, XGB_val_model.predict(x_val))
    XGB_val_balanced_accuracy = balanced_accuracy_score(y_val, XGB_val_model.predict(x_val))
    XGB_val_f1 = f1_score(y_val, XGB_val_model.predict(x_val))
    XGB_val_ROC = roc_auc_score(y_val, XGB_val_model.predict_proba(x_val)[:,1])
    print(f"XGB Train: Accuracy = {XGB_train_accuracy:.4f} | Balanced = {XGB_train_balanced_accuracy:.4f} | F1 = {XGB_train_f1:.4f} | ROC AUC = {XGB_train_ROC:.4f}")
    print(f"XGB Oversampled Train: Accuracy = {XGB_train_oversampled_accuracy:.4f} | Balanced = {XGB_train_oversampled_balanced_accuracy:.4f} | F1 = {XGB_train_oversampled_f1:.4f} | ROC AUC = {XGB_train_oversampled_ROC:.4f}")
    print(f"XGB Val: Accuracy = {XGB_val_accuracy:.4f} | Balanced = {XGB_val_balanced_accuracy:.4f} | F1 = {XGB_val_f1:.4f} | ROC AUC = {XGB_val_ROC:.4f}")

    SVM_val_model = SVM_tuned_model_pipeline.fit(x_train_oversampled, y_train_oversampled)
    SVM_train_accuracy = accuracy_score(y_train_processed, SVM_val_model.predict(x_train_processed))
    SVM_train_balanced_accuracy = balanced_accuracy_score(y_train_processed, SVM_val_model.predict(x_train_processed))
    SVM_train_f1 = f1_score(y_train_processed, SVM_val_model.predict(x_train_processed))
    SVM_train_ROC = roc_auc_score(y_train_processed, SVM_val_model.decision_function(x_train_processed))
    SVM_train_oversampled_accuracy = accuracy_score(y_train_oversampled, SVM_val_model.predict(x_train_oversampled))
    SVM_train_oversampled_balanced_accuracy = balanced_accuracy_score(y_train_oversampled, SVM_val_model.predict(x_train_oversampled))
    SVM_train_oversampled_f1 = f1_score(y_train_oversampled, SVM_val_model.predict(x_train_oversampled))
    SVM_train_oversampled_ROC = roc_auc_score(y_train_oversampled, SVM_val_model.decision_function(x_train_oversampled))
    SVM_val_accuracy = accuracy_score(y_val_processed, SVM_val_model.predict(x_val_processed))
    SVM_val_balanced_accuracy = balanced_accuracy_score(y_val_processed, SVM_val_model.predict(x_val_processed))
    SVM_val_f1 = f1_score(y_val_processed, SVM_val_model.predict(x_val_processed))
    SVM_val_ROC = roc_auc_score(y_val_processed, SVM_val_model.decision_function(x_val_processed))
    print(f"SVM Train: Accuracy = {SVM_train_accuracy:.4f} | Balanced = {SVM_train_balanced_accuracy:.4f} | F1 = {SVM_train_f1:.4f} | ROC AUC = {SVM_train_ROC:.4f}")
    print(f"SVM Oversampled Train: Accuracy = {SVM_train_oversampled_accuracy:.4f} | Balanced = {SVM_train_oversampled_balanced_accuracy:.4f} | F1 = {SVM_train_oversampled_f1:.4f} | ROC AUC = {SVM_train_oversampled_ROC:.4f}")
    print(f"SVM Val: Accuracy = {SVM_val_accuracy:.4f} | Balanced = {SVM_val_balanced_accuracy:.4f} | F1 = {SVM_val_f1:.4f} | ROC AUC = {SVM_val_ROC:.4f}")


#----Model Testing----
model_test = False
if model_test:
    LR_test_model = LR_tuned_model_pipeline.fit(x_tval_oversampled, y_tval_oversampled)
    LR_tval_accuracy = accuracy_score(y_tval_processed, LR_test_model.predict(x_tval_processed))
    LR_tval_balanced_accuracy = balanced_accuracy_score(y_tval_processed, LR_test_model.predict(x_tval_processed))
    LR_tval_f1 = f1_score(y_tval_processed, LR_test_model.predict(x_tval_processed))
    LR_tval_ROC = roc_auc_score(y_tval_processed, LR_test_model.predict_proba(x_tval_processed)[:,1])
    LR_tval_oversampled_accuracy = accuracy_score(y_tval_oversampled, LR_test_model.predict(x_tval_oversampled))
    LR_tval_oversampled_balanced_accuracy = balanced_accuracy_score(y_tval_oversampled, LR_test_model.predict(x_tval_oversampled))
    LR_tval_oversampled_f1 = f1_score(y_tval_oversampled, LR_test_model.predict(x_tval_oversampled))
    LR_tval_oversampled_ROC = roc_auc_score(y_tval_oversampled, LR_test_model.predict_proba(x_tval_oversampled)[:,1])
    LR_test_accuracy = accuracy_score(y_test_processed, LR_test_model.predict(x_test_processed))
    LR_test_balanced_accuracy = balanced_accuracy_score(y_test_processed, LR_test_model.predict(x_test_processed))
    LR_test_f1 = f1_score(y_test_processed, LR_test_model.predict(x_test_processed))
    LR_test_ROC = roc_auc_score(y_test_processed, LR_test_model.predict_proba(x_test_processed)[:,1])
    print(f"LR Tval: Accuracy = {LR_tval_accuracy:.4f} | Balanced = {LR_tval_balanced_accuracy:.4f} | F1 = {LR_tval_f1:.4f} | ROC AUC = {LR_tval_ROC:.4f}")
    print(f"LR Oversampled Tval: Accuracy = {LR_tval_oversampled_accuracy:.4f} | Balanced = {LR_tval_oversampled_balanced_accuracy:.4f} | F1 = {LR_tval_oversampled_f1:.4f} | ROC AUC = {LR_tval_oversampled_ROC:.4f}")
    print(f"LR Test: Accuracy = {LR_test_accuracy:.4f} | Balanced = {LR_test_balanced_accuracy:.4f} | F1 = {LR_test_f1:.4f} | ROC AUC = {LR_test_ROC:.4f}")

    XGB_test_model = XGB_tuned_model_pipeline.fit(x_tval_oversampled_XGB, y_tval_oversampled_XGB)
    XGB_tval_accuracy = accuracy_score(y_tval, XGB_test_model.predict(x_tval))
    XGB_tval_balanced_accuracy = balanced_accuracy_score(y_tval, XGB_test_model.predict(x_tval))
    XGB_tval_f1 = f1_score(y_tval, XGB_test_model.predict(x_tval))
    XGB_tval_ROC = roc_auc_score(y_tval, XGB_test_model.predict_proba(x_tval)[:,1])
    XGB_tval_oversampled_accuracy = accuracy_score(y_tval_oversampled_XGB, XGB_test_model.predict(x_tval_oversampled_XGB))
    XGB_tval_oversampled_balanced_accuracy = balanced_accuracy_score(y_tval_oversampled_XGB, XGB_test_model.predict(x_tval_oversampled_XGB))
    XGB_tval_oversampled_f1 = f1_score(y_tval_oversampled_XGB, XGB_test_model.predict(x_tval_oversampled_XGB))
    XGB_tval_oversampled_ROC = roc_auc_score(y_tval_oversampled_XGB, XGB_test_model.predict_proba(x_tval_oversampled_XGB)[:,1])
    XGB_test_accuracy = accuracy_score(y_test, XGB_test_model.predict(x_test))
    XGB_test_balanced_accuracy = balanced_accuracy_score(y_test, XGB_test_model.predict(x_test))
    XGB_test_f1 = f1_score(y_test, XGB_test_model.predict(x_test))
    XGB_test_ROC = roc_auc_score(y_test, XGB_test_model.predict_proba(x_test)[:,1])
    print(f"XGB Tval: Accuracy = {XGB_tval_accuracy:.4f} | Balanced = {XGB_tval_balanced_accuracy:.4f} | F1 = {XGB_tval_f1:.4f} | ROC AUC = {XGB_tval_ROC:.4f}")
    print(f"XGB Oversampled Tval: Accuracy = {XGB_tval_oversampled_accuracy:.4f} | Balanced = {XGB_tval_oversampled_balanced_accuracy:.4f} | F1 = {XGB_tval_oversampled_f1:.4f} | ROC AUC = {XGB_tval_oversampled_ROC:.4f}")
    print(f"XGB Test: Accuracy = {XGB_test_accuracy:.4f} | Balanced = {XGB_test_balanced_accuracy:.4f} | F1 = {XGB_test_f1:.4f} | ROC AUC = {XGB_test_ROC:.4f}")

    SVM_test_model = SVM_tuned_model_pipeline.fit(x_tval_oversampled, y_tval_oversampled)
    SVM_tval_accuracy = accuracy_score(y_tval_processed, SVM_test_model.predict(x_tval_processed))
    SVM_tval_balanced_accuracy = balanced_accuracy_score(y_tval_processed, SVM_test_model.predict(x_tval_processed))
    SVM_tval_f1 = f1_score(y_tval_processed, SVM_test_model.predict(x_tval_processed))
    SVM_tval_ROC = roc_auc_score(y_tval_processed, SVM_test_model.decision_function(x_tval_processed))
    SVM_tval_oversampled_accuracy = accuracy_score(y_tval_oversampled, SVM_test_model.predict(x_tval_oversampled))
    SVM_tval_oversampled_balanced_accuracy = balanced_accuracy_score(y_tval_oversampled, SVM_test_model.predict(x_tval_oversampled))
    SVM_tval_oversampled_f1 = f1_score(y_tval_oversampled, SVM_test_model.predict(x_tval_oversampled))
    SVM_tval_oversampled_ROC = roc_auc_score(y_tval_oversampled, SVM_test_model.decision_function(x_tval_oversampled))
    SVM_test_accuracy = accuracy_score(y_test_processed, SVM_test_model.predict(x_test_processed))
    SVM_test_balanced_accuracy = balanced_accuracy_score(y_test_processed, SVM_test_model.predict(x_test_processed))
    SVM_test_f1 = f1_score(y_test_processed, SVM_test_model.predict(x_test_processed))
    SVM_test_ROC = roc_auc_score(y_test_processed, SVM_test_model.decision_function(x_test_processed))
    print(f"SVM Tval: Accuracy = {SVM_tval_accuracy:.4f} | Balanced = {SVM_tval_balanced_accuracy:.4f} | F1 = {SVM_tval_f1:.4f} | ROC AUC = {SVM_tval_ROC:.4f}")
    print(f"SVM Oversampled Tval: Accuracy = {SVM_tval_oversampled_accuracy:.4f} | Balanced = {SVM_tval_oversampled_balanced_accuracy:.4f} | F1 = {SVM_tval_oversampled_f1:.4f} | ROC AUC = {SVM_tval_oversampled_ROC:.4f}")
    print(f"SVM Test: Accuracy = {SVM_test_accuracy:.4f} | Balanced = {SVM_test_balanced_accuracy:.4f} | F1 = {SVM_test_f1:.4f} | ROC AUC = {SVM_test_ROC:.4f}")



#----ROC AUC curves----
plot_ROC = False
if plot_ROC:
    LR_plot_ROC_model = LR_tuned_model_pipeline.fit(x_tval_oversampled, y_tval_oversampled)
    XGB_plot_ROC_model = XGB_tuned_model_pipeline.fit(x_tval_oversampled_XGB, y_tval_oversampled_XGB)
    SVM_plot_ROC_model = SVM_tuned_model_pipeline.fit(x_tval_oversampled, y_tval_oversampled)
    
    axes = plt.gca()
    RocCurveDisplay.from_predictions(y_true=y_test_processed, y_score=LR_plot_ROC_model.predict_proba(x_test_processed)[:,1], ax=axes, name="Logisstic Regression")
    RocCurveDisplay.from_predictions(y_true=y_test, y_score=XGB_plot_ROC_model.predict_proba(x_test)[:,1], ax=axes, name="XGBoost")
    RocCurveDisplay.from_predictions(y_true=y_test_processed, y_score=SVM_plot_ROC_model.decision_function(x_test_processed), ax=axes, name="SVM")

    plt.tight_layout()
    plt.show()


#----Data/model Analysis----
# Feature Importance:
#   Statistical:
#   - Mutual Information
#   Model Based:
#   - SHAP
#   - Permutation Imbalance (Remove/shuffle individual features and recalculate score) (week 9)
#   - Logistic Regression coefficients/weights (week 9)
#   - XGBoost Feature Gain (week 9)

#   Mutual Information
MI = False
if MI:
    MI_preprocessing = ColumnTransformer(
        transformers=[
            ("ordinal", IterativeImputer(max_iter=5, initial_strategy="median", random_state=0)),
            ("ordinal", SimpleImputer(strategy="most-frequent"))
        ]
    )
    mutual_info = pd.Series(
        mutual_info_classif(
            MI_preprocessing.fit_transform(x),
            y,
            discrete_features= np.isin(x.columns,nominal_features+ordinal_features)
        ),index=x.columns).sort_values(ascending=False)
    mutual_info.plot.bar()
    plt.title("Mutual Information Feature Importance")
    plt.ylabel("MI Score")
    plt.tight_layout()
    plt.show()

#   SHAP
SHAP = False
if SHAP:
    LR_test_model = LR_tuned_model_pipeline.fit(x_tval_oversampled, y_tval_oversampled)
    LR_explainer = shap.Explainer(LR_test_model.predict_proba, x_tval_processed)
    LR_SHAP_values = LR_explainer(x_test_processed)[:, :, 1]
    LR_SHAP_importance = np.abs(LR_SHAP_values.values).mean(axis=0)

    XGB_test_model = XGB_tuned_model_pipeline.fit(x_tval_oversampled_XGB, y_tval_oversampled_XGB)
    XGB_explainer = shap.Explainer(XGB_test_model.predict_proba, x_tval)
    XGB_SHAP_values = XGB_explainer(x_test)[:, :, 1]
    XGB_SHAP_importance = np.abs(XGB_SHAP_values.values).mean(axis=0)
    
    SVM_test_model = SVM_tuned_model_pipeline.fit(x_tval_oversampled, y_tval_oversampled)
    SVM_explainer = shap.Explainer(SVM_test_model.decision_function, x_tval_processed)
    SVM_SHAP_values = SVM_explainer(x_test_processed)[:, :, 1]
    SVM_SHAP_importance = np.abs(SVM_SHAP_values.values).mean(axis=0)

    # Plot bars by shifting x-positions
    x_axis_locations = np.arange(len(x.columns))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(x_axis_locations - width/2, LR_SHAP_importance, width=width, label="Logistic Regression")
    ax.bar(x_axis_locations, XGB_SHAP_importance, width=width, label="XGBoost")
    ax.bar(x_axis_locations + width/2, SVM_SHAP_importance, width=width, label="SVM")

    # Add labels and formatting
    ax.set_xticks(x_axis_locations, rotation=60, ha="right", labels=x.columns)
    ax.legend()
    plt.tight_layout()
    plt.show()
        


#----Improvements----
#   Data synthesis (SMOTE/tVAE)
#   model based feature importance analysis
#   tSNE for dataset visualisation
