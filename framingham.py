import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.feature_selection import mutual_info_classif
import shap


file = Path(__file__).with_name("framingham.csv")
df = pd.read_csv(file)

x = df.drop(columns="TenYearCHD")
y = df["TenYearCHD"]

# cigsPerDay=0 when cuurentSmoker=0
x.loc[x['currentSmoker'] == 0, 'cigsPerDay'] = 0

# missings indicator feature
missings_feature = False
if missings_feature:
    df["Nissings"] = x.isna().sum(axis=1)

#----pandas analysis----
# ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
# contains missing: ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate','glucose']
numeric_features = ['age', 'sysBP', 'diaBP', 'totChol', 'BMI', 'heartRate', 'glucose']
binary_features = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
discrete_features = {"male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"}

num_rows = df.shape[0]
nan_rows=df.isna().any(axis=1).sum()
print(nan_rows/num_rows)
#print(df.isna().sum())
#print(df["education"].describe())
#print(df.columns)
#print(len(df.columns))
#print(df.loc[:,df.isna().any()].columns)
#print(y.head)


# train test split
x_tval, x_test, y_tval, y_test = train_test_split(x, y, test_size=0.15, random_state=0, stratify=y, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_tval, y_tval, test_size=0.1, random_state=0, stratify=y_tval)

class_imbalance_weight = (y_train == 0).sum() / (y_train == 1).sum()


#----Logistic Regression----

# preprocessing pipeline
def edu_step_function(values):
    rounded = np.rint(values)
    return np.clip(rounded, 1, 4)

def rounding_function(values):
    rounded = np.rint(values)
    return np.clip(rounded, 0, None)

LR_numeric_pipeline = Pipeline([
    ('imputer', IterativeImputer(max_iter=5, initial_strategy='median', random_state=0)),
    ('scaler', StandardScaler())
])

LR_int_pipeline = Pipeline([
    ('imputer', IterativeImputer(max_iter=5, initial_strategy='median', random_state=0)),
    ('round', FunctionTransformer(rounding_function, validate=False)),
    ('scaler', StandardScaler())
])

LR_education_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('bucket', FunctionTransformer(edu_step_function, validate=False))
])

LR_binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('numeric', LR_numeric_pipeline, numeric_features),
        ('cigs', LR_int_pipeline, ['cigsPerDay']),
        ('education', LR_education_pipeline, ['education']),
        ('binary', LR_binary_pipeline, binary_features)
    ],
    remainder=StandardScaler()
)

# base (untuned) model
LR_base_model_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", LogisticRegression())
    ])

# tuned model
LR_tuned_model_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('model', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=5000,
        random_state=0,
        n_jobs=-1,
        # tuned paarams:
        C=1,
        l1_ratio=0.0,
        class_weight=None
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
        n_iter=100,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
        random_state=0,
        refit=True
    )

    # model comparison
    LR_base_model = LR_base_model_pipeline.fit(x_train, y_train)
    LR_base_score = roc_auc_score(y_val, LR_base_model.predict_proba(x_val)[:,1])

    LR_pretuned_model = LR_tuned_model_pipeline.fit(x_train, y_train)
    LR_pretuned_score = roc_auc_score(y_val, LR_pretuned_model.predict_proba(x_val)[:,1])

    LR_search.fit(x_train, y_train)
    LR_tuned_model = LR_search.best_estimator_
    LR_tuned_score = roc_auc_score(y_val, LR_tuned_model.predict_proba(x_val)[:,1])

    print("Logistic Regression:")
    print(f"Best score: {LR_search.best_score_}")
    print(f"ROC_AUC Base: {LR_base_score}; Untuned:{LR_pretuned_score}; Tuned: {LR_tuned_score}")
    print(LR_search.best_params_)


#----XGBoost----
# no preprocessing required:
# no scaling necessary for trees
# xgboost handles missing/nan

# base (untuned) model
XGB_base_model_pipeline = XGBClassifier()

# tuned model
XGB_tuned_model_pipeline = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    random_state=0,
    n_jobs=-1,
    #tuned params:
    n_estimators=475,
    learning_rate=0.007948263801032772,
    max_depth=2,
    min_child_weight=6,
    subsample=0.6767872227893332,
    colsample_bytree=0.6,
    gamma=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    scale_pos_weight=None
)

# Hyperparameter Tuning
# XGB_hp_tuning = True to run hyperparameter tuning
XGB_hp_tuning = False

if XGB_hp_tuning:
    # XGBoost Tuning
    XGB_param_space = {
        "n_estimators": Integer(300, 600),
        "learning_rate": Real(0.005, 0.1, prior="log-uniform"),
        "max_depth": Integer(2, 6),
        "min_child_weight": Integer(2, 6),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.6, 1.0),
        "gamma": Real(0.0, 1.0),
        "reg_alpha": Real(0.0, 1.0),
        "reg_lambda": Real(0.0, 5.0),
        "scale_pos_weight": Categorical([None, class_imbalance_weight])
    }

    XGB_search = BayesSearchCV(
        estimator=XGB_tuned_model_pipeline,
        search_spaces=XGB_param_space,
        n_iter=250,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
        random_state=0,
        refit=True
    )

    # model comparison
    XGB_base_model = XGB_base_model_pipeline.fit(x_train, y_train)
    XGB_base_score = roc_auc_score(y_val, XGB_base_model.predict_proba(x_val)[:,1])

    XGB_pretuned_model = XGB_tuned_model_pipeline.fit(x_train, y_train)
    XGB_pretuned_score = roc_auc_score(y_val, XGB_pretuned_model.predict_proba(x_val)[:,1])

    XGB_search.fit(x_train, y_train)
    XGB_tuned_model = XGB_search.best_estimator_
    XGB_tuned_score = roc_auc_score(y_val, XGB_tuned_model.predict_proba(x_val)[:,1])

    print("XGBoost:")
    print(f"Best score: {XGB_search.best_score_}")
    print(f"ROC_AUC Base: {XGB_base_score}; Untuned:{XGB_pretuned_score}; Tuned: {XGB_tuned_score}")
    print(XGB_search.best_params_)

#----Testing Models----
plot_ROC = False
if plot_ROC:
    LR_test_model = LR_tuned_model_pipeline.fit(x_tval, y_tval)
    LR_train_score = roc_auc_score(y_tval, LR_test_model.predict_proba(x_tval)[:,1])
    LR_test_score = roc_auc_score(y_test, LR_test_model.predict_proba(x_test)[:,1])

    XGB_test_model = XGB_tuned_model_pipeline.fit(x_tval, y_tval)
    XGB_train_score = roc_auc_score(y_tval, XGB_test_model.predict_proba(x_tval)[:,1])
    XGB_test_score = roc_auc_score(y_test, XGB_test_model.predict_proba(x_test)[:,1])

    print(f"LR ROC AUC| Train: {LR_train_score} Test: {LR_test_score}")
    print(f"XGB ROC AUC| Train: {XGB_train_score} Test: {XGB_test_score}")

    axes = plt.gca()
    RocCurveDisplay.from_predictions(y_true=y_test, y_score=LR_test_model.predict_proba(x_test)[:,1], ax=axes, name="Logisstic Regression")
    RocCurveDisplay.from_predictions(y_true=y_test, y_score=XGB_test_model.predict_proba(x_test)[:,1], ax=axes, name="XGBoost")

    plt.tight_layout()
    plt.show()

#??? compare imputing vs removing missings

#----Data Analysis----
# Feature Importance:
#   Statistical:
#   - Mutual Information
#   Model Based:
#   - SHAP
#   - Permutation Imbalance (Remove/shuffle individual features and recalculate score) (week 9)
#   - Logistic Regression coefficients/weights (week 9)
#   - XGBoost Feature Gain (week 9)

#   Mutual Information
MI = True
if MI:
    mutual_info = pd.Series(
        mutual_info_classif(
            preprocessing_pipeline.fit_transform(x),
            y,
            discrete_features=[feature in discrete_features for feature in x.columns]
        ),index=x.columns).sort_values(ascending=False)
    mutual_info.plot.bar()
    plt.title("Mutual Information Feature Importance")
    plt.ylabel("MI Score")
    plt.tight_layout()
    plt.show()

#   SHAP
SHAP = True
if SHAP:
    LR_test_model = LR_tuned_model_pipeline.fit(x_tval, y_tval)
    LR_explainer = shap.Explainer(LR_test_model.predict_proba, x_tval)
    LR_SHAP_values = LR_explainer(x_test)[:, :, 1]
    LR_SHAP_importance = np.abs(LR_SHAP_values.values).mean(axis=0)

    XGB_test_model = XGB_tuned_model_pipeline.fit(x_tval, y_tval)
    XGB_explainer = shap.Explainer(XGB_test_model.predict_proba, x_tval)
    XGB_SHAP_values = XGB_explainer(x_test)[:, :, 1]
    XGB_SHAP_importance = np.abs(XGB_SHAP_values.values).mean(axis=0)
    
    # Plot bars by shifting x-positions
    x_axis_locations = np.arange(len(x.columns))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x_axis_locations - width/2, LR_SHAP_importance, width=width, label="Logistic Regression")
    ax.bar(x_axis_locations + width/2, XGB_SHAP_importance, width=width, label="XGBoost")

    # Add labels and formatting
    ax.set_xticks(x_axis_locations, rotation=60, ha="right", labels=x.columns)
    ax.legend()
    plt.tight_layout()
    plt.show()
        


#----Week 9 additions----
#   model based feature importance analysis
#   logistic regression model from scratch
#   plot training and compare to other models
#   MLP using pytorch (small dataset so won't perform well)
