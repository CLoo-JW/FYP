import pandas as pd  # For reading CSV files
import numpy as np  # Used to combine outputs for meta classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from torch.nn import CrossEntropyLoss
from tqdm import tqdm  # For progress bars
from nltk.sentiment import SentimentIntensityAnalyzer  # VADER
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # RoBERTa (Tokeniser and Classifier)
from scipy.special import softmax  # To convert into probability
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Converts text into number for SVM
from sklearn.svm import SVC  # SVM
from sklearn.model_selection import train_test_split  # Splits dataset
from sklearn.metrics import classification_report, f1_score  # Output metrics
from sklearn.pipeline import Pipeline  # Chains TFIDF (preprocessing) and SVM (model) together
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.metrics import accuracy_score  # Output metrics
from imblearn.over_sampling import RandomOverSampler  # Balance classes
from sklearn.model_selection import cross_val_predict  # Out of fold training
import optuna # Hyperparameter tuning
from sklearn.model_selection import cross_val_score  # OOF training
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
import torch
from huggingface_hub import login
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import ftfy
import html
import re
from sklearn.svm import LinearSVC
import os
import shap

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
# ----------------------------------------------------------------------------- Start
# BASE MODELS
# ----------------------------------------------------------------------------- 
base_vader_train_scores_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_train_probabilities.csv")
base_vader_train_scores = base_vader_train_scores_df[["base_vader_neg", "base_vader_neu", "base_vader_pos", "base_vader_compound"]].to_numpy()

base_vader_test_scores_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_test_probabilities.csv")
base_vader_test_scores = base_vader_test_scores_df[["base_vader_neg", "base_vader_neu", "base_vader_pos", "base_vader_compound"]].to_numpy()

base_vader_test_sentiment_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_test_sentiment.csv")
base_vader_test_sentiment = base_vader_test_sentiment_df["base_vader_sentiment"].to_numpy()


base_svm_train_probabilities_df = pd.read_csv("Base_Learner/Results/SVM/Base/base_svm_train_probabilities.csv")
base_svm_train_probabilities = base_svm_train_probabilities_df[["base_svm_neg", "base_svm_neu", "base_svm_pos"]].to_numpy()

base_svm_test_probabilities_df = pd.read_csv("Base_Learner/Results/SVM/Base/base_svm_test_probabilities.csv")
base_svm_test_probabilities = base_svm_test_probabilities_df[["base_svm_neg", "base_svm_neu", "base_svm_pos"]].to_numpy()

base_svm_test_sentiment_df = pd.read_csv("Base_Learner/Results/SVM/Base/base_svm_test_sentiment.csv")
base_svm_test_sentiment = base_svm_test_sentiment_df["base_svm_sentiment"].to_numpy()


base_nb_train_probabilities_df = pd.read_csv("Base_Learner/Results/NB/Base/base_nb_train_probabilities.csv")
base_nb_train_probabilities = base_nb_train_probabilities_df[["base_nb_neg", "base_nb_neu", "base_nb_pos"]].to_numpy()

base_nb_test_probabilities_df = pd.read_csv("Base_Learner/Results/NB/Base/base_nb_test_probabilities.csv")
base_nb_test_probabilities = base_nb_test_probabilities_df[["base_nb_neg", "base_nb_neu", "base_nb_pos"]].to_numpy()

base_nb_test_sentiment_df = pd.read_csv("Base_Learner/Results/NB/Base/base_nb_test_sentiment.csv")
base_nb_test_sentiment = base_nb_test_sentiment_df["base_nb_sentiment"].to_numpy()


base_roberta_train_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_train_probabilities.csv")
base_roberta_train_probabilities = base_roberta_train_probabilities_df[["base_roberta_neg", "base_roberta_neu", "base_roberta_pos"]].to_numpy()

base_roberta_test_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_test_probabilities.csv")
base_roberta_test_probabilities = base_roberta_test_probabilities_df[["base_roberta_neg", "base_roberta_neu", "base_roberta_pos"]].to_numpy()

base_roberta_test_sentiment_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_test_sentiment.csv")
base_roberta_test_sentiment = base_roberta_test_sentiment_df["base_roberta_sentiment"].to_numpy()
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# ENHANCED MODELS
# ----------------------------------------------------------------------------- 
enhanced_vader_train_scores_df = pd.read_csv("Base_Learner/Results/VADER/Enhanced/enhanced_vader_train_probabilities.csv")
enhanced_vader_train_scores = enhanced_vader_train_scores_df[["enhanced_vader_neg", "enhanced_vader_neu", "enhanced_vader_pos", "enhanced_vader_compound"]].to_numpy()

enhanced_vader_test_scores_df = pd.read_csv("Base_Learner/Results/VADER/Enhanced/enhanced_vader_test_probabilities.csv")
enhanced_vader_test_scores = enhanced_vader_test_scores_df[["enhanced_vader_neg", "enhanced_vader_neu", "enhanced_vader_pos", "enhanced_vader_compound"]].to_numpy()

enhanced_vader_test_sentiment_df = pd.read_csv("Base_Learner/Results/VADER/Enhanced/enhanced_vader_test_sentiment.csv")
enhanced_vader_test_sentiment = enhanced_vader_test_sentiment_df["enhanced_vader_sentiment"].to_numpy()


enhanced_svm_train_probabilities_df = pd.read_csv("Base_Learner/Results/SVM/Enhanced/enhanced_svm_train_probabilities.csv")
enhanced_svm_train_probabilities = enhanced_svm_train_probabilities_df[["enhanced_svm_neg", "enhanced_svm_neu", "enhanced_svm_pos"]].to_numpy()

enhanced_svm_test_probabilities_df = pd.read_csv("Base_Learner/Results/SVM/Enhanced/enhanced_svm_test_probabilities.csv")
enhanced_svm_test_probabilities = enhanced_svm_test_probabilities_df[["enhanced_svm_neg", "enhanced_svm_neu", "enhanced_svm_pos"]].to_numpy()

enhanced_svm_test_sentiment_df = pd.read_csv("Base_Learner/Results/SVM/Enhanced/enhanced_svm_test_sentiment.csv")
enhanced_svm_test_sentiment = enhanced_svm_test_sentiment_df["enhanced_svm_sentiment"].to_numpy()


enhanced_nb_train_probabilities_df = pd.read_csv("Base_Learner/Results/NB/Enhanced/enhanced_nb_train_probabilities.csv")
enhanced_nb_train_probabilities = enhanced_nb_train_probabilities_df[["enhanced_nb_neg", "enhanced_nb_neu", "enhanced_nb_pos"]].to_numpy()

enhanced_nb_test_probabilities_df = pd.read_csv("Base_Learner/Results/NB/Enhanced/enhanced_nb_test_probabilities.csv")
enhanced_nb_test_probabilities = enhanced_nb_test_probabilities_df[["enhanced_nb_neg", "enhanced_nb_neu", "enhanced_nb_pos"]].to_numpy()

enhanced_nb_test_sentiment_df = pd.read_csv("Base_Learner/Results/NB/Enhanced/enhanced_nb_test_sentiment.csv")
enhanced_nb_test_sentiment = enhanced_nb_test_sentiment_df["enhanced_nb_sentiment"].to_numpy()


enhanced_roberta_train_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Enhanced/enhanced_roberta_train_probabilities.csv")
enhanced_roberta_train_probabilities = enhanced_roberta_train_probabilities_df[["enhanced_roberta_neg", "enhanced_roberta_neu", "enhanced_roberta_pos"]].to_numpy()

enhanced_roberta_test_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Enhanced/enhanced_roberta_test_probabilities.csv")
enhanced_roberta_test_probabilities = enhanced_roberta_test_probabilities_df[["enhanced_roberta_neg", "enhanced_roberta_neu", "enhanced_roberta_pos"]].to_numpy()

enhanced_roberta_test_sentiment_df = pd.read_csv("Base_Learner/Results/RoBERTa/Enhanced/enhanced_roberta_test_sentiment.csv")
enhanced_roberta_test_sentiment = enhanced_roberta_test_sentiment_df["enhanced_roberta_sentiment"].to_numpy()
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# SENTIMENT AND EXTRA FEATURES
# ----------------------------------------------------------------------------- 
extra_train_meta_features_df = pd.read_csv("Dataset/Meta_Features/train_meta_polarity_features.csv")
extra_test_meta_features_df = pd.read_csv("Dataset/Meta_Features/test_meta_polarity_features.csv")

extra_train_meta_features = extra_train_meta_features_df.to_numpy()
extra_test_meta_features = extra_test_meta_features_df.to_numpy()

train_split_df = pd.read_csv("Dataset/Preprocessed/train_split.csv")
val_split_df = pd.read_csv("Dataset/Preprocessed/val_split.csv")
test_split_df = pd.read_csv("Dataset/Preprocessed/test_split.csv")

for split_name, split_df in [
    ("TRAIN SET", train_split_df),
    ("VALIDATION SET", val_split_df),
    ("TEST SET", test_split_df)
]:
    print(f"{split_name} Missing Sentiment:", split_df["Sentiment"].isna().sum())

sentiment_train = train_split_df["Sentiment"]

sentiment_val = val_split_df["Sentiment"]

sentiment_test = test_split_df["Sentiment"]

dataset_size = len(sentiment_test) + len(sentiment_train) + len(sentiment_val)

print("\n========== DATASET SPLIT ==========")
print(str(int((len(sentiment_train) / dataset_size) * 100)) + "% TRAIN SPLIT: "
      + str(len(sentiment_train)) + "\n" + str(sentiment_train.value_counts())
      )
print("\n" + str(int((len(sentiment_val) / dataset_size) * 100)) + "% VALIDATION SPLIT: "
      + str(len(sentiment_val)) + "\n" + str(sentiment_val.value_counts())
      )
print("\n" + str(int((len(sentiment_test) / dataset_size) * 100)) + "% TEST SPLIT: "
      + str(len(sentiment_test)) + "\n" + str(sentiment_test.value_counts())
      )
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END


# ================================================================================================================ START
# META XGBOOST
# ======================================================================================================================
# ----------------------------------------------------------------------------- Start
# META XGBOOST SETUP
# ----------------------------------------------------------------------------- 
optuna_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

label_map = {
    "neg": 0,
    "neu": 1,
    "pos": 2
}

reverse_label_map = {
    0: "neg",
    1: "neu",
    2: "pos"
}

sentiment_train_num = sentiment_train.map(label_map)

assert sentiment_train_num.isna().sum() == 0, "Some sentiment labels were not mapped correctly."

sentiment_train_num = sentiment_train_num.astype(int)

base_meta_features_train = np.hstack([base_svm_train_probabilities, base_nb_train_probabilities, base_roberta_train_probabilities, base_vader_train_scores])
base_meta_features_test = np.hstack([base_svm_test_probabilities, base_nb_test_probabilities, base_roberta_test_probabilities, base_vader_test_scores])

mixed_meta_features_train = np.hstack([enhanced_svm_train_probabilities, 
                                      enhanced_nb_train_probabilities, 
                                      enhanced_roberta_train_probabilities, 
                                      enhanced_vader_train_scores])
mixed_meta_features_test = np.hstack([enhanced_svm_test_probabilities, 
                                     enhanced_nb_test_probabilities, 
                                     enhanced_roberta_test_probabilities, 
                                     enhanced_vader_test_scores])

enhanced_meta_features_train = np.hstack([enhanced_svm_train_probabilities, 
                                      enhanced_nb_train_probabilities, 
                                      enhanced_roberta_train_probabilities, 
                                      enhanced_vader_train_scores, extra_train_meta_features])
enhanced_meta_features_test = np.hstack([enhanced_svm_test_probabilities, 
                                     enhanced_nb_test_probabilities, 
                                     enhanced_roberta_test_probabilities, 
                                     enhanced_vader_test_scores, extra_test_meta_features])
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# XAI FEATURE NAMES
# -----------------------------------------------------------------------------
base_feature_names = [
    "base_svm_neg", "base_svm_neu", "base_svm_pos",
    "base_nb_neg", "base_nb_neu", "base_nb_pos",
    "base_roberta_neg", "base_roberta_neu", "base_roberta_pos",
    "base_vader_neg", "base_vader_neu", "base_vader_pos", "base_vader_compound"
]

mixed_feature_names = [
    "enhanced_svm_neg", "enhanced_svm_neu", "enhanced_svm_pos",
    "enhanced_nb_neg", "enhanced_nb_neu", "enhanced_nb_pos",
    "enhanced_roberta_neg", "enhanced_roberta_neu", "enhanced_roberta_pos",
    "enhanced_vader_neg", "enhanced_vader_neu", "enhanced_vader_pos", "enhanced_vader_compound"
]

extra_feature_names = extra_train_meta_features_df.columns.tolist()

enhanced_feature_names = mixed_feature_names + extra_feature_names

assert len(base_feature_names) == base_meta_features_train.shape[1]
assert len(mixed_feature_names) == mixed_meta_features_train.shape[1]
assert len(enhanced_feature_names) == enhanced_meta_features_train.shape[1]

assert base_meta_features_train.shape[0] == len(sentiment_train)
assert base_meta_features_test.shape[0] == len(sentiment_test)

assert mixed_meta_features_train.shape[0] == len(sentiment_train)
assert mixed_meta_features_test.shape[0] == len(sentiment_test)

assert enhanced_meta_features_train.shape[0] == len(sentiment_train)
assert enhanced_meta_features_test.shape[0] == len(sentiment_test)

print("\n========== XGB XAI FEATURE CHECK ==========")
print("Base XGB XAI feature count:", len(base_feature_names))
print("Mixed XGB XAI feature count:", len(mixed_feature_names))
print("Enhanced XGB XAI feature count:", len(enhanced_feature_names))
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# BASE META XGBOOST
# -----------------------------------------------------------------------------
def base_meta_xgb_optuna(trial):
    optuna_xgb_meta_model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(
        optuna_xgb_meta_model,
        base_meta_features_train,
        sentiment_train_num,
        cv=optuna_cv,
        scoring="f1_macro"
    )
    return scores.mean()

print("\n========== BASE META XGBOOST MODEL ==========")
base_meta_xgb_study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
    )
base_meta_xgb_study.optimize(
    base_meta_xgb_optuna,
    n_trials=30
)

base_meta_xgb_best = base_meta_xgb_study.best_params

base_meta_xgb_model = XGBClassifier(
    n_estimators=base_meta_xgb_best['n_estimators'],
    learning_rate=base_meta_xgb_best['learning_rate'],
    max_depth=base_meta_xgb_best['max_depth'],
    min_child_weight=base_meta_xgb_best['min_child_weight'],
    gamma=base_meta_xgb_best['gamma'],
    subsample=base_meta_xgb_best['subsample'],
    colsample_bytree=base_meta_xgb_best['colsample_bytree'],
    reg_alpha=base_meta_xgb_best['reg_alpha'],
    reg_lambda=base_meta_xgb_best['reg_lambda'],
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    num_class=3,
    random_state=42,
    n_jobs=-1
)
base_meta_xgb_model.fit(base_meta_features_train, sentiment_train_num)
base_meta_xgb_test_predictions = base_meta_xgb_model.predict(base_meta_features_test)
base_meta_xgb_test_sentiment = [reverse_label_map[i] for i in base_meta_xgb_test_predictions]
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# MIXED META XGBOOST
# -----------------------------------------------------------------------------
def mixed_meta_xgb_optuna(trial):
    optuna_xgb_meta_model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(
        optuna_xgb_meta_model,
        mixed_meta_features_train,
        sentiment_train_num,
        cv=optuna_cv,
        scoring="f1_macro"
    )
    return scores.mean()

print("\n========== MIXED META XGBOOST MODEL ==========")
mixed_meta_xgb_study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
    )
mixed_meta_xgb_study.optimize(
    mixed_meta_xgb_optuna,
    n_trials=30
)

mixed_meta_xgb_best = mixed_meta_xgb_study.best_params

mixed_meta_xgb_model = XGBClassifier(
    n_estimators=mixed_meta_xgb_best['n_estimators'],
    learning_rate=mixed_meta_xgb_best['learning_rate'],
    max_depth=mixed_meta_xgb_best['max_depth'],
    min_child_weight=mixed_meta_xgb_best['min_child_weight'],
    gamma=mixed_meta_xgb_best['gamma'],
    subsample=mixed_meta_xgb_best['subsample'],
    colsample_bytree=mixed_meta_xgb_best['colsample_bytree'],
    reg_alpha=mixed_meta_xgb_best['reg_alpha'],
    reg_lambda=mixed_meta_xgb_best['reg_lambda'],
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    num_class=3,
    random_state=42,
    n_jobs=-1
)
mixed_meta_xgb_model.fit(mixed_meta_features_train, sentiment_train_num)
mixed_meta_xgb_test_predictions = mixed_meta_xgb_model.predict(mixed_meta_features_test)
mixed_meta_xgb_test_sentiment = [reverse_label_map[i] for i in mixed_meta_xgb_test_predictions]
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# ENHANCED META XGBOOST
# -----------------------------------------------------------------------------
def enhanced_meta_xgb_optuna(trial):
    optuna_xgb_meta_model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(
        optuna_xgb_meta_model,
        enhanced_meta_features_train,
        sentiment_train_num,
        cv=optuna_cv,
        scoring="f1_macro"
    )
    return scores.mean()

print("\n========== ENHANCED META XGBOOST MODEL ==========")
enhanced_meta_xgb_study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
    )
enhanced_meta_xgb_study.optimize(
    enhanced_meta_xgb_optuna,
    n_trials=30
)

enhanced_meta_xgb_best = enhanced_meta_xgb_study.best_params

enhanced_meta_xgb_model = XGBClassifier(
    n_estimators=enhanced_meta_xgb_best['n_estimators'],
    learning_rate=enhanced_meta_xgb_best['learning_rate'],
    max_depth=enhanced_meta_xgb_best['max_depth'],
    min_child_weight=enhanced_meta_xgb_best['min_child_weight'],
    gamma=enhanced_meta_xgb_best['gamma'],
    subsample=enhanced_meta_xgb_best['subsample'],
    colsample_bytree=enhanced_meta_xgb_best['colsample_bytree'],
    reg_alpha=enhanced_meta_xgb_best['reg_alpha'],
    reg_lambda=enhanced_meta_xgb_best['reg_lambda'],
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    num_class=3,
    random_state=42,
    n_jobs=-1
)
enhanced_meta_xgb_model.fit(enhanced_meta_features_train, sentiment_train_num)
enhanced_meta_xgb_test_predictions = enhanced_meta_xgb_model.predict(enhanced_meta_features_test)
enhanced_meta_xgb_test_sentiment = [reverse_label_map[i] for i in enhanced_meta_xgb_test_predictions]
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# READABLE XGBOOST SHAP XAI FUNCTIONS
# -----------------------------------------------------------------------------
def get_meta_feature_group(feature_name):
    feature_name = str(feature_name).lower()

    if "roberta" in feature_name:
        return "RoBERTa"
    elif "svm" in feature_name:
        return "SVM"
    elif "nb" in feature_name:
        return "Naive Bayes"
    elif "vader" in feature_name:
        return "VADER"
    elif feature_name.startswith("meta_"):
        return "Polarity Shift Features"
    else:
        return "Other"


def make_feature_name_readable(feature_name):
    readable_name = str(feature_name)

    readable_name = readable_name.replace("base_", "Base ")
    readable_name = readable_name.replace("enhanced_", "Enhanced ")
    readable_name = readable_name.replace("meta_", "Meta ")
    readable_name = readable_name.replace("_", " ")

    readable_name = readable_name.title()

    readable_name = readable_name.replace("Svm", "SVM")
    readable_name = readable_name.replace("Nb", "NB")
    readable_name = readable_name.replace("Roberta", "RoBERTa")
    readable_name = readable_name.replace("Vader", "VADER")

    return readable_name


def format_feature_contributions(feature_df, contribution_column, top_n=5):
    if feature_df.empty:
        return "None"

    formatted_features = []

    for _, row in feature_df.head(top_n).iterrows():
        formatted_features.append(
            make_feature_name_readable(row["feature"])
            + " = "
            + str(round(row["raw_value"], 4))
            + " (SHAP contribution: "
            + str(round(row[contribution_column], 4))
            + ")"
        )

    return ";\n".join(formatted_features)


def get_row_class_shap_values(shap_values, row_position, class_index, n_rows, n_features, n_classes):
    if isinstance(shap_values, list):
        return shap_values[class_index][row_position]

    shap_values_array = np.asarray(shap_values)

    if shap_values_array.ndim == 3:
        # Common multiclass format: (n_samples, n_features, n_classes)
        if shap_values_array.shape[0] == n_rows and shap_values_array.shape[1] == n_features:
            return shap_values_array[row_position, :, class_index]

        # Alternative format: (n_classes, n_samples, n_features)
        if shap_values_array.shape[0] == n_classes and shap_values_array.shape[2] == n_features:
            return shap_values_array[class_index, row_position, :]

    if shap_values_array.ndim == 2:
        return shap_values_array[row_position]

    raise ValueError("Unexpected SHAP values shape: " + str(shap_values_array.shape))


def save_readable_local_xgb_shap_report(
        model,
        feature_values,
        feature_names,
        test_split_df,
        true_sentiments,
        output_folder,
        file_prefix,
        row_indices,
        reverse_label_map,
        top_n=5
):
    selected_feature_values = feature_values[row_indices]

    predicted_class_numbers = model.predict(selected_feature_values).astype(int)
    predicted_probabilities = model.predict_proba(selected_feature_values)

    class_numbers = list(model.classes_)
    n_classes = len(class_numbers)
    n_rows = selected_feature_values.shape[0]
    n_features = selected_feature_values.shape[1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(selected_feature_values)

    true_sentiments = np.asarray(true_sentiments)

    readable_rows = []

    correct_text_report_blocks = []
    wrong_text_report_blocks = []

    for local_position, row_index in enumerate(row_indices):
        review_text = test_split_df.iloc[row_index]["Text"]

        true_sentiment = true_sentiments[row_index]

        predicted_class_number = int(predicted_class_numbers[local_position])
        predicted_sentiment = reverse_label_map[predicted_class_number]

        if true_sentiment == predicted_sentiment:
            prediction_result = "correct"
        else:
            prediction_result = "wrong"

        predicted_class_index = class_numbers.index(predicted_class_number)
        prediction_confidence = predicted_probabilities[local_position][predicted_class_index]

        row_shap_values = get_row_class_shap_values(
            shap_values=shap_values,
            row_position=local_position,
            class_index=predicted_class_index,
            n_rows=n_rows,
            n_features=n_features,
            n_classes=n_classes
        )

        local_xai_df = pd.DataFrame({
            "feature": feature_names,
            "readable_feature": [make_feature_name_readable(feature) for feature in feature_names],
            "feature_group": [get_meta_feature_group(feature) for feature in feature_names],
            "raw_value": selected_feature_values[local_position],
            "shap_value_for_predicted_class": row_shap_values,
            "abs_shap_value": np.abs(row_shap_values)
        })

        supporting_features_df = (
            local_xai_df[local_xai_df["shap_value_for_predicted_class"] > 0]
            .sort_values("shap_value_for_predicted_class", ascending=False)
        )

        opposing_features_df = (
            local_xai_df[local_xai_df["shap_value_for_predicted_class"] < 0]
            .sort_values("shap_value_for_predicted_class", ascending=True)
        )

        group_contribution_df = (
            local_xai_df
            .groupby("feature_group")["shap_value_for_predicted_class"]
            .sum()
            .reset_index()
            .sort_values("shap_value_for_predicted_class", ascending=False)
        )

        positive_group_contribution_df = group_contribution_df[
            group_contribution_df["shap_value_for_predicted_class"] > 0
        ]

        if positive_group_contribution_df.empty:
            strongest_feature_group = "None"
        else:
            strongest_feature_group = positive_group_contribution_df.iloc[0]["feature_group"]

        if supporting_features_df.empty:
            strongest_individual_feature = "None"
        else:
            strongest_individual_feature = supporting_features_df.iloc[0]["readable_feature"]

        top_supporting_features = format_feature_contributions(
            supporting_features_df,
            contribution_column="shap_value_for_predicted_class",
            top_n=top_n
        )

        top_opposing_features = format_feature_contributions(
            opposing_features_df,
            contribution_column="shap_value_for_predicted_class",
            top_n=top_n
        )

        supporting_feature_names = supporting_features_df.head(top_n)["readable_feature"].tolist()
        opposing_feature_names = opposing_features_df.head(1)["readable_feature"].tolist()

        if len(supporting_feature_names) > 0:
            supporting_feature_text = ", ".join(supporting_feature_names)
        else:
            supporting_feature_text = "no strong supporting features"

        if len(opposing_feature_names) > 0:
            opposing_feature_text = opposing_feature_names[0]
        else:
            opposing_feature_text = "no strong opposing signal"

        explanation = (
            "The meta XGBoost model predicted '"
            + str(predicted_sentiment)
            + "' mainly because the strongest individual supporting feature was "
            + str(strongest_individual_feature)
            + ", while the strongest overall feature group was "
            + str(strongest_feature_group)
            + ". The top supporting features were "
            + supporting_feature_text
            + ". The main opposing signal was "
            + opposing_feature_text
            + "."
        )

        readable_rows.append({
            "row_index": row_index,
            "prediction_result": prediction_result,
            "review_text": review_text,
            "true_sentiment": true_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "prediction_confidence": round(prediction_confidence, 4),
            "strongest_individual_feature": strongest_individual_feature,
            "strongest_feature_group": strongest_feature_group,
            "top_supporting_features": top_supporting_features,
            "top_opposing_features": top_opposing_features,
            "explanation": explanation
        })

        text_block = (
            "row_index:\n"
            + str(row_index)
            + "\n\nprediction_result:\n"
            + str(prediction_result)
            + "\n\nreview_text:\n\""
            + str(review_text)
            + "\"\n\ntrue_sentiment:\n"
            + str(true_sentiment)
            + "\n\npredicted_sentiment:\n"
            + str(predicted_sentiment)
            + "\n\nprediction_confidence:\n"
            + str(round(prediction_confidence, 4))
            + "\n\nstrongest_individual_feature:\n"
            + str(strongest_individual_feature)
            + "\n\nstrongest_feature_group:\n"
            + str(strongest_feature_group)
            + "\n\ntop_supporting_features:\n"
            + str(top_supporting_features)
            + "\n\ntop_opposing_features:\n"
            + str(top_opposing_features)
            + "\n\nexplanation:\n"
            + str(explanation)
            + "\n\n"
            + "=" * 100
            + "\n\n"
        )

        if prediction_result == "correct":
            correct_text_report_blocks.append(text_block)
        else:
            wrong_text_report_blocks.append(text_block)

    readable_xai_report_df = pd.DataFrame(readable_rows)

    readable_xai_report_df[
        readable_xai_report_df["prediction_result"] == "correct"
    ].to_csv(
        os.path.join(output_folder, file_prefix + "_correct_readable_xai_report.csv"),
        index=False
    )

    readable_xai_report_df[
        readable_xai_report_df["prediction_result"] == "wrong"
    ].to_csv(
        os.path.join(output_folder, file_prefix + "_wrong_readable_xai_report.csv"),
        index=False
    )

    with open(
        os.path.join(output_folder, file_prefix + "_correct_readable_xai_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write("".join(correct_text_report_blocks))

    with open(
        os.path.join(output_folder, file_prefix + "_wrong_readable_xai_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write("".join(wrong_text_report_blocks))

    return readable_xai_report_df
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# EVALUATE META XGBOOST
# ----------------------------------------------------------------------------- 
print("\nBASE META XGBOOST BEST PARAMETERS: " + str(base_meta_xgb_study.best_value))
print(base_meta_xgb_best)
print("BASE META XGBOOST MODEL ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_meta_xgb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_meta_xgb_test_sentiment, digits=4))

print("\nBASE VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_vader_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_vader_test_sentiment, digits=4))

print("\nBASE NAIVE BAYES ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_nb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_nb_test_sentiment, digits=4))

print("\nBASE SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_svm_test_sentiment, digits=4))

print("\nBASE RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_roberta_test_sentiment, digits=4))


print("\nMIXED META XGBOOST BEST PARAMETERS: " + str(mixed_meta_xgb_study.best_value))
print(mixed_meta_xgb_best)
print("MIXED META XGBOOST MODEL ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, mixed_meta_xgb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, mixed_meta_xgb_test_sentiment, digits=4))

print("\nMIXED VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_vader_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_vader_test_sentiment, digits=4))

print("\nMIXED NAIVE BAYES ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_nb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_nb_test_sentiment, digits=4))

print("\nMIXED SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_svm_test_sentiment, digits=4))

print("\nMIXED RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_roberta_test_sentiment, digits=4))


print("\nENHANCED META XGBOOST BEST PARAMETERS: " + str(enhanced_meta_xgb_study.best_value))
print(enhanced_meta_xgb_best)
print("ENHANCED META XGBOOST MODEL ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_meta_xgb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_meta_xgb_test_sentiment, digits=4))

print("\nENHANCED VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_vader_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_vader_test_sentiment, digits=4))

print("\nENHANCED NAIVE BAYES ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_nb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_nb_test_sentiment, digits=4))

print("\nENHANCED SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_svm_test_sentiment, digits=4))

print("\nENHANCED RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_roberta_test_sentiment, digits=4))
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# SAVE OUTPUT / XAI
# ----------------------------------------------------------------------------- 
xai_output_folder = "Meta_Model/XAI/XGB"
os.makedirs(xai_output_folder, exist_ok=True)

base_readable_xai_report_df = save_readable_local_xgb_shap_report(
    model=base_meta_xgb_model,
    feature_values=base_meta_features_test,
    feature_names=base_feature_names,
    test_split_df=test_split_df,
    true_sentiments=sentiment_test,
    output_folder=xai_output_folder,
    file_prefix="base_meta_xgb",
    row_indices=list(range(len(sentiment_test))),
    reverse_label_map=reverse_label_map,
    top_n=5
)

print("\n========== BASE META XGB CORRECT XAI EXAMPLES ==========")
print(base_readable_xai_report_df[
    base_readable_xai_report_df["prediction_result"] == "correct"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\n========== BASE META XGB WRONG XAI EXAMPLES ==========")
print(base_readable_xai_report_df[
    base_readable_xai_report_df["prediction_result"] == "wrong"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))


mixed_readable_xai_report_df = save_readable_local_xgb_shap_report(
    model=mixed_meta_xgb_model,
    feature_values=mixed_meta_features_test,
    feature_names=mixed_feature_names,
    test_split_df=test_split_df,
    true_sentiments=sentiment_test,
    output_folder=xai_output_folder,
    file_prefix="mixed_meta_xgb",
    row_indices=list(range(len(sentiment_test))),
    reverse_label_map=reverse_label_map,
    top_n=5
)

print("\n========== MIXED META XGB CORRECT XAI EXAMPLES ==========")
print(mixed_readable_xai_report_df[
    mixed_readable_xai_report_df["prediction_result"] == "correct"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\n========== MIXED META XGB WRONG XAI EXAMPLES ==========")
print(mixed_readable_xai_report_df[
    mixed_readable_xai_report_df["prediction_result"] == "wrong"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))


enhanced_readable_xai_report_df = save_readable_local_xgb_shap_report(
    model=enhanced_meta_xgb_model,
    feature_values=enhanced_meta_features_test,
    feature_names=enhanced_feature_names,
    test_split_df=test_split_df,
    true_sentiments=sentiment_test,
    output_folder=xai_output_folder,
    file_prefix="enhanced_meta_xgb",
    row_indices=list(range(len(sentiment_test))),
    reverse_label_map=reverse_label_map,
    top_n=5
)

print("\n========== ENHANCED META XGB CORRECT XAI EXAMPLES ==========")
print(enhanced_readable_xai_report_df[
    enhanced_readable_xai_report_df["prediction_result"] == "correct"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\n========== ENHANCED META XGB WRONG XAI EXAMPLES ==========")
print(enhanced_readable_xai_report_df[
    enhanced_readable_xai_report_df["prediction_result"] == "wrong"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\nSaved XGBoost SHAP XAI Report to:", xai_output_folder)

output_folder = "Meta_Model/Classification_Report/XGB"
os.makedirs(output_folder, exist_ok=True)

base_meta_xgb_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_meta_xgb_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

mixed_meta_xgb_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        mixed_meta_xgb_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_meta_xgb_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        enhanced_meta_xgb_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_meta_xgb_test_report_df.to_csv(
    os.path.join(output_folder, "base_meta_xgb_test_classification_report.csv"),
    index_label="class"
)

mixed_meta_xgb_test_report_df.to_csv(
    os.path.join(output_folder, "mixed_meta_xgb_test_classification_report.csv"),
    index_label="class"
)

enhanced_meta_xgb_test_report_df.to_csv(
    os.path.join(output_folder, "enhanced_meta_xgb_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_meta_xgb_test_report_df.values,
    rowLabels=base_meta_xgb_test_report_df.index,
    colLabels=base_meta_xgb_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_meta_xgb_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=mixed_meta_xgb_test_report_df.values,
    rowLabels=mixed_meta_xgb_test_report_df.index,
    colLabels=mixed_meta_xgb_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "mixed_meta_xgb_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_meta_xgb_test_report_df.values,
    rowLabels=enhanced_meta_xgb_test_report_df.index,
    colLabels=enhanced_meta_xgb_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_meta_xgb_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved XGBoost Classification Report to:", output_folder)

output_folder = "Meta_Model/Results"
os.makedirs(output_folder, exist_ok=True)

meta_xgb_optuna_summary = pd.DataFrame([
    {
        "hyperparameter": "n_estimators",
        "search_range": "200 to 800",
        "base best_value": base_meta_xgb_best["n_estimators"],
        "mixed best_value": mixed_meta_xgb_best["n_estimators"],
        "enhanced best_value": enhanced_meta_xgb_best["n_estimators"]
    },
    {
        "hyperparameter": "learning_rate",
        "search_range": "0.01 to 0.2, logarithmic",
        "base best_value": base_meta_xgb_best["learning_rate"],
        "mixed best_value": mixed_meta_xgb_best["learning_rate"],
        "enhanced best_value": enhanced_meta_xgb_best["learning_rate"]
    },
    {
        "hyperparameter": "max_depth",
        "search_range": "2 to 6",
        "base best_value": base_meta_xgb_best["max_depth"],
        "mixed best_value": mixed_meta_xgb_best["max_depth"],
        "enhanced best_value": enhanced_meta_xgb_best["max_depth"]
    },
    {
        "hyperparameter": "min_child_weight",
        "search_range": "1 to 30",
        "base best_value": base_meta_xgb_best["min_child_weight"],
        "mixed best_value": mixed_meta_xgb_best["min_child_weight"],
        "enhanced best_value": enhanced_meta_xgb_best["min_child_weight"]
    },
    {
        "hyperparameter": "gamma",
        "search_range": "0.0 to 5.0",
        "base best_value": base_meta_xgb_best["gamma"],
        "mixed best_value": mixed_meta_xgb_best["gamma"],
        "enhanced best_value": enhanced_meta_xgb_best["gamma"]
    },
    {
        "hyperparameter": "subsample",
        "search_range": "0.7 to 1.0",
        "base best_value": base_meta_xgb_best["subsample"],
        "mixed best_value": mixed_meta_xgb_best["subsample"],
        "enhanced best_value": enhanced_meta_xgb_best["subsample"]
    },
    {
        "hyperparameter": "colsample_bytree",
        "search_range": "0.7 to 1.0",
        "base best_value": base_meta_xgb_best["colsample_bytree"],
        "mixed best_value": mixed_meta_xgb_best["colsample_bytree"],
        "enhanced best_value": enhanced_meta_xgb_best["colsample_bytree"]
    },
    {
        "hyperparameter": "reg_alpha",
        "search_range": "0.0 to 5.0",
        "base best_value": base_meta_xgb_best["reg_alpha"],
        "mixed best_value": mixed_meta_xgb_best["reg_alpha"],
        "enhanced best_value": enhanced_meta_xgb_best["reg_alpha"]
    },
    {
        "hyperparameter": "reg_lambda",
        "search_range": "0.1 to 10.0, logarithmic",
        "base best_value": base_meta_xgb_best["reg_lambda"],
        "mixed best_value": mixed_meta_xgb_best["reg_lambda"],
        "enhanced best_value": enhanced_meta_xgb_best["reg_lambda"]
    }
])

meta_xgb_optuna_summary.to_csv(
    os.path.join(output_folder, "meta_xgb_optuna_parameters.csv"),
    index=False
)

print("Saved Meta XGBoost Optuna Parameters to:" ,output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END