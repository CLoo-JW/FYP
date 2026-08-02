import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import os

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
# META SVM
# ======================================================================================================================
# ----------------------------------------------------------------------------- Start
# META SVM SETUP
# ----------------------------------------------------------------------------- 
optuna_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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

base_extra_meta_features_train = np.hstack([base_svm_train_probabilities,
                                        base_nb_train_probabilities,
                                        base_roberta_train_probabilities,
                                        base_vader_train_scores,
                                        extra_train_meta_features])

base_extra_meta_features_test = np.hstack([base_svm_test_probabilities,
                                        base_nb_test_probabilities,
                                        base_roberta_test_probabilities,
                                        base_vader_test_scores,
                                        extra_test_meta_features])

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

base_extra_feature_names = base_feature_names + extra_feature_names

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

print("\n========== SVM XAI FEATURE CHECK ==========")
print("Base SVM XAI feature count:", len(base_feature_names))
print("Base Extra SVM XAI feature count:",len(base_extra_feature_names))
print("Mixed SVM XAI feature count:", len(mixed_feature_names))
print("Enhanced SVM XAI feature count:", len(enhanced_feature_names))
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# BASE META SVM
# ----------------------------------------------------------------------------- 
def base_meta_svm_optuna(trial):
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
        C=trial.suggest_float("C", 0.001, 10.0, log=True),
        max_iter=30000,
        dual=False,
        random_state=42
        ))
    ])

    scores = cross_val_score(
        svm_model,
        base_meta_features_train,
        sentiment_train,
        cv=optuna_cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    return scores.mean()

print("\n========== BASE META SVM MODEL ==========")
base_meta_svm_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
    )
base_meta_svm_study.optimize(
    base_meta_svm_optuna,
    n_trials=30
)
base_meta_svm_best = base_meta_svm_study.best_params

base_meta_svm = LinearSVC(
    C=base_meta_svm_best['C'],
    max_iter=30000,
    dual=False,
    random_state=42
)

base_meta_svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", base_meta_svm)
])
base_meta_svm_model.fit(base_meta_features_train, sentiment_train)
base_meta_svm_test_sentiment = base_meta_svm_model.predict(base_meta_features_test)
# ----------------------------------------------------------------------------- END

# -----------------------------------------------------------------------------
# BASE EXTRA META SVM
# -----------------------------------------------------------------------------
def base_extra_meta_svm_optuna(trial):
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=trial.suggest_float(
                "C",
                0.001,
                10.0,
                log=True
            ),
            max_iter=30000,
            dual=False,
            random_state=42
        ))
    ])

    scores = cross_val_score(
        svm_model,
        base_extra_meta_features_train,
        sentiment_train,
        cv=optuna_cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    return scores.mean()


print("\n========== BASE EXTRA META SVM MODEL ==========")

base_extra_meta_svm_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

base_extra_meta_svm_study.optimize(
    base_extra_meta_svm_optuna,
    n_trials=30
)

base_extra_meta_svm_best = base_extra_meta_svm_study.best_params

base_extra_meta_svm = LinearSVC(
    C=base_extra_meta_svm_best["C"],
    max_iter=30000,
    dual=False,
    random_state=42
)

base_extra_meta_svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", base_extra_meta_svm)
])

base_extra_meta_svm_model.fit(
    base_extra_meta_features_train,
    sentiment_train
)

base_extra_meta_svm_test_sentiment = (
    base_extra_meta_svm_model.predict(
        base_extra_meta_features_test
    )
)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# MIXED META SVM
# ----------------------------------------------------------------------------- 
def mixed_meta_svm_optuna(trial):
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
        C=trial.suggest_float("C", 0.001, 10.0, log=True),
        max_iter=30000,
        dual=False,
        random_state=42
        ))
    ])

    scores = cross_val_score(
        svm_model,
        mixed_meta_features_train,
        sentiment_train,
        cv=optuna_cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    return scores.mean()

print("\n========== MIXED META SVM MODEL ==========")
mixed_meta_svm_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
    )
mixed_meta_svm_study.optimize(
    mixed_meta_svm_optuna,
    n_trials=30
)
mixed_meta_svm_best = mixed_meta_svm_study.best_params

mixed_meta_svm = LinearSVC(
    C=mixed_meta_svm_best['C'],
    max_iter=30000,
    dual=False,
    random_state=42
)

mixed_meta_svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", mixed_meta_svm)
])
mixed_meta_svm_model.fit(mixed_meta_features_train, sentiment_train)
mixed_meta_svm_test_sentiment = mixed_meta_svm_model.predict(mixed_meta_features_test)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# ENHANCED META SVM
# ----------------------------------------------------------------------------- 
def enhanced_meta_svm_optuna(trial):
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
        C=trial.suggest_float("C", 0.001, 10.0, log=True),
        max_iter=30000,
        dual=False,
        random_state=42
        ))
    ])

    scores = cross_val_score(
        svm_model,
        enhanced_meta_features_train,
        sentiment_train,
        cv=optuna_cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    return scores.mean()

print("\n========== ENHANCED META SVM MODEL ==========")
enhanced_meta_svm_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
    )
enhanced_meta_svm_study.optimize(
    enhanced_meta_svm_optuna,
    n_trials=30
)
enhanced_meta_svm_best = enhanced_meta_svm_study.best_params

enhanced_meta_svm = LinearSVC(
    C=enhanced_meta_svm_best['C'],
    max_iter=30000,
    dual=False,
    random_state=42
)

enhanced_meta_svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", enhanced_meta_svm)
])
enhanced_meta_svm_model.fit(enhanced_meta_features_train, sentiment_train)
enhanced_meta_svm_test_sentiment = enhanced_meta_svm_model.predict(enhanced_meta_features_test)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# READABLE LINEAR SVM XAI FUNCTIONS
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
            + " (contribution: "
            + str(round(row[contribution_column], 4))
            + ")"
        )

    return ";\n".join(formatted_features)

def generate_coefficient_xai_charts(
        model,
        feature_values,
        feature_names,
        true_labels,
        output_folder,
        file_prefix,
        correct_row_index=None,
        incorrect_row_index=None,
        global_top_n=20,
        local_top_n=10
):
    os.makedirs(output_folder, exist_ok=True)

    scaler = model.named_steps["scaler"]

    if "lr" in model.named_steps:
        linear_model = model.named_steps["lr"]
        model_name = "Logistic Regression"
    elif "svm" in model.named_steps:
        linear_model = model.named_steps["svm"]
        model_name = "Linear SVM"
    else:
        raise ValueError(
            "The pipeline must contain either an 'lr' or 'svm' step."
        )

    scaled_features = scaler.transform(feature_values)

    predicted_labels = model.predict(feature_values)
    true_labels = np.asarray(true_labels)

    class_names = list(linear_model.classes_)

    predicted_class_indices = np.array([
        class_names.index(label)
        for label in predicted_labels
    ])

    predicted_class_coefficients = linear_model.coef_[
        predicted_class_indices
    ]

    # Each local contribution is:
    # standardized feature value × predicted-class coefficient
    contribution_matrix = (
        scaled_features * predicted_class_coefficients
    )

    readable_feature_names = [
        make_feature_name_readable(feature)
        for feature in feature_names
    ]

    feature_groups = np.array([
        get_meta_feature_group(feature)
        for feature in feature_names
    ])

    # =========================================================
    # 1. GLOBAL FEATURE CONTRIBUTIONS
    # =========================================================
    mean_absolute_contributions = np.mean(
        np.abs(contribution_matrix),
        axis=0
    )

    global_feature_df = pd.DataFrame({
        "feature": feature_names,
        "readable_feature": readable_feature_names,
        "feature_group": feature_groups,
        "mean_absolute_contribution":
            mean_absolute_contributions
    }).sort_values(
        "mean_absolute_contribution",
        ascending=False
    )

    global_feature_df.to_csv(
        os.path.join(
            output_folder,
            file_prefix
            + "_global_feature_contributions.csv"
        ),
        index=False
    )

    plotted_global_feature_df = (
        global_feature_df
        .head(global_top_n)
        .sort_values("mean_absolute_contribution")
    )

    plt.figure(
        figsize=(
            10,
            max(5, len(plotted_global_feature_df) * 0.4)
        )
    )

    plt.barh(
        plotted_global_feature_df["readable_feature"],
        plotted_global_feature_df[
            "mean_absolute_contribution"
        ]
    )

    plt.xlabel("Mean Absolute Coefficient Contribution")
    plt.ylabel("Meta-Feature")
    plt.title(
        model_name
        + ": Global Mean Absolute Contribution by Feature"
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            file_prefix
            + "_global_feature_contributions.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================
    # 2. GLOBAL GROUP CONTRIBUTIONS
    # =========================================================
    group_rows = []

    for feature_group in pd.unique(feature_groups):
        group_mask = feature_groups == feature_group

        row_group_contributions = contribution_matrix[
            :, group_mask
        ].sum(axis=1)

        group_rows.append({
            "feature_group": feature_group,
            "feature_count": int(group_mask.sum()),
            "mean_absolute_group_contribution": np.mean(
                np.abs(row_group_contributions)
            )
        })

    global_group_df = pd.DataFrame(group_rows).sort_values(
        "mean_absolute_group_contribution",
        ascending=False
    )

    global_group_df.to_csv(
        os.path.join(
            output_folder,
            file_prefix
            + "_global_group_contributions.csv"
        ),
        index=False
    )

    plotted_group_df = global_group_df.sort_values(
        "mean_absolute_group_contribution"
    )

    plt.figure(
        figsize=(
            9,
            max(4, len(plotted_group_df) * 0.7)
        )
    )

    plt.barh(
        plotted_group_df["feature_group"],
        plotted_group_df[
            "mean_absolute_group_contribution"
        ]
    )

    plt.xlabel("Mean Absolute Group Contribution")
    plt.ylabel("Feature Group")
    plt.title(
        model_name
        + ": Global Mean Absolute Contribution by Feature Group"
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            file_prefix
            + "_global_group_contributions.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================
    # SELECT CORRECT AND INCORRECT EXAMPLES
    # =========================================================
    correct_indices = np.flatnonzero(
        predicted_labels == true_labels
    )

    incorrect_indices = np.flatnonzero(
        predicted_labels != true_labels
    )

    if correct_row_index is None:
        if len(correct_indices) == 0:
            raise ValueError(
                "No correct prediction was found."
            )

        correct_row_index = int(correct_indices[0])

    if incorrect_row_index is None:
        if len(incorrect_indices) == 0:
            raise ValueError(
                "No incorrect prediction was found."
            )

        incorrect_row_index = int(incorrect_indices[0])

    # =========================================================
    # LOCAL CHART
    # =========================================================
    def save_local_contribution_chart(
            row_index,
            prediction_result
    ):
        local_df = pd.DataFrame({
            "feature": feature_names,
            "readable_feature": readable_feature_names,
            "feature_group": feature_groups,
            "raw_value": feature_values[row_index],
            "scaled_value": scaled_features[row_index],
            "coefficient":
                predicted_class_coefficients[row_index],
            "contribution":
                contribution_matrix[row_index],
            "absolute_contribution": np.abs(
                contribution_matrix[row_index]
            )
        })

        local_df = (
            local_df
            .sort_values(
                "absolute_contribution",
                ascending=False
            )
            .head(local_top_n)
            .sort_values("contribution")
        )

        local_df.to_csv(
            os.path.join(
                output_folder,
                file_prefix
                + "_local_"
                + prediction_result
                + "_row_"
                + str(row_index)
                + ".csv"
            ),
            index=False
        )

        bar_colours = [
            "tab:blue"
            if contribution >= 0
            else "tab:orange"
            for contribution in local_df["contribution"]
        ]

        plt.figure(
            figsize=(
                10,
                max(5, len(local_df) * 0.5)
            )
        )

        plt.barh(
            local_df["readable_feature"],
            local_df["contribution"],
            color=bar_colours
        )

        plt.axvline(
            x=0,
            linewidth=0.8,
            color="black"
        )

        plt.xlabel("Contribution to Predicted Class")
        plt.ylabel("Meta-Feature")

        plt.title(
            model_name
            + ": "
            + prediction_result.title()
            + " Prediction\n"
            + "True = "
            + str(true_labels[row_index])
            + ", Predicted = "
            + str(predicted_labels[row_index])
            + ", Row = "
            + str(row_index)
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_folder,
                file_prefix
                + "_local_"
                + prediction_result
                + "_row_"
                + str(row_index)
                + ".png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        return local_df

    correct_local_df = save_local_contribution_chart(
        row_index=correct_row_index,
        prediction_result="correct"
    )

    incorrect_local_df = save_local_contribution_chart(
        row_index=incorrect_row_index,
        prediction_result="incorrect"
    )

    print(
        "\nSaved coefficient-based XAI charts to:",
        output_folder
    )
    print("Correct example row:", correct_row_index)
    print("Incorrect example row:", incorrect_row_index)

    return {
        "global_feature_contributions":
            global_feature_df,
        "global_group_contributions":
            global_group_df,
        "correct_local_contributions":
            correct_local_df,
        "incorrect_local_contributions":
            incorrect_local_df,
        "correct_row_index":
            correct_row_index,
        "incorrect_row_index":
            incorrect_row_index
    }

def save_readable_local_svm_xai_report(
        model,
        feature_values,
        feature_names,
        test_split_df,
        true_labels,
        output_folder,
        file_prefix,
        row_indices,
        top_n=5
):
    scaler = model.named_steps["scaler"]
    svm_model = model.named_steps["svm"]

    scaled_features = scaler.transform(feature_values)

    predicted_labels = model.predict(feature_values)

    decision_scores = svm_model.decision_function(scaled_features)
    decision_confidences = softmax(decision_scores, axis=1)

    true_labels = np.asarray(true_labels)
    class_names = list(svm_model.classes_)

    readable_rows = []

    correct_text_report_blocks = []
    wrong_text_report_blocks = []

    for row_index in row_indices:
        review_text = test_split_df.iloc[row_index]["Text"]
        true_sentiment = true_labels[row_index]
        predicted_sentiment = predicted_labels[row_index]

        if true_sentiment == predicted_sentiment:
            prediction_result = "correct"
        else:
            prediction_result = "wrong"

        predicted_class_index = class_names.index(predicted_sentiment)

        prediction_decision_score = decision_scores[row_index][predicted_class_index]
        softmax_normalized_decision_score = decision_confidences[row_index][predicted_class_index]

        class_coefficients = svm_model.coef_[predicted_class_index]

        contributions = scaled_features[row_index] * class_coefficients

        local_xai_df = pd.DataFrame({
            "feature": feature_names,
            "readable_feature": [make_feature_name_readable(feature) for feature in feature_names],
            "feature_group": [get_meta_feature_group(feature) for feature in feature_names],
            "raw_value": feature_values[row_index],
            "scaled_value": scaled_features[row_index],
            "coefficient_for_predicted_class": class_coefficients,
            "contribution_to_predicted_class": contributions,
            "abs_contribution": np.abs(contributions)
        })

        supporting_features_df = (
            local_xai_df[local_xai_df["contribution_to_predicted_class"] > 0]
            .sort_values("contribution_to_predicted_class", ascending=False)
        )

        opposing_features_df = (
            local_xai_df[local_xai_df["contribution_to_predicted_class"] < 0]
            .sort_values("contribution_to_predicted_class", ascending=True)
        )

        group_contribution_df = (
            local_xai_df
            .groupby("feature_group")["contribution_to_predicted_class"]
            .sum()
            .reset_index()
            .sort_values("contribution_to_predicted_class", ascending=False)
        )

        positive_group_contribution_df = group_contribution_df[
            group_contribution_df["contribution_to_predicted_class"] > 0
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
            contribution_column="contribution_to_predicted_class",
            top_n=top_n
        )

        top_opposing_features = format_feature_contributions(
            opposing_features_df,
            contribution_column="contribution_to_predicted_class",
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
            "The meta SVM predicted '"
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
            "softmax_normalized_decision_score": round(softmax_normalized_decision_score, 4),
            "prediction_decision_score": round(prediction_decision_score, 4),
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
            + "\n\nsoftmax_normalized_decision_score:\n"
            + str(round(softmax_normalized_decision_score, 4))
            + "\n\nprediction_decision_score:\n"
            + str(round(prediction_decision_score, 4))
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
# EVALUATE META SVM
# ----------------------------------------------------------------------------- 
print("\nBASE META SVM BEST PARAMETERS: " + str(base_meta_svm_study.best_value))
print(base_meta_svm_best)
print("BASE META SVM MODEL ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_meta_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_meta_svm_test_sentiment, digits=4))

print("BASE EXTRA META SVM BEST PARAMETERS: " + str(base_extra_meta_svm_study.best_value))
print(base_extra_meta_svm_best)
print("BASE EXTRA META SVM MODEL ON TEST: ACCURACY = "
    + str(round(accuracy_score(sentiment_test, base_extra_meta_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_extra_meta_svm_test_sentiment, digits=4))

print("\nBASE VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_vader_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_vader_test_sentiment, digits=4))

print("\nBASE NAIVE BAYES ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_nb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_nb_test_sentiment, digits=4))

print("\nBASE SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_svm_test_sentiment, digits=4))

print("\nBASE RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_roberta_test_sentiment, digits=4))


print("\nMIXED META SVM BEST PARAMETERS: " + str(mixed_meta_svm_study.best_value))
print(mixed_meta_svm_best)
print("MIXED META SVM MODEL ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, mixed_meta_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, mixed_meta_svm_test_sentiment, digits=4))

print("\nMIXED VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_vader_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_vader_test_sentiment, digits=4))

print("\nMIXED NAIVE BAYES ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_nb_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_nb_test_sentiment, digits=4))

print("\nMIXED SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_svm_test_sentiment, digits=4))

print("\nMIXED RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_roberta_test_sentiment, digits=4))


print("\nENHANCED META SVM BEST PARAMETERS: " + str(enhanced_meta_svm_study.best_value))
print(enhanced_meta_svm_best)
print("ENHANCED META SVM MODEL ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_meta_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_meta_svm_test_sentiment, digits=4))

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
output_folder = "Meta_Model/XAI/SVM"
os.makedirs(output_folder, exist_ok=True)

base_readable_xai_report_df = save_readable_local_svm_xai_report(
    model=base_meta_svm_model,
    feature_values=base_meta_features_test,
    feature_names=base_feature_names,
    test_split_df=test_split_df,
    true_labels=sentiment_test,
    output_folder=output_folder,
    file_prefix="base_meta_svm",
    row_indices=list(range(len(sentiment_test))),
    top_n=5
)

print("\n========== BASE META SVM CORRECT XAI EXAMPLES ==========")
print(base_readable_xai_report_df[
    base_readable_xai_report_df["prediction_result"] == "correct"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "softmax_normalized_decision_score",
    "prediction_decision_score",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\n========== BASE META SVM WRONG XAI EXAMPLES ==========")
print(base_readable_xai_report_df[
    base_readable_xai_report_df["prediction_result"] == "wrong"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "softmax_normalized_decision_score",
    "prediction_decision_score",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))


base_extra_readable_xai_report_df = (
    save_readable_local_svm_xai_report(
        model=base_extra_meta_svm_model,
        feature_values=base_extra_meta_features_test,
        feature_names=base_extra_feature_names,
        test_split_df=test_split_df,
        true_labels=sentiment_test,
        output_folder=output_folder,
        file_prefix="base_extra_meta_svm",
        row_indices=list(range(len(sentiment_test))),
        top_n=5
    )
)

print("\n========== BASE EXTRA META SVM CORRECT XAI EXAMPLES ==========")

print(
    base_extra_readable_xai_report_df[
        base_extra_readable_xai_report_df[
            "prediction_result"
        ] == "correct"
    ][[
        "row_index",
        "true_sentiment",
        "predicted_sentiment",
        "softmax_normalized_decision_score",
        "prediction_decision_score",
        "strongest_individual_feature",
        "strongest_feature_group",
        "explanation"
    ]].head(5)
)

print("\n========== BASE EXTRA META SVM WRONG XAI EXAMPLES ==========")

print(
    base_extra_readable_xai_report_df[
        base_extra_readable_xai_report_df[
            "prediction_result"
        ] == "wrong"
    ][[
        "row_index",
        "true_sentiment",
        "predicted_sentiment",
        "softmax_normalized_decision_score",
        "prediction_decision_score",
        "strongest_individual_feature",
        "strongest_feature_group",
        "explanation"
    ]].head(5)
)


mixed_readable_xai_report_df = save_readable_local_svm_xai_report(
    model=mixed_meta_svm_model,
    feature_values=mixed_meta_features_test,
    feature_names=mixed_feature_names,
    test_split_df=test_split_df,
    true_labels=sentiment_test,
    output_folder=output_folder,
    file_prefix="mixed_meta_svm",
    row_indices=list(range(len(sentiment_test))),
    top_n=5
)

print("\n========== MIXED META SVM CORRECT XAI EXAMPLES ==========")
print(mixed_readable_xai_report_df[
    mixed_readable_xai_report_df["prediction_result"] == "correct"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "softmax_normalized_decision_score",
    "prediction_decision_score",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\n========== MIXED META SVM WRONG XAI EXAMPLES ==========")
print(mixed_readable_xai_report_df[
    mixed_readable_xai_report_df["prediction_result"] == "wrong"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "softmax_normalized_decision_score",
    "prediction_decision_score",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))


enhanced_readable_xai_report_df = save_readable_local_svm_xai_report(
    model=enhanced_meta_svm_model,
    feature_values=enhanced_meta_features_test,
    feature_names=enhanced_feature_names,
    test_split_df=test_split_df,
    true_labels=sentiment_test,
    output_folder=output_folder,
    file_prefix="enhanced_meta_svm",
    row_indices=list(range(len(sentiment_test))),
    top_n=5
)

print("\n========== ENHANCED META SVM CORRECT XAI EXAMPLES ==========")
print(enhanced_readable_xai_report_df[
    enhanced_readable_xai_report_df["prediction_result"] == "correct"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "softmax_normalized_decision_score",
    "prediction_decision_score",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\n========== ENHANCED META SVM WRONG XAI EXAMPLES ==========")
print(enhanced_readable_xai_report_df[
    enhanced_readable_xai_report_df["prediction_result"] == "wrong"
][[
    "row_index",
    "true_sentiment",
    "predicted_sentiment",
    "softmax_normalized_decision_score",
    "prediction_decision_score",
    "strongest_individual_feature",
    "strongest_feature_group",
    "explanation"
]].head(5))

print("\nSaved SVM XAI Report to:", output_folder)

output_folder = "Meta_Model/Classification_Report/SVM"
os.makedirs(output_folder, exist_ok=True)

base_meta_svm_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_meta_svm_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_extra_meta_svm_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_extra_meta_svm_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

mixed_meta_svm_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        mixed_meta_svm_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_meta_svm_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        enhanced_meta_svm_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_meta_svm_test_report_df.to_csv(
    os.path.join(output_folder, "base_meta_svm_test_classification_report.csv"),
    index_label="class"
)

base_extra_meta_svm_test_report_df.to_csv(
    os.path.join(
        output_folder,
        "base_extra_meta_svm_test_classification_report.csv"
    ),
    index_label="class"
)

mixed_meta_svm_test_report_df.to_csv(
    os.path.join(output_folder, "mixed_meta_svm_test_classification_report.csv"),
    index_label="class"
)

enhanced_meta_svm_test_report_df.to_csv(
    os.path.join(output_folder, "enhanced_meta_svm_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_meta_svm_test_report_df.values,
    rowLabels=base_meta_svm_test_report_df.index,
    colLabels=base_meta_svm_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_meta_svm_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_extra_meta_svm_test_report_df.values,
    rowLabels=base_extra_meta_svm_test_report_df.index,
    colLabels=base_extra_meta_svm_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(
        output_folder,
        "base_extra_meta_svm_test_classification_report.png"
    ),
    bbox_inches="tight",
    dpi=300
)

plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=mixed_meta_svm_test_report_df.values,
    rowLabels=mixed_meta_svm_test_report_df.index,
    colLabels=mixed_meta_svm_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "mixed_meta_svm_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_meta_svm_test_report_df.values,
    rowLabels=enhanced_meta_svm_test_report_df.index,
    colLabels=enhanced_meta_svm_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_meta_svm_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved SVM Classification Report to:", output_folder)

output_folder = "Meta_Model/Results/SVM"
os.makedirs(output_folder, exist_ok=True)

meta_svm_optuna_summary = pd.DataFrame([
    {
        "hyperparameter": "C",
        "search_range": "0.001 to 10.0, logarithmic",
        "base best_value": base_meta_svm_best["C"],
        "base_extra best_value": base_extra_meta_svm_best["C"],
        "mixed best_value": mixed_meta_svm_best["C"],
        "enhanced best_value": enhanced_meta_svm_best["C"]
    }
])

meta_svm_optuna_summary.to_csv(
    os.path.join(output_folder, "meta_svm_optuna_parameters.csv"),
    index=False
)

print("Saved Meta SVM Optuna Parameters to:" ,output_folder)


output_folder = "Meta_Model/XAI/SVM"
os.makedirs(output_folder, exist_ok=True)

base_svm_coefficient_xai = (
    generate_coefficient_xai_charts(
        model=base_meta_svm_model,
        feature_values=base_meta_features_test,
        feature_names=base_feature_names,
        true_labels=sentiment_test,
        output_folder=output_folder,
        file_prefix="base_meta_svm",
        global_top_n=20,
        local_top_n=10
    )
)

base_extra_svm_coefficient_xai = (
    generate_coefficient_xai_charts(
        model=base_extra_meta_svm_model,
        feature_values=base_extra_meta_features_test,
        feature_names=base_extra_feature_names,
        true_labels=sentiment_test,
        output_folder=output_folder,
        file_prefix="base_extra_meta_svm",
        global_top_n=20,
        local_top_n=10
    )
)

mixed_svm_coefficient_xai = (
    generate_coefficient_xai_charts(
        model=mixed_meta_svm_model,
        feature_values=mixed_meta_features_test,
        feature_names=mixed_feature_names,
        true_labels=sentiment_test,
        output_folder=output_folder,
        file_prefix="mixed_meta_svm",
        global_top_n=20,
        local_top_n=10
    )
)

enhanced_svm_coefficient_xai = (
    generate_coefficient_xai_charts(
        model=enhanced_meta_svm_model,
        feature_values=enhanced_meta_features_test,
        feature_names=enhanced_feature_names,
        true_labels=sentiment_test,
        output_folder=output_folder,
        file_prefix="enhanced_meta_svm",
        global_top_n=20,
        local_top_n=10
    )
)

print(
    "Saved All Meta SVM XAI Charts to:",
    output_folder
)


output_folder = "Meta_Model/Results/SVM"
os.makedirs(output_folder, exist_ok=True)

base_meta_svm_test_sentiment_df = pd.DataFrame({
    "base_meta_svm_test_sentiment": base_meta_svm_test_sentiment
})

base_meta_svm_test_sentiment_df.to_csv(
    os.path.join(output_folder, "base_meta_svm_test_sentiment.csv"),
    index=False
)

base_extra_meta_svm_test_sentiment_df = pd.DataFrame({
    "base_extra_meta_svm_test_sentiment": base_extra_meta_svm_test_sentiment
})

base_extra_meta_svm_test_sentiment_df.to_csv(
    os.path.join(output_folder, "base_extra_meta_svm_test_sentiment.csv"),
    index=False
)

mixed_meta_svm_test_sentiment_df = pd.DataFrame({
    "mixed_meta_svm_test_sentiment": mixed_meta_svm_test_sentiment
})

mixed_meta_svm_test_sentiment_df.to_csv(
    os.path.join(output_folder, "mixed_meta_svm_test_sentiment.csv"),
    index=False
)

enhanced_meta_svm_test_sentiment_df = pd.DataFrame({
    "enhanced_meta_svm_test_sentiment": enhanced_meta_svm_test_sentiment
})

enhanced_meta_svm_test_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_meta_svm_test_sentiment.csv"),
    index=False
)
print("Saved Meta SVM Sentiments to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END