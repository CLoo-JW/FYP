import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from scipy.stats import binomtest
import os

# =============================================================================
# LOAD TRUE TEST LABELS
# =============================================================================

test_split_df = pd.read_csv("Dataset/Preprocessed/test_split.csv")
sentiment_test = test_split_df["Sentiment"]


# =============================================================================
# LOAD LOGISTIC REGRESSION PREDICTIONS
# =============================================================================

base_meta_logistic_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/LG/base_meta_logistic_test_sentiment.csv"
)
base_meta_logistic_test_sentiment = (
    base_meta_logistic_test_sentiment_df[
        "base_meta_logistic_test_sentiment"
    ]
)

base_extra_meta_logistic_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/LG/base_extra_meta_logistic_test_sentiment.csv"
)
base_extra_meta_logistic_test_sentiment = (
    base_extra_meta_logistic_test_sentiment_df[
        "base_extra_meta_logistic_test_sentiment"
    ]
)

mixed_meta_logistic_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/LG/mixed_meta_logistic_test_sentiment.csv"
)
mixed_meta_logistic_test_sentiment = (
    mixed_meta_logistic_test_sentiment_df[
        "mixed_meta_logistic_test_sentiment"
    ]
)

enhanced_meta_logistic_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/LG/enhanced_meta_logistic_test_sentiment.csv"
)
enhanced_meta_logistic_test_sentiment = (
    enhanced_meta_logistic_test_sentiment_df[
        "enhanced_meta_logistic_test_sentiment"
    ]
)


# =============================================================================
# LOAD RANDOM FOREST PREDICTIONS
# =============================================================================

base_meta_rf_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/RF/base_meta_rf_test_sentiment.csv"
)
base_meta_rf_test_sentiment = (
    base_meta_rf_test_sentiment_df[
        "base_meta_rf_test_sentiment"
    ]
)

base_extra_meta_rf_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/RF/base_extra_meta_rf_test_sentiment.csv"
)
base_extra_meta_rf_test_sentiment = (
    base_extra_meta_rf_test_sentiment_df[
        "base_extra_meta_rf_test_sentiment"
    ]
)

mixed_meta_rf_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/RF/mixed_meta_rf_test_sentiment.csv"
)
mixed_meta_rf_test_sentiment = (
    mixed_meta_rf_test_sentiment_df[
        "mixed_meta_rf_test_sentiment"
    ]
)

enhanced_meta_rf_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/RF/enhanced_meta_rf_test_sentiment.csv"
)
enhanced_meta_rf_test_sentiment = (
    enhanced_meta_rf_test_sentiment_df[
        "enhanced_meta_rf_test_sentiment"
    ]
)


# =============================================================================
# LOAD SVM PREDICTIONS
# =============================================================================

base_meta_svm_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/SVM/base_meta_svm_test_sentiment.csv"
)
base_meta_svm_test_sentiment = (
    base_meta_svm_test_sentiment_df[
        "base_meta_svm_test_sentiment"
    ]
)

base_extra_meta_svm_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/SVM/base_extra_meta_svm_test_sentiment.csv"
)
base_extra_meta_svm_test_sentiment = (
    base_extra_meta_svm_test_sentiment_df[
        "base_extra_meta_svm_test_sentiment"
    ]
)

mixed_meta_svm_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/SVM/mixed_meta_svm_test_sentiment.csv"
)
mixed_meta_svm_test_sentiment = (
    mixed_meta_svm_test_sentiment_df[
        "mixed_meta_svm_test_sentiment"
    ]
)

enhanced_meta_svm_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/SVM/enhanced_meta_svm_test_sentiment.csv"
)
enhanced_meta_svm_test_sentiment = (
    enhanced_meta_svm_test_sentiment_df[
        "enhanced_meta_svm_test_sentiment"
    ]
)


# =============================================================================
# LOAD XGBOOST PREDICTIONS
# =============================================================================

base_meta_xgb_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/XGB/base_meta_xgb_test_sentiment.csv"
)
base_meta_xgb_test_sentiment = (
    base_meta_xgb_test_sentiment_df[
        "base_meta_xgb_test_sentiment"
    ]
)

base_extra_meta_xgb_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/XGB/base_extra_meta_xgb_test_sentiment.csv"
)
base_extra_meta_xgb_test_sentiment = (
    base_extra_meta_xgb_test_sentiment_df[
        "base_extra_meta_xgb_test_sentiment"
    ]
)

mixed_meta_xgb_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/XGB/mixed_meta_xgb_test_sentiment.csv"
)
mixed_meta_xgb_test_sentiment = (
    mixed_meta_xgb_test_sentiment_df[
        "mixed_meta_xgb_test_sentiment"
    ]
)

enhanced_meta_xgb_test_sentiment_df = pd.read_csv(
    "Meta_Model/Results/XGB/enhanced_meta_xgb_test_sentiment.csv"
)
enhanced_meta_xgb_test_sentiment = (
    enhanced_meta_xgb_test_sentiment_df[
        "enhanced_meta_xgb_test_sentiment"
    ]
)


# =============================================================================
# LOAD BASE LEARNER PREDICTIONS
# =============================================================================

base_nb_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/NB/Base/base_nb_test_sentiment.csv"
)
base_nb_test_sentiment = (
    base_nb_test_sentiment_df[
        "base_nb_sentiment"
    ]
)
enhanced_nb_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/NB/Enhanced/enhanced_nb_test_sentiment.csv"
)
enhanced_nb_test_sentiment = (
    enhanced_nb_test_sentiment_df[
        "enhanced_nb_sentiment"
    ]
)

base_svm_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/SVM/Base/base_svm_test_sentiment.csv"
)
base_svm_test_sentiment = (
    base_svm_test_sentiment_df[
        "base_svm_sentiment"
    ]
)
enhanced_svm_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/SVM/Enhanced/enhanced_svm_test_sentiment.csv"
)
enhanced_svm_test_sentiment = (
    enhanced_svm_test_sentiment_df[
        "enhanced_svm_sentiment"
    ]
)

base_roberta_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/RoBERTa/Base/base_roberta_test_sentiment.csv"
)
base_roberta_test_sentiment = (
    base_roberta_test_sentiment_df[
        "base_roberta_sentiment"
    ]
)
enhanced_roberta_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/RoBERTa/Enhanced/enhanced_roberta_test_sentiment.csv"
)
enhanced_roberta_test_sentiment = (
    enhanced_roberta_test_sentiment_df[
        "enhanced_roberta_sentiment"
    ]
)

base_vader_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/VADER/Base/base_vader_test_sentiment.csv"
)
base_vader_test_sentiment = (
    base_vader_test_sentiment_df[
        "base_vader_sentiment"
    ]
)
enhanced_vader_test_sentiment_df = pd.read_csv(
    "Base_Learner/Results/VADER/Enhanced/enhanced_vader_test_sentiment.csv"
)
enhanced_vader_test_sentiment = (
    enhanced_vader_test_sentiment_df[
        "enhanced_vader_sentiment"
    ]
)


# =============================================================================
# OUTPUT FOLDER
# =============================================================================

output_folder = "Model_Comparison/Results"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "meta_configuration_audit.txt"
)

confusion_matrix_folder = os.path.join(
    output_folder,
    "Confusion_Matrices"
)

base_confusion_matrix_folder = os.path.join(
    confusion_matrix_folder,
    "Base_Learners"
)

meta_confusion_matrix_folder = os.path.join(
    confusion_matrix_folder,
    "Meta_Learners"
)

os.makedirs(
    base_confusion_matrix_folder,
    exist_ok=True
)

os.makedirs(
    meta_confusion_matrix_folder,
    exist_ok=True
)


# =============================================================================
# AUDIT FUNCTION
# =============================================================================

def audit_against_baseline(
    model_name,
    comparison_name,
    true_labels,
    baseline_predictions,
    comparison_predictions
):
    y_true = np.asarray(true_labels)
    y_base = np.asarray(baseline_predictions)
    y_compare = np.asarray(comparison_predictions)

    lengths = {
        "true_labels": len(y_true),
        "baseline_predictions": len(y_base),
        "comparison_predictions": len(y_compare)
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Row-count mismatch for {model_name}, "
            f"{comparison_name}: {lengths}"
        )

    if pd.isna(y_true).any():
        raise ValueError(
            "The true test labels contain missing values."
        )

    if pd.isna(y_base).any():
        raise ValueError(
            f"{model_name} baseline predictions contain missing values."
        )

    if pd.isna(y_compare).any():
        raise ValueError(
            f"{model_name} {comparison_name} predictions contain "
            f"missing values."
        )

    baseline_correct_mask = y_base == y_true
    comparison_correct_mask = y_compare == y_true
    same_prediction_mask = y_base == y_compare

    corrected_mask = (
        (~baseline_correct_mask)
        & comparison_correct_mask
    )

    harmed_mask = (
        baseline_correct_mask
        & (~comparison_correct_mask)
    )

    changed_still_wrong_mask = (
        (~baseline_correct_mask)
        & (~comparison_correct_mask)
        & (~same_prediction_mask)
    )

    unchanged_correct_mask = (
        baseline_correct_mask
        & comparison_correct_mask
        & same_prediction_mask
    )

    unchanged_wrong_mask = (
        (~baseline_correct_mask)
        & (~comparison_correct_mask)
        & same_prediction_mask
    )

    corrected = int(corrected_mask.sum())
    harmed = int(harmed_mask.sum())
    changed_still_wrong = int(
        changed_still_wrong_mask.sum()
    )
    unchanged_correct = int(
        unchanged_correct_mask.sum()
    )
    unchanged_wrong = int(
        unchanged_wrong_mask.sum()
    )

    net_corrections = corrected - harmed

    baseline_correct_count = int(
        baseline_correct_mask.sum()
    )
    comparison_correct_count = int(
        comparison_correct_mask.sum()
    )

    correct_count_change = (
        comparison_correct_count
        - baseline_correct_count
    )

    if net_corrections != correct_count_change:
        raise AssertionError(
            f"Audit check failed for {model_name}, "
            f"{comparison_name}. "
            f"Net corrections={net_corrections}, "
            f"correct-count change={correct_count_change}."
        )

    total_reviews = len(y_true)

    accounted_total = (
        corrected
        + harmed
        + changed_still_wrong
        + unchanged_correct
        + unchanged_wrong
    )

    if accounted_total != total_reviews:
        raise AssertionError(
            f"Audit categories account for {accounted_total} "
            f"reviews, but {total_reviews} were expected."
        )

    baseline_accuracy = accuracy_score(
        y_true,
        y_base
    )

    comparison_accuracy = accuracy_score(
        y_true,
        y_compare
    )

    baseline_macro_f1 = f1_score(
        y_true,
        y_base,
        average="macro"
    )

    comparison_macro_f1 = f1_score(
        y_true,
        y_compare,
        average="macro"
    )

    total_changed = int(
        (~same_prediction_mask).sum()
    )

    total_unchanged = int(
        same_prediction_mask.sum()
    )

    return {
        "Model": model_name,
        "Comparison": f"{comparison_name} vs Baseline",
        "Total Reviews": total_reviews,
        "Baseline Correct": baseline_correct_count,
        "Comparison Correct": comparison_correct_count,
        "Corrected": corrected,
        "Harmed": harmed,
        "Net Corrections": net_corrections,
        "Changed But Still Wrong": changed_still_wrong,
        "Unchanged Correct": unchanged_correct,
        "Unchanged Wrong": unchanged_wrong,
        "Total Predictions Changed": total_changed,
        "Total Predictions Unchanged": total_unchanged,
        "Baseline Accuracy": baseline_accuracy,
        "Comparison Accuracy": comparison_accuracy,
        "Accuracy Change": (
            comparison_accuracy
            - baseline_accuracy
        ),
        "Accuracy Change Percentage Points": (
            comparison_accuracy
            - baseline_accuracy
        ) * 100,
        "Baseline Macro F1": baseline_macro_f1,
        "Comparison Macro F1": comparison_macro_f1,
        "Macro F1 Change": (
            comparison_macro_f1
            - baseline_macro_f1
        ),
        "Macro F1 Change Percentage Points": (
            comparison_macro_f1
            - baseline_macro_f1
        ) * 100
    }


# =============================================================================
# GROUP PREDICTIONS BY BASE LEARNER
# =============================================================================

base_learner_predictions = {
    "Naive Bayes": {
        "Base": base_nb_test_sentiment,
        "Enhanced": enhanced_nb_test_sentiment
    },
    "SVM": {
        "Base": base_svm_test_sentiment,
        "Enhanced": enhanced_svm_test_sentiment
    },
    "RoBERTa": {
        "Base": base_roberta_test_sentiment,
        "Enhanced": enhanced_roberta_test_sentiment
    },
    "VADER": {
        "Base": base_vader_test_sentiment,
        "Enhanced": enhanced_vader_test_sentiment
    }
}


# =============================================================================
# GROUP PREDICTIONS BY META LEARNER
# =============================================================================

meta_model_predictions = {
    "Logistic Regression": {
        "Baseline": base_meta_logistic_test_sentiment,
        "Baseline Extra": base_extra_meta_logistic_test_sentiment,
        "Mixed": mixed_meta_logistic_test_sentiment,
        "Enhanced": enhanced_meta_logistic_test_sentiment
    },
    "Random Forest": {
        "Baseline": base_meta_rf_test_sentiment,
        "Baseline Extra": base_extra_meta_rf_test_sentiment,
        "Mixed": mixed_meta_rf_test_sentiment,
        "Enhanced": enhanced_meta_rf_test_sentiment
    },
    "SVM": {
        "Baseline": base_meta_svm_test_sentiment,
        "Baseline Extra": base_extra_meta_svm_test_sentiment,
        "Mixed": mixed_meta_svm_test_sentiment,
        "Enhanced": enhanced_meta_svm_test_sentiment
    },
    "XGBoost": {
        "Baseline": base_meta_xgb_test_sentiment,
        "Baseline Extra": base_extra_meta_xgb_test_sentiment,
        "Mixed": mixed_meta_xgb_test_sentiment,
        "Enhanced": enhanced_meta_xgb_test_sentiment
    }
}


# =============================================================================
# CONFUSION MATRIX FUNCTIONS
# =============================================================================

LABEL_NORMALISATION_MAP = {
    "negative": "neg",
    "neg": "neg",
    "0": "neg",
    "neutral": "neu",
    "neu": "neu",
    "1": "neu",
    "positive": "pos",
    "pos": "pos",
    "2": "pos"
}

CONFUSION_MATRIX_LABELS = [
    "neg",
    "neu",
    "pos"
]

CONFUSION_MATRIX_DISPLAY_LABELS = [
    "Negative",
    "Neutral",
    "Positive"
]


def normalise_confusion_matrix_labels(
    values,
    source_name
):

    values_series = pd.Series(
        np.asarray(values)
    )

    cleaned_values = (
        values_series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    normalised_values = cleaned_values.map(
        LABEL_NORMALISATION_MAP
    )

    invalid_mask = normalised_values.isna()

    if invalid_mask.any():
        invalid_values = sorted(
            cleaned_values[
                invalid_mask
            ].unique().tolist()
        )

        raise ValueError(
            f"Unsupported sentiment labels in {source_name}: "
            f"{invalid_values}"
        )

    return normalised_values.to_numpy()


def make_safe_filename(value):

    return (
        value
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_confusion_matrix(
    learner_type,
    model_name,
    configuration_name,
    true_labels,
    predicted_labels,
    destination_folder
):

    y_true = normalise_confusion_matrix_labels(
        true_labels,
        "true test labels"
    )

    y_pred = normalise_confusion_matrix_labels(
        predicted_labels,
        f"{model_name} {configuration_name} predictions"
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Row-count mismatch for {model_name}, "
            f"{configuration_name}: "
            f"true labels={len(y_true)}, "
            f"predictions={len(y_pred)}"
        )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CONFUSION_MATRIX_LABELS
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CONFUSION_MATRIX_DISPLAY_LABELS
    )

    matrix_display.plot(
        ax=axis,
        values_format="d",
        colorbar=False
    )

    axis.set_title(
        f"{learner_type}: {model_name} - "
        f"{configuration_name}"
    )

    axis.set_xlabel(
        "Predicted sentiment"
    )

    axis.set_ylabel(
        "True sentiment"
    )

    figure.tight_layout()

    model_filename = make_safe_filename(
        model_name
    )

    configuration_filename = make_safe_filename(
        configuration_name
    )

    image_filename = (
        f"{model_filename}_"
        f"{configuration_filename}_"
        f"confusion_matrix.png"
    )

    image_path = os.path.join(
        destination_folder,
        image_filename
    )

    figure.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    return {
        "Learner Type": learner_type,
        "Model": model_name,
        "Configuration": configuration_name,
        "Matrix": matrix,
        "Image Path": image_path
    }


# =============================================================================
# RUN ALL 12 AUDITS
# =============================================================================

audit_results = []

for model_name, predictions in meta_model_predictions.items():
    baseline_predictions = predictions["Baseline"]

    for comparison_name in [
        "Baseline Extra",
        "Mixed",
        "Enhanced"
    ]:
        result = audit_against_baseline(
            model_name=model_name,
            comparison_name=comparison_name,
            true_labels=sentiment_test,
            baseline_predictions=baseline_predictions,
            comparison_predictions=predictions[
                comparison_name
            ]
        )

        audit_results.append(result)


audit_df = pd.DataFrame(audit_results)


# =============================================================================
# ROUND DECIMAL RESULTS
# =============================================================================

decimal_columns = [
    "Baseline Accuracy",
    "Comparison Accuracy",
    "Accuracy Change",
    "Accuracy Change Percentage Points",
    "Baseline Macro F1",
    "Comparison Macro F1",
    "Macro F1 Change",
    "Macro F1 Change Percentage Points"
]

audit_df[decimal_columns] = (
    audit_df[decimal_columns].round(6)
)


# =============================================================================
# SAVE ALL RESULTS TO ONE TEXT FILE
# =============================================================================

summary_columns = [
    "Model",
    "Comparison",
    "Corrected",
    "Harmed",
    "Net Corrections",
    "Changed But Still Wrong",
    "Unchanged Correct",
    "Unchanged Wrong",
    "Accuracy Change Percentage Points",
    "Macro F1 Change Percentage Points"
]

with open(output_file, "w", encoding="utf-8") as file:
    file.write(
        "META-LEARNER CONFIGURATION AUDIT\n"
    )
    file.write(
        "=" * 120 + "\n\n"
    )

    file.write(
        "Reference configuration: Baseline\n\n"
    )

    file.write(
        "Corrected: Baseline was wrong and the compared "
        "configuration was correct.\n"
    )

    file.write(
        "Harmed: Baseline was correct and the compared "
        "configuration was wrong.\n"
    )

    file.write(
        "Net Corrections: Corrected - Harmed.\n"
    )

    file.write(
        "Changed But Still Wrong: Both configurations were "
        "wrong, but predicted different incorrect classes.\n"
    )

    file.write(
        "Unchanged Correct: Both configurations produced the "
        "same correct prediction.\n"
    )

    file.write(
        "Unchanged Wrong: Both configurations produced the "
        "same incorrect prediction.\n\n"
    )

    file.write(
        "SUMMARY OF ALL 12 COMPARISONS\n"
    )
    file.write(
        "-" * 120 + "\n"
    )

    file.write(
        audit_df[summary_columns].to_string(
            index=False
        )
    )

    file.write("\n\n")
    file.write(
        "=" * 120 + "\n"
    )
    file.write(
        "DETAILED RESULTS BY META LEARNER\n"
    )
    file.write(
        "=" * 120 + "\n"
    )

    for model_name in meta_model_predictions:
        model_results = audit_df[
            audit_df["Model"] == model_name
        ]

        file.write(
            f"\n\n{model_name.upper()}\n"
        )
        file.write(
            "-" * 120 + "\n"
        )

        for _, row in model_results.iterrows():
            file.write(
                f"\nComparison: {row['Comparison']}\n"
            )
            file.write(
                f"Total reviews: {row['Total Reviews']}\n"
            )
            file.write(
                f"Baseline correct: "
                f"{row['Baseline Correct']}\n"
            )
            file.write(
                f"Comparison correct: "
                f"{row['Comparison Correct']}\n"
            )
            file.write(
                f"Corrected: {row['Corrected']}\n"
            )
            file.write(
                f"Harmed: {row['Harmed']}\n"
            )
            file.write(
                f"Net corrections: "
                f"{row['Net Corrections']}\n"
            )
            file.write(
                f"Changed but still wrong: "
                f"{row['Changed But Still Wrong']}\n"
            )
            file.write(
                f"Unchanged correct: "
                f"{row['Unchanged Correct']}\n"
            )
            file.write(
                f"Unchanged wrong: "
                f"{row['Unchanged Wrong']}\n"
            )
            file.write(
                f"Total predictions changed: "
                f"{row['Total Predictions Changed']}\n"
            )
            file.write(
                f"Total predictions unchanged: "
                f"{row['Total Predictions Unchanged']}\n"
            )
            file.write(
                f"Baseline accuracy: "
                f"{row['Baseline Accuracy']:.6f}\n"
            )
            file.write(
                f"Comparison accuracy: "
                f"{row['Comparison Accuracy']:.6f}\n"
            )
            file.write(
                f"Accuracy change: "
                f"{row['Accuracy Change']:.6f}\n"
            )
            file.write(
                f"Accuracy change percentage points: "
                f"{row['Accuracy Change Percentage Points']:.6f}\n"
            )
            file.write(
                f"Baseline macro F1: "
                f"{row['Baseline Macro F1']:.6f}\n"
            )
            file.write(
                f"Comparison macro F1: "
                f"{row['Comparison Macro F1']:.6f}\n"
            )
            file.write(
                f"Macro F1 change: "
                f"{row['Macro F1 Change']:.6f}\n"
            )
            file.write(
                f"Macro F1 change percentage points: "
                f"{row['Macro F1 Change Percentage Points']:.6f}\n"
            )


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print(
    "\n========== META CONFIGURATION AUDIT =========="
)

print(
    audit_df[summary_columns].to_string(
        index=False
    )
)

print(
    f"\nAudit saved to:\n{output_file}"
)

# =============================================================================
# CREATE BASE-LEARNER CONFUSION MATRICES
# 4 BASE LEARNERS x 2 CONFIGURATIONS = 8 MATRICES
# =============================================================================

base_confusion_matrix_results = []

for model_name, configurations in base_learner_predictions.items():
    for configuration_name, predictions in configurations.items():
        matrix_result = save_confusion_matrix(
            learner_type="Base Learner",
            model_name=model_name,
            configuration_name=configuration_name,
            true_labels=sentiment_test,
            predicted_labels=predictions,
            destination_folder=base_confusion_matrix_folder
        )

        base_confusion_matrix_results.append(
            matrix_result
        )


# =============================================================================
# CREATE META-LEARNER CONFUSION MATRICES
# 4 META LEARNERS x 4 CONFIGURATIONS = 16 MATRICES
# =============================================================================

meta_confusion_matrix_results = []

for model_name, configurations in meta_model_predictions.items():
    for configuration_name, predictions in configurations.items():
        matrix_result = save_confusion_matrix(
            learner_type="Meta Learner",
            model_name=model_name,
            configuration_name=configuration_name,
            true_labels=sentiment_test,
            predicted_labels=predictions,
            destination_folder=meta_confusion_matrix_folder
        )

        meta_confusion_matrix_results.append(
            matrix_result
        )


all_confusion_matrix_results = (
    base_confusion_matrix_results
    + meta_confusion_matrix_results
)


# =============================================================================
# APPEND ALL RAW CONFUSION-MATRIX COUNTS TO THE AUDIT TEXT FILE
# =============================================================================

with open(
    output_file,
    "a",
    encoding="utf-8"
) as file:
    file.write("\n\n")
    file.write(
        "=" * 120 + "\n"
    )
    file.write(
        "BASE- AND META-LEARNER CONFUSION MATRICES\n"
    )
    file.write(
        "=" * 120 + "\n"
    )

    file.write(
        "\nRows represent true sentiment classes. "
        "Columns represent predicted sentiment classes.\n"
    )

    for result in all_confusion_matrix_results:
        matrix_df = pd.DataFrame(
            result["Matrix"],
            index=[
                "True Negative",
                "True Neutral",
                "True Positive"
            ],
            columns=[
                "Predicted Negative",
                "Predicted Neutral",
                "Predicted Positive"
            ]
        )

        file.write(
            f"\n\n{result['Learner Type']} - "
            f"{result['Model']} - "
            f"{result['Configuration']}\n"
        )

        file.write(
            "-" * 120 + "\n"
        )

        file.write(
            matrix_df.to_string()
        )

        file.write(
            f"\n\nImage saved to: "
            f"{result['Image Path']}\n"
        )


# =============================================================================
# DISPLAY CONFUSION-MATRIX OUTPUT LOCATIONS
# =============================================================================

print(
    f"\nSaved {len(base_confusion_matrix_results)} "
    f"base-learner confusion matrices to:\n"
    f"{base_confusion_matrix_folder}"
)

print(
    f"\nSaved {len(meta_confusion_matrix_results)} "
    f"meta-learner confusion matrices to:\n"
    f"{meta_confusion_matrix_folder}"
)

print(
    f"\nSaved {len(all_confusion_matrix_results)} "
    f"confusion matrices in total."
)

print(
    "\nThe raw confusion-matrix counts were also appended to:\n"
    f"{output_file}"
)

# =============================================================================
# EXACT MCNEMAR TEST WITH BONFERRONI CORRECTION
# =============================================================================

def exact_mcnemar_comparison(
    learner_type,
    model_name,
    reference_name,
    comparison_name,
    true_labels,
    reference_predictions,
    comparison_predictions
):

    y_true = normalise_confusion_matrix_labels(
        true_labels,
        "true test labels"
    )

    y_reference = normalise_confusion_matrix_labels(
        reference_predictions,
        f"{model_name} {reference_name} predictions"
    )

    y_comparison = normalise_confusion_matrix_labels(
        comparison_predictions,
        f"{model_name} {comparison_name} predictions"
    )

    lengths = {
        "true_labels": len(y_true),
        "reference_predictions": len(y_reference),
        "comparison_predictions": len(y_comparison)
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Row-count mismatch for {model_name}, "
            f"{comparison_name} vs {reference_name}: {lengths}"
        )

    reference_correct = (
        y_reference == y_true
    )

    comparison_correct = (
        y_comparison == y_true
    )

    corrected = int(
        (
            (~reference_correct)
            & comparison_correct
        ).sum()
    )

    harmed = int(
        (
            reference_correct
            & (~comparison_correct)
        ).sum()
    )

    discordant_pairs = (
        corrected + harmed
    )

    if discordant_pairs == 0:
        exact_p_value = 1.0
    else:
        exact_p_value = binomtest(
            k=min(corrected, harmed),
            n=discordant_pairs,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    net_corrections = (
        corrected - harmed
    )

    if net_corrections > 0:
        direction = (
            "Comparison produced more corrections than harms"
        )
    elif net_corrections < 0:
        direction = (
            "Comparison produced more harms than corrections"
        )
    else:
        direction = (
            "Corrections and harms were equal"
        )

    return {
        "Learner Type": learner_type,
        "Model": model_name,
        "Reference": reference_name,
        "Comparison Configuration": comparison_name,
        "Comparison": (
            f"{comparison_name} vs {reference_name}"
        ),
        "Corrected": corrected,
        "Harmed": harmed,
        "Discordant Pairs": discordant_pairs,
        "Net Corrections": net_corrections,
        "Exact McNemar P-Value": exact_p_value,
        "Direction": direction
    }


# =============================================================================
# BASE-LEARNER MCNEMAR TESTS
# 4 TESTS: ENHANCED VS BASE
# =============================================================================

base_mcnemar_results = []

for model_name, configurations in base_learner_predictions.items():
    result = exact_mcnemar_comparison(
        learner_type="Base Learner",
        model_name=model_name,
        reference_name="Base",
        comparison_name="Enhanced",
        true_labels=sentiment_test,
        reference_predictions=configurations["Base"],
        comparison_predictions=configurations["Enhanced"]
    )

    base_mcnemar_results.append(
        result
    )


base_mcnemar_df = pd.DataFrame(
    base_mcnemar_results
)

base_alpha = 0.05
base_number_of_tests = len(
    base_mcnemar_df
)

base_bonferroni_threshold = (
    base_alpha / base_number_of_tests
)

base_mcnemar_df[
    "Comparison Family Size"
] = base_number_of_tests

base_mcnemar_df[
    "Bonferroni Threshold"
] = base_bonferroni_threshold

base_mcnemar_df[
    "Bonferroni-Adjusted P-Value"
] = np.minimum(
    base_mcnemar_df[
        "Exact McNemar P-Value"
    ] * base_number_of_tests,
    1.0
)

base_mcnemar_df[
    "Significant at Alpha 0.05"
] = (
    base_mcnemar_df[
        "Exact McNemar P-Value"
    ] < base_alpha
)

base_mcnemar_df[
    "Significant After Bonferroni"
] = (
    base_mcnemar_df[
        "Exact McNemar P-Value"
    ] < base_bonferroni_threshold
)


# =============================================================================
# META-LEARNER MCNEMAR TESTS
# 12 TESTS: EACH ALTERNATIVE CONFIGURATION VS BASELINE
# =============================================================================

meta_mcnemar_results = []

for model_name, configurations in meta_model_predictions.items():
    reference_predictions = configurations[
        "Baseline"
    ]

    for comparison_name in [
        "Baseline Extra",
        "Mixed",
        "Enhanced"
    ]:
        result = exact_mcnemar_comparison(
            learner_type="Meta Learner",
            model_name=model_name,
            reference_name="Baseline",
            comparison_name=comparison_name,
            true_labels=sentiment_test,
            reference_predictions=reference_predictions,
            comparison_predictions=configurations[
                comparison_name
            ]
        )

        meta_mcnemar_results.append(
            result
        )


meta_mcnemar_df = pd.DataFrame(
    meta_mcnemar_results
)

meta_alpha = 0.05
meta_number_of_tests = len(
    meta_mcnemar_df
)

meta_bonferroni_threshold = (
    meta_alpha / meta_number_of_tests
)

meta_mcnemar_df[
    "Comparison Family Size"
] = meta_number_of_tests

meta_mcnemar_df[
    "Bonferroni Threshold"
] = meta_bonferroni_threshold

meta_mcnemar_df[
    "Bonferroni-Adjusted P-Value"
] = np.minimum(
    meta_mcnemar_df[
        "Exact McNemar P-Value"
    ] * meta_number_of_tests,
    1.0
)

meta_mcnemar_df[
    "Significant at Alpha 0.05"
] = (
    meta_mcnemar_df[
        "Exact McNemar P-Value"
    ] < meta_alpha
)

meta_mcnemar_df[
    "Significant After Bonferroni"
] = (
    meta_mcnemar_df[
        "Exact McNemar P-Value"
    ] < meta_bonferroni_threshold
)


# =============================================================================
# COMBINE AND SAVE STATISTICAL RESULTS
# =============================================================================

all_mcnemar_df = pd.concat(
    [
        base_mcnemar_df,
        meta_mcnemar_df
    ],
    ignore_index=True
)

mcnemar_csv_file = os.path.join(
    output_folder,
    "exact_mcnemar_bonferroni_results.csv"
)

all_mcnemar_df.to_csv(
    mcnemar_csv_file,
    index=False
)


# =============================================================================
# APPEND MCNEMAR RESULTS TO THE MAIN AUDIT TEXT FILE
# =============================================================================

mcnemar_display_columns = [
    "Model",
    "Comparison",
    "Corrected",
    "Harmed",
    "Net Corrections",
    "Discordant Pairs",
    "Exact McNemar P-Value",
    "Bonferroni-Adjusted P-Value",
    "Bonferroni Threshold",
    "Significant After Bonferroni"
]

with open(
    output_file,
    "a",
    encoding="utf-8"
) as file:
    file.write("\n\n")
    file.write(
        "=" * 140 + "\n"
    )
    file.write(
        "EXACT MCNEMAR TESTS WITH BONFERRONI CORRECTION\n"
    )
    file.write(
        "=" * 140 + "\n"
    )

    file.write(
        "\nExact two-sided McNemar tests were calculated from "
        "the corrected and harmed prediction counts.\n"
    )

    file.write(
        "Null hypothesis: corrected and harmed outcomes are "
        "equally likely for the paired configurations.\n"
    )

    file.write(
        "A positive net correction count indicates that the "
        "comparison configuration corrected more baseline "
        "errors than it harmed.\n"
    )

    file.write(
        "\nBASE-LEARNER COMPARISON FAMILY\n"
    )
    file.write(
        "-" * 140 + "\n"
    )

    file.write(
        f"Number of tests: {base_number_of_tests}\n"
    )

    file.write(
        f"Unadjusted alpha: {base_alpha:.6f}\n"
    )

    file.write(
        f"Bonferroni threshold: "
        f"{base_bonferroni_threshold:.6f}\n\n"
    )

    file.write(
        base_mcnemar_df[
            mcnemar_display_columns
        ].to_string(
            index=False,
            formatters={
                "Exact McNemar P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni-Adjusted P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni Threshold": (
                    lambda value: f"{value:.6f}"
                )
            }
        )
    )

    file.write("\n\n")
    file.write(
        "META-LEARNER COMPARISON FAMILY\n"
    )
    file.write(
        "-" * 140 + "\n"
    )

    file.write(
        f"Number of tests: {meta_number_of_tests}\n"
    )

    file.write(
        f"Unadjusted alpha: {meta_alpha:.6f}\n"
    )

    file.write(
        f"Bonferroni threshold: "
        f"{meta_bonferroni_threshold:.6f}\n\n"
    )

    file.write(
        meta_mcnemar_df[
            mcnemar_display_columns
        ].to_string(
            index=False,
            formatters={
                "Exact McNemar P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni-Adjusted P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni Threshold": (
                    lambda value: f"{value:.6f}"
                )
            }
        )
    )

    file.write("\n\n")
    file.write(
        "Interpretation note: statistical significance should "
        "be considered together with accuracy, macro F1 and "
        "net-correction magnitude. A statistically significant "
        "difference may still be practically small.\n"
    )

    file.write(
        "The tests compare the final separately optimised "
        "pipelines and therefore do not isolate polarity-shift "
        "handling from differences in selected hyperparameters.\n"
    )


# =============================================================================
# DISPLAY MCNEMAR RESULTS
# =============================================================================

print(
    "\n========== BASE-LEARNER EXACT MCNEMAR TESTS =========="
)

print(
    base_mcnemar_df[
        mcnemar_display_columns
    ].to_string(
        index=False,
        formatters={
            "Exact McNemar P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni-Adjusted P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni Threshold": (
                lambda value: f"{value:.6f}"
            )
        }
    )
)

print(
    "\n========== META-LEARNER EXACT MCNEMAR TESTS =========="
)

print(
    meta_mcnemar_df[
        mcnemar_display_columns
    ].to_string(
        index=False,
        formatters={
            "Exact McNemar P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni-Adjusted P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni Threshold": (
                lambda value: f"{value:.6f}"
            )
        }
    )
)

print(
    f"\nMcNemar and Bonferroni results appended to:\n"
    f"{output_file}"
)

print(
    f"\nMcNemar and Bonferroni CSV saved to:\n"
    f"{mcnemar_csv_file}"
)

# =============================================================================
# EXACT MCNEMAR TESTS: ROBERTA VS META LEARNERS
# =============================================================================

def calculate_accuracy_from_predictions(
    true_labels,
    predictions,
    source_name
):

    y_true = normalise_confusion_matrix_labels(
        true_labels,
        "true test labels"
    )

    y_pred = normalise_confusion_matrix_labels(
        predictions,
        source_name
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Row-count mismatch for {source_name}: "
            f"true labels={len(y_true)}, predictions={len(y_pred)}"
        )

    return accuracy_score(
        y_true,
        y_pred
    )


def run_roberta_vs_meta_mcnemar_family(
    family_name,
    roberta_configuration_name,
    meta_configuration_name,
    true_labels,
    roberta_predictions,
    meta_predictions_by_model,
    alpha=0.05
):

    family_results = []

    roberta_accuracy = calculate_accuracy_from_predictions(
        true_labels=true_labels,
        predictions=roberta_predictions,
        source_name=(
            f"{roberta_configuration_name} RoBERTa predictions"
        )
    )

    for meta_model_name, meta_predictions in (
        meta_predictions_by_model.items()
    ):
        result = exact_mcnemar_comparison(
            learner_type=family_name,
            model_name=(
                f"{meta_model_name} Meta Learner vs RoBERTa"
            ),
            reference_name=(
                f"{roberta_configuration_name} RoBERTa"
            ),
            comparison_name=(
                f"{meta_configuration_name} "
                f"{meta_model_name} Meta Learner"
            ),
            true_labels=true_labels,
            reference_predictions=roberta_predictions,
            comparison_predictions=meta_predictions
        )

        meta_accuracy = calculate_accuracy_from_predictions(
            true_labels=true_labels,
            predictions=meta_predictions,
            source_name=(
                f"{meta_configuration_name} "
                f"{meta_model_name} meta predictions"
            )
        )

        result.update({
            "Comparison Family": family_name,
            "RoBERTa Configuration": roberta_configuration_name,
            "Meta Learner": meta_model_name,
            "Meta Configuration": meta_configuration_name,
            "RoBERTa Accuracy": roberta_accuracy,
            "Meta Accuracy": meta_accuracy,
            "Accuracy Difference Percentage Points": (
                meta_accuracy - roberta_accuracy
            ) * 100
        })

        family_results.append(
            result
        )

    family_df = pd.DataFrame(
        family_results
    )

    number_of_tests = len(
        family_df
    )

    if number_of_tests == 0:
        raise ValueError(
            f"No comparisons were generated for {family_name}."
        )

    bonferroni_threshold = (
        alpha / number_of_tests
    )

    family_df[
        "Comparison Family Size"
    ] = number_of_tests

    family_df[
        "Bonferroni Threshold"
    ] = bonferroni_threshold

    family_df[
        "Bonferroni-Adjusted P-Value"
    ] = np.minimum(
        family_df[
            "Exact McNemar P-Value"
        ] * number_of_tests,
        1.0
    )

    family_df[
        "Significant at Alpha 0.05"
    ] = (
        family_df[
            "Exact McNemar P-Value"
        ] < alpha
    )

    family_df[
        "Significant After Bonferroni"
    ] = (
        family_df[
            "Exact McNemar P-Value"
        ] < bonferroni_threshold
    )

    return family_df


# =============================================================================
# PREPARE BASELINE AND ENHANCED META-LEARNER PREDICTIONS
# =============================================================================

baseline_meta_predictions = {
    model_name: configurations["Baseline"]
    for model_name, configurations
    in meta_model_predictions.items()
}

enhanced_meta_predictions = {
    model_name: configurations["Enhanced"]
    for model_name, configurations
    in meta_model_predictions.items()
}


# =============================================================================
# RUN 4 BASELINE ROBERTA-VS-META TESTS
# =============================================================================

baseline_roberta_vs_meta_df = run_roberta_vs_meta_mcnemar_family(
    family_name="Baseline RoBERTa vs Baseline Meta",
    roberta_configuration_name="Baseline",
    meta_configuration_name="Baseline",
    true_labels=sentiment_test,
    roberta_predictions=base_roberta_test_sentiment,
    meta_predictions_by_model=baseline_meta_predictions,
    alpha=0.05
)


# =============================================================================
# RUN 4 ENHANCED ROBERTA-VS-META TESTS
# =============================================================================

enhanced_roberta_vs_meta_df = run_roberta_vs_meta_mcnemar_family(
    family_name="Enhanced RoBERTa vs Enhanced Meta",
    roberta_configuration_name="Enhanced",
    meta_configuration_name="Enhanced",
    true_labels=sentiment_test,
    roberta_predictions=enhanced_roberta_test_sentiment,
    meta_predictions_by_model=enhanced_meta_predictions,
    alpha=0.05
)


# =============================================================================
# COMBINE AND SAVE ROBERTA-VS-META MCNEMAR RESULTS
# =============================================================================

roberta_vs_meta_mcnemar_df = pd.concat(
    [
        baseline_roberta_vs_meta_df,
        enhanced_roberta_vs_meta_df
    ],
    ignore_index=True
)

roberta_vs_meta_csv_file = os.path.join(
    output_folder,
    "roberta_vs_meta_exact_mcnemar_bonferroni_results.csv"
)

roberta_vs_meta_mcnemar_df.to_csv(
    roberta_vs_meta_csv_file,
    index=False
)


# =============================================================================
# SAVE ROBERTA-VS-META RESULTS TO A DEDICATED TEXT FILE
# =============================================================================

roberta_vs_meta_display_columns = [
    "Comparison Family",
    "Meta Learner",
    "RoBERTa Accuracy",
    "Meta Accuracy",
    "Accuracy Difference Percentage Points",
    "Corrected",
    "Harmed",
    "Net Corrections",
    "Discordant Pairs",
    "Exact McNemar P-Value",
    "Bonferroni-Adjusted P-Value",
    "Bonferroni Threshold",
    "Significant After Bonferroni"
]

roberta_vs_meta_txt_file = os.path.join(
    output_folder,
    "roberta_vs_meta_exact_mcnemar_bonferroni_results.txt"
)

with open(
    roberta_vs_meta_txt_file,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        "EXACT MCNEMAR TESTS: ROBERTA VS META LEARNERS\n"
    )
    file.write(
        "=" * 170 + "\n"
    )

    file.write(
        "\nRoBERTa is the reference model and each meta learner "
        "is the comparison model.\n"
    )

    file.write(
        "Corrected: RoBERTa was wrong and the meta learner "
        "was correct.\n"
    )

    file.write(
        "Harmed: RoBERTa was correct and the meta learner "
        "was wrong.\n"
    )

    file.write(
        "Net Corrections: Corrected - Harmed.\n"
    )

    file.write(
        "A positive accuracy difference and positive net-correction "
        "count favour the meta learner.\n"
    )

    file.write(
        "Each comparison family contains four tests, giving a "
        "Bonferroni threshold of 0.05 / 4 = 0.0125.\n"
    )

    file.write(
        "\nBASELINE ROBERTA VS BASELINE META LEARNERS\n"
    )
    file.write(
        "-" * 170 + "\n"
    )

    file.write(
        baseline_roberta_vs_meta_df[
            roberta_vs_meta_display_columns
        ].to_string(
            index=False,
            formatters={
                "RoBERTa Accuracy": (
                    lambda value: f"{value:.6f}"
                ),
                "Meta Accuracy": (
                    lambda value: f"{value:.6f}"
                ),
                "Accuracy Difference Percentage Points": (
                    lambda value: f"{value:+.6f}"
                ),
                "Exact McNemar P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni-Adjusted P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni Threshold": (
                    lambda value: f"{value:.6f}"
                )
            }
        )
    )

    file.write(
        "\n\nENHANCED ROBERTA VS ENHANCED META LEARNERS\n"
    )
    file.write(
        "-" * 170 + "\n"
    )

    file.write(
        enhanced_roberta_vs_meta_df[
            roberta_vs_meta_display_columns
        ].to_string(
            index=False,
            formatters={
                "RoBERTa Accuracy": (
                    lambda value: f"{value:.6f}"
                ),
                "Meta Accuracy": (
                    lambda value: f"{value:.6f}"
                ),
                "Accuracy Difference Percentage Points": (
                    lambda value: f"{value:+.6f}"
                ),
                "Exact McNemar P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni-Adjusted P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni Threshold": (
                    lambda value: f"{value:.6f}"
                )
            }
        )
    )

    file.write(
        "\n\nInterpretation note: these tests evaluate paired "
        "differences in correctness and therefore support claims "
        "about accuracy or test-set error rate. They do not "
        "directly test differences in macro F1.\n"
    )


# =============================================================================
# APPEND ROBERTA-VS-META RESULTS TO THE MAIN AUDIT TEXT FILE
# =============================================================================

def write_roberta_vs_meta_family_to_file(
    file,
    heading,
    family_df
):

    family_size = int(
        family_df[
            "Comparison Family Size"
        ].iloc[0]
    )

    threshold = float(
        family_df[
            "Bonferroni Threshold"
        ].iloc[0]
    )

    file.write(
        f"\n{heading}\n"
    )

    file.write(
        "-" * 170 + "\n"
    )

    file.write(
        f"Number of paired comparisons: {family_size}\n"
    )

    file.write(
        "Unadjusted alpha: 0.050000\n"
    )

    file.write(
        f"Bonferroni threshold: {threshold:.6f}\n\n"
    )

    file.write(
        family_df[
            roberta_vs_meta_display_columns
        ].to_string(
            index=False,
            formatters={
                "RoBERTa Accuracy": (
                    lambda value: f"{value:.6f}"
                ),
                "Meta Accuracy": (
                    lambda value: f"{value:.6f}"
                ),
                "Accuracy Difference Percentage Points": (
                    lambda value: f"{value:+.6f}"
                ),
                "Exact McNemar P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni-Adjusted P-Value": (
                    lambda value: f"{value:.10g}"
                ),
                "Bonferroni Threshold": (
                    lambda value: f"{value:.6f}"
                )
            }
        )
    )

    file.write("\n")


with open(
    output_file,
    "a",
    encoding="utf-8"
) as file:
    file.write("\n\n")
    file.write(
        "=" * 170 + "\n"
    )
    file.write(
        "EXACT MCNEMAR TESTS: ROBERTA VS META LEARNERS\n"
    )
    file.write(
        "=" * 170 + "\n"
    )

    file.write(
        "\nRoBERTa is the reference model and each meta learner "
        "is the comparison model.\n"
    )

    file.write(
        "Corrected means that RoBERTa was wrong and the meta "
        "learner was correct.\n"
    )

    file.write(
        "Harmed means that RoBERTa was correct and the meta "
        "learner was wrong.\n"
    )

    file.write(
        "A positive net-correction count and positive accuracy "
        "difference favour the meta learner.\n"
    )

    file.write(
        "Baseline and enhanced comparisons are treated as two "
        "separate planned families. Each family contains four "
        "tests and uses a Bonferroni threshold of "
        "0.05 / 4 = 0.0125.\n"
    )

    write_roberta_vs_meta_family_to_file(
        file=file,
        heading=(
            "BASELINE ROBERTA VS BASELINE META LEARNERS"
        ),
        family_df=baseline_roberta_vs_meta_df
    )

    write_roberta_vs_meta_family_to_file(
        file=file,
        heading=(
            "ENHANCED ROBERTA VS ENHANCED META LEARNERS"
        ),
        family_df=enhanced_roberta_vs_meta_df
    )

    file.write(
        "\nInterpretation note: these tests evaluate paired "
        "differences in correctness and therefore support "
        "claims about test-set error rate or accuracy. They do "
        "not directly test differences in macro F1.\n"
    )


# =============================================================================
# DISPLAY ROBERTA-VS-META MCNEMAR RESULTS
# =============================================================================

print(
    "\n========== BASELINE ROBERTA VS BASELINE META =========="
)

print(
    baseline_roberta_vs_meta_df[
        roberta_vs_meta_display_columns
    ].to_string(
        index=False,
        formatters={
            "RoBERTa Accuracy": (
                lambda value: f"{value:.6f}"
            ),
            "Meta Accuracy": (
                lambda value: f"{value:.6f}"
            ),
            "Accuracy Difference Percentage Points": (
                lambda value: f"{value:+.6f}"
            ),
            "Exact McNemar P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni-Adjusted P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni Threshold": (
                lambda value: f"{value:.6f}"
            )
        }
    )
)

print(
    "\n========== ENHANCED ROBERTA VS ENHANCED META =========="
)

print(
    enhanced_roberta_vs_meta_df[
        roberta_vs_meta_display_columns
    ].to_string(
        index=False,
        formatters={
            "RoBERTa Accuracy": (
                lambda value: f"{value:.6f}"
            ),
            "Meta Accuracy": (
                lambda value: f"{value:.6f}"
            ),
            "Accuracy Difference Percentage Points": (
                lambda value: f"{value:+.6f}"
            ),
            "Exact McNemar P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni-Adjusted P-Value": (
                lambda value: f"{value:.10g}"
            ),
            "Bonferroni Threshold": (
                lambda value: f"{value:.6f}"
            )
        }
    )
)

print(
    "\nRoBERTa-versus-meta exact McNemar results appended to:\n"
    f"{output_file}"
)

print(
    "\nRoBERTa-versus-meta exact McNemar CSV saved to:\n"
    f"{roberta_vs_meta_csv_file}"
)

print(
    "\nRoBERTa-versus-meta exact McNemar text report saved to:\n"
    f"{roberta_vs_meta_txt_file}"
)