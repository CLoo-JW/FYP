import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import os
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
train_split_df = pd.read_csv("Dataset/Preprocessed/train_split.csv")
val_split_df = pd.read_csv("Dataset/Preprocessed/val_split.csv")
test_split_df = pd.read_csv("Dataset/Preprocessed/test_split.csv")

for split_name, split_df in [
    ("TRAIN SET", train_split_df),
    ("VALIDATION SET", val_split_df),
    ("TEST SET", test_split_df)
]:
    print(f"\n{split_name} Missing Text:", split_df["Text"].isna().sum())
    print(f"{split_name} Empty Text:", split_df["Text"].fillna("").str.strip().eq("").sum())
    print(f"{split_name} Missing Sentiment:", split_df["Sentiment"].isna().sum())

text_train = train_split_df["Text"]
sentiment_train = train_split_df["Sentiment"]


OPTUNA_TRAIN_SIZE = min(90000, len(text_train))

if OPTUNA_TRAIN_SIZE < len(text_train):
    text_train_optuna, _, sentiment_train_optuna, _ = train_test_split(
        text_train,
        sentiment_train,
        train_size=OPTUNA_TRAIN_SIZE,
        stratify=sentiment_train,
        random_state=42
    )
else:
    text_train_optuna = text_train
    sentiment_train_optuna = sentiment_train

print("\n========== BASE SVM OPTUNA SUBSET ==========")
print("Optuna training rows:", len(text_train_optuna))
print(sentiment_train_optuna.value_counts())

text_val = val_split_df["Text"]
sentiment_val = val_split_df["Sentiment"]

text_test = test_split_df["Text"]
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
# SVM
# ======================================================================================================================
# ----------------------------------------------------------------------------- Start
# BASE SVM SETUP
# ----------------------------------------------------------------------------- 
base_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=242)

ngram_map = {
    "1_1": (1, 1),
    "1_2": (1, 2),
    "1_3": (1, 3)
}
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# TQDM PROGRESS BARS
# ----------------------------------------------------------------------------- 
def predict_with_progress(model, texts, batch_size=5000, desc="Predicting"):
    predictions = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        end = start + batch_size
        batch_texts = texts.iloc[start:end]

        batch_predictions = model.predict(batch_texts)
        predictions.extend(batch_predictions)

    return np.array(predictions)

def predict_proba_with_progress(model, texts, batch_size=5000, desc="Predicting probabilities"):
    probabilities = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        end = start + batch_size
        batch_texts = texts.iloc[start:end]

        batch_probabilities = model.predict_proba(batch_texts)
        probabilities.append(batch_probabilities)

    return np.vstack(probabilities)

def cross_val_predict_proba_with_progress(model, text, sentiment, cv, desc="Generating OOF probabilities"):
    text = text.reset_index(drop=True)
    sentiment = sentiment.reset_index(drop=True)

    classes = np.array(sorted(sentiment.unique()))
    class_to_index = {
        class_label: index
        for index, class_label in enumerate(classes)
    }

    oof_probabilities = np.zeros((len(text), len(classes)))

    for train_index, val_index in tqdm(
        cv.split(text, sentiment),
        total=cv.get_n_splits(),
        desc=desc
    ):
        text_fold_train = text.iloc[train_index]
        sentiment_fold_train = sentiment.iloc[train_index]

        text_fold_val = text.iloc[val_index]

        fold_model = clone(model)

        fold_model.fit(
            text_fold_train,
            sentiment_fold_train
        )

        fold_probabilities = fold_model.predict_proba(
            text_fold_val
        )

        fold_classes = fold_model.classes_

        for fold_class_index, class_label in enumerate(fold_classes):
            target_class_index = class_to_index[class_label]
            oof_probabilities[val_index, target_class_index] = fold_probabilities[:, fold_class_index]

    return oof_probabilities, classes
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# BASE SVM HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------- 
def base_svm_optuna(trial):
    # vectorizer_type = trial.suggest_categorical(
    #     "vectorizer_type",
    #     ["tfidf", "count"]
    # )

    # if vectorizer_type == "tfidf":
    optuna_vectorizer = TfidfVectorizer(
        max_features=trial.suggest_categorical(
            "max_features",
            [30000, 50000, 100000]
        ),
        ngram_range=ngram_map[
            trial.suggest_categorical(
                "ngram_range",
                ["1_1", "1_2", "1_3"]
            )
        ],
        min_df=trial.suggest_categorical(
            "min_df",
            [10, 20, 50]
        ),
        max_df=trial.suggest_categorical(
            "max_df",
            [0.90, 0.95, 0.98]
        ),
        sublinear_tf=trial.suggest_categorical(
            "sublinear_tf",
            [True, False]
        )
    )

    # else:
    #     optuna_vectorizer = CountVectorizer(
    #         max_features=trial.suggest_categorical(
    #             "max_features",
    #             [30000, 50000, 100000, 150000]
    #         ),
    #         ngram_range=ngram_map[
    #             trial.suggest_categorical(
    #                 "ngram_range",
    #                 ["1_1", "1_2"]
    #             )
    #         ],
    #         min_df=trial.suggest_categorical(
    #             "min_df",
    #             [10, 20, 50]
    #         ),
    #         max_df=trial.suggest_categorical(
    #             "max_df",
    #             [0.90, 0.95]
    #         ),
    #         binary=trial.suggest_categorical(
    #             "binary",
    #             [True, False]
    #         )
    #     )

    optuna_svc = LinearSVC(
        C=trial.suggest_float(
            "C",
            0.1,
            10,
            log=True
        ),
        random_state=42,
        max_iter=5000,
        tol=0.001,
        dual="auto"
    )

    optuna_svm_pipeline = ImbPipeline([
        ("vec", optuna_vectorizer),
        ("clf", optuna_svc)
    ])

    optuna_svm_pipeline.fit(
        text_train_optuna,
        sentiment_train_optuna
    )

    validation_predictions = optuna_svm_pipeline.predict(
        text_val
    )

    validation_macro_f1 = f1_score(
        sentiment_val,
        validation_predictions,
        average="macro"
    )

    return validation_macro_f1
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE BASE SVM
# -----------------------------------------------------------------------------
print("\n========== BASE SVM ==========")
base_svm_study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
base_svm_study.optimize(
    base_svm_optuna,
    n_trials=20,
    n_jobs=1,
    gc_after_trial=True,
    show_progress_bar=True
)
base_svm_best = base_svm_study.best_params

# if base_svm_best['vectorizer_type'] == 'tfidf':
base_svm_vectorizer = TfidfVectorizer(
    max_features=base_svm_best['max_features'],
    ngram_range=ngram_map[base_svm_best['ngram_range']],
    min_df=base_svm_best['min_df'],
    max_df=base_svm_best['max_df'],
    sublinear_tf=base_svm_best['sublinear_tf']
)
# else:
#     base_svm_vectorizer = CountVectorizer(
#         max_features=base_svm_best['max_features'],
#         ngram_range=ngram_map[base_svm_best['ngram_range']],
#         min_df=base_svm_best['min_df'],
#         max_df=base_svm_best['max_df'],
#         binary=base_svm_best['binary']
#     )

base_svc = LinearSVC(
    C=base_svm_best['C'],
    random_state=42,
    tol=0.0001,
    dual='auto',
    max_iter=10000
)

base_svm_uncalibrated_pipeline = ImbPipeline([
    ("vec", base_svm_vectorizer),
    ("clf", base_svc)
])

base_svm_calibrated_pipeline = CalibratedClassifierCV(
    estimator=base_svm_uncalibrated_pipeline,
    cv=calibration_cv,
    method='sigmoid', 
    ensemble=False,
    n_jobs=3
)

base_svm_pipeline = base_svm_calibrated_pipeline

progress = tqdm(total=1, desc="Base SVM")
progress.set_description("Fitting Base SVM")
base_svm_pipeline.fit(text_train, sentiment_train)
progress.update(1)
progress.close()

base_svm_val_sentiment = predict_with_progress(
    base_svm_pipeline,
    text_val,
    batch_size=5000,
    desc="Predicting Validation Sentiments With Base SVM"
)

base_svm_test_sentiment = predict_with_progress(
    base_svm_pipeline,
    text_test,
    batch_size=5000,
    desc="Predicting Test Sentiments With Base SVM"
)

print("\nBASE SVM BEST PARAMETERS: " + str(base_svm_study.best_value))
print(base_svm_study.best_params)
print("\nBASE SVM ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_svm_val_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_val, base_svm_val_sentiment, digits=4))
print("\nBASE SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_svm_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_svm_test_sentiment, digits=4))

base_svm_val_probabilities = predict_proba_with_progress(
    base_svm_pipeline,
    text_val,
    batch_size=5000,
    desc="Predicting Validation Probabilities With Base SVM"
)

base_svm_test_probabilities = predict_proba_with_progress(
    base_svm_pipeline,
    text_test,
    batch_size=5000,
    desc="Predicting Test Probabilities With Base SVM"
)

base_svm_train_probabilities, base_svm_probability_classes = cross_val_predict_proba_with_progress(
    base_svm_pipeline,
    text_train,
    sentiment_train,
    base_cv,
    desc="Predicting OOF Train Probabilities With Base SVM"
)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
svm_classes = np.array(["neg", "neu", "pos"])

assert np.array_equal(
    svm_classes,
    base_svm_probability_classes
), "Class order mismatch!"

assert np.array_equal(
    svm_classes,
    base_svm_pipeline.classes_
), "Class order mismatch!"

output_folder = "Base_Learner/Results/SVM/Base"
os.makedirs(output_folder, exist_ok=True)

base_svm_val_sentiment_df = pd.DataFrame({
    "base_svm_sentiment": base_svm_val_sentiment
})

base_svm_test_sentiment_df = pd.DataFrame({
    "base_svm_sentiment": base_svm_test_sentiment
})

base_svm_val_sentiment_df.to_csv(
    os.path.join(output_folder, "base_svm_val_sentiment.csv"),
    index=False
)

base_svm_test_sentiment_df.to_csv(
    os.path.join(output_folder, "base_svm_test_sentiment.csv"),
    index=False
)

print("Saved SVM Sentiment CSV Files to:", output_folder)

assert np.array_equal(
    base_svm_probability_classes,
    base_svm_pipeline.classes_
), "Class order mismatch!"

base_svm_val_probabilities_df = pd.DataFrame(
    base_svm_val_probabilities,
    columns=[
        "base_svm_" + class_label
        for class_label in base_svm_probability_classes
    ]
)

base_svm_test_probabilities_df = pd.DataFrame(
    base_svm_test_probabilities,
    columns=[
        "base_svm_" + class_label
        for class_label in base_svm_probability_classes
    ]
)

base_svm_train_probabilities_df = pd.DataFrame(
    base_svm_train_probabilities,
    columns=[
        "base_svm_" + class_label
        for class_label in base_svm_probability_classes
    ]
)

base_svm_train_probabilities_df.to_csv(
    os.path.join(output_folder, "base_svm_train_probabilities.csv"),
    index=False
)

base_svm_val_probabilities_df.to_csv(
    os.path.join(output_folder, "base_svm_val_probabilities.csv"),
    index=False
)

base_svm_test_probabilities_df.to_csv(
    os.path.join(output_folder, "base_svm_test_probabilities.csv"),
    index=False
)

print("Saved SVM Probabilities CSV Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/SVM/Base"
os.makedirs(output_folder, exist_ok=True)

base_svm_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        base_svm_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_svm_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_svm_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)


base_svm_val_report_df.to_csv(
    os.path.join(output_folder, "base_svm_validation_classification_report.csv"),
    index_label="class"
)

base_svm_test_report_df.to_csv(
    os.path.join(output_folder, "base_svm_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_svm_val_report_df.values,
    rowLabels=base_svm_val_report_df.index,
    colLabels=base_svm_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_svm_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_svm_test_report_df.values,
    rowLabels=base_svm_test_report_df.index,
    colLabels=base_svm_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_svm_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Base SVM Classification Report to:", output_folder)

output_folder = "Base_Learner/Results/SVM/Base"
os.makedirs(output_folder, exist_ok=True)

base_svm_optuna_summary = pd.DataFrame([
    {
        "hyperparameter": "max_features",
        "search_range": "30000, 50000, 100000",
        "best_value": base_svm_best["max_features"]
    },
    {
        "hyperparameter": "ngram_range",
        "search_range": "1_1, 1_2, 1_3",
        "best_value": base_svm_best["ngram_range"]
    },
    {
        "hyperparameter": "min_df",
        "search_range": "10, 20, 50",
        "best_value": base_svm_best["min_df"]
    },
    {
        "hyperparameter": "max_df",
        "search_range": "0.90, 0.95, 0.98",
        "best_value": base_svm_best["max_df"]
    },
    {
        "hyperparameter": "sublinear_tf",
        "search_range": "True, False",
        "best_value": base_svm_best["sublinear_tf"]
    },
    {
        "hyperparameter": "C",
        "search_range": "0.1 to 10.0, logarithmic",
        "best_value": base_svm_best["C"]
    }
])

base_svm_optuna_summary.to_csv(
    os.path.join(output_folder, "base_svm_optuna_parameters.csv"),
    index=False
)

print("Saved Base SVM Optuna Parameters to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END