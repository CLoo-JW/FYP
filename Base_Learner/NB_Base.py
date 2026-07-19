import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

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
# NAIVE BAYES
# ======================================================================================================================
# ----------------------------------------------------------------------------- Start
# BASE NB SETUP
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
# BASE NB HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------- 
def base_nb_optuna(trial):
    vectorizer_type = trial.suggest_categorical(
        "vectorizer_type",
        ["tfidf", "count"]
    )

    model_type = trial.suggest_categorical(
        "model_type",
        ["multinomial", "complement"]
    )

    optuna_nb_pipeline_steps = []

    if vectorizer_type == "tfidf":
        optuna_nb_pipeline_steps.append(
            ("vec", TfidfVectorizer(
                max_features=trial.suggest_categorical(
                    "max_features",
                    [30000, 50000, 100000, 150000]
                ),
                ngram_range=ngram_map[
                    trial.suggest_categorical(
                        "ngram_range",
                        ["1_1", "1_2", "1_3"]
                    )
                ],
                min_df=trial.suggest_categorical(
                    "min_df",
                    [5, 10, 20, 50]
                ),
                max_df=trial.suggest_categorical(
                    "max_df",
                    [0.90, 0.95, 0.98]
                ),
                sublinear_tf=trial.suggest_categorical(
                    "sublinear_tf",
                    [True, False]
                )
            ))
        )

    else:
        optuna_nb_pipeline_steps.append(
            ("vec", CountVectorizer(
                max_features=trial.suggest_categorical(
                    "max_features",
                    [30000, 50000, 100000, 150000]
                ),
                ngram_range=ngram_map[
                    trial.suggest_categorical(
                        "ngram_range",
                        ["1_1", "1_2", "1_3"]
                    )
                ],
                min_df=trial.suggest_categorical(
                    "min_df",
                    [5, 10, 20, 50]
                ),
                max_df=trial.suggest_categorical(
                    "max_df",
                    [0.90, 0.95, 0.98]
                ),
                binary=trial.suggest_categorical(
                    "binary",
                    [True, False]
                )
            ))
        )

    if model_type == "multinomial":
        optuna_nb_pipeline_steps.append(
            ("clf", MultinomialNB(
                alpha=trial.suggest_float(
                    "alpha",
                    0.01,
                    2.0,
                    log=True
                ),
                fit_prior=trial.suggest_categorical(
                    "fit_prior",
                    [True, False]
                )
            ))
        )

    else:
        optuna_nb_pipeline_steps.append(
            ("clf", ComplementNB(
                alpha=trial.suggest_float(
                    "alpha",
                    0.01,
                    2.0,
                    log=True
                ),
                fit_prior=trial.suggest_categorical(
                    "fit_prior",
                    [True, False]
                )
            ))
        )

    optuna_nb_pipeline = ImbPipeline(optuna_nb_pipeline_steps)

    optuna_nb_pipeline.fit(
        text_train,
        sentiment_train
    )

    validation_predictions = optuna_nb_pipeline.predict(
        text_val
    )

    validation_macro_f1 = f1_score(
        sentiment_val,
        validation_predictions,
        average="macro"
    )

    return validation_macro_f1
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# EVALUATE BASE NB
# ----------------------------------------------------------------------------- 
print("\n========== BASE NAIVE BAYES ==========")
base_nb_study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
base_nb_study.optimize(
    base_nb_optuna,
    n_trials=20,
    n_jobs=1
)
base_nb_best = base_nb_study.best_params

base_nb_pipeline_steps = []
if base_nb_best['vectorizer_type'] == 'tfidf':
    base_nb_pipeline_steps.append(
        ("vec", TfidfVectorizer(
            max_features=base_nb_best['max_features'],
            ngram_range=ngram_map[base_nb_best['ngram_range']],
            min_df=base_nb_best['min_df'],
            max_df=base_nb_best['max_df'],
            sublinear_tf=base_nb_best['sublinear_tf'],
        ))
    )
else:
    base_nb_pipeline_steps.append(
        ("vec", CountVectorizer(
            max_features=base_nb_best['max_features'],
            ngram_range=ngram_map[base_nb_best['ngram_range']],
            min_df=base_nb_best['min_df'],
            max_df=base_nb_best['max_df'],
            binary=base_nb_best['binary']
        ))
    )

if base_nb_best['model_type'] == "multinomial":
    base_nb_pipeline_steps.append(
        ('clf', MultinomialNB(
            alpha=base_nb_best['alpha'],
            fit_prior=base_nb_best['fit_prior']
        ))
    )
else:
    base_nb_pipeline_steps.append(
        ('clf', ComplementNB(
            alpha=base_nb_best['alpha'],
            fit_prior=base_nb_best['fit_prior']
        ))
    )

base_nb_uncalibrated_pipeline = ImbPipeline(base_nb_pipeline_steps)

base_nb_calibrated_pipeline = CalibratedClassifierCV(
    estimator=base_nb_uncalibrated_pipeline,
    cv=calibration_cv,
    method="sigmoid",
    ensemble=False,
    n_jobs=3
)
base_nb_pipeline = base_nb_calibrated_pipeline

progress = tqdm(total=1, desc="Base NB")
progress.set_description("Fitting Base NB")
base_nb_pipeline.fit(text_train, sentiment_train)
progress.update(1)
progress.close()

base_nb_val_sentiment = predict_with_progress(
    base_nb_pipeline,
    text_val,
    batch_size=5000,
    desc="Predicting Validation Sentiments With Base NB"
)

base_nb_test_sentiment = predict_with_progress(
    base_nb_pipeline,
    text_test,
    batch_size=5000,
    desc="Predicting Test Sentiments With Base NB"
)

print("\nBASE NAIVE BAYES BEST PARAMETERS: " + str(base_nb_study.best_value))
print(base_nb_study.best_params)
print("\nBASE NAIVE BAYES ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_nb_val_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_val, base_nb_val_sentiment, digits=4))
print("\nBASE NAIVE BAYES ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_nb_test_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_test, base_nb_test_sentiment, digits=4))

base_nb_val_probabilities = predict_proba_with_progress(
    base_nb_pipeline,
    text_val,
    batch_size=5000,
    desc="Predicting Validation Probabilities With Base NB"
)

base_nb_test_probabilities = predict_proba_with_progress(
    base_nb_pipeline,
    text_test,
    batch_size=5000,
    desc="Predicting Test Probabilities With Base NB"
)

base_nb_train_probabilities, base_nb_probability_classes = cross_val_predict_proba_with_progress(
    base_nb_pipeline,
    text_train,
    sentiment_train,
    base_cv,
    desc="Predicting OOF Train Probabilities With Base NB"
)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
nb_classes = np.array(["neg", "neu", "pos"])

assert np.array_equal(
    nb_classes,
    base_nb_probability_classes
), "Class order mismatch!"

assert np.array_equal(
    nb_classes,
    base_nb_pipeline.classes_
), "Class order mismatch!"

output_folder = "Base_Learner/Results/NB/Base"
os.makedirs(output_folder, exist_ok=True)

base_nb_val_sentiment_df = pd.DataFrame({
    "base_nb_sentiment": base_nb_val_sentiment
})

base_nb_test_sentiment_df = pd.DataFrame({
    "base_nb_sentiment": base_nb_test_sentiment
})

base_nb_val_sentiment_df.to_csv(
    os.path.join(output_folder, "base_nb_val_sentiment.csv"),
    index=False
)

base_nb_test_sentiment_df.to_csv(
    os.path.join(output_folder, "base_nb_test_sentiment.csv"),
    index=False
)

print("Saved Naive Bayes Sentiment CSV Files to:", output_folder)

assert np.array_equal(
    base_nb_probability_classes,
    base_nb_pipeline.classes_
), "Class order mismatch!"

base_nb_val_probabilities_df = pd.DataFrame(
    base_nb_val_probabilities,
    columns=[
        "base_nb_" + class_label
        for class_label in base_nb_probability_classes
    ]
)

base_nb_test_probabilities_df = pd.DataFrame(
    base_nb_test_probabilities,
    columns=[
        "base_nb_" + class_label
        for class_label in base_nb_probability_classes
    ]
)

base_nb_train_probabilities_df = pd.DataFrame(
    base_nb_train_probabilities,
    columns=[
        "base_nb_" + class_label
        for class_label in base_nb_probability_classes
    ]
)

base_nb_train_probabilities_df.to_csv(
    os.path.join(output_folder, "base_nb_train_probabilities.csv"),
    index=False
)

base_nb_val_probabilities_df.to_csv(
    os.path.join(output_folder, "base_nb_val_probabilities.csv"),
    index=False
)

base_nb_test_probabilities_df.to_csv(
    os.path.join(output_folder, "base_nb_test_probabilities.csv"),
    index=False
)

print("Saved Naive Bayes Probabilities CSV Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/NB/Base"
os.makedirs(output_folder, exist_ok=True)

base_nb_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        base_nb_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_nb_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_nb_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)


base_nb_val_report_df.to_csv(
    os.path.join(output_folder, "base_nb_validation_classification_report.csv"),
    index_label="class"
)

base_nb_test_report_df.to_csv(
    os.path.join(output_folder, "base_nb_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_nb_val_report_df.values,
    rowLabels=base_nb_val_report_df.index,
    colLabels=base_nb_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_nb_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_nb_test_report_df.values,
    rowLabels=base_nb_test_report_df.index,
    colLabels=base_nb_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_nb_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Base Naive Bayes Classification Report to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END