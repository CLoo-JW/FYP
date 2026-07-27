import pandas as pd
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score
import optuna
from sklearn.model_selection import StratifiedKFold
import torch
import matplotlib.pyplot as plt
import os

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
# ROBERTA
# ======================================================================================================================
# ----------------------------------------------------------------------------- START
# BASE ROBERTA SETUP
# ----------------------------------------------------------------------------- 
base_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=242)

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
sentiment_test_num = sentiment_test.map(label_map)
sentiment_val_num = sentiment_val.map(label_map)

ROBERTA_OPTUNA_TRAIN_SIZE = min(60000, len(text_train))

roberta_train_df = train_split_df[["Text", "Sentiment", "Stratify"]].copy()
roberta_train_df["sentiment_num"] = (roberta_train_df["Sentiment"].map(label_map))

if ROBERTA_OPTUNA_TRAIN_SIZE < len(text_train):
    train_roberta_optuna, _ = train_test_split(
        roberta_train_df,
        train_size=ROBERTA_OPTUNA_TRAIN_SIZE,
        stratify=roberta_train_df["Stratify"],
        random_state=42
    )
    train_roberta_optuna = (train_roberta_optuna.reset_index(drop=True))
    text_train_roberta_optuna = (train_roberta_optuna["Text"])
    sentiment_train_num_roberta_optuna = (train_roberta_optuna["sentiment_num"].astype(int))
else:
    text_train_roberta_optuna = text_train
    sentiment_train_num_roberta_optuna = sentiment_train_num

print("\n========== BASE RoBERTa OPTUNA SUBSET ==========")
print("Optuna training rows:", len(text_train_roberta_optuna))
print(sentiment_train_num_roberta_optuna.value_counts())
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# DATASET WRAPPER
# ----------------------------------------------------------------------------- 
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# TEMPERATURE SCALING HELPERS
# ----------------------------------------------------------------------------- 
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        temperature = torch.exp(self.log_temperature)
        return logits / temperature

    def get_temperature(self):
        return torch.exp(self.log_temperature).item()


def fit_temperature_scaler(logits, sentiment):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    sentiment_tensor = torch.tensor(sentiment, dtype=torch.long)

    temperature_scaler = TemperatureScaler()
    loss_function = CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        temperature_scaler.parameters(),
        lr=0.01,
        max_iter=50
    )

    def closure():
        optimizer.zero_grad()
        scaled_logits = temperature_scaler(logits_tensor)
        loss = loss_function(scaled_logits, sentiment_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature_scaler


def apply_temperature_scaling(logits, temperature_scaler):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)

    with torch.no_grad():
        scaled_logits = temperature_scaler(logits_tensor)
        calibrated_probabilities = torch.softmax(scaled_logits, dim=1)

    return calibrated_probabilities.numpy()
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# BASE ROBERTA HELPERS
# ----------------------------------------------------------------------------- 
def cleanup_trainer(trainer):
    try:
        trainer.model.to("cpu")
    except Exception:
        pass

    del trainer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict_roberta_logits(
        trainer,
        predict_text,
        predict_sentiment,
        tokenizer,
        max_length
):
    predict_dataset = build_roberta_dataset(
        text_series=predict_text,
        sentiment_series=predict_sentiment,
        tokenizer=tokenizer,
        max_length=max_length
    )

    logits = trainer.predict(
        predict_dataset
    ).predictions

    return logits

def build_roberta_dataset(text_series, sentiment_series, tokenizer, max_length):
    encodings = tokenizer(
        list(text_series),
        truncation=True,
        padding=True,
        max_length=max_length
    )

    dataset = ReviewDataset(
        encodings,
        sentiment_series.tolist()
    )

    return dataset

def make_roberta_training_args(output_dir, roberta_params, number_of_training_rows):
    batch_size = roberta_params["per_device_train_batch_size"]

    steps_per_epoch = max(
        1,
        int(np.ceil(number_of_training_rows / batch_size))
    )

    total_steps = steps_per_epoch * roberta_params["num_train_epochs"]

    warmup_steps = int(
        roberta_params["warmup_ratio"] * total_steps
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=roberta_params["learning_rate"],
        per_device_train_batch_size=roberta_params["per_device_train_batch_size"],
        per_device_eval_batch_size=roberta_params["per_device_train_batch_size"],
        num_train_epochs=roberta_params["num_train_epochs"],
        weight_decay=roberta_params["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=roberta_params["lr_scheduler_type"],
        logging_strategy="no",
        save_strategy="no",
        report_to="none",

        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    )

def train_roberta_model(
        train_text,
        train_sentiment,
        tokenizer,
        model_name,
        roberta_params,
        output_dir
):
    train_dataset = build_roberta_dataset(
        text_series=train_text,
        sentiment_series=train_sentiment,
        tokenizer=tokenizer,
        max_length=roberta_params["max_length"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    training_args = make_roberta_training_args(
        output_dir=output_dir,
        roberta_params=roberta_params,
        number_of_training_rows=len(train_text)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

    return trainer
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# BASE ROBERTA HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------- 
def base_roberta_optuna(trial):
    optuna_learning_rate = trial.suggest_float(
        "learning_rate",
        1e-6,
        5e-5,
        log=True
    )

    optuna_per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size",
        [8, 16]
    )

    optuna_num_train_epochs = trial.suggest_int(
        "num_train_epochs",
        1,
        3
    )

    optuna_weight_decay = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1
    )

    optuna_max_length = trial.suggest_categorical(
        "max_length",
        [64, 128, 256]
    )

    optuna_lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler_type",
        ["linear", "cosine"]
    )

    optuna_warmup_ratio = trial.suggest_float(
        "warmup_ratio",
        0.0,
        0.1
    )

    optuna_model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    optuna_params = {
        "model": optuna_model_name,
        "learning_rate": optuna_learning_rate,
        "per_device_train_batch_size": optuna_per_device_train_batch_size,
        "num_train_epochs": optuna_num_train_epochs,
        "weight_decay": optuna_weight_decay,
        "max_length": optuna_max_length,
        "lr_scheduler_type": optuna_lr_scheduler_type,
        "warmup_ratio": optuna_warmup_ratio
    }

    optuna_tokenizer = AutoTokenizer.from_pretrained(
        optuna_model_name
    )

    optuna_trainer = train_roberta_model(
        train_text=text_train_roberta_optuna,
        train_sentiment=sentiment_train_num_roberta_optuna,
        tokenizer=optuna_tokenizer,
        model_name=optuna_model_name,
        roberta_params=optuna_params,
        output_dir="./tmp"
    )

    optuna_val_logits = predict_roberta_logits(
        trainer=optuna_trainer,
        predict_text=text_val,
        predict_sentiment=sentiment_val_num,
        tokenizer=optuna_tokenizer,
        max_length=optuna_max_length
    )

    optuna_val_probabilities = softmax(
        optuna_val_logits,
        axis=1
    )

    optuna_val_prediction_ids = np.argmax(
        optuna_val_probabilities,
        axis=1
    )

    optuna_val_macro_f1 = f1_score(
        sentiment_val_num,
        optuna_val_prediction_ids,
        average="macro"
    )

    cleanup_trainer(optuna_trainer)

    return optuna_val_macro_f1

roberta_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

roberta_study.optimize(
    base_roberta_optuna,
    n_trials=20,
    n_jobs=1,
    gc_after_trial=True,
    show_progress_bar=True
)

base_roberta_best = roberta_study.best_params

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("\nBASE RoBERTa BEST PARAMETERS: " + str(roberta_study.best_value))
print(base_roberta_best)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# TEMPERATURE SCALING
# ----------------------------------------------------------------------------- 
print("\n===== BASE RoBERTa TEMPERATURE SCALING =====")

roberta_train_calibration_logits = np.zeros(
    (len(text_train), 3)
)

for calibration_fold, (calibration_train_idx, calibration_val_idx) in enumerate(
        calibration_cv.split(text_train, sentiment_train_num)
    ):
    print(f"\n----- TEMPERATURE SCALING FOLD {calibration_fold + 1}/{calibration_cv.n_splits} -----")

    text_calibration_train = text_train.iloc[calibration_train_idx]
    text_calibration_val = text_train.iloc[calibration_val_idx]

    sentiment_calibration_train = sentiment_train_num.iloc[calibration_train_idx]
    sentiment_calibration_val = sentiment_train_num.iloc[calibration_val_idx]

    calibration_trainer = train_roberta_model(
        train_text=text_calibration_train,
        train_sentiment=sentiment_calibration_train,
        tokenizer=tokenizer,
        model_name=MODEL,
        roberta_params=base_roberta_best,
        output_dir="./tmp"
    )

    calibration_val_logits = predict_roberta_logits(
        trainer=calibration_trainer,
        predict_text=text_calibration_val,
        predict_sentiment=sentiment_calibration_val,
        tokenizer=tokenizer,
        max_length=base_roberta_best["max_length"]
    )

    roberta_train_calibration_logits[calibration_val_idx] = calibration_val_logits

    cleanup_trainer(calibration_trainer)

base_roberta_temperature_scaler = fit_temperature_scaler(
    logits=roberta_train_calibration_logits,
    sentiment=sentiment_train_num.to_numpy()
)

print("\nBase RoBERTa Temperature:", base_roberta_temperature_scaler.get_temperature())
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE BASE ROBERTA
# ----------------------------------------------------------------------------- 
print("\n========== BASE RoBERTa ==========")
base_roberta_trainer = train_roberta_model(
    train_text=text_train,
    train_sentiment=sentiment_train_num,
    tokenizer=tokenizer,
    model_name=MODEL,
    roberta_params=base_roberta_best,
    output_dir="./tmp"
)

base_roberta_test_logits = predict_roberta_logits(
    trainer=base_roberta_trainer,
    predict_text=text_test,
    predict_sentiment=sentiment_test_num,
    tokenizer=tokenizer,
    max_length=base_roberta_best["max_length"]
)

base_roberta_val_logits = predict_roberta_logits(
    trainer=base_roberta_trainer,
    predict_text=text_val,
    predict_sentiment=sentiment_val_num,
    tokenizer=tokenizer,
    max_length=base_roberta_best["max_length"]
)

base_roberta_val_probabilities = apply_temperature_scaling(
    logits=base_roberta_val_logits,
    temperature_scaler=base_roberta_temperature_scaler
)

base_roberta_test_probabilities = apply_temperature_scaling(
    logits=base_roberta_test_logits,
    temperature_scaler=base_roberta_temperature_scaler
)

base_roberta_val_sentiment_ids = np.argmax(
    base_roberta_val_probabilities,
    axis=1
)

base_roberta_test_sentiment_ids = np.argmax(
    base_roberta_test_probabilities,
    axis=1
)

base_roberta_val_sentiment = [
    reverse_label_map[sentiment_id]
    for sentiment_id in base_roberta_val_sentiment_ids
]

base_roberta_test_sentiment = [
    reverse_label_map[sentiment_id]
    for sentiment_id in base_roberta_test_sentiment_ids
]

print("\nBASE RoBERTa ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_roberta_val_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_val, base_roberta_val_sentiment, digits=4))

print("\nBASE RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_roberta_test_sentiment, digits=4))

cleanup_trainer(base_roberta_trainer)

print("\n===== BASE RoBERTa OOF META FEATURES =====")

roberta_oof_probabilities = np.zeros(
    (len(text_train), 3)
)

for fold, (train_idx, fold_val_idx) in enumerate(
        base_cv.split(text_train, sentiment_train_num)
    ):
    print(f"\n----- OOF META FEATURES FOLD {fold + 1}/{base_cv.n_splits} -----")

    text_fold_train = text_train.iloc[train_idx]
    text_fold_val = text_train.iloc[fold_val_idx]

    sentiment_fold_train = sentiment_train_num.iloc[train_idx]
    sentiment_fold_val = sentiment_train_num.iloc[fold_val_idx]

    fold_trainer = train_roberta_model(
        train_text=text_fold_train,
        train_sentiment=sentiment_fold_train,
        tokenizer=tokenizer,
        model_name=MODEL,
        roberta_params=base_roberta_best,
        output_dir="./tmp"
    )

    fold_val_logits = predict_roberta_logits(
        trainer=fold_trainer,
        predict_text=text_fold_val,
        predict_sentiment=sentiment_fold_val,
        tokenizer=tokenizer,
        max_length=base_roberta_best["max_length"]
    )

    calibrated_fold_val_probabilities = apply_temperature_scaling(
        logits=fold_val_logits,
        temperature_scaler=base_roberta_temperature_scaler
    )

    roberta_oof_probabilities[fold_val_idx] = calibrated_fold_val_probabilities

    cleanup_trainer(fold_trainer)

base_roberta_train_probabilities = roberta_oof_probabilities
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
output_folder = "Base_Learner/Results/RoBERTa/Base"
os.makedirs(output_folder, exist_ok=True)

base_roberta_val_sentiment_df = pd.DataFrame({
    "base_roberta_sentiment": base_roberta_val_sentiment
})

base_roberta_test_sentiment_df = pd.DataFrame({
    "base_roberta_sentiment": base_roberta_test_sentiment
})

base_roberta_val_sentiment_df.to_csv(
    os.path.join(output_folder, "base_roberta_val_sentiment.csv"),
    index=False
)

base_roberta_test_sentiment_df.to_csv(
    os.path.join(output_folder, "base_roberta_test_sentiment.csv"),
    index=False
)

print("Saved RoBERTa Sentiment CSV Files to:", output_folder)

base_roberta_probability_classes = np.array([
    reverse_label_map[0],
    reverse_label_map[1],
    reverse_label_map[2]
])

assert np.array_equal(
    base_roberta_probability_classes,
    np.array(["neg", "neu", "pos"])
), "Class order mismatch!"

base_roberta_val_probabilities_df = pd.DataFrame(
    base_roberta_val_probabilities,
    columns=[
        "base_roberta_" + class_label
        for class_label in base_roberta_probability_classes
    ]
)

base_roberta_test_probabilities_df = pd.DataFrame(
    base_roberta_test_probabilities,
    columns=[
        "base_roberta_" + class_label
        for class_label in base_roberta_probability_classes
    ]
)

base_roberta_train_probabilities_df = pd.DataFrame(
    base_roberta_train_probabilities,
    columns=[
        "base_roberta_" + class_label
        for class_label in base_roberta_probability_classes
    ]
)

base_roberta_train_probabilities_df.to_csv(
    os.path.join(output_folder, "base_roberta_train_probabilities.csv"),
    index=False
)

base_roberta_val_probabilities_df.to_csv(
    os.path.join(output_folder, "base_roberta_val_probabilities.csv"),
    index=False
)

base_roberta_test_probabilities_df.to_csv(
    os.path.join(output_folder, "base_roberta_test_probabilities.csv"),
    index=False
)

print("Saved RoBERTa Probabilities CSV Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/RoBERTa/Base"
os.makedirs(output_folder, exist_ok=True)

base_roberta_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        base_roberta_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_roberta_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_roberta_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)


base_roberta_val_report_df.to_csv(
    os.path.join(output_folder, "base_roberta_validation_classification_report.csv"),
    index_label="class"
)

base_roberta_test_report_df.to_csv(
    os.path.join(output_folder, "base_roberta_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_roberta_val_report_df.values,
    rowLabels=base_roberta_val_report_df.index,
    colLabels=base_roberta_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_roberta_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_roberta_test_report_df.values,
    rowLabels=base_roberta_test_report_df.index,
    colLabels=base_roberta_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_roberta_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Base RoBERTa Classification Report to:", output_folder)

output_folder = "Base_Learner/Results/RoBERTa/Base"
os.makedirs(output_folder, exist_ok=True)

base_roberta_optuna_summary = pd.DataFrame([
    {
        "hyperparameter": "learning_rate",
        "search_range": "1e-6 to 5e-5, logarithmic",
        "best_value": base_roberta_best["learning_rate"]
    },
    {
        "hyperparameter": "per_device_train_batch_size",
        "search_range": "8, 16",
        "best_value": base_roberta_best[
            "per_device_train_batch_size"
        ]
    },
    {
        "hyperparameter": "num_train_epochs",
        "search_range": "1 to 3",
        "best_value": base_roberta_best["num_train_epochs"]
    },
    {
        "hyperparameter": "weight_decay",
        "search_range": "0.0 to 0.1",
        "best_value": base_roberta_best["weight_decay"]
    },
    {
        "hyperparameter": "max_length",
        "search_range": "64, 128, 256",
        "best_value": base_roberta_best["max_length"]
    },
    {
        "hyperparameter": "lr_scheduler_type",
        "search_range": "linear, cosine",
        "best_value": base_roberta_best["lr_scheduler_type"]
    },
    {
        "hyperparameter": "warmup_ratio",
        "search_range": "0.0 to 0.1",
        "best_value": base_roberta_best["warmup_ratio"]
    }
])

base_roberta_optuna_summary.to_csv(
    os.path.join(output_folder, "base_roberta_optuna_parameters.csv"),
    index=False
)

print("Saved Base RoBERTa Optuna Parameters to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END
