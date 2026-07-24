import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
# import nltk
# nltk.download("vader_lexicon")

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
# BASE VADER
# ======================================================================================================================
# ----------------------------------------------------------------------------- START
# BASE VADER SETUP
# -----------------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def vader_label(compound):
    if compound >= 0.05:
        return "pos"

    if compound <= -0.05:
        return "neg"

    return "neu"
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE BASE VADER
# -----------------------------------------------------------------------------
print("\n========== BASE VADER ==========")
base_vader_train_scores = [sia.polarity_scores(str(review))
                    for review in tqdm(text_train, desc="Predicting Train Scores With Base VADER")
                    ]
base_vader_train_sentiment = [vader_label(score["compound"])
                        for score in base_vader_train_scores
                        ]
                        
base_vader_val_scores = [sia.polarity_scores(str(review))
                    for review in tqdm(text_val, desc="Predicting Validation Scores With Base VADER")
                    ]
base_vader_val_sentiment = [vader_label(score["compound"])
                        for score in base_vader_val_scores
                        ]

base_vader_test_scores = [sia.polarity_scores(str(review))
                    for review in tqdm(text_test, desc="Predicting Test Scores With Base VADER")
                    ]
base_vader_test_sentiment = [vader_label(score["compound"])
                        for score in base_vader_test_scores
                        ]

base_vader_train_probabilities = np.asarray([[score["neg"], score["neu"], score["pos"], score["compound"]]
                                  for score in base_vader_train_scores
                                  ], dtype=np.float32)

base_vader_val_probabilities = np.asarray([[score["neg"], score["neu"], score["pos"], score["compound"]]
                                  for score in base_vader_val_scores
                                  ], dtype=np.float32)

base_vader_test_probabilities = np.asarray([[score["neg"], score["neu"], score["pos"], score["compound"]]
                                  for score in base_vader_test_scores
                                  ], dtype=np.float32)



print("\nBASE VADER ON TRAIN: ACCURACY = " + str(round(accuracy_score(sentiment_train, base_vader_train_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_train, base_vader_train_sentiment, digits=4))

print("\nBASE VADER ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_vader_val_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_val, base_vader_val_sentiment, digits=4))

print("\nBASE VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_vader_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_vader_test_sentiment, digits=4))
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
output_folder = "Base_Learner/Results/VADER/Base"
os.makedirs(output_folder, exist_ok=True)

base_vader_train_scores_df = pd.DataFrame(base_vader_train_scores)
base_vader_val_scores_df = pd.DataFrame(base_vader_val_scores)
base_vader_test_scores_df = pd.DataFrame(base_vader_test_scores)

base_vader_train_scores_df.to_csv(
    os.path.join(output_folder, "base_vader_train_scores.csv"),
    index=False
)

base_vader_val_scores_df.to_csv(
    os.path.join(output_folder, "base_vader_val_scores.csv"),
    index=False
)

base_vader_test_scores_df.to_csv(
    os.path.join(output_folder, "base_vader_test_scores.csv"),
    index=False
)

print("\nSaved VADER Score CSV Files to:", output_folder)

base_vader_train_sentiment_df = pd.DataFrame({
    "base_vader_sentiment": base_vader_train_sentiment
})

base_vader_val_sentiment_df = pd.DataFrame({
    "base_vader_sentiment": base_vader_val_sentiment
})

base_vader_test_sentiment_df = pd.DataFrame({
    "base_vader_sentiment": base_vader_test_sentiment
})

base_vader_train_sentiment_df.to_csv(
    os.path.join(output_folder, "base_vader_train_sentiment.csv"),
    index=False
)

base_vader_val_sentiment_df.to_csv(
    os.path.join(output_folder, "base_vader_val_sentiment.csv"),
    index=False
)

base_vader_test_sentiment_df.to_csv(
    os.path.join(output_folder, "base_vader_test_sentiment.csv"),
    index=False
)

print("Saved VADER Sentiment CSV Files to:", output_folder)

base_vader_train_probabilities_df = pd.DataFrame(base_vader_train_probabilities).rename(columns={
    0: "base_vader_neg",
    1: "base_vader_neu",
    2: "base_vader_pos",
    3: "base_vader_compound"
})

base_vader_val_probabilities_df = pd.DataFrame(base_vader_val_probabilities).rename(columns={
    0: "base_vader_neg",
    1: "base_vader_neu",
    2: "base_vader_pos",
    3: "base_vader_compound"
})

base_vader_test_probabilities_df = pd.DataFrame(base_vader_test_probabilities).rename(columns={
    0: "base_vader_neg",
    1: "base_vader_neu",
    2: "base_vader_pos",
    3: "base_vader_compound"
})

base_vader_train_probabilities_df.to_csv(
    os.path.join(output_folder, "base_vader_train_probabilities.csv"),
    index=False
)

base_vader_val_probabilities_df.to_csv(
    os.path.join(output_folder, "base_vader_val_probabilities.csv"),
    index=False
)

base_vader_test_probabilities_df.to_csv(
    os.path.join(output_folder, "base_vader_test_probabilities.csv"),
    index=False
)

print("Saved VADER Probabilities CSV Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/VADER/Base"
os.makedirs(output_folder, exist_ok=True)

base_vader_train_report_df = pd.DataFrame(
    classification_report(
        sentiment_train,
        base_vader_train_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_vader_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        base_vader_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_vader_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        base_vader_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

base_vader_train_report_df.to_csv(
    os.path.join(output_folder, "base_vader_train_classification_report.csv"),
    index_label="class"
)

base_vader_val_report_df.to_csv(
    os.path.join(output_folder, "base_vader_validation_classification_report.csv"),
    index_label="class"
)

base_vader_test_report_df.to_csv(
    os.path.join(output_folder, "base_vader_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_vader_train_report_df.values,
    rowLabels=base_vader_train_report_df.index,
    colLabels=base_vader_train_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_vader_train_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_vader_val_report_df.values,
    rowLabels=base_vader_val_report_df.index,
    colLabels=base_vader_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_vader_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=base_vader_test_report_df.values,
    rowLabels=base_vader_test_report_df.index,
    colLabels=base_vader_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "base_vader_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Base VADER Classification Report to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END
