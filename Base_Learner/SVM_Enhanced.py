import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import re
import spacy
from sklearn.svm import LinearSVC
from tqdm.contrib.concurrent import process_map
import os
from sklearn.base import clone
import io
import contextlib

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
base_svm_val_sentiment_df = pd.read_csv("Base_Learner/Results/SVM/Base/base_svm_val_sentiment.csv")
base_svm_test_sentiment_df = pd.read_csv("Base_Learner/Results/SVM/Base/base_svm_test_sentiment.csv")

base_svm_test_sentiment = base_svm_test_sentiment_df["base_svm_sentiment"].to_numpy()
base_svm_val_sentiment = base_svm_val_sentiment_df["base_svm_sentiment"].to_numpy()

base_svm_val_probabilities_df = pd.read_csv("Base_Learner/Results/SVM/Base/base_svm_val_probabilities.csv")

base_svm_val_probabilities = base_svm_val_probabilities_df[["base_svm_neg", "base_svm_neu", "base_svm_pos"]].to_numpy()

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
# SVM
# ======================================================================================================================
# ----------------------------------------------------------------------------- Start
# ENHANCED SVM SETUP
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

# ----------------------------------------------------------------------------- START
# SINGLE-WORD POLARITY SHIFTER
# -----------------------------------------------------------------------------
nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner"]
)

NEGATION_WORDS = {
    "not", "no", "never",
    "cannot", "cant", "can't",
    "dont", "don't",
    "doesnt", "doesn't",
    "didnt", "didn't",
    "isnt", "isn't",
    "wasnt", "wasn't",
    "werent", "weren't",
    "wont", "won't",
    "wouldnt", "wouldn't",
    "shouldnt", "shouldn't",
    "couldnt", "couldn't", 
    "n't", "nt",
    "aint", "ain't",
    "arent", "aren't",
    "hasnt", "hasn't",
    "havent", "haven't",
    "hadnt", "hadn't",
    "without"
}

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "highly",
    "super", "ultra", "absolutely", "completely", "totally",
    "surprisingly", "ridiculously", "seriously", "terribly"
}

DIMINISHERS = {
    "slightly", "somewhat", "mildly", "partly",
    "partially", "kinda", "sorta", "barely", "hardly"
}

POST_CONTRAST_MARKERS = {
    "but",
    "however",
    "yet",
    "nevertheless",
    "nonetheless"
}

CONCESSIVE_STARTERS = {
    "although",
    "though"
}
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# PHRASES AND REGEX
# -----------------------------------------------------------------------------
NEG_AUX = (
    r"(?:"
    r"not|"
    r"cannot|can\s+not|can't|cant|"
    r"do\s+not|don't|dont|"
    r"does\s+not|doesn't|doesnt|"
    r"did\s+not|didn't|didnt|"
    r"is\s+not|isn't|isnt|"
    r"was\s+not|wasn't|wasnt|"
    r"are\s+not|aren't|arent|"
    r"were\s+not|weren't|werent|"
    r"will\s+not|won't|wont|"
    r"would\s+not|wouldn't|wouldnt|"
    r"should\s+not|shouldn't|shouldnt|"
    r"could\s+not|couldn't|couldnt|"
    r"has\s+not|hasn't|hasnt|"
    r"have\s+not|haven't|havent|"
    r"had\s+not|hadn't|hadnt"
    r")"
)

OPTIONAL_DEGREE = r"(?:really\s+|very\s+|so\s+|too\s+|quite\s+|at\s+all\s+)?"

UNIVERSAL_PHRASE_RULES = [
    {
        "rule_key": "phrase:neutral_good_but_not_great",
        "feature": "FEAT_PHRASE_neutral_good_but_not_great",
        "pattern": re.compile(
            r"\b(?:good|decent|okay|ok|fine)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect|the\s+best)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_not_bad_not_great",
        "feature": "FEAT_PHRASE_neutral_not_bad_not_great",
        "pattern": re.compile(
            r"\bnot\s+(?:bad|terrible|awful)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_okay_but_issues",
        "feature": "FEAT_PHRASE_neutral_okay_but_issues",
        "pattern": re.compile(
            r"\b(?:okay|ok|fine|decent|good)\s+but\s+"
            r"(?:has|have|had|with)\s+(?:some\s+)?"
            r"(?:issues|problems|flaws|drawbacks|downsides)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_works_but",
        "feature": "FEAT_PHRASE_neutral_works_but",
        "pattern": re.compile(
            r"\b(?:works|worked|work)\s+but\s+"
            r"(?:not\s+perfect|not\s+great|has\s+issues|could\s+be\s+better|"
            r"there\s+are\s+issues|with\s+some\s+problems)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_decent_for_price",
        "feature": "FEAT_PHRASE_neutral_decent_for_price",
        "pattern": re.compile(
            r"\b(?:decent|okay|ok|fine|acceptable|reasonable)\s+"
            r"(?:for|given)\s+(?:the\s+)?(?:price|money|cost)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_average_nothing_special",
        "feature": "FEAT_PHRASE_neutral_average_nothing_special",
        "pattern": re.compile(
            r"\b(?:average|mediocre|ordinary)\s+"
            r"(?:product|item|quality|book|read|purchase)\b|"
            r"\bnothing\s+(?:special|amazing|great|exceptional)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_pros_and_cons",
        "feature": "FEAT_PHRASE_neutral_pros_and_cons",
        "pattern": re.compile(
            r"\b(?:pros\s+and\s+cons|good\s+and\s+bad|"
            r"some\s+good\s+and\s+some\s+bad|mixed\s+feelings|mixed\s+review)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_somewhat_disappointed",
        "feature": "FEAT_PHRASE_neutral_somewhat_disappointed",
        "pattern": re.compile(
            r"\b(?:somewhat|slightly|a\s+little|kind\s+of|kinda)\s+"
            r"(?:disappointed|underwhelmed)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_expected_more",
        "feature": "FEAT_PHRASE_neutral_expected_more",
        "pattern": re.compile(
            r"\b(?:expected|was\s+expecting)\s+"
            r"(?:a\s+)?(?:little\s+)?more\b|"
            r"\bnot\s+(?:quite|really)\s+what\s+i\s+expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_worth_it",
        "feature": "FEAT_PHRASE_not_worth_it",
        "pattern": re.compile(
            r"\bnot\s+" + OPTIONAL_DEGREE +
            r"worth\s+(?:it|the\s+money|the\s+price|buying|getting|keeping)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:waste_of_money",
        "feature": "FEAT_PHRASE_waste_of_money",
        "pattern": re.compile(
            r"\b(?:a\s+)?(?:complete\s+|total\s+|real\s+|absolute\s+)?"
            r"waste\s+of\s+(?:money|time)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:low_quality",
        "feature": "FEAT_PHRASE_low_quality",
        "pattern": re.compile(
            r"\b(?:low|poor|bad|terrible|awful|horrible)\s+quality\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:cheaply_made",
        "feature": "FEAT_PHRASE_cheaply_made",
        "pattern": re.compile(
            r"\b(?:cheaply\s+made|"
            r"(?:feel|feels|felt|feeling)\s+cheap|"
            r"(?:material|fabric|plastic|product|item)\s+"
            r"(?:feel|feels|felt|feeling)\s+cheap)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:fell_apart",
        "feature": "FEAT_PHRASE_fell_apart",
        "pattern": re.compile(
            r"\b(?:fell|came|comes|coming)\s+apart\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_lasting",
        "feature": "FEAT_PHRASE_not_lasting",
        "pattern": re.compile(
            r"\b(?:"
            r"(?:did\s+not|didn't|didnt)\s+last|"
            r"not\s+lasting|"
            r"only\s+lasted|"
            r"lasted\s+(?:only\s+)?(?:a\s+)?(?:day|week|month|few\s+days|few\s+weeks)|"
            r"broke\s+(?:after|within)"
            r")\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_as_described",
        "feature": "FEAT_PHRASE_not_as_described",
        "pattern": re.compile(
            r"\b(?:not|isn't|isnt|wasn't|wasnt|is\s+not|was\s+not)\s+"
            r"(?:as\s+)?described\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_item",
        "feature": "FEAT_PHRASE_wrong_item",
        "pattern": re.compile(
            r"\bwrong\s+(?:item|product|model|version|book|charger|case)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:missing_parts",
        "feature": "FEAT_PHRASE_missing_parts",
        "pattern": re.compile(
            r"\bmissing\s+(?:parts?|pieces?|accessories|components|items?)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:never_received",
        "feature": "FEAT_PHRASE_never_received",
        "pattern": re.compile(
            r"\b(?:never\s+received|"
            r"did\s+not\s+receive|didn't\s+receive|didnt\s+receive|"
            r"have\s+not\s+received|haven't\s+received|havent\s+received)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_delivered",
        "feature": "FEAT_PHRASE_not_delivered",
        "pattern": re.compile(
            r"\b(?:not|never|was\s+not|wasn't|wasnt|"
            r"has\s+not\s+been|hasn't\s+been|hasnt\s+been)"
            r"\s+delivered\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:had_to_return",
        "feature": "FEAT_PHRASE_had_to_return",
        "pattern": re.compile(
            r"\b(?:had\s+to\s+return|"
            r"returned\s+(?:it|this|the\s+item|the\s+product|the\s+book))\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:want_refund",
        "feature": "FEAT_PHRASE_want_refund",
        "pattern": re.compile(
            r"\b(?:want|wanted|need|needed|request(?:ed)?|asking\s+for)"
            r"\s+(?:a\s+)?refund\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_satisfied",
        "feature": "FEAT_PHRASE_not_satisfied",
        "pattern": re.compile(
            r"\bnot\s+(?:very\s+|really\s+|fully\s+|completely\s+)?"
            r"(?:satisfied|happy)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_not_recommend",
        "feature": "FEAT_PHRASE_would_not_recommend",
        "pattern": re.compile(
            r"\b(?:would\s+not|wouldn't|wouldnt)\s+recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_bad",
        "feature": "FEAT_PHRASE_not_bad",
        "pattern": re.compile(
            r"\bnot\s+(?:too\s+|that\s+|so\s+|very\s+)?bad\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_complaints",
        "feature": "FEAT_PHRASE_no_complaints",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?complaints?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_issues",
        "feature": "FEAT_PHRASE_no_issues",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?issues?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_problems",
        "feature": "FEAT_PHRASE_no_problems",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?problems?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_regrets",
        "feature": "FEAT_PHRASE_no_regrets",
        "pattern": re.compile(
            r"\bno\s+regrets?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_great",
        "feature": "FEAT_PHRASE_works_great",
        "pattern": re.compile(
            r"\bworks?\s+(?:really\s+|very\s+|so\s+)?great\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_perfectly",
        "feature": "FEAT_PHRASE_works_perfectly",
        "pattern": re.compile(
            r"\bworks?\s+(?:perfectly|flawlessly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_as_expected",
        "feature": "FEAT_PHRASE_works_as_expected",
        "pattern": re.compile(
            r"\bwork(?:s|ed)?\s+(?:exactly\s+)?as\s+expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:highly_recommend",
        "feature": "FEAT_PHRASE_highly_recommend",
        "pattern": re.compile(
            r"\b(?:highly|strongly|definitely)\s+recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_recommend",
        "feature": "FEAT_PHRASE_would_recommend",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+|highly\s+)?recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_buy_again",
        "feature": "FEAT_PHRASE_would_buy_again",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+)?buy\s+(?:it\s+|this\s+)?again\b|"
            r"\bbuy\s+(?:it\s+|this\s+)?again\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:worth_every_penny",
        "feature": "FEAT_PHRASE_worth_every_penny",
        "pattern": re.compile(
            r"\bworth\s+every\s+(?:penny|cent)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:better_than_expected",
        "feature": "FEAT_PHRASE_better_than_expected",
        "pattern": re.compile(
            r"\bbetter\s+than\s+(?:i\s+)?expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:exceeded_expectations",
        "feature": "FEAT_PHRASE_exceeded_expectations",
        "pattern": re.compile(
            r"\bexceeded\s+(?:my\s+)?expectations\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:could_not_be_happier",
        "feature": "FEAT_PHRASE_could_not_be_happier",
        "pattern": re.compile(
            r"\b(?:could\s+not|couldn't|couldnt)\s+be\s+happier\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:cannot_recommend_enough",
        "feature": "FEAT_PHRASE_cannot_recommend_enough",
        "pattern": re.compile(
            r"\b(?:cannot|can\s+not|can't|cant)\s+recommend"
            r"(?:\s+(?:it|this|these|them))?\s+enough\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:does_not_work",
        "feature": "FEAT_PHRASE_does_not_work",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:stopped_working",
        "feature": "FEAT_PHRASE_stopped_working",
        "pattern": re.compile(
            r"\b(?:stopped|stop|stops|quit|quits)\s+working\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_longer_works",
        "feature": "FEAT_PHRASE_no_longer_works",
        "pattern": re.compile(
            r"\bno\s+longer\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:dead_on_arrival",
        "feature": "FEAT_PHRASE_dead_on_arrival",
        "pattern": re.compile(
            r"\b(?:dead\s+on\s+arrival|doa)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_powering_on",
        "feature": "FEAT_PHRASE_not_powering_on",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:power\s+on|powering\s+on)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_turning_on",
        "feature": "FEAT_PHRASE_not_turning_on",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:turn\s+on|turning\s+on)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_charging",
        "feature": "FEAT_PHRASE_not_charging",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+charg(?:e|es|ed|ing)\b|"
            r"\b(?:stopped|stop|stops|quit|quits)\s+charging\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:battery_drains_fast",
        "feature": "FEAT_PHRASE_battery_drains_fast",
        "pattern": re.compile(
            r"\b(?:battery|batteries)\s+"
            r"(?:drain|drains|drained|die|dies|died)\s+"
            r"(?:too\s+)?(?:fast|quickly|rapidly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:does_not_hold_charge",
        "feature": "FEAT_PHRASE_does_not_hold_charge",
        "pattern": re.compile(
            rf"\b(?:battery\s+)?(?:{NEG_AUX})\s+hold\s+"
            r"(?:a\s+|the\s+)?charge\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:keeps_disconnecting",
        "feature": "FEAT_PHRASE_keeps_disconnecting",
        "pattern": re.compile(
            r"\b(?:keep|keeps|kept)\s+disconnecting\b|"
            r"\blosing\s+connection\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poor_connection",
        "feature": "FEAT_PHRASE_poor_connection",
        "pattern": re.compile(
            r"\b(?:poor|bad|weak|unstable)\s+(?:connection|signal)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:overheats_quickly",
        "feature": "FEAT_PHRASE_overheats_quickly",
        "pattern": re.compile(
            r"\b(?:overheat|overheats|overheated|overheating)\s+"
            r"(?:too\s+)?(?:quickly|fast|easily)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:screen_cracked",
        "feature": "FEAT_PHRASE_screen_cracked",
        "pattern": re.compile(
            r"\b(?:screen\s+cracked|cracked\s+screen)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:flickering_screen",
        "feature": "FEAT_PHRASE_flickering_screen",
        "pattern": re.compile(
            r"\b(?:screen|display)\s+(?:is\s+|was\s+|keeps\s+|kept\s+|started\s+)?"
            r"(?:flickering|flickers|flicker)\b|"
            r"\b(?:flickering|flickers|flicker)\s+(?:screen|display)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:touchscreen_not_responsive",
        "feature": "FEAT_PHRASE_touchscreen_not_responsive",
        "pattern": re.compile(
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:is\s+|was\s+)?not\s+responsive\b|"
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:does\s+not|doesn't|doesnt)\s+respond\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_fitting",
        "feature": "FEAT_PHRASE_not_fitting",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+fit(?:s|ted|ting)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_size",
        "feature": "FEAT_PHRASE_wrong_size",
        "pattern": re.compile(
            r"\bwrong\s+size\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_color",
        "feature": "FEAT_PHRASE_wrong_color",
        "pattern": re.compile(
            r"\bwrong\s+(?:color|colour)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:runs_small",
        "feature": "FEAT_PHRASE_runs_small",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?small\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:runs_large",
        "feature": "FEAT_PHRASE_runs_large",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?(?:large|big)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poor_fit",
        "feature": "FEAT_PHRASE_poor_fit",
        "pattern": re.compile(
            r"\b(?:poor|bad|terrible|awkward|weird)\s+fit\b|"
            r"\bfit(?:s|ted)?\s+(?:poorly|badly|terribly|awkwardly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:see_through",
        "feature": "FEAT_PHRASE_see_through",
        "pattern": re.compile(
            r"\b(?:see\s*through|see-through|too\s+sheer|transparent)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:shrunk_after_wash",
        "feature": "FEAT_PHRASE_shrunk_after_wash",
        "pattern": re.compile(
            r"\b(?:shrank|shrunk|shrinked)\s+"
            r"(?:after|following)\s+(?:a\s+)?(?:wash|washing)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:color_faded",
        "feature": "FEAT_PHRASE_color_faded",
        "pattern": re.compile(
            r"\b(?:color|colour|colors|colours)\s+(?:faded|fades|fade)\b|"
            r"\b(?:faded|fades|fade)\s+(?:color|colour|colors|colours)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:fabric_feels_cheap",
        "feature": "FEAT_PHRASE_fabric_feels_cheap",
        "pattern": re.compile(
            r"\b(?:fabric|material)\s+(?:feel|feels|felt|feeling)\s+cheap\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:seam_ripped",
        "feature": "FEAT_PHRASE_seam_ripped",
        "pattern": re.compile(
            r"\bseam\s+(?:ripped|torn)\b|"
            r"\bstitching\s+(?:came\s+loose|undone|ripped|torn)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:zipper_broken",
        "feature": "FEAT_PHRASE_zipper_broken",
        "pattern": re.compile(
            r"\bzipper\s+(?:broken|stuck|jammed|broke)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:true_to_size",
        "feature": "FEAT_PHRASE_true_to_size",
        "pattern": re.compile(
            r"\btrue\s+to\s+size\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:comfortable_fit",
        "feature": "FEAT_PHRASE_comfortable_fit",
        "pattern": re.compile(
            r"\bcomfortable\s+fit\b|"
            r"\bfit(?:s|ted)?\s+comfortably\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:missing_pages",
        "feature": "FEAT_PHRASE_missing_pages",
        "pattern": re.compile(
            r"\bmissing\s+pages?\b|"
            r"\bpages?\s+(?:is\s+|are\s+|was\s+|were\s+)?missing\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poorly_written",
        "feature": "FEAT_PHRASE_poorly_written",
        "pattern": re.compile(
            r"\b(?:poorly|badly|terribly)\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:poor|bad|terrible|awful)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:hard_to_follow",
        "feature": "FEAT_PHRASE_hard_to_follow",
        "pattern": re.compile(
            r"\b(?:hard|difficult|confusing)\s+to\s+follow\b|"
            r"\bnot\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:bad_translation",
        "feature": "FEAT_PHRASE_bad_translation",
        "pattern": re.compile(
            r"\b(?:bad|poor|terrible|awful)\s+translation\b|"
            r"\btranslation\s+(?:is\s+|was\s+)?(?:bad|poor|terrible|awful)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:printing_error",
        "feature": "FEAT_PHRASE_printing_error",
        "pattern": re.compile(
            r"\b(?:printing|print)\s+errors?\b|"
            r"\b(?:misprint|misprinted|misprints)\b|"
            r"\bpages?\s+(?:printed|print)\s+incorrectly\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:great_read",
        "feature": "FEAT_PHRASE_great_read",
        "pattern": re.compile(
            r"\bgreat\s+read\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:well_written",
        "feature": "FEAT_PHRASE_well_written",
        "pattern": re.compile(
            r"\bwell\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:excellent|great|clear)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:easy_to_follow",
        "feature": "FEAT_PHRASE_easy_to_follow",
        "pattern": re.compile(
            r"\b(?:easy|clear)\s+to\s+follow\b|"
            r"\bclear\s+and\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:highly_informative",
        "feature": "FEAT_PHRASE_highly_informative",
        "pattern": re.compile(
            r"\b(?:highly|very|really)\s+informative\b|"
            r"\binformative\s+and\s+useful\b",
            flags=re.IGNORECASE
        )
    }
]
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# RULE / FEATURE PRUNING
# -----------------------------------------------------------------------------
SVM_CULL_RULE_KEYS = {
    # Add based on validation set results
}
SVM_BLOCKED_EXACT_FEATURES = {
    # Add based on validation set results
}
SVM_CONDITIONAL_RULE_KEYS = {
    # "phrase:does_not_work"
}

SVM_POSITIVE_CONTEXT_FEATURES = {
    # "FEAT_PHRASE_works_great",
    # "FEAT_PHRASE_works_perfectly",
    # "FEAT_PHRASE_works_as_expected",
    # "FEAT_PHRASE_would_recommend",
    # "FEAT_PHRASE_highly_recommend",
    # "FEAT_PHRASE_would_buy_again",
    # "FEAT_PHRASE_worth_every_penny",
    # "FEAT_PHRASE_better_than_expected",
    # "FEAT_PHRASE_exceeded_expectations",
    # "FEAT_PHRASE_no_issues",
    # "FEAT_PHRASE_no_problems",
    # "FEAT_PHRASE_no_complaints",
    # "FEAT_PHRASE_not_bad",
    # "FEAT_PHRASE_true_to_size",
    # "FEAT_PHRASE_comfortable_fit",
    # "FEAT_PHRASE_great_read",
    # "FEAT_PHRASE_well_written",
    # "FEAT_PHRASE_easy_to_follow",
    # "FEAT_PHRASE_highly_informative",
}
SVM_BLOCKED_DYNAMIC_SUFFIXES = {
    # "character",
    # "story",
    # "fan",
    # "purchase",
    # "software",
    # "interested",
    # "instead",
    # "rest",
    # "music",
    # "wonder",
    # "definitely",
}

def is_blocked_svm_dynamic_feature(feature):
    dynamic_prefixes = (
        # "FEAT_NEG_SCOPE_",
        # "FEAT_NEG_DEP_",
        # "FEAT_INTENSIFIED_",
        # "FEAT_INTENSIFIED_DEP_",
        # "FEAT_DIMINISHED_",
        # "FEAT_DIMINISHED_DEP_",
        # "FEAT_AFTER_CONTRAST_",
        # "FEAT_AFTER_CONCESSION_"
    )

    if not feature.startswith(dynamic_prefixes):
        return False

    for blocked_suffix in SVM_BLOCKED_DYNAMIC_SUFFIXES:
        if feature.endswith("_" + blocked_suffix):
            return True

    return False

def filter_svm_polarity_features(polarity_features):
    filtered_features = []
    polarity_feature_set = set(polarity_features)
    has_positive_context = len(polarity_feature_set.intersection(SVM_POSITIVE_CONTEXT_FEATURES)) > 0

    for feature in polarity_features:
        if feature in SVM_BLOCKED_EXACT_FEATURES:
            continue

        if feature.startswith("FEAT_CAPS_") and feature != "FEAT_CAPS_EMPHASIS":
            continue

        if is_blocked_svm_dynamic_feature(feature):
            continue

        rule_key = get_rule_key_from_feature(feature)
        if rule_key in SVM_CULL_RULE_KEYS:
            continue

        if (rule_key in SVM_CONDITIONAL_RULE_KEYS and has_positive_context):
            continue

        filtered_features.append(feature)

    return filtered_features
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# TEXT NORMALISATION
# -----------------------------------------------------------------------------
def normalise_for_spacy_svm(text):
    text = str(text)

    NEGATION_CONTRACTION_REPLACEMENTS = {
        r"\bcan't\b": "can not",
        r"\bcant\b": "can not",
        r"\bcannot\b": "can not",
        r"\bwon't\b": "will not",
        r"\bwont\b": "will not",
        r"\bdon't\b": "do not",
        r"\bdont\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdoesnt\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bdidnt\b": "did not",
        r"\bisn't\b": "is not",
        r"\bisnt\b": "is not",
        r"\baren't\b": "are not",
        r"\barent\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bwasnt\b": "was not",
        r"\bweren't\b": "were not",
        r"\bwerent\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhasnt\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhavent\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bhadnt\b": "had not",
        r"\bwouldn't\b": "would not",
        r"\bwouldnt\b": "would not",
        r"\bshouldn't\b": "should not",
        r"\bshouldnt\b": "should not",
        r"\bcouldn't\b": "could not",
        r"\bcouldnt\b": "could not",
        r"\bmustn't\b": "must not",
        r"\bmustnt\b": "must not",
        r"\bneedn't\b": "need not",
        r"\bneednt\b": "need not",
        r"\bshan't\b": "shall not",
        r"\bshant\b": "shall not",
        r"\bain't\b": "is not",
        r"\baint\b": "is not",
    }

    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
            .lower()
    )

    for pattern, replacement in NEGATION_CONTRACTION_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)

    return text

# ----------------------------------------------------------------------------- end

# ----------------------------------------------------------------------------- START
# FEATURE EXTRACTION
# -----------------------------------------------------------------------------
def extract_regex_polarity_features_and_spans(text):
    phrase_spans = []

    normalised_text = normalise_for_spacy_svm(text)

    for rule in UNIVERSAL_PHRASE_RULES:
        for match in rule["pattern"].finditer(normalised_text):
            phrase_spans.append({
                "feature": rule["feature"],
                "rule_key": rule["rule_key"],
                "start": match.start(),
                "end": match.end(),
                "length": match.end() - match.start()
            })

    phrase_spans = sorted(
        phrase_spans,
        key=lambda item: (item["start"], -item["length"])
    )

    selected_phrase_spans = []
    occupied_ranges = []

    for phrase_span in phrase_spans:
        start = phrase_span["start"]
        end = phrase_span["end"]

        overlaps = False

        for occupied_start, occupied_end in occupied_ranges:
            if start < occupied_end and end > occupied_start:
                overlaps = True
                break

        if overlaps:
            continue

        selected_phrase_spans.append(phrase_span)
        occupied_ranges.append((start, end))

    regex_features = [
        phrase_span["feature"]
        for phrase_span in selected_phrase_spans
    ]

    regex_features = list(dict.fromkeys(regex_features))

    return regex_features, selected_phrase_spans

def token_is_inside_phrase_span(token, phrase_spans):
    token_start = token.idx
    token_end = token.idx + len(token.text)

    for phrase_span in phrase_spans:
        phrase_start = phrase_span["start"]
        phrase_end = phrase_span["end"]

        if token_start < phrase_end and token_end > phrase_start:
            return True

    return False

def clean_spacy_token(token):
    lemma = token.lemma_.lower()

    if lemma == "-pron-":
        lemma = token.text.lower()

    lemma = lemma.replace(" ", "_")

    return lemma

def collect_dependency_targets(doc):
    negated_targets = set()
    intensified_targets = set()
    diminished_targets = set()

    for token in doc:
        word = token.text.lower().replace("’", "'")

        if token.dep_ == "neg" or word in NEGATION_WORDS:
            head = token.head

            if head is not None and head.i != token.i:
                negated_targets.add(head.i)

        if word in INTENSIFIERS:
            head = token.head

            if head is not None and head.i != token.i:
                intensified_targets.add(head.i)

        if word in DIMINISHERS:
            head = token.head

            if head is not None and head.i != token.i:
                diminished_targets.add(head.i)

    return negated_targets, intensified_targets, diminished_targets

def is_useful_scope_token(token, processed_token):
    if not processed_token:
        return False

    if len(processed_token) < 3:
        return False

    if token.is_stop:
        return False

    if token.like_num:
        return False
    
    if processed_token in {
        # Decide based on validation results
    }:
        return False

    if token.pos_ not in {
        "ADJ",
        "ADV",
        "VERB",
        "NOUN",
        "PROPN"
    }:
        return False

    return True

def extract_caps_features(text):
    tokens = str(text).split()
    caps_features = []

    for token in tokens:
        clean_token = token.strip(".,!?;:\"'()[]{}")

        if clean_token.isupper() and len(clean_token) > 1:
            caps_features.append("CAPS_EMPHASIS")

    return caps_features

# Main feature extraction function
def extract_polarity_feature_tokens(text):
    features = []
    regex_features, phrase_spans = extract_regex_polarity_features_and_spans(text)
    caps_features = extract_caps_features(text)
    
    for regex_feature in regex_features:
        features.append(regex_feature)

    for caps_feature in caps_features:
        features.append(
            "FEAT_" + caps_feature
        )

    normalised_text = normalise_for_spacy_svm(text)
    doc = nlp(normalised_text)
    negated_targets, intensified_targets, diminished_targets = (collect_dependency_targets(doc))

    negation_scope = 0
    intensifier_scope = 0
    diminisher_scope = 0

    after_contrast = False
    after_contrast_marker = None
    inside_concession_lead = False
    after_concession = False
    concession_marker = None

    skip_token_indexes = set()

    for token in doc:
        if token.i in skip_token_indexes:
            continue

        word = token.text.lower().replace("’", "'")
        inside_phrase_span = token_is_inside_phrase_span(token, phrase_spans)

        # Punctuation
        if token.is_punct:
            negation_scope = 0
            intensifier_scope = 0
            diminisher_scope = 0

            if token.text in {".", "!", "?"}:
                after_contrast = False
                after_contrast_marker = None
                inside_concession_lead = False
                after_concession = False
                concession_marker = None

            elif token.text == ";":
                after_contrast = False
                after_contrast_marker = None
                inside_concession_lead = False
                after_concession = False
                concession_marker = None

            elif token.text == ",":
                if inside_concession_lead:
                    inside_concession_lead = False
                    after_concession = True

            continue

        if token.is_space:
            continue

        if inside_phrase_span:
            continue

        if word in POST_CONTRAST_MARKERS:
            clean_contrast_marker = str(word).lower()
            features.append("FEAT_CONTRAST_" + clean_contrast_marker)

            negation_scope = 0
            intensifier_scope = 0
            diminisher_scope = 0

            after_contrast = True
            after_contrast_marker = clean_contrast_marker
            inside_concession_lead = False
            after_concession = False
            concession_marker = None

            continue

        # "even though" exception
        if (
            word == "even"
            and token.i + 1 < len(doc)
            and doc[token.i + 1].text.lower().replace("’", "'") == "though"
        ):
            features.append("FEAT_CONCESSIVE_even_though")

            negation_scope = 0
            intensifier_scope = 0
            diminisher_scope = 0

            after_contrast = False
            after_contrast_marker = None
            inside_concession_lead = True
            after_concession = False
            concession_marker = "even_though"

            skip_token_indexes.add(token.i + 1)

            continue

        if word in CONCESSIVE_STARTERS:
            clean_concession_marker = str(word).lower()

            features.append(
                "FEAT_CONCESSIVE_" + clean_concession_marker
            )

            negation_scope = 0
            intensifier_scope = 0
            diminisher_scope = 0

            after_contrast = False
            after_contrast_marker = None

            inside_concession_lead = True
            after_concession = False
            concession_marker = clean_concession_marker

            continue

        # Negation
        if word in NEGATION_WORDS or token.dep_ == "neg":
            features.append("FEAT_NEGATOR_" + word.replace("'", ""))
            negation_scope = 4

            continue

        # Intensifiers
        if word in INTENSIFIERS:
            features.append("FEAT_INTENSIFIER_" + word)
            intensifier_scope = 2

            continue

        # Diminishers
        if word in DIMINISHERS:
            features.append("FEAT_DIMINISHER_" + word)
            diminisher_scope = 2

            continue

        processed_token = clean_spacy_token(token)
        if not processed_token:
            continue

        # Negation scope
        if negation_scope > 0:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_NEG_SCOPE_" + processed_token)

            negation_scope -= 1

        elif token.i in negated_targets:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_NEG_DEP_" + processed_token)

        # Intensifier scope
        if intensifier_scope > 0:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_INTENSIFIED_" + processed_token)

            intensifier_scope -= 1

        elif token.i in intensified_targets:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_INTENSIFIED_DEP_" + processed_token)

        # Diminisher scope
        if diminisher_scope > 0:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_DIMINISHED_" + processed_token)

            diminisher_scope -= 1

        elif token.i in diminished_targets:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_DIMINISHED_DEP_" + processed_token)

        # Contrast scope
        if after_contrast and after_contrast_marker is not None:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_AFTER_CONTRAST_" + after_contrast_marker + "_" + processed_token)

        # Concession scope
        if after_concession and concession_marker is not None:
            if is_useful_scope_token(token, processed_token):
                features.append("FEAT_AFTER_CONCESSION_" + concession_marker + "_" + processed_token)

    features = list(dict.fromkeys(features))

    return features

def svm_append_polarity_features(text):
    original_text = normalise_for_spacy_svm(text)

    polarity_features = extract_polarity_feature_tokens(text)
    polarity_features = filter_svm_polarity_features(polarity_features)

    if polarity_features:
        return original_text + " " + " ".join(polarity_features)

    return original_text
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# POLARITY SHIFT FEATURE MAPPING
# -----------------------------------------------------------------------------
def get_registered_svm_rules_df():
    registered_rows = []

    for word in sorted(NEGATION_WORDS):
        clean_word = str(word).lower()
        registered_rows.append({
            "rule_type": "negator",
            "rule_name": clean_word,
            "rule_key": "negator:" + clean_word
        })

    for word in sorted(INTENSIFIERS):
        clean_word = str(word).lower()
        registered_rows.append({
            "rule_type": "intensifier",
            "rule_name": clean_word,
            "rule_key": "intensifier:" + clean_word
        })

    for word in sorted(DIMINISHERS):
        clean_word = str(word).lower()
        registered_rows.append({
            "rule_type": "diminisher",
            "rule_name": clean_word,
            "rule_key": "diminisher:" + clean_word
        })

    for word in sorted(POST_CONTRAST_MARKERS):
        clean_word = str(word).lower()
        registered_rows.append({
            "rule_type": "post_contrast_marker",
            "rule_name": clean_word,
            "rule_key": "post_contrast_marker:" + clean_word
        })

    for word in sorted(CONCESSIVE_STARTERS):
        clean_word = str(word).lower()
        registered_rows.append({
            "rule_type": "concessive_marker",
            "rule_name": clean_word,
            "rule_key": "concessive_marker:" + clean_word
        })

    # "even though" exception
    registered_rows.append({
        "rule_type": "concessive_marker",
        "rule_name": "even_though",
        "rule_key": "concessive_marker:even_though"
    })

    for regex_rule in UNIVERSAL_PHRASE_RULES:
        rule_key = regex_rule["rule_key"]
        if rule_key.startswith("phrase:"):
            phrase_name = rule_key.replace("phrase:", "")

            registered_rows.append({
                "rule_type": "phrase",
                "rule_name": phrase_name,
                "rule_key": rule_key
            })

    registered_rules_df = pd.DataFrame(registered_rows)

    registered_rules_df = (
        registered_rules_df
        .drop_duplicates(subset=["rule_key"])
        .sort_values(["rule_type", "rule_name"])
        .reset_index(drop=True)
    )

    return registered_rules_df

def get_rule_key_from_feature(feature):
    if feature.startswith("FEAT_PHRASE_"):
        return "phrase:" + feature.replace("FEAT_PHRASE_", "")

    if feature.startswith("FEAT_NEGATOR_"):
        return "negator:" + feature.replace("FEAT_NEGATOR_", "")

    if feature.startswith("FEAT_INTENSIFIER_"):
        return "intensifier:" + feature.replace("FEAT_INTENSIFIER_", "")

    if feature.startswith("FEAT_DIMINISHER_"):
        return "diminisher:" + feature.replace("FEAT_DIMINISHER_", "")

    if feature.startswith("FEAT_CONTRAST_"):
        return "post_contrast_marker:" + feature.replace("FEAT_CONTRAST_", "")

    if feature.startswith("FEAT_CONCESSIVE_"):
        return "concessive_marker:" + feature.replace("FEAT_CONCESSIVE_", "")

    return None
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# REVIEW LEVEL AUDIT
# -----------------------------------------------------------------------------
def get_probability_for_label(probabilities, classes, label):
    class_to_index = {
        class_label: index
        for index, class_label in enumerate(classes)
    }

    return probabilities[class_to_index[label]]

def get_true_class_margin(probabilities, classes, true_label):
    class_to_index = {class_label: index
                      for index, class_label in enumerate(classes)
                     }

    true_index = class_to_index[true_label]
    true_probability = probabilities[true_index]
    other_probabilities = np.delete(probabilities, true_index)
    strongest_other_probability = other_probabilities.max()

    return true_probability - strongest_other_probability

def classify_svm_effect(true_label, base_prediction, enhanced_prediction):
    base_correct = base_prediction == true_label
    enhanced_correct = enhanced_prediction == true_label
    prediction_changed = base_prediction != enhanced_prediction

    if (not base_correct) and enhanced_correct:
        return "corrected"

    if base_correct and (not enhanced_correct):
        return "harmed"

    if base_correct and enhanced_correct:
        return "stayed_correct"

    if prediction_changed:
        return "wrong_to_wrong"

    return "stayed_wrong"

def create_svm_rule_review_audit_df(
        texts,
        true_labels,
        base_predictions,
        enhanced_predictions,
        base_probabilities,
        enhanced_probabilities,
        base_classes,
        enhanced_classes
):
    audit_rows = []
    texts = texts.reset_index(drop=True)
    true_labels = true_labels.reset_index(drop=True)

    for review_index, text in enumerate(texts):
        true_label = true_labels.iloc[review_index]
        base_prediction = base_predictions[review_index]
        enhanced_prediction = enhanced_predictions[review_index]

        changed_text = svm_append_polarity_features(text)

        polarity_features = extract_polarity_feature_tokens(text)
        filtered_polarity_features = filter_svm_polarity_features(polarity_features)

        rule_keys = sorted(set(
            rule_key
            for rule_key in [
                get_rule_key_from_feature(feature)
                for feature in filtered_polarity_features
            ]
            if rule_key is not None
        ))

        base_margin = get_true_class_margin(
            base_probabilities[review_index],
            base_classes,
            true_label
        )

        enhanced_margin = get_true_class_margin(
            enhanced_probabilities[review_index],
            enhanced_classes,
            true_label
        )

        margin_change = enhanced_margin - base_margin

        base_true_probability = get_probability_for_label(
            base_probabilities[review_index],
            base_classes,
            true_label
        )

        enhanced_true_probability = get_probability_for_label(
            enhanced_probabilities[review_index],
            enhanced_classes,
            true_label
        )

        true_probability_change = enhanced_true_probability - base_true_probability

        base_neg_probability = get_probability_for_label(
            base_probabilities[review_index],
            base_classes,
            "neg"
        )

        base_neu_probability = get_probability_for_label(
            base_probabilities[review_index],
            base_classes,
            "neu"
        )

        base_pos_probability = get_probability_for_label(
            base_probabilities[review_index],
            base_classes,
            "pos"
        )

        enhanced_neg_probability = get_probability_for_label(
            enhanced_probabilities[review_index],
            enhanced_classes,
            "neg"
        )

        enhanced_neu_probability = get_probability_for_label(
            enhanced_probabilities[review_index],
            enhanced_classes,
            "neu"
        )

        enhanced_pos_probability = get_probability_for_label(
            enhanced_probabilities[review_index],
            enhanced_classes,
            "pos"
        )

        effect = classify_svm_effect(
            true_label,
            base_prediction,
            enhanced_prediction
        )

        audit_rows.append({
            "review_index": review_index,
            "original_text": text,
            "changed_text": changed_text,

            "true_label": true_label,
            "base_prediction": base_prediction,
            "enhanced_prediction": enhanced_prediction,
            "base_correct": base_prediction == true_label,
            "enhanced_correct": enhanced_prediction == true_label,
            "prediction_changed": base_prediction != enhanced_prediction,
            "effect": effect,

            "rule_keys": rule_keys,
            "number_of_rules": len(rule_keys),
            "polarity_features": filtered_polarity_features,
            "number_of_features": len(filtered_polarity_features),

            "base_neg_probability": base_neg_probability,
            "base_neu_probability": base_neu_probability,
            "base_pos_probability": base_pos_probability,
            "enhanced_neg_probability": enhanced_neg_probability,
            "enhanced_neu_probability": enhanced_neu_probability,
            "enhanced_pos_probability": enhanced_pos_probability,
            "neg_probability_change": enhanced_neg_probability - base_neg_probability,
            "neu_probability_change": enhanced_neu_probability - base_neu_probability,
            "pos_probability_change": enhanced_pos_probability - base_pos_probability,

            "base_true_class_probability": base_true_probability,
            "enhanced_true_class_probability": enhanced_true_probability,
            "true_class_probability_change": true_probability_change,

            "base_margin": base_margin,
            "enhanced_margin": enhanced_margin,
            "margin_change": margin_change,

            "text_changed": text != changed_text,
            "score_changed": not np.allclose(
                base_probabilities[review_index],
                enhanced_probabilities[review_index],
                atol=1e-12,
                rtol=0.0
            )
        })

    return pd.DataFrame(audit_rows)

def print_short_svm_review_audit_summary(audit_df):
    total_reviews = len(audit_df)

    with_rules_df = audit_df[audit_df["number_of_rules"] > 0]
    without_rules_df = audit_df[audit_df["number_of_rules"] == 0]
    corrected_df = audit_df[audit_df["effect"] == "corrected"]
    harmed_df = audit_df[audit_df["effect"] == "harmed"]
    wrong_to_wrong_df = audit_df[audit_df["effect"] == "wrong_to_wrong"]
    stayed_correct_df = audit_df[audit_df["effect"] == "stayed_correct"]
    stayed_wrong_df = audit_df[audit_df["effect"] == "stayed_wrong"]
    corrected_with_rules = corrected_df[corrected_df["number_of_rules"] > 0]
    corrected_without_rules = corrected_df[corrected_df["number_of_rules"] == 0]
    harmed_with_rules = harmed_df[harmed_df["number_of_rules"] > 0]
    harmed_without_rules = harmed_df[harmed_df["number_of_rules"] == 0]
    wrong_to_wrong_with_rules = wrong_to_wrong_df[wrong_to_wrong_df["number_of_rules"] > 0]
    wrong_to_wrong_without_rules = wrong_to_wrong_df[wrong_to_wrong_df["number_of_rules"] == 0]
    sentiment_changed_df = audit_df[audit_df["prediction_changed"]]

    net_corrections = len(corrected_df) - len(harmed_df)
    net_corrections_with_rules = len(corrected_with_rules) - len(harmed_with_rules)
    net_corrections_without_rules = len(corrected_without_rules) - len(harmed_without_rules)

    summary_rows = [
        {
            "metric": "Total reviews",
            "count": total_reviews
        },
        {
            "metric": "Reviews with at least 1 rule applied",
            "count": len(with_rules_df)
        },
        {
            "metric": "Reviews with no rules applied",
            "count": len(without_rules_df)
        },
        {
            "metric": "Reviews whose sentiment changed",
            "count": len(sentiment_changed_df)
        },
        {
            "metric": "Corrected reviews total",
            "count": len(corrected_df)
        },
        {
            "metric": "Corrected reviews with rules applied",
            "count": len(corrected_with_rules)
        },
        {
            "metric": "Corrected reviews with no rules applied",
            "count": len(corrected_without_rules)
        },
        {
            "metric": "Harmed reviews total",
            "count": len(harmed_df)
        },
        {
            "metric": "Harmed reviews with rules applied",
            "count": len(harmed_with_rules)
        },
        {
            "metric": "Harmed reviews with no rules applied",
            "count": len(harmed_without_rules)
        },
        {
            "metric": "Wrong to wrong changed reviews total",
            "count": len(wrong_to_wrong_df)
        },
        {
            "metric": "Wrong to wrong changed with rules applied",
            "count": len(wrong_to_wrong_with_rules)
        },
        {
            "metric": "Wrong to wrong changed with no rules applied",
            "count": len(wrong_to_wrong_without_rules)
        },
        {
            "metric": "Stayed correct reviews",
            "count": len(stayed_correct_df)
        },
        {
            "metric": "Stayed wrong reviews",
            "count": len(stayed_wrong_df)
        },
        {
            "metric": "Net corrections total",
            "count": net_corrections
        },
        {
            "metric": "Net corrections with rules applied",
            "count": net_corrections_with_rules
        },
        {
            "metric": "Net corrections with no rules applied",
            "count": net_corrections_without_rules
        }
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df["percent"] = (summary_df["count"] / total_reviews * 100).round(2)

    print("\n========== SHORT SVM REVIEW-LEVEL AUDIT SUMMARY ==========")
    print(summary_df.to_string(index=False))

    return summary_df
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# FEATURE LEVEL AUDIT
# -----------------------------------------------------------------------------
def get_svm_feature_family(feature):
    if feature.startswith("FEAT_PHRASE_"):
        return "phrase"

    if feature.startswith("FEAT_NEGATOR_"):
        return "negator"

    if feature.startswith("FEAT_NEG_DEP_"):
        return "negation_dependency"

    if feature.startswith("FEAT_NEG_SCOPE_"):
        return "negation_scope"

    if feature.startswith("FEAT_INTENSIFIER_"):
        return "intensifier"

    if feature.startswith("FEAT_INTENSIFIED_DEP_"):
        return "intensified_dependency"

    if feature.startswith("FEAT_INTENSIFIED_"):
        return "intensified_scope"

    if feature.startswith("FEAT_DIMINISHER_"):
        return "diminisher"

    if feature.startswith("FEAT_DIMINISHED_DEP_"):
        return "diminished_dependency"

    if feature.startswith("FEAT_DIMINISHED_"):
        return "diminished_scope"

    if feature.startswith("FEAT_CONTRAST_"):
        return "contrast_marker"

    if feature.startswith("FEAT_AFTER_CONTRAST_"):
        return "after_contrast_scope"

    if feature.startswith("FEAT_CONCESSIVE_"):
        return "concessive_marker"

    if feature.startswith("FEAT_AFTER_CONCESSION_"):
        return "after_concession_scope"

    if feature == "FEAT_CAPS_EMPHASIS":
        return "caps_emphasis"

    if feature.startswith("FEAT_CAPS_"):
        return "caps_specific_token"

    return "other"

def create_svm_feature_summary_df(svm_rule_review_audit_df):
    exploded_df = svm_rule_review_audit_df.explode("polarity_features")
    exploded_df = exploded_df[exploded_df["polarity_features"].notna()].copy()

    if exploded_df.empty:
        return pd.DataFrame()

    feature_summary_df = (
        exploded_df
        .groupby("polarity_features", as_index=False)
        .agg(
            applications=("review_index", "nunique"),
            corrected=("effect", lambda values: (values == "corrected").sum()),
            harmed=("effect", lambda values: (values == "harmed").sum()),
            mean_margin_change=("margin_change", "mean")
        )
        .rename(columns={
            "polarity_features": "feature"
        })
    )

    feature_summary_df["feature_family"] = feature_summary_df["feature"].apply(get_svm_feature_family)
    feature_summary_df["net_corrections"] = (feature_summary_df["corrected"] - feature_summary_df["harmed"])
    feature_summary_df["decisive_cases"] = (feature_summary_df["corrected"] + feature_summary_df["harmed"])

    feature_summary_df["correction_precision"] = np.where(
        feature_summary_df["decisive_cases"] > 0,
        feature_summary_df["corrected"] / feature_summary_df["decisive_cases"],
        np.nan
    )

    feature_summary_df = (
        feature_summary_df
        .sort_values(
            [
                "net_corrections",
                "applications",
                "mean_margin_change"
            ],
            ascending=[
                True,
                False,
                True
            ]
        )
        .reset_index(drop=True)
    )

    return feature_summary_df

def decide_svm_feature_action(row):
    applications = row["applications"]
    decisive_cases = row["decisive_cases"]
    net_corrections = row["net_corrections"]
    correction_precision = row["correction_precision"]
    mean_margin_change = row["mean_margin_change"]

    if applications < 30:
        return "REVIEW"

    if decisive_cases < 5:
        return "REVIEW"

    if (net_corrections < 0
        and correction_precision < 0.40
        and mean_margin_change < 0.01
    ):
        return "BLOCK"

    if (
        net_corrections >= 3
        and correction_precision >= 0.65
        and mean_margin_change > 0
    ):
        return "KEEP"

    return "REVIEW"


def add_svm_feature_decisions(feature_summary_df):
    feature_summary_df = feature_summary_df.copy()

    feature_summary_df["decision"] = feature_summary_df.apply(
        decide_svm_feature_action,
        axis=1
    )

    return feature_summary_df

def print_svm_feature_audit(feature_summary_df, top_n=100):
    print("\n========== SVM FEATURE-LEVEL AUDIT ==========")
    print("Total unique polarity features:", len(feature_summary_df))

    print("\n========== MOST HARMFUL FEATURES ==========")
    harmful_features_df = (
        feature_summary_df
        .sort_values(
            [
                "net_corrections",
                "correction_precision",
                "mean_margin_change"
            ],
            ascending=[
                True,
                True,
                True
            ]
        )
        .head(top_n)
    )

    print(
        harmful_features_df[
            [
                "feature",
                "feature_family",
                "applications",
                "corrected",
                "harmed",
                "net_corrections",
                "decisive_cases",
                "correction_precision",
                "mean_margin_change",
                "decision"
            ]
        ].to_string(index=False)
    )

    print("\n========== MOST HELPFUL FEATURES ==========")
    helpful_features_df = (
        feature_summary_df
        .sort_values(
            [
                "net_corrections",
                "correction_precision",
                "mean_margin_change"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .head(top_n)
    )

    print(
        helpful_features_df[
            [
                "feature",
                "feature_family",
                "applications",
                "corrected",
                "harmed",
                "net_corrections",
                "decisive_cases",
                "correction_precision",
                "mean_margin_change",
                "decision"
            ]
        ].to_string(index=False)
    )

    print("\n========== BLOCK FEATURES ==========")
    blocked_features_df = (
        feature_summary_df[
            feature_summary_df["decision"] == "BLOCK"
        ]
        .sort_values(
            [
                "net_corrections",
                "correction_precision",
                "mean_margin_change"
            ],
            ascending=[
                True,
                True,
                True
            ]
        )
        .reset_index(drop=True)
    )

    if blocked_features_df.empty:
        print("No features were marked as BLOCK.")
    else:
        print(
            blocked_features_df[
                [
                    "feature",
                    "feature_family",
                    "applications",
                    "corrected",
                    "harmed",
                    "net_corrections",
                    "decisive_cases",
                    "correction_precision",
                    "mean_margin_change",
                    "decision"
                ]
            ].to_string(index=False)
        )

    print("\n========== FEATURE FAMILY SUMMARY ==========")
    family_summary_df = (
        feature_summary_df
        .groupby("feature_family", as_index=False)
        .agg(
            features=("feature", "nunique"),
            applications=("applications", "sum"),
            corrected=("corrected", "sum"),
            harmed=("harmed", "sum"),
            net_corrections=("net_corrections", "sum"),
            mean_margin_change=("mean_margin_change", "mean")
        )
    )
    family_summary_df["decisive_cases"] = (family_summary_df["corrected"] + family_summary_df["harmed"])
    family_summary_df["correction_precision"] = np.where(
        family_summary_df["decisive_cases"] > 0,
        family_summary_df["corrected"] / family_summary_df["decisive_cases"],
        np.nan
    )
    family_summary_df = (
        family_summary_df
        .sort_values(
            ["net_corrections", "applications"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )

    print(
        family_summary_df[
            [
                "feature_family",
                "features",
                "applications",
                "corrected",
                "harmed",
                "net_corrections",
                "decisive_cases",
                "correction_precision",
                "mean_margin_change"
            ]
        ].to_string(index=False)
    )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# RULE LEVEL AUDIT
# -----------------------------------------------------------------------------
def create_svm_rule_summary_df(rule_review_audit_df):
    registered_rules_df = get_registered_svm_rules_df()
    exploded_df = rule_review_audit_df.explode("rule_keys")
    exploded_df = exploded_df[exploded_df["rule_keys"].notna()].copy()

    if exploded_df.empty:
        registered_rules_df["applications"] = 0
        registered_rules_df["corrected"] = 0
        registered_rules_df["harmed"] = 0
        registered_rules_df["net_corrections"] = 0
        registered_rules_df["correction_precision"] = np.nan
        registered_rules_df["mean_margin_change"] = np.nan
        registered_rules_df["status"] = "UNUSED"

        return registered_rules_df

    rule_usage_df = (
        exploded_df
        .groupby("rule_keys", as_index=False)
        .agg(
            applications=("review_index", "nunique"),
            corrected=("effect", lambda values: (values == "corrected").sum()),
            harmed=("effect", lambda values: (values == "harmed").sum()),
            mean_margin_change=("margin_change", "mean")
        )
        .rename(columns={
            "rule_keys": "rule_key"
        })
    )

    rule_summary_df = registered_rules_df.merge(
        rule_usage_df,
        on="rule_key",
        how="left"
    )

    for column in ["applications", "corrected", "harmed"]:
        rule_summary_df[column] = (rule_summary_df[column].fillna(0).astype(int))

    rule_summary_df["net_corrections"] = (rule_summary_df["corrected"] - rule_summary_df["harmed"])
    decisive_cases = (rule_summary_df["corrected"] + rule_summary_df["harmed"])

    rule_summary_df["correction_precision"] = np.where(
        decisive_cases > 0,
        rule_summary_df["corrected"] / decisive_cases,
        np.nan
    )
    rule_summary_df["status"] = np.where(
        rule_summary_df["applications"] > 0,
        "USED",
        "UNUSED"
    )

    rule_summary_df = (
        rule_summary_df
        .sort_values(
            [
                "net_corrections",
                "applications",
                "mean_margin_change"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return rule_summary_df

def print_svm_rule_audit(rule_summary_df):
    total_rules = len(rule_summary_df)
    rules_used = int((rule_summary_df["applications"] > 0).sum())
    rules_not_used = int((rule_summary_df["applications"] == 0).sum())

    print("\n========== SVM RULE-LEVEL AUDIT ==========")
    print("Total rules:", total_rules)
    print("Rules used:", rules_used)
    print("Rules not used:", rules_not_used)

    print("\n========== RULE SUMMARY TABLE ==========")
    print(
        rule_summary_df[
            [
                "rule_key",
                "applications",
                "corrected",
                "harmed",
                "net_corrections",
                "correction_precision",
                "mean_margin_change",
                "status"
            ]
        ].to_string(index=False)
    )

    print("\n========== RULES NOT USED ==========")
    unused_rules_df = rule_summary_df[
        rule_summary_df["applications"] == 0
    ]

    if unused_rules_df.empty:
        print("Every registered rule was used at least once.")
    else:
        for rule in unused_rules_df['rule_key']:
            print(rule)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EXCLUSIVE RULE AUDIT
# -----------------------------------------------------------------------------
def decide_rule_action(row):
    exclusive_applications = row["exclusive_applications"]
    decisive_changes = row["decisive_changes"]
    net_corrections = row["net_corrections"]
    correction_precision = row["correction_precision"]
    mean_margin_change = row["mean_margin_change"]

    if net_corrections < 0:
        return "CULL"
    if exclusive_applications < 30:
        return "REVIEW"
    if decisive_changes < 10:
        return "REVIEW"
    if pd.isna(correction_precision):
        return "REVIEW"
    if (net_corrections >= 0 and correction_precision >= 0.60 and mean_margin_change > 0):
        return "KEEP"

    return "REVIEW"

def create_exclusive_svm_rule_summary_df(rule_review_audit_df):
    registered_rules_df = get_registered_svm_rules_df()
    overall_rule_summary_df = create_svm_rule_summary_df(rule_review_audit_df)

    overall_usage_df = overall_rule_summary_df[[
        "rule_key", "applications"
        ]].rename(columns={
            "applications": "all_applications"
            })

    exclusive_df = rule_review_audit_df[rule_review_audit_df["number_of_rules"] == 1].copy()

    if exclusive_df.empty:
        exclusive_summary_df = registered_rules_df.merge(
            overall_usage_df,
            on="rule_key",
            how="left"
        )

        exclusive_summary_df["all_applications"] = (
            exclusive_summary_df["all_applications"]
            .fillna(0)
            .astype(int)
        )

        exclusive_summary_df["exclusive_applications"] = 0
        exclusive_summary_df["corrected"] = 0
        exclusive_summary_df["harmed"] = 0
        exclusive_summary_df["prediction_changed"] = 0
        exclusive_summary_df["decisive_changes"] = 0
        exclusive_summary_df["net_corrections"] = 0
        exclusive_summary_df["correction_precision"] = np.nan
        exclusive_summary_df["mean_margin_change"] = np.nan
        exclusive_summary_df["exclusive_status"] = "NO_EXCLUSIVE_EVIDENCE"
        exclusive_summary_df["decision"] = "REVIEW"

        return exclusive_summary_df

    exclusive_df["exclusive_rule"] = exclusive_df["rule_keys"].apply(lambda rules: rules[0])

    exclusive_usage_df = (
        exclusive_df
        .groupby("exclusive_rule", as_index=False)
        .agg(
            exclusive_applications=("review_index", "nunique"),
            corrected=("effect", lambda values: (values == "corrected").sum()),
            harmed=("effect", lambda values: (values == "harmed").sum()),
            prediction_changed=("prediction_changed", "sum"),
            mean_margin_change=("margin_change", "mean")
        )
        .rename(columns={
            "exclusive_rule": "rule_key"
        })
    )

    exclusive_summary_df = registered_rules_df.merge(
        overall_usage_df,
        on="rule_key",
        how="left"
    )

    exclusive_summary_df = exclusive_summary_df.merge(
        exclusive_usage_df,
        on="rule_key",
        how="left"
    )

    count_columns = [
        "all_applications",
        "exclusive_applications",
        "corrected",
        "harmed",
        "prediction_changed"
    ]

    for column in count_columns:
        exclusive_summary_df[column] = (
            exclusive_summary_df[column]
            .fillna(0)
            .astype(int)
        )

    exclusive_summary_df["decisive_changes"] = (exclusive_summary_df["corrected"] + exclusive_summary_df["harmed"])
    exclusive_summary_df["net_corrections"] = (exclusive_summary_df["corrected"] - exclusive_summary_df["harmed"])

    exclusive_summary_df["correction_precision"] = np.where(
        exclusive_summary_df["decisive_changes"] > 0,
        exclusive_summary_df["corrected"] / exclusive_summary_df["decisive_changes"],
        np.nan
    )
    exclusive_summary_df["exclusive_status"] = np.where(
        exclusive_summary_df["exclusive_applications"] > 0,
        "HAS_EXCLUSIVE_EVIDENCE",
        "NO_EXCLUSIVE_EVIDENCE"
    )
    exclusive_summary_df["decision"] = exclusive_summary_df.apply(
        decide_rule_action,
        axis=1
    )

    exclusive_summary_df = (
        exclusive_summary_df
        .sort_values(
            [
                "net_corrections",
                "correction_precision",
                "exclusive_applications"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return exclusive_summary_df

def print_exclusive_svm_rule_summary(exclusive_summary_df):
    total_rules = len(exclusive_summary_df)

    rules_used_anywhere = int((exclusive_summary_df["all_applications"] > 0).sum())
    rules_with_exclusive_evidence = int((exclusive_summary_df["exclusive_applications"] > 0).sum())
    rules_without_exclusive_evidence = int((exclusive_summary_df["exclusive_applications"] == 0).sum())

    print("\n========== EXCLUSIVE SVM RULE SUMMARY ==========")
    print("Total registered rules:", total_rules)
    print("Rules used anywhere:", rules_used_anywhere)
    print("Rules with exclusive evidence:", rules_with_exclusive_evidence)
    print("Rules without exclusive evidence:", rules_without_exclusive_evidence)

    print("\n========== EXCLUSIVE-RULE TABLE ==========")
    print(
        exclusive_summary_df[
            [
                "rule_key",
                "all_applications",
                "exclusive_applications",
                "corrected",
                "harmed",
                "decisive_changes",
                "net_corrections",
                "correction_precision",
                "mean_margin_change",
                "exclusive_status",
                "decision"
            ]
        ].to_string(index=False)
    )

    for decision in ["KEEP", "CULL", "REVIEW"]:
        print("\n========== EXCLUSIVE " + decision + " ==========")

        selected_rules = exclusive_summary_df[
            exclusive_summary_df["decision"] == decision
        ]["rule_key"].tolist()

        if len(selected_rules) == 0:
            print("None")
        else:
            for rule in selected_rules:
                print(rule)

# -----------------------------------------------------------------------------
# START
# SCOPED-EXCLUSIVE RULE AUDIT
# -----------------------------------------------------------------------------
def get_svm_rule_scope(rule_key):
    rule_key = str(rule_key).lower()

    if rule_key.startswith("phrase:"):
        return "phrase"

    if rule_key.startswith("negator:"):
        return "negator"

    if rule_key.startswith("intensifier:"):
        return "intensifier"

    if rule_key.startswith("diminisher:"):
        return "diminisher"

    if rule_key.startswith("post_contrast_marker:"):
        return "contrast"

    if rule_key.startswith("concessive_marker:"):
        return "contrast"

    return "other"


def get_svm_rules_in_scope(rule_keys, target_scope):
    return [
        rule_key
        for rule_key in rule_keys
        if get_svm_rule_scope(rule_key) == target_scope
    ]


def create_scoped_exclusive_svm_rule_summary_df(
        rule_review_audit_df,
        target_scope
):
    registered_rules_df = get_registered_svm_rules_df()
    registered_rules_df = registered_rules_df.copy()

    registered_rules_df["scope"] = registered_rules_df["rule_key"].apply(
        get_svm_rule_scope
    )

    scoped_registered_rules_df = registered_rules_df[
        registered_rules_df["scope"] == target_scope
    ].copy()

    overall_rule_summary_df = create_svm_rule_summary_df(
        rule_review_audit_df
    )

    overall_usage_df = overall_rule_summary_df[
        [
            "rule_key",
            "applications"
        ]
    ].rename(columns={
        "applications": "all_applications"
    })

    scoped_rows = []

    for _, rule_row in scoped_registered_rules_df.iterrows():
        rule_key = rule_row["rule_key"]

        scoped_exclusive_rows = []

        for _, review_row in rule_review_audit_df.iterrows():
            rule_keys = review_row["rule_keys"]

            scoped_rules = get_svm_rules_in_scope(
                rule_keys,
                target_scope
            )

            is_scoped_exclusive = (
                len(scoped_rules) == 1
                and scoped_rules[0] == rule_key
            )

            if is_scoped_exclusive:
                scoped_exclusive_rows.append(review_row)

        if len(scoped_exclusive_rows) == 0:
            scoped_rows.append({
                "scope": target_scope,
                "rule_key": rule_key,
                "all_applications": 0,
                "scoped_exclusive_applications": 0,
                "corrected": 0,
                "harmed": 0,
                "prediction_changed": 0,
                "decisive_changes": 0,
                "net_corrections": 0,
                "correction_precision": np.nan,
                "mean_margin_change": np.nan,
                "scoped_exclusive_status": "NO_SCOPED_EXCLUSIVE_EVIDENCE",
                "decision": "REVIEW"
            })

            continue

        scoped_exclusive_df = pd.DataFrame(scoped_exclusive_rows)

        corrected = int((scoped_exclusive_df["effect"] == "corrected").sum())
        harmed = int((scoped_exclusive_df["effect"] == "harmed").sum())

        prediction_changed = int(
            scoped_exclusive_df["prediction_changed"].sum()
        )

        scoped_exclusive_applications = int(
            scoped_exclusive_df["review_index"].nunique()
        )

        decisive_changes = corrected + harmed
        net_corrections = corrected - harmed

        if decisive_changes > 0:
            correction_precision = corrected / decisive_changes
        else:
            correction_precision = np.nan

        mean_margin_change = float(
            scoped_exclusive_df["margin_change"].mean()
        )

        scoped_rows.append({
            "scope": target_scope,
            "rule_key": rule_key,
            "all_applications": 0,
            "scoped_exclusive_applications": scoped_exclusive_applications,
            "corrected": corrected,
            "harmed": harmed,
            "prediction_changed": prediction_changed,
            "decisive_changes": decisive_changes,
            "net_corrections": net_corrections,
            "correction_precision": correction_precision,
            "mean_margin_change": mean_margin_change,
            "scoped_exclusive_status": "HAS_SCOPED_EXCLUSIVE_EVIDENCE",
            "decision": "REVIEW"
        })

    scoped_summary_df = pd.DataFrame(scoped_rows)

    scoped_summary_df = scoped_summary_df.merge(
        overall_usage_df,
        on="rule_key",
        how="left",
        suffixes=("", "_overall")
    )

    scoped_summary_df["all_applications"] = (
        scoped_summary_df["all_applications_overall"]
        .fillna(0)
        .astype(int)
    )

    scoped_summary_df = scoped_summary_df.drop(
        columns=["all_applications_overall"]
    )

    # Reuse your existing exclusive decision function by temporarily
    # giving it the column name it expects.
    decision_input_df = scoped_summary_df.copy()
    decision_input_df["exclusive_applications"] = (
        decision_input_df["scoped_exclusive_applications"]
    )

    scoped_summary_df["decision"] = decision_input_df.apply(
        decide_rule_action,
        axis=1
    )

    scoped_summary_df = (
        scoped_summary_df
        .sort_values(
            [
                "net_corrections",
                "correction_precision",
                "scoped_exclusive_applications"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return scoped_summary_df


def print_scoped_exclusive_svm_rule_summary(scoped_summary_df, target_scope):
    total_rules = len(scoped_summary_df)

    rules_used_anywhere = int(
        (scoped_summary_df["all_applications"] > 0).sum()
    )

    rules_with_scoped_exclusive_evidence = int(
        (scoped_summary_df["scoped_exclusive_applications"] > 0).sum()
    )

    rules_without_scoped_exclusive_evidence = int(
        (scoped_summary_df["scoped_exclusive_applications"] == 0).sum()
    )

    print(
        "\n========== SCOPED-EXCLUSIVE "
        + target_scope.upper()
        + " RULE SUMMARY =========="
    )

    print("Total registered rules:", total_rules)
    print("Rules used anywhere:", rules_used_anywhere)
    print("Rules with scoped-exclusive evidence:", rules_with_scoped_exclusive_evidence)
    print("Rules without scoped-exclusive evidence:", rules_without_scoped_exclusive_evidence)

    print(
        "\n========== SCOPED-EXCLUSIVE "
        + target_scope.upper()
        + " RULE TABLE =========="
    )

    print(
        scoped_summary_df[
            [
                "scope",
                "rule_key",
                "all_applications",
                "scoped_exclusive_applications",
                "corrected",
                "harmed",
                "decisive_changes",
                "net_corrections",
                "correction_precision",
                "mean_margin_change",
                "scoped_exclusive_status",
                "decision"
            ]
        ].to_string(index=False)
    )

    for decision in ["KEEP", "CULL", "REVIEW"]:
        print(
            "\n========== SCOPED-EXCLUSIVE "
            + target_scope.upper()
            + " "
            + decision
            + " =========="
        )

        selected_rules = scoped_summary_df[
            scoped_summary_df["decision"] == decision
        ]["rule_key"].tolist()

        if len(selected_rules) == 0:
            print("None")
        else:
            for rule in selected_rules:
                print(rule)


def print_all_scoped_exclusive_svm_rule_summaries(rule_review_audit_df):
    scoped_tables = {}

    for target_scope in [
        "phrase",
        "negator",
        "intensifier",
        "diminisher",
        "contrast"
    ]:
        scoped_summary_df = create_scoped_exclusive_svm_rule_summary_df(
            rule_review_audit_df=rule_review_audit_df,
            target_scope=target_scope
        )

        scoped_tables[target_scope] = scoped_summary_df

        print_scoped_exclusive_svm_rule_summary(
            scoped_summary_df=scoped_summary_df,
            target_scope=target_scope
        )

    return scoped_tables
# ----------------------------------------------------------------------------- END
# SCOPED-EXCLUSIVE RULE AUDIT
# -----------------------------------------------------------------------------
def get_svm_feature_operation_text(feature):
    rule_key = get_rule_key_from_feature(feature)

    if feature.startswith("FEAT_PHRASE_"):
        return feature + " | added phrase polarity feature | " + str(rule_key)

    if feature.startswith("FEAT_NEGATOR_"):
        return feature + " | detected negation word | " + str(rule_key)

    if feature.startswith("FEAT_NEG_SCOPE_"):
        return feature + " | added negation-scope feature"

    if feature.startswith("FEAT_NEG_DEP_"):
        return feature + " | added dependency negation feature"

    if feature.startswith("FEAT_INTENSIFIER_"):
        return feature + " | detected intensifier word | " + str(rule_key)

    if feature.startswith("FEAT_INTENSIFIED_"):
        return feature + " | added intensified-scope feature"

    if feature.startswith("FEAT_INTENSIFIED_DEP_"):
        return feature + " | added dependency intensifier feature"

    if feature.startswith("FEAT_DIMINISHER_"):
        return feature + " | detected diminisher word | " + str(rule_key)

    if feature.startswith("FEAT_DIMINISHED_"):
        return feature + " | added diminished-scope feature"

    if feature.startswith("FEAT_DIMINISHED_DEP_"):
        return feature + " | added dependency diminisher feature"

    if feature.startswith("FEAT_CONTRAST_"):
        return feature + " | detected contrast marker | " + str(rule_key)

    if feature.startswith("FEAT_AFTER_CONTRAST_"):
        return feature + " | added after-contrast scope feature"

    if feature.startswith("FEAT_CONCESSIVE_"):
        return feature + " | detected concessive marker | " + str(rule_key)

    if feature.startswith("FEAT_AFTER_CONCESSION_"):
        return feature + " | added after-concession scope feature"

    if feature == "FEAT_CAPS_EMPHASIS":
        return feature + " | detected capital-letter emphasis"

    return feature + " | polarity feature added"

def svm_review_example_block(row):
    print("\n" + "-" * 120)

    print("REVIEW INDEX:", row["review_index"])

    print("\nORIGINAL REVIEW:")
    print('"' + str(row["original_text"]) + '"')

    print("\nCHANGED REVIEW:")
    print('"' + str(row["changed_text"]) + '"')

    print("\nRULES APPLIED:")
    if len(row["rule_keys"]) == 0:
        print("No registered rules applied.")
    else:
        print(row["rule_keys"])

    print("\nFULL OPERATIONS:")
    if len(row["polarity_features"]) == 0:
        print("- No polarity features added")
    else:
        for feature in row["polarity_features"]:
            print("- " + get_svm_feature_operation_text(feature))

    print("\nSENTIMENT:")
    print("True label:", row["true_label"])
    print(
        "Raw prediction:",
        row["base_prediction"],
        "| Correct:",
        row["base_correct"]
    )
    print(
        "Enhanced prediction:",
        row["enhanced_prediction"],
        "| Correct:",
        row["enhanced_correct"]
    )

    print("\nSCORE CHANGE (Raw -> Enhanced):")
    print(
        "Negative:",
        round(row["base_neg_probability"], 4),
        "->",
        round(row["enhanced_neg_probability"], 4),
        "| Change:",
        round(row["neg_probability_change"], 4)
    )
    print(
        "Neutral:",
        round(row["base_neu_probability"], 4),
        "->",
        round(row["enhanced_neu_probability"], 4),
        "| Change:",
        round(row["neu_probability_change"], 4)
    )
    print(
        "Positive:",
        round(row["base_pos_probability"], 4),
        "->",
        round(row["enhanced_pos_probability"], 4),
        "| Change:",
        round(row["pos_probability_change"], 4)
    )

    print("\nTRUE-CLASS MARGIN:")
    print(
        "Raw margin:",
        round(row["base_margin"], 4),
        "-> Enhanced margin:",
        round(row["enhanced_margin"], 4),
        "| Margin change:",
        round(row["margin_change"], 4)
    )

    print("\nFLAGS:")
    print(
        "Text changed:",
        row["text_changed"],
        "| Score changed:",
        row["score_changed"],
        "| Prediction changed:",
        row["prediction_changed"]
    )

def svm_review_examples(title, selected_df, number=0):
    if number > 0:
        sample_size = min(number, len(selected_df))
    else:
        sample_size = len(selected_df)
    
    print("\n========== " + title + " ==========")
    print("Count:", len(selected_df))

    if selected_df.empty:
        print("No examples found.")
        return

    selected_df = selected_df.copy()
    selected_df = selected_df.sort_values(
        [
            "number_of_rules",
            "prediction_changed",
            "margin_change"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    selected_df = selected_df.head(sample_size)

    for _, row in selected_df.iterrows():
        svm_review_example_block(row)

def print_all_short_svm_review_examples(audit_df, number=0):
    corrected_df = audit_df[audit_df["effect"] == "corrected"]
    harmed_df = audit_df[audit_df["effect"] == "harmed"]
    wrong_to_wrong_df = audit_df[audit_df["effect"] == "wrong_to_wrong"]
    stayed_wrong_df = audit_df[audit_df["effect"] == "stayed_wrong"]
    stayed_correct_df = audit_df[audit_df["effect"] == "stayed_correct"]

    no_rule_but_changed_df = audit_df[
        (audit_df["number_of_rules"] == 0)
        & (audit_df["prediction_changed"])
    ]

    svm_review_examples(
        "CORRECTED REVIEWS",
        corrected_df,
        number=number
    )
    svm_review_examples(
        "HARMED REVIEWS",
        harmed_df,
        number=number
    )
    svm_review_examples(
        "WRONG TO WRONG REVIEWS",
        wrong_to_wrong_df,
        number=number
    )
    svm_review_examples(
        "STAYED WRONG REVIEWS",
        stayed_wrong_df,
        number=number
    )
    svm_review_examples(
        "STAYED CORRECT REVIEWS",
        stayed_correct_df,
        number=number
    )
    svm_review_examples(
        "NO RULES APPLIED BUT SENTIMENT CHANGED",
        no_rule_but_changed_df,
        number=number
    )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- Start
# ENHANCED SVM HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------- 
N_WORKERS = min(8, os.cpu_count())
print("Precomputing Enhanced Validation Text...")
enhanced_text_val = pd.Series(
    process_map(
        svm_append_polarity_features,
        text_val.tolist(),
        max_workers=N_WORKERS,
        chunksize=100,
        desc="Validation Set"
    ),
    index=text_val.index
)
print("Precomputing Enhanced Training Text...")
enhanced_text_train = pd.Series(
    process_map(
        svm_append_polarity_features,
        text_train.tolist(),
        max_workers=N_WORKERS,
        chunksize=100,
        desc="Train Set"
    ),
    index=text_train.index
)
print("Precomputing Enhanced Test Text...")
enhanced_text_test = pd.Series(
    process_map(
        svm_append_polarity_features,
        text_test.tolist(),
        max_workers=N_WORKERS,
        chunksize=100,
        desc="Test Set"
    ),
    index=text_test.index
)

def enhanced_svm_optuna(trial):
    vectorizer_type = trial.suggest_categorical("vectorizer_type", ["tfidf", "count"])

    if vectorizer_type == 'tfidf':
        optuna_vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            max_features=trial.suggest_categorical(
                'max_features',
                [30000, 50000, 100000, 150000]
            ),
            ngram_range=ngram_map[trial.suggest_categorical(
                'ngram_range',
                ["1_1", "1_2", "1_3"]
            )],
            min_df=trial.suggest_categorical(
                'min_df',
                [5, 10, 20, 50]
            ),
            max_df=trial.suggest_categorical(
                "max_df",
                [0.90, 0.95, 0.98]
            ),
            sublinear_tf=trial.suggest_categorical(
                'sublinear_tf',
                [True, False]
            )
        )
    else:
        optuna_vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            max_features=trial.suggest_categorical(
                "max_features",
                [30000, 50000, 100000, 150000]
            ),
            ngram_range=ngram_map[trial.suggest_categorical(
                "ngram_range",
                ["1_1", "1_2", "1_3"]
            )],
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
        )

    optuna_svc = LinearSVC(
        C=trial.suggest_float(
            'C',
            0.1,
            10,
            log=True
        ),
        random_state=42,
        max_iter=30000
    )

    optuna_svm_pipeline = ImbPipeline([
        ('vec', optuna_vectorizer),
        ('clf', optuna_svc)
    ])

    optuna_svm_pipeline.fit(
        enhanced_text_train,
        sentiment_train
    )

    validation_predictions = optuna_svm_pipeline.predict(
        enhanced_text_val
    )

    validation_macro_f1 = f1_score(
        sentiment_val,
        validation_predictions,
        average="macro"
    )

    return validation_macro_f1
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE ENHANCED SVM
# -----------------------------------------------------------------------------
print("\n========== ENHANCED SVM ==========")
enhanced_svm_study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
enhanced_svm_study.optimize(
    enhanced_svm_optuna,
    n_trials=20,
    n_jobs=1
)
enhanced_svm_best = enhanced_svm_study.best_params

if enhanced_svm_best['vectorizer_type'] == 'tfidf':
    enhanced_svm_vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            max_features=enhanced_svm_best['max_features'],
            ngram_range=ngram_map[enhanced_svm_best['ngram_range']],
            min_df=enhanced_svm_best['min_df'],
            max_df=enhanced_svm_best['max_df'],
            sublinear_tf=enhanced_svm_best['sublinear_tf']
    )
else:
    enhanced_svm_vectorizer = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        max_features=enhanced_svm_best['max_features'],
        ngram_range=ngram_map[enhanced_svm_best['ngram_range']],
        min_df=enhanced_svm_best['min_df'],
        max_df=enhanced_svm_best['max_df'],
        binary=enhanced_svm_best['binary']
    )

enhanced_svc = LinearSVC(
    C=enhanced_svm_best['C'],
    random_state=42,
    max_iter=30000
)

enhanced_svm_uncalibrated_pipeline = ImbPipeline([
    ("vec", enhanced_svm_vectorizer),
    ("clf", enhanced_svc)
])

enhanced_svm_calibrated_pipeline = CalibratedClassifierCV(
    estimator=enhanced_svm_uncalibrated_pipeline,
    cv=calibration_cv,
    method='sigmoid', 
    ensemble=False,
    n_jobs=3
)

enhanced_svm_pipeline = enhanced_svm_calibrated_pipeline

progress = tqdm(total=1, desc="Enhanced SVM")
progress.set_description("Fitting Enhanced SVM")
enhanced_svm_pipeline.fit(enhanced_text_train, sentiment_train)
progress.update(1)
progress.close()

enhanced_svm_val_sentiment = predict_with_progress(
    enhanced_svm_pipeline,
    enhanced_text_val,
    batch_size=5000,
    desc="Predicting Validation Sentiments With Enhanced SVM"
)

enhanced_svm_test_sentiment = predict_with_progress(
    enhanced_svm_pipeline,
    enhanced_text_test,
    batch_size=5000,
    desc="Predicting Test Sentiments With Enhanced SVM"
)

print("\nENHANCED SVM BEST PARAMETERS: " + str(enhanced_svm_study.best_value))
print(enhanced_svm_study.best_params)

print("\nBASE SVM ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_svm_val_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_val, base_svm_val_sentiment, digits=4))
print("ENHANCED SVM ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, enhanced_svm_val_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_val, enhanced_svm_val_sentiment, digits=4))

print("\nBASE SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_svm_test_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_test, base_svm_test_sentiment, digits=4))
print("ENHANCED SVM ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_svm_test_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_test, enhanced_svm_test_sentiment, digits=4))

enhanced_svm_val_probabilities = predict_proba_with_progress(
    enhanced_svm_pipeline,
    enhanced_text_val,
    batch_size=5000,
    desc="Predicting Validation Probabilities With Enhanced SVM"
)

enhanced_svm_test_probabilities = predict_proba_with_progress(
    enhanced_svm_pipeline,
    enhanced_text_test,
    batch_size=5000,
    desc="Predicting Test Probabilities With Enhanced SVM"
)

enhanced_svm_train_probabilities, enhanced_svm_probability_classes = cross_val_predict_proba_with_progress(
    enhanced_svm_pipeline,
    enhanced_text_train,
    sentiment_train,
    base_cv,
    desc="Predicting OOF Train Probabilities With Enhanced SVM"
)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SVM AUDIT ON VALIDATION SET
# -----------------------------------------------------------------------------
svm_classes = np.array(["neg", "neu", "pos"])

assert np.array_equal(
    svm_classes,
    enhanced_svm_probability_classes
), "Class order mismatch!"

assert np.array_equal(
    svm_classes,
    enhanced_svm_pipeline.classes_
), "Class order mismatch!"

svm_rule_review_audit_df = create_svm_rule_review_audit_df(
    texts=text_val,
    true_labels=sentiment_val,
    base_predictions=base_svm_val_sentiment,
    enhanced_predictions=enhanced_svm_val_sentiment,
    base_probabilities=base_svm_val_probabilities,
    enhanced_probabilities=enhanced_svm_val_probabilities,
    base_classes=svm_classes,
    enhanced_classes=svm_classes
)

short_svm_review_summary_df = print_short_svm_review_audit_summary(svm_rule_review_audit_df)

svm_rule_summary_df = create_svm_rule_summary_df(svm_rule_review_audit_df)
print_svm_rule_audit(svm_rule_summary_df)

exclusive_svm_rule_summary_df = create_exclusive_svm_rule_summary_df(svm_rule_review_audit_df)
print_exclusive_svm_rule_summary(exclusive_svm_rule_summary_df)

svm_feature_summary_df = create_svm_feature_summary_df(svm_rule_review_audit_df)
svm_feature_summary_df = add_svm_feature_decisions(svm_feature_summary_df)
print_svm_feature_audit(svm_feature_summary_df, top_n=100)

scoped_exclusive_svm_rule_tables = print_all_scoped_exclusive_svm_rule_summaries(
    svm_rule_review_audit_df
)

print_all_short_svm_review_examples(svm_rule_review_audit_df, number=1)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
output_folder = "Base_Learner/Results/SVM/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_svm_train_probabilities_df = pd.DataFrame(enhanced_svm_train_probabilities).rename(columns={
    0: "enhanced_svm_neg",
    1: "enhanced_svm_neu",
    2: "enhanced_svm_pos"
})

enhanced_svm_val_probabilities_df = pd.DataFrame(enhanced_svm_val_probabilities).rename(columns={
    0: "enhanced_svm_neg",
    1: "enhanced_svm_neu",
    2: "enhanced_svm_pos"
})

enhanced_svm_test_probabilities_df = pd.DataFrame(enhanced_svm_test_probabilities).rename(columns={
    0: "enhanced_svm_neg",
    1: "enhanced_svm_neu",
    2: "enhanced_svm_pos"
})

enhanced_svm_train_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_train_probabilities.csv"),
    index=False
)

enhanced_svm_val_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_val_probabilities.csv"),
    index=False
)

enhanced_svm_test_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_test_probabilities.csv"),
    index=False
)

print("\nSaved SVM Probabilities CSV Files to:", output_folder)

enhanced_svm_val_sentiment_df = pd.DataFrame({
    "enhanced_svm_sentiment": enhanced_svm_val_sentiment
})

enhanced_svm_test_sentiment_df = pd.DataFrame({
    "enhanced_svm_sentiment": enhanced_svm_test_sentiment
})

enhanced_svm_val_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_val_sentiment.csv"),
    index=False
)

enhanced_svm_test_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_test_sentiment.csv"),
    index=False
)

print("Saved SVM Sentiment CSV Files to:", output_folder)

output_folder = "Base_Learner/Rule_Decisions/SVM"
os.makedirs(output_folder, exist_ok=True)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_short_svm_review_examples(svm_rule_review_audit_df, number=0)
text_output = output.getvalue()
with open(os.path.join(output_folder, "rule_affected_reviews.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_svm_rule_audit(svm_rule_summary_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "all_rule_usage_table.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_exclusive_svm_rule_summary(exclusive_svm_rule_summary_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "exclusive_rule_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_svm_feature_audit(svm_feature_summary_df, top_n=100)
text_output = output.getvalue()
with open(os.path.join(output_folder, "feature_audit_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_scoped_exclusive_svm_rule_summaries(svm_rule_review_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "scoped_exclusive_rule_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_short_svm_review_audit_summary(svm_rule_review_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "enhanced_svm_audit_summary.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

print("Saved SVM Audit Text Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/SVM/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_svm_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        enhanced_svm_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_svm_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        enhanced_svm_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_svm_val_report_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_validation_classification_report.csv"),
    index_label="class"
)

enhanced_svm_test_report_df.to_csv(
    os.path.join(output_folder, "enhanced_svm_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_svm_val_report_df.values,
    rowLabels=enhanced_svm_val_report_df.index,
    colLabels=enhanced_svm_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_svm_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_svm_test_report_df.values,
    rowLabels=enhanced_svm_test_report_df.index,
    colLabels=enhanced_svm_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_svm_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Enhanced SVM Classification Report to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END