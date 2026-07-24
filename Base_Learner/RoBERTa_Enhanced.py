import pandas as pd  # For reading CSV files
import numpy as np  # Used to combine outputs for meta classifier
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # RoBERTa (Tokeniser and Classifier)
from scipy.special import softmax  # To convert into probability
from sklearn.model_selection import train_test_split  # Splits dataset
from sklearn.metrics import classification_report, f1_score  # Output metrics
from sklearn.metrics import accuracy_score  # Output metrics
import optuna # Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold
import torch
import matplotlib.pyplot as plt
import re
import os
import io
import contextlib
from tqdm.contrib.concurrent import process_map

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
base_roberta_val_sentiment_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_val_sentiment.csv")
base_roberta_test_sentiment_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_test_sentiment.csv")

base_roberta_test_sentiment = base_roberta_test_sentiment_df["base_roberta_sentiment"].to_numpy()
base_roberta_val_sentiment = base_roberta_val_sentiment_df["base_roberta_sentiment"].to_numpy()

base_roberta_val_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_val_probabilities.csv")

base_roberta_val_probabilities = base_roberta_val_probabilities_df[["base_roberta_neg", "base_roberta_neu", "base_roberta_pos"]].to_numpy()

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
# ----------------------------------------------------------------------------- Start
# ENHANCED ROBERTA SETUP
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
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# AFFIRMATIVE INTERPRETATION RULES
# ----------------------------------------------------------------------------- 
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
    "aint", "ain't",
    "arent", "aren't",
    "hasnt", "hasn't",
    "havent", "haven't",
    "hadnt", "hadn't",
    "n't", "nt",
    "without"
}

INTENSIFIER_WORDS = {
    "very", "really", "extremely", "incredibly", "highly",
    "super", "ultra", "absolutely", "completely", "totally",
    "surprisingly", "ridiculously", "seriously", "terribly"
}

DIMINISHER_WORDS = {
    "slightly", "somewhat", "mildly", "partly",
    "partially", "kinda", "sorta", "barely", "hardly",
    "almost", "fairly"
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

CONTRAST_WORDS = POST_CONTRAST_MARKERS.union(CONCESSIVE_STARTERS)

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
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:good|decent|okay|ok|fine)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect|the\s+best)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_not_bad_not_great",
        "polarity": "neu",
        "pattern": re.compile(
            r"\bnot\s+(?:bad|terrible|awful)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_okay_but_issues",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:okay|ok|fine|decent|good)\s+but\s+"
            r"(?:has|have|had|with)\s+(?:some\s+)?"
            r"(?:issues|problems|flaws|drawbacks|downsides)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_works_but",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:works|worked|work)\s+but\s+"
            r"(?:not\s+perfect|not\s+great|has\s+issues|could\s+be\s+better|"
            r"there\s+are\s+issues|with\s+some\s+problems)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_decent_for_price",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:decent|okay|ok|fine|acceptable|reasonable)\s+"
            r"(?:for|given)\s+(?:the\s+)?(?:price|money|cost)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_average_nothing_special",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:average|mediocre|ordinary)\s+"
            r"(?:product|item|quality|book|read|purchase)\b|"
            r"\bnothing\s+(?:special|amazing|great|exceptional)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_pros_and_cons",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:pros\s+and\s+cons|good\s+and\s+bad|"
            r"some\s+good\s+and\s+some\s+bad|mixed\s+feelings|mixed\s+review)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_somewhat_disappointed",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:somewhat|slightly|a\s+little|kind\s+of|kinda)\s+"
            r"(?:disappointed|underwhelmed)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_expected_more",
        "polarity": "neu",
        "pattern": re.compile(
            r"\b(?:expected|was\s+expecting)\s+"
            r"(?:a\s+)?(?:little\s+)?more\b|"
            r"\bnot\s+(?:quite|really)\s+what\s+i\s+expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_worth_it",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bnot\s+" + OPTIONAL_DEGREE +
            r"worth\s+(?:it|the\s+money|the\s+price|buying|getting|keeping)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:waste_of_money",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:a\s+)?(?:complete\s+|total\s+|real\s+|absolute\s+)?"
            r"waste\s+of\s+(?:money|time)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:low_quality",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:low|poor|bad|terrible|awful|horrible)\s+quality\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:cheaply_made",
        "polarity": "neg",
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
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:fell|came|comes|coming)\s+apart\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_lasting",
        "polarity": "neg",
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
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:not|isn't|isnt|wasn't|wasnt|is\s+not|was\s+not)\s+"
            r"(?:as\s+)?described\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_item",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bwrong\s+(?:item|product|model|version|book|charger|case)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:missing_parts",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bmissing\s+(?:parts?|pieces?|accessories|components|items?)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:never_received",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:never\s+received|"
            r"did\s+not\s+receive|didn't\s+receive|didnt\s+receive|"
            r"have\s+not\s+received|haven't\s+received|havent\s+received)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_delivered",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:not|never|was\s+not|wasn't|wasnt|"
            r"has\s+not\s+been|hasn't\s+been|hasnt\s+been)"
            r"\s+delivered\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:had_to_return",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:had\s+to\s+return|"
            r"returned\s+(?:it|this|the\s+item|the\s+product|the\s+book))\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:want_refund",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:want|wanted|need|needed|request(?:ed)?|asking\s+for)"
            r"\s+(?:a\s+)?refund\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_satisfied",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bnot\s+(?:very\s+|really\s+|fully\s+|completely\s+)?"
            r"(?:satisfied|happy)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_not_recommend",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:would\s+not|wouldn't|wouldnt)\s+recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_bad",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bnot\s+(?:too\s+|that\s+|so\s+|very\s+)?bad\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_complaints",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?complaints?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_issues",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?issues?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_problems",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?problems?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_regrets",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bno\s+regrets?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_great",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bworks?\s+(?:really\s+|very\s+|so\s+)?great\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_perfectly",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bworks?\s+(?:perfectly|flawlessly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_as_expected",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bwork(?:s|ed)?\s+(?:exactly\s+)?as\s+expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:highly_recommend",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:highly|strongly|definitely)\s+recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_recommend",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+|highly\s+)?recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_buy_again",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+)?buy\s+(?:it\s+|this\s+)?again\b|"
            r"\bbuy\s+(?:it\s+|this\s+)?again\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:worth_every_penny",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bworth\s+every\s+(?:penny|cent)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:better_than_expected",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bbetter\s+than\s+(?:i\s+)?expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:exceeded_expectations",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bexceeded\s+(?:my\s+)?expectations\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:could_not_be_happier",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:could\s+not|couldn't|couldnt)\s+be\s+happier\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:cannot_recommend_enough",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:cannot|can\s+not|can't|cant)\s+recommend"
            r"(?:\s+(?:it|this|these|them))?\s+enough\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:does_not_work",
        "polarity": "neg",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:stopped_working",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:stopped|stop|stops|quit|quits)\s+working\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_longer_works",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bno\s+longer\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:dead_on_arrival",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:dead\s+on\s+arrival|doa)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_powering_on",
        "polarity": "neg",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:power\s+on|powering\s+on)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_turning_on",
        "polarity": "neg",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:turn\s+on|turning\s+on)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_charging",
        "polarity": "neg",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+charg(?:e|es|ed|ing)\b|"
            r"\b(?:stopped|stop|stops|quit|quits)\s+charging\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:battery_drains_fast",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:battery|batteries)\s+"
            r"(?:drain|drains|drained|die|dies|died)\s+"
            r"(?:too\s+)?(?:fast|quickly|rapidly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:does_not_hold_charge",
        "polarity": "neg",
        "pattern": re.compile(
            rf"\b(?:battery\s+)?(?:{NEG_AUX})\s+hold\s+"
            r"(?:a\s+|the\s+)?charge\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:keeps_disconnecting",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:keep|keeps|kept)\s+disconnecting\b|"
            r"\blosing\s+connection\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poor_connection",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:poor|bad|weak|unstable)\s+(?:connection|signal)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:overheats_quickly",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:overheat|overheats|overheated|overheating)\s+"
            r"(?:too\s+)?(?:quickly|fast|easily)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:screen_cracked",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:screen\s+cracked|cracked\s+screen)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:flickering_screen",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:screen|display)\s+(?:is\s+|was\s+|keeps\s+|kept\s+|started\s+)?"
            r"(?:flickering|flickers|flicker)\b|"
            r"\b(?:flickering|flickers|flicker)\s+(?:screen|display)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:touchscreen_not_responsive",
        "polarity": "neg",
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
        "polarity": "neg",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+fit(?:s|ted|ting)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_size",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bwrong\s+size\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_color",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bwrong\s+(?:color|colour)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:runs_small",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?small\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:runs_large",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?(?:large|big)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poor_fit",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:poor|bad|terrible|awkward|weird)\s+fit\b|"
            r"\bfit(?:s|ted)?\s+(?:poorly|badly|terribly|awkwardly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:see_through",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:see\s*through|see-through|too\s+sheer|transparent)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:shrunk_after_wash",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:shrank|shrunk|shrinked)\s+"
            r"(?:after|following)\s+(?:a\s+)?(?:wash|washing)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:color_faded",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:color|colour|colors|colours)\s+(?:faded|fades|fade)\b|"
            r"\b(?:faded|fades|fade)\s+(?:color|colour|colors|colours)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:fabric_feels_cheap",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:fabric|material)\s+(?:feel|feels|felt|feeling)\s+cheap\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:seam_ripped",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bseam\s+(?:ripped|torn)\b|"
            r"\bstitching\s+(?:came\s+loose|undone|ripped|torn)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:zipper_broken",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bzipper\s+(?:broken|stuck|jammed|broke)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:true_to_size",
        "polarity": "pos",
        "pattern": re.compile(
            r"\btrue\s+to\s+size\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:comfortable_fit",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bcomfortable\s+fit\b|"
            r"\bfit(?:s|ted)?\s+comfortably\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:missing_pages",
        "polarity": "neg",
        "pattern": re.compile(
            r"\bmissing\s+pages?\b|"
            r"\bpages?\s+(?:is\s+|are\s+|was\s+|were\s+)?missing\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poorly_written",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:poorly|badly|terribly)\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:poor|bad|terrible|awful)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:hard_to_follow",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:hard|difficult|confusing)\s+to\s+follow\b|"
            r"\bnot\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:bad_translation",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:bad|poor|terrible|awful)\s+translation\b|"
            r"\btranslation\s+(?:is\s+|was\s+)?(?:bad|poor|terrible|awful)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:printing_error",
        "polarity": "neg",
        "pattern": re.compile(
            r"\b(?:printing|print)\s+errors?\b|"
            r"\b(?:misprint|misprinted|misprints)\b|"
            r"\bpages?\s+(?:printed|print)\s+incorrectly\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:great_read",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bgreat\s+read\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:well_written",
        "polarity": "pos",
        "pattern": re.compile(
            r"\bwell\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:excellent|great|clear)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:easy_to_follow",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:easy|clear)\s+to\s+follow\b|"
            r"\bclear\s+and\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:highly_informative",
        "polarity": "pos",
        "pattern": re.compile(
            r"\b(?:highly|very|really)\s+informative\b|"
            r"\binformative\s+and\s+useful\b",
            flags=re.IGNORECASE
        )
    }
]

CUSTOM_PHRASE_INTERPRETATIONS = {
    "phrase:neutral_good_but_not_great": "the product is acceptable but not excellent",
    "phrase:neutral_not_bad_not_great": "the product is average",
    "phrase:neutral_okay_but_issues": "the product is acceptable but has issues",
    "phrase:neutral_works_but": "the product works but has limitations",
    "phrase:neutral_decent_for_price": "the product is acceptable for the price",
    "phrase:neutral_average_nothing_special": "the product is average",
    "phrase:neutral_pros_and_cons": "the review expresses mixed feelings",
    "phrase:neutral_somewhat_disappointed": "the customer is mildly disappointed",
    "phrase:neutral_expected_more": "the customer expected more",
    "phrase:not_worth_it": "the product is a waste of money",
    "phrase:waste_of_money": "the product is a waste of money",
    "phrase:low_quality": "the product quality is poor",
    "phrase:cheaply_made": "the product is cheaply made",
    "phrase:fell_apart": "the product broke apart",
    "phrase:not_lasting": "the product does not last",
    "phrase:not_as_described": "the product is not as described",
    "phrase:wrong_item": "the wrong item was received",
    "phrase:missing_parts": "the product has missing parts",
    "phrase:never_received": "the customer did not receive the product",
    "phrase:not_delivered": "the product was not delivered",
    "phrase:had_to_return": "the customer returned the product",
    "phrase:want_refund": "the customer wants a refund",
    "phrase:not_satisfied": "the customer is dissatisfied",
    "phrase:would_not_recommend": "the customer recommends avoiding this product",
    "phrase:not_bad": "the product is acceptable",
    "phrase:no_complaints": "the customer has no complaints",
    "phrase:no_issues": "the product works without issues",
    "phrase:no_problems": "the product works without problems",
    "phrase:no_regrets": "the customer does not regret the purchase",
    "phrase:works_great": "the product works very well",
    "phrase:works_perfectly": "the product works perfectly",
    "phrase:works_as_expected": "the product works as expected",
    "phrase:highly_recommend": "the customer strongly recommends this product",
    "phrase:would_recommend": "the customer recommends this product",
    "phrase:would_buy_again": "the customer would buy the product again",
    "phrase:worth_every_penny": "the product is worth the money",
    "phrase:better_than_expected": "the product exceeded expectations",
    "phrase:exceeded_expectations": "the product exceeded expectations",
    "phrase:could_not_be_happier": "the customer is very happy",
    "phrase:cannot_recommend_enough": "the customer strongly recommends this product",
    "phrase:does_not_work": "the product is defective",
    "phrase:stopped_working": "the product stopped working",
    "phrase:no_longer_works": "the product no longer works",
    "phrase:dead_on_arrival": "the product arrived defective",
    "phrase:not_powering_on": "the product does not power on",
    "phrase:not_turning_on": "the product does not turn on",
    "phrase:not_charging": "the product does not charge",
    "phrase:battery_drains_fast": "the battery drains quickly",
    "phrase:does_not_hold_charge": "the battery does not hold charge",
    "phrase:keeps_disconnecting": "the product keeps disconnecting",
    "phrase:poor_connection": "the product has a poor connection",
    "phrase:overheats_quickly": "the product overheats quickly",
    "phrase:screen_cracked": "the screen is cracked",
    "phrase:flickering_screen": "the screen is flickering",
    "phrase:touchscreen_not_responsive": "the touchscreen is not responsive",
    "phrase:not_fitting": "the item has a poor fit",
    "phrase:wrong_size": "the item has the wrong size",
    "phrase:wrong_color": "the item has the wrong colour",
    "phrase:runs_small": "the item runs small",
    "phrase:runs_large": "the item runs large",
    "phrase:poor_fit": "the item has a poor fit",
    "phrase:see_through": "the material is see through",
    "phrase:shrunk_after_wash": "the item shrank after washing",
    "phrase:color_faded": "the colour faded",
    "phrase:fabric_feels_cheap": "the fabric feels cheap",
    "phrase:seam_ripped": "the seam ripped",
    "phrase:zipper_broken": "the zipper is broken",
    "phrase:true_to_size": "the item is true to size",
    "phrase:comfortable_fit": "the item fits comfortably",
    "phrase:missing_pages": "the book has missing pages",
    "phrase:poorly_written": "the book is poorly written",
    "phrase:hard_to_follow": "the book is difficult to follow",
    "phrase:bad_translation": "the translation is poor",
    "phrase:printing_error": "the book has printing errors",
    "phrase:great_read": "the book is a great read",
    "phrase:well_written": "the book is well written",
    "phrase:easy_to_follow": "the book is easy to follow",
    "phrase:highly_informative": "the book is very informative",
}

INTENSIFIER_PATTERN = r"(?:very|really|extremely|incredibly|highly|super|ultra|absolutely|completely|totally|seriously|terribly)"
DIMINISHER_PATTERN = r"(?:slightly|somewhat|mildly|partly|partially|kinda|sorta|barely|hardly|almost|fairly)"

AFFIRMATIVE_MODIFIER_RULES = [
    {
        "rule_key": "negator:not_good",
        "rule_group": "negator",
        "pattern": re.compile(rf"\b(?:{NEG_AUX})\s+good\b", flags=re.IGNORECASE),
        "interpretation": "the product is bad"
    },
    {
        "rule_key": "negator:not_great",
        "rule_group": "negator",
        "pattern": re.compile(rf"\b(?:{NEG_AUX})\s+great\b", flags=re.IGNORECASE),
        "interpretation": "the product is not great"
    },
    {
        "rule_key": "negator:not_useful",
        "rule_group": "negator",
        "pattern": re.compile(rf"\b(?:{NEG_AUX})\s+useful\b", flags=re.IGNORECASE),
        "interpretation": "the product is not useful"
    },
    {
        "rule_key": "negator:not_clear",
        "rule_group": "negator",
        "pattern": re.compile(rf"\b(?:{NEG_AUX})\s+clear\b", flags=re.IGNORECASE),
        "interpretation": "the explanation is unclear"
    },
    {
        "rule_key": "intensifier:strong_positive",
        "rule_group": "intensifier",
        "pattern": re.compile(
            rf"\b{INTENSIFIER_PATTERN}\s+(?:good|great|excellent|amazing|perfect|useful|comfortable|informative)\b",
            flags=re.IGNORECASE
        ),
        "interpretation": "the review expresses strong positive sentiment"
    },
    {
        "rule_key": "intensifier:strong_negative",
        "rule_group": "intensifier",
        "pattern": re.compile(
            rf"\b{INTENSIFIER_PATTERN}\s+(?:bad|terrible|awful|horrible|poor|disappointed|uncomfortable|confusing)\b",
            flags=re.IGNORECASE
        ),
        "interpretation": "the review expresses strong negative sentiment"
    },
    {
        "rule_key": "diminisher:mild_positive",
        "rule_group": "diminisher",
        "pattern": re.compile(
            rf"\b{DIMINISHER_PATTERN}\s+(?:good|useful|comfortable|helpful|informative)\b",
            flags=re.IGNORECASE
        ),
        "interpretation": "the review expresses mild positive sentiment"
    },
    {
        "rule_key": "diminisher:mild_negative",
        "rule_group": "diminisher",
        "pattern": re.compile(
            rf"\b{DIMINISHER_PATTERN}\s+(?:bad|disappointed|underwhelmed|confusing|uncomfortable)\b",
            flags=re.IGNORECASE
        ),
        "interpretation": "the review expresses mild negative sentiment"
    },
]
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# TEXT NORMALISATION
# ----------------------------------------------------------------------------- 
def normalise_negation_contractions(text):
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

    for pattern, replacement in NEGATION_CONTRACTION_REPLACEMENTS.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# PHRASE AND CONTRAST EXTRACTION
# ----------------------------------------------------------------------------- 
def find_non_overlapping_phrase_matches(text):
    original_text = str(text)
    phrase_matches = []

    for rule in UNIVERSAL_PHRASE_RULES:
        rule_key = rule["rule_key"]

        for match in rule["pattern"].finditer(original_text):
            interpretation = CUSTOM_PHRASE_INTERPRETATIONS.get(rule_key)

            if interpretation is None:
                if rule["polarity"] == "neg":
                    interpretation = "the review expresses a negative product issue"
                elif rule["polarity"] == "pos":
                    interpretation = "the review expresses a positive product evaluation"
                else:
                    interpretation = "the review expresses a mixed or neutral evaluation"

            phrase_matches.append({
                "rule_key": rule_key,
                "rule_group": "phrase",
                "start": match.start(),
                "end": match.end(),
                "matched_text": match.group(0),
                "interpretation": interpretation,
                "length": match.end() - match.start()
            })

    phrase_matches = sorted(
        phrase_matches,
        key=lambda item: (
            item["start"],
            -item["length"]
        )
    )

    selected_matches = []
    occupied_ranges = []

    for phrase_match in phrase_matches:
        start = phrase_match["start"]
        end = phrase_match["end"]

        overlaps_existing_match = False

        for occupied_start, occupied_end in occupied_ranges:
            if start < occupied_end and end > occupied_start:
                overlaps_existing_match = True
                break

        if overlaps_existing_match:
            continue

        selected_matches.append(phrase_match)
        occupied_ranges.append((start, end))

    return selected_matches

def split_into_sentences(text):
    return re.split(
        r"(?<=[.!?])\s+",
        str(text).strip()
    )

def get_contrast_focus_clauses(text):
    sentences = split_into_sentences(text)
    focus_clauses = []

    post_contrast_pattern = re.compile(
        r"\b(?:but|however|yet|nevertheless|nonetheless)\b",
        flags=re.IGNORECASE
    )

    concessive_start_pattern = re.compile(
        r"^\s*(?:although|though)\s+(.+?),\s+(.+)$",
        flags=re.IGNORECASE
    )

    trailing_concessive_pattern = re.compile(
        r"^\s*(.+?),\s*(?:although|though)\s+(.+)$",
        flags=re.IGNORECASE
    )

    for sentence in sentences:
        match = concessive_start_pattern.search(sentence)

        if match:
            focus_clauses.append(match.group(2).strip())
            continue

        match = trailing_concessive_pattern.search(sentence)

        if match:
            focus_clauses.append(match.group(1).strip())
            continue

        matches = list(post_contrast_pattern.finditer(sentence))

        for index, match in enumerate(matches):
            start = match.end()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(sentence)

            focus_clause = sentence[start:end].strip(" ,;:")

            if focus_clause:
                focus_clauses.append(focus_clause)

    return focus_clauses

def extract_universal_word_rules(text):
    text = normalise_negation_contractions(text)

    tokens = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]",
        text
    )

    applied_rules = set()

    clean_negation_words = {
        str(word).lower()
        for word in NEGATION_WORDS
    }

    clean_intensifier_words = {
        str(word).lower()
        for word in INTENSIFIER_WORDS
    }

    clean_diminisher_words = {
        str(word).lower()
        for word in DIMINISHER_WORDS
    }

    clean_post_contrast_markers = {
        str(word).lower()
        for word in POST_CONTRAST_MARKERS
    }

    clean_concessive_starters = {
        str(word).lower()
        for word in CONCESSIVE_STARTERS
    }

    for token in tokens:
        clean_token = str(token).lower()

        if clean_token in clean_negation_words:
            applied_rules.add("negator:" + clean_token)

        if clean_token in clean_intensifier_words:
            applied_rules.add("intensifier:" + clean_token)

        if clean_token in clean_diminisher_words:
            applied_rules.add("diminisher:" + clean_token)

        if clean_token in clean_post_contrast_markers:
            applied_rules.add("contrast:" + clean_token)

        if clean_token in clean_concessive_starters:
            applied_rules.add("contrast:" + clean_token)

    return applied_rules
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# AFFIRMATIVE INTERPRETATION CREATION
# ----------------------------------------------------------------------------- 
def find_affirmative_interpretations(text):
    original_text = str(text)
    normalised_text = normalise_negation_contractions(original_text)

    interpretations = []
    applied_rules = set()
    seen_core_interpretations = set()

    def add_interpretation(interpretation):
        if interpretation not in seen_core_interpretations:
            interpretations.append(interpretation)
            seen_core_interpretations.add(interpretation)

    phrase_matches = find_non_overlapping_phrase_matches(normalised_text)

    for phrase_match in phrase_matches:
        add_interpretation(phrase_match["interpretation"])
        applied_rules.add(phrase_match["rule_key"])

    applied_rules.update(
        extract_universal_word_rules(normalised_text)
    )

    for rule in AFFIRMATIVE_MODIFIER_RULES:
        if rule["pattern"].search(normalised_text):
            add_interpretation(rule["interpretation"])
            applied_rules.add(rule["rule_key"])

    contrast_focus_clauses = get_contrast_focus_clauses(normalised_text)

    for clause in contrast_focus_clauses:
        clause_phrase_matches = find_non_overlapping_phrase_matches(clause)

        for phrase_match in clause_phrase_matches:
            core_interpretation = phrase_match["interpretation"]

            if core_interpretation not in seen_core_interpretations:
                add_interpretation(
                    "the main opinion is that " + core_interpretation
                )

            applied_rules.add(phrase_match["rule_key"])

        for rule in AFFIRMATIVE_MODIFIER_RULES:
            if rule["pattern"].search(clause):
                core_interpretation = rule["interpretation"]

                if core_interpretation not in seen_core_interpretations:
                    add_interpretation(
                        "the main opinion is that " + core_interpretation
                    )

                applied_rules.add(rule["rule_key"])

    return interpretations, sorted(applied_rules)

def add_affirmative_interpretation(text, max_interpretations=4):
    original_text = str(text)

    interpretations, _ = find_affirmative_interpretations(original_text)

    unique_interpretations = []
    seen = set()

    for interpretation in interpretations:
        if interpretation not in seen:
            unique_interpretations.append(interpretation)
            seen.add(interpretation)

    unique_interpretations = unique_interpretations[:max_interpretations]

    if not unique_interpretations:
        return original_text

    interpretation_text = ". ".join(unique_interpretations)

    return (
        "Interpretation: "
        + interpretation_text
        + ". Review: "
        + original_text
    )

def extract_affirmative_interpretation_rules(text):
    _, applied_rules = find_affirmative_interpretations(text)
    return applied_rules
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# RULE TABLES
# ----------------------------------------------------------------------------- 
def build_affirmative_roberta_rule_catalog():
    catalog_rows = []

    for word in sorted(NEGATION_WORDS):
        catalog_rows.append({
            "rule_key": "negator:" + str(word).lower(),
            "rule_group": "negator",
            "marker": "CONTEXT_SIGNAL",
            "description": "Negator word detected: " + str(word),
            "polarity": None
        })

    for word in sorted(INTENSIFIER_WORDS):
        catalog_rows.append({
            "rule_key": "intensifier:" + str(word).lower(),
            "rule_group": "intensifier",
            "marker": "CONTEXT_SIGNAL",
            "description": "Intensifier word detected: " + str(word),
            "polarity": None
        })

    for word in sorted(DIMINISHER_WORDS):
        catalog_rows.append({
            "rule_key": "diminisher:" + str(word).lower(),
            "rule_group": "diminisher",
            "marker": "CONTEXT_SIGNAL",
            "description": "Diminisher word detected: " + str(word),
            "polarity": None
        })

    for word in sorted(POST_CONTRAST_MARKERS):
        catalog_rows.append({
            "rule_key": "contrast:" + str(word).lower(),
            "rule_group": "contrast",
            "marker": "CONTEXT_SIGNAL",
            "description": "Post-contrast marker detected: " + str(word),
            "polarity": None
        })

    for word in sorted(CONCESSIVE_STARTERS):
        catalog_rows.append({
            "rule_key": "contrast:" + str(word).lower(),
            "rule_group": "contrast",
            "marker": "CONTEXT_SIGNAL",
            "description": "Concessive marker detected: " + str(word),
            "polarity": None
        })

    # Phrase rules directly create affirmative interpretation text.
    for rule in UNIVERSAL_PHRASE_RULES:
        catalog_rows.append({
            "rule_key": rule["rule_key"],
            "rule_group": "phrase",
            "marker": "AFFIRMATIVE_INTERPRETATION",
            "description": rule["pattern"].pattern,
            "polarity": rule["polarity"]
        })

    # Modifier interpretation rules also directly create affirmative interpretation text.
    for rule in AFFIRMATIVE_MODIFIER_RULES:
        catalog_rows.append({
            "rule_key": rule["rule_key"],
            "rule_group": rule["rule_group"],
            "marker": "AFFIRMATIVE_INTERPRETATION",
            "description": rule["pattern"].pattern,
            "polarity": None
        })

    catalog_df = pd.DataFrame(catalog_rows)

    catalog_df = (
        catalog_df
        .drop_duplicates(subset=["rule_key"])
        .sort_values(["rule_group", "rule_key"])
        .reset_index(drop=True)
    )

    return catalog_df

def count_rule_usage(rule_sets):
    rule_counts = {}

    for rule_set in rule_sets:
        for rule_key in rule_set:
            rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1

    return rule_counts

def build_roberta_rule_usage_table(
        rule_catalog_df,
        train_rule_sets,
        val_rule_sets,
        test_rule_sets
):
    train_counts = count_rule_usage(train_rule_sets)
    val_counts = count_rule_usage(val_rule_sets)
    test_counts = count_rule_usage(test_rule_sets)

    rule_usage_df = rule_catalog_df.copy()

    rule_usage_df["train_applications"] = rule_usage_df["rule_key"].map(train_counts).fillna(0).astype(int)
    rule_usage_df["val_applications"] = rule_usage_df["rule_key"].map(val_counts).fillna(0).astype(int)
    rule_usage_df["test_applications"] = rule_usage_df["rule_key"].map(test_counts).fillna(0).astype(int)

    rule_usage_df["total_applications"] = (
        rule_usage_df["train_applications"]
        + rule_usage_df["val_applications"]
        + rule_usage_df["test_applications"]
    )

    rule_usage_df["usage_status"] = np.where(
        rule_usage_df["total_applications"] > 0,
        "USED",
        "UNUSED"
    )

    rule_usage_df = rule_usage_df.sort_values(
        ["total_applications", "rule_group", "rule_key"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    return rule_usage_df

def print_roberta_rule_usage_table(rule_usage_df):

    print("\n========== ENHANCED RoBERTa RULE USAGE TABLE ==========")
    print(rule_usage_df.to_string(index=False))

    print("\n========== ENHANCED RoBERTa UNUSED RULES ==========")
    print(
        rule_usage_df[
            rule_usage_df["usage_status"] == "UNUSED"
        ][["rule_key"]].to_string(index=False)
    )

    return rule_usage_df

def build_roberta_all_rules_summary_table(rule_usage_df):
    summary_rows = []

    for rule_group, group_df in rule_usage_df.groupby("rule_group"):
        total_rules = len(group_df)
        used_rules = int((group_df["usage_status"] == "USED").sum())
        unused_rules = int((group_df["usage_status"] == "UNUSED").sum())

        summary_rows.append({
            "rule_group": rule_group,
            "total_rules": total_rules,
            "used_rules": used_rules,
            "unused_rules": unused_rules
        })

    total_rules = len(rule_usage_df)
    used_rules = int((rule_usage_df["usage_status"] == "USED").sum())
    unused_rules = int((rule_usage_df["usage_status"] == "UNUSED").sum())

    summary_rows.append({
        "rule_group": "ALL_RULES",
        "total_rules": total_rules,
        "used_rules": used_rules,
        "unused_rules": unused_rules
    })

    return pd.DataFrame(summary_rows)

def print_roberta_all_rules_summary_table(rule_usage_df):
    all_rules_summary_df = build_roberta_all_rules_summary_table(
        rule_usage_df
    )

    print("\n========== AFFIRMATIVE RoBERTa ALL-RULES SUMMARY ==========")
    print(all_rules_summary_df.to_string(index=False))

    return all_rules_summary_df

def print_roberta_general_correction_summary(
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    general_correction_summary_df = build_roberta_general_correction_summary(
        val_rule_sets=val_rule_sets,
        sentiment_true=sentiment_true,
        base_sentiment=base_sentiment,
        enhanced_sentiment=enhanced_sentiment
    )

    print("\n========== ENHANCED RoBERTa GENERAL RULE IMPACT SUMMARY ==========")
    print(general_correction_summary_df.to_string(index=False))

    return general_correction_summary_df

def build_roberta_general_correction_summary(
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    sentiment_true = np.asarray(sentiment_true)
    base_sentiment = np.asarray(base_sentiment)
    enhanced_sentiment = np.asarray(enhanced_sentiment)

    affected_mask = np.array([
        len(rule_set) > 0
        for rule_set in val_rule_sets
    ])

    baseline_correct = base_sentiment == sentiment_true
    enhanced_correct = enhanced_sentiment == sentiment_true

    corrected_mask = affected_mask & (~baseline_correct) & enhanced_correct
    harmed_mask = affected_mask & baseline_correct & (~enhanced_correct)
    stayed_correct_mask = affected_mask & baseline_correct & enhanced_correct
    wrong_to_wrong_mask = (
        affected_mask
        & (~baseline_correct)
        & (~enhanced_correct)
        & (base_sentiment != enhanced_sentiment)
    )
    stayed_wrong_mask = (
        affected_mask
        & (~baseline_correct)
        & (~enhanced_correct)
        & (base_sentiment == enhanced_sentiment)
    )

    affected = int(affected_mask.sum())
    corrected = int(corrected_mask.sum())
    harmed = int(harmed_mask.sum())
    stayed_correct = int(stayed_correct_mask.sum())
    wrong_to_wrong = int(wrong_to_wrong_mask.sum())
    stayed_wrong = int(stayed_wrong_mask.sum())
    net_correction = corrected - harmed

    if affected > 0:
        affected_accuracy_baseline = float(np.mean(baseline_correct[affected_mask]))
        affected_accuracy_enhanced = float(np.mean(enhanced_correct[affected_mask]))
    else:
        affected_accuracy_baseline = np.nan
        affected_accuracy_enhanced = np.nan

    return pd.DataFrame([{
        "affected_reviews": affected,
        "corrected": corrected,
        "harmed": harmed,
        "net_correction": net_correction,
        "stayed_correct": stayed_correct,
        "wrong_to_wrong": wrong_to_wrong,
        "stayed_wrong": stayed_wrong,
        "baseline_accuracy_on_affected": affected_accuracy_baseline,
        "enhanced_accuracy_on_affected": affected_accuracy_enhanced
    }])

def make_exclusive_rule_decision(
        applications,
        corrected,
        harmed,
        net_correction,
        correction_precision
):
    decisive_cases = corrected + harmed

    if applications == 0:
        return "UNUSED"
    if applications < 30:
        return "REVIEW"
    if decisive_cases < 10:
        return "REVIEW"
    if net_correction < 0:
        return "CULL"
    if net_correction > 0 and correction_precision >= 0.60:
        return "KEEP"

    return "REVIEW"

def build_roberta_exclusive_rule_table(
        rule_catalog_df,
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    sentiment_true = np.asarray(sentiment_true)
    base_sentiment = np.asarray(base_sentiment)
    enhanced_sentiment = np.asarray(enhanced_sentiment)

    baseline_correct = base_sentiment == sentiment_true
    enhanced_correct = enhanced_sentiment == sentiment_true

    exclusive_rows = []

    for _, rule_row in rule_catalog_df.iterrows():
        rule_key = rule_row["rule_key"]
        rule_group = rule_row["rule_group"]

        if rule_group not in {
            "phrase",
            "contrast",
            "negator",
            "intensifier",
            "diminisher"
        }:
            continue

        exclusive_mask = np.array([
            len(rule_set) == 1 and rule_set[0] == rule_key
            for rule_set in val_rule_sets
        ])

        applications = int(exclusive_mask.sum())

        corrected = int(np.sum(
            exclusive_mask
            & (~baseline_correct)
            & enhanced_correct
        ))

        harmed = int(np.sum(
            exclusive_mask
            & baseline_correct
            & (~enhanced_correct)
        ))

        decisive_cases = corrected + harmed
        net_correction = corrected - harmed

        if decisive_cases > 0:
            correction_precision = corrected / decisive_cases
        else:
            correction_precision = np.nan

        decision = make_exclusive_rule_decision(
            applications=applications,
            corrected=corrected,
            harmed=harmed,
            net_correction=net_correction,
            correction_precision=correction_precision
        )

        exclusive_rows.append({
            "rule_key": rule_key,
            "rule_group": rule_group,
            "exclusive_applications": applications,
            "corrected": corrected,
            "harmed": harmed,
            "net_correction": net_correction,
            "decisive_cases": decisive_cases,
            "correction_precision": correction_precision,
            "decision": decision
        })

    exclusive_df = pd.DataFrame(exclusive_rows)

    decision_order = {
        "KEEP": 0,
        "REVIEW": 1,
        "CULL": 2,
        "UNUSED": 3
    }

    exclusive_df["decision_order"] = exclusive_df["decision"].map(decision_order)

    exclusive_df = exclusive_df.sort_values(
        ["decision_order", "exclusive_applications", "net_correction"],
        ascending=[True, False, False]
    ).drop(columns=["decision_order"]).reset_index(drop=True)

    return exclusive_df

def print_roberta_exclusive_rule_table(
        rule_catalog_df,
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    exclusive_rule_df = build_roberta_exclusive_rule_table(
        rule_catalog_df=rule_catalog_df,
        val_rule_sets=val_rule_sets,
        sentiment_true=sentiment_true,
        base_sentiment=base_sentiment,
        enhanced_sentiment=enhanced_sentiment
    )

    print("\n========== ENHANCED RoBERTa EXCLUSIVE RULE TABLE ==========")
    print(exclusive_rule_df.to_string(index=False))

    print("\n========== ENHANCED RoBERTa EXCLUSIVE KEEP RULES ==========")
    print(
        exclusive_rule_df[
            exclusive_rule_df["decision"] == "KEEP"
        ].to_string(index=False)
    )

    print("\n========== ENHANCED RoBERTa EXCLUSIVE REVIEW RULES ==========")
    print(
        exclusive_rule_df[
            exclusive_rule_df["decision"] == "REVIEW"
        ].to_string(index=False)
    )

    print("\n========== ENHANCED RoBERTa EXCLUSIVE CULL RULES ==========")
    print(
        exclusive_rule_df[
            exclusive_rule_df["decision"] == "CULL"
        ].to_string(index=False)
    )

    return exclusive_rule_df

def build_roberta_non_exclusive_all_rule_table(
        rule_catalog_df,
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    sentiment_true = np.asarray(sentiment_true)
    base_sentiment = np.asarray(base_sentiment)
    enhanced_sentiment = np.asarray(enhanced_sentiment)

    baseline_correct = base_sentiment == sentiment_true
    enhanced_correct = enhanced_sentiment == sentiment_true

    all_rule_rows = []

    for _, rule_row in rule_catalog_df.iterrows():
        rule_key = rule_row["rule_key"]

        rule_mask = np.array([
            rule_key in rule_set
            for rule_set in val_rule_sets
        ])

        applications = int(rule_mask.sum())

        corrected = int(np.sum(
            rule_mask
            & (~baseline_correct)
            & enhanced_correct
        ))

        harmed = int(np.sum(
            rule_mask
            & baseline_correct
            & (~enhanced_correct)
        ))

        stayed_correct = int(np.sum(
            rule_mask
            & baseline_correct
            & enhanced_correct
        ))

        wrong_to_wrong = int(np.sum(
            rule_mask
            & (~baseline_correct)
            & (~enhanced_correct)
            & (base_sentiment != enhanced_sentiment)
        ))

        stayed_wrong = int(np.sum(
            rule_mask
            & (~baseline_correct)
            & (~enhanced_correct)
            & (base_sentiment == enhanced_sentiment)
        ))

        decisive_cases = corrected + harmed
        net_correction = corrected - harmed

        if decisive_cases > 0:
            correction_precision = corrected / decisive_cases
        else:
            correction_precision = np.nan

        if applications > 0:
            baseline_rule_accuracy = float(np.mean(baseline_correct[rule_mask]))
            enhanced_rule_accuracy = float(np.mean(enhanced_correct[rule_mask]))
            accuracy_change = enhanced_rule_accuracy - baseline_rule_accuracy
        else:
            baseline_rule_accuracy = np.nan
            enhanced_rule_accuracy = np.nan
            accuracy_change = np.nan

        all_rule_rows.append({
            "rule_key": rule_key,
            "rule_group": rule_row["rule_group"],
            "marker": rule_row["marker"],
            "applications": applications,
            "corrected": corrected,
            "harmed": harmed,
            "net_correction": net_correction,
            "stayed_correct": stayed_correct,
            "wrong_to_wrong": wrong_to_wrong,
            "stayed_wrong": stayed_wrong,
            "decisive_cases": decisive_cases,
            "correction_precision": correction_precision,
            "baseline_rule_accuracy": baseline_rule_accuracy,
            "enhanced_rule_accuracy": enhanced_rule_accuracy,
            "accuracy_change": accuracy_change
        })

    all_rule_df = pd.DataFrame(all_rule_rows)

    all_rule_df = all_rule_df.sort_values(
        ["applications", "net_correction"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return all_rule_df

def print_roberta_non_exclusive_all_rule_table(
        rule_catalog_df,
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    non_exclusive_all_rule_df = build_roberta_non_exclusive_all_rule_table(
        rule_catalog_df=rule_catalog_df,
        val_rule_sets=val_rule_sets,
        sentiment_true=sentiment_true,
        base_sentiment=base_sentiment,
        enhanced_sentiment=enhanced_sentiment
    )

    print("\n========== ENHANCED RoBERTa NON-EXCLUSIVE ALL-RULES IMPACT TABLE ==========")
    print(non_exclusive_all_rule_df.to_string(index=False))

    return non_exclusive_all_rule_df

def get_roberta_rule_scope(rule_key):
    rule_key = str(rule_key).lower()

    if rule_key.startswith("phrase:"):
        return "phrase"

    if rule_key.startswith("negator:"):
        return "negator"

    if rule_key.startswith("intensifier:"):
        return "intensifier"

    if rule_key.startswith("diminisher:"):
        return "diminisher"

    if rule_key.startswith("contrast:"):
        return "contrast"

    return "other"

def get_roberta_rules_in_scope(rule_set, target_scope):
    return [
        rule_key
        for rule_key in rule_set
        if get_roberta_rule_scope(rule_key) == target_scope
    ]

def build_roberta_scoped_exclusive_rule_table(
        rule_catalog_df,
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment,
        target_scope
):
    sentiment_true = np.asarray(sentiment_true)
    base_sentiment = np.asarray(base_sentiment)
    enhanced_sentiment = np.asarray(enhanced_sentiment)

    baseline_correct = base_sentiment == sentiment_true
    enhanced_correct = enhanced_sentiment == sentiment_true

    scoped_catalog_df = rule_catalog_df.copy()

    scoped_catalog_df["scope"] = scoped_catalog_df["rule_key"].apply(
        get_roberta_rule_scope
    )

    scoped_catalog_df = scoped_catalog_df[
        scoped_catalog_df["scope"] == target_scope
    ].copy()

    scoped_rows = []

    for _, rule_row in scoped_catalog_df.iterrows():
        rule_key = rule_row["rule_key"]

        scoped_exclusive_mask_values = []

        for rule_set in val_rule_sets:
            scoped_rules = get_roberta_rules_in_scope(
                rule_set,
                target_scope
            )

            is_scoped_exclusive = (
                len(scoped_rules) == 1
                and scoped_rules[0] == rule_key
            )

            scoped_exclusive_mask_values.append(is_scoped_exclusive)

        scoped_exclusive_mask = np.array(scoped_exclusive_mask_values)

        applications = int(scoped_exclusive_mask.sum())

        corrected = int(np.sum(
            scoped_exclusive_mask
            & (~baseline_correct)
            & enhanced_correct
        ))

        harmed = int(np.sum(
            scoped_exclusive_mask
            & baseline_correct
            & (~enhanced_correct)
        ))

        decisive_cases = corrected + harmed
        net_correction = corrected - harmed

        if decisive_cases > 0:
            correction_precision = corrected / decisive_cases
        else:
            correction_precision = np.nan

        decision = make_exclusive_rule_decision(
            applications=applications,
            corrected=corrected,
            harmed=harmed,
            net_correction=net_correction,
            correction_precision=correction_precision
        )

        scoped_rows.append({
            "scope": target_scope,
            "rule_key": rule_key,
            "rule_group": rule_row["rule_group"],
            "scoped_exclusive_applications": applications,
            "corrected": corrected,
            "harmed": harmed,
            "net_correction": net_correction,
            "decisive_cases": decisive_cases,
            "correction_precision": correction_precision,
            "decision": decision
        })

    scoped_df = pd.DataFrame(scoped_rows)

    if scoped_df.empty:
        return scoped_df

    decision_order = {
        "KEEP": 0,
        "REVIEW": 1,
        "CULL": 2,
        "UNUSED": 3
    }

    scoped_df["decision_order"] = scoped_df["decision"].map(decision_order)

    scoped_df = (
        scoped_df
        .sort_values(
            [
                "decision_order",
                "scoped_exclusive_applications",
                "net_correction"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
        .drop(columns=["decision_order"])
        .reset_index(drop=True)
    )

    return scoped_df


def print_all_scoped_exclusive_roberta_rule_tables(
        rule_catalog_df,
        val_rule_sets,
        sentiment_true,
        base_sentiment,
        enhanced_sentiment
):
    scoped_tables = {}

    for scope_name in [
        "negator",
        "intensifier",
        "diminisher",
        "contrast",
        "phrase"
    ]:
        scoped_df = build_roberta_scoped_exclusive_rule_table(
            rule_catalog_df=rule_catalog_df,
            val_rule_sets=val_rule_sets,
            sentiment_true=sentiment_true,
            base_sentiment=base_sentiment,
            enhanced_sentiment=enhanced_sentiment,
            target_scope=scope_name
        )

        scoped_tables[scope_name] = scoped_df

        print("\n========== RoBERTa SCOPED-EXCLUSIVE "
              + scope_name.upper()
              + " RULE TABLE ==========")

        if scoped_df.empty:
            print("No scoped-exclusive rules found for scope:", scope_name)
        else:
            print(scoped_df.to_string(index=False))

            print("\n========== RoBERTa SCOPED-EXCLUSIVE "
                  + scope_name.upper()
                  + " KEEP RULES ==========")
            print(
                scoped_df[
                    scoped_df["decision"] == "KEEP"
                ].to_string(index=False)
            )

            print("\n========== RoBERTa SCOPED-EXCLUSIVE "
                  + scope_name.upper()
                  + " REVIEW RULES ==========")
            print(
                scoped_df[
                    scoped_df["decision"] == "REVIEW"
                ].to_string(index=False)
            )

            print("\n========== RoBERTa SCOPED-EXCLUSIVE "
                  + scope_name.upper()
                  + " CULL RULES ==========")
            print(
                scoped_df[
                    scoped_df["decision"] == "CULL"
                ].to_string(index=False)
            )

    return scoped_tables
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# AUDIT SUMMARY AND EXAMPLES
# ----------------------------------------------------------------------------- 
def get_roberta_true_class_margin(probabilities, reverse_label_map, true_label):
    class_to_index = {
        class_label: class_id
        for class_id, class_label in reverse_label_map.items()
    }

    true_index = class_to_index[true_label]
    true_probability = probabilities[true_index]
    other_probabilities = np.delete(probabilities, true_index)
    strongest_other_probability = other_probabilities.max()

    return true_probability - strongest_other_probability

def classify_roberta_effect(true_label, base_prediction, enhanced_prediction):
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

def create_roberta_rule_review_audit_df(
        original_text,
        enhanced_text,
        true_labels,
        base_sentiment,
        enhanced_sentiment,
        rule_sets,
        base_probabilities,
        enhanced_probabilities,
        reverse_label_map
):
    audit_rows = []

    original_text = original_text.reset_index(drop=True)
    enhanced_text = enhanced_text.reset_index(drop=True)
    true_labels = true_labels.reset_index(drop=True)
    base_sentiment = pd.Series(base_sentiment).reset_index(drop=True)
    enhanced_sentiment = pd.Series(enhanced_sentiment).reset_index(drop=True)
    rule_sets = rule_sets.reset_index(drop=True)

    base_probabilities = np.asarray(base_probabilities)
    enhanced_probabilities = np.asarray(enhanced_probabilities)

    for review_index, text in enumerate(original_text):
        true_label = true_labels.iloc[review_index]
        base_prediction = base_sentiment.iloc[review_index]
        enhanced_prediction = enhanced_sentiment.iloc[review_index]
        rules_applied = list(rule_sets.iloc[review_index])

        base_margin = get_roberta_true_class_margin(
            probabilities=base_probabilities[review_index],
            reverse_label_map=reverse_label_map,
            true_label=true_label
        )

        enhanced_margin = get_roberta_true_class_margin(
            probabilities=enhanced_probabilities[review_index],
            reverse_label_map=reverse_label_map,
            true_label=true_label
        )

        margin_change = enhanced_margin - base_margin

        effect = classify_roberta_effect(
            true_label=true_label,
            base_prediction=base_prediction,
            enhanced_prediction=enhanced_prediction
        )

        true_class_id = None
        for class_id, class_label in reverse_label_map.items():
            if class_label == true_label:
                true_class_id = class_id
                break

        if true_class_id is None:
            base_true_probability = np.nan
            enhanced_true_probability = np.nan
            true_probability_change = np.nan
        else:
            base_true_probability = base_probabilities[review_index][true_class_id]
            enhanced_true_probability = enhanced_probabilities[review_index][true_class_id]
            true_probability_change = enhanced_true_probability - base_true_probability

        audit_rows.append({
            "review_index": review_index,
            "original_text": text,
            "enhanced_text": enhanced_text.iloc[review_index],
            "true_label": true_label,
            "base_prediction": base_prediction,
            "enhanced_prediction": enhanced_prediction,
            "base_correct": base_prediction == true_label,
            "enhanced_correct": enhanced_prediction == true_label,
            "prediction_changed": base_prediction != enhanced_prediction,
            "effect": effect,
            "rule_keys": rules_applied,
            "number_of_rules": len(rules_applied),
            "base_margin": base_margin,
            "enhanced_margin": enhanced_margin,
            "margin_change": margin_change,
            "base_true_class_probability": base_true_probability,
            "enhanced_true_class_probability": enhanced_true_probability,
            "true_class_probability_change": true_probability_change,
            "base_neg_probability": base_probabilities[review_index][0],
            "base_neu_probability": base_probabilities[review_index][1],
            "base_pos_probability": base_probabilities[review_index][2],
            "enhanced_neg_probability": enhanced_probabilities[review_index][0],
            "enhanced_neu_probability": enhanced_probabilities[review_index][1],
            "enhanced_pos_probability": enhanced_probabilities[review_index][2],
            "neg_probability_change": enhanced_probabilities[review_index][0] - base_probabilities[review_index][0],
            "neu_probability_change": enhanced_probabilities[review_index][1] - base_probabilities[review_index][1],
            "pos_probability_change": enhanced_probabilities[review_index][2] - base_probabilities[review_index][2],
            "text_changed": text != enhanced_text.iloc[review_index],
            "score_changed": not np.allclose(
                base_probabilities[review_index],
                enhanced_probabilities[review_index],
                atol=1e-12,
                rtol=0.0
            )
        })

    return pd.DataFrame(audit_rows)

def print_short_roberta_review_audit_summary(audit_df):
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

    total_reviews = len(audit_df)
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
            "metric": "Wrong to wrong reviews total",
            "count": len(wrong_to_wrong_df)
        },
        {
            "metric": "Wrong to wrong with rules applied",
            "count": len(wrong_to_wrong_with_rules)
        },
        {
            "metric": "Wrong to wrong with no rules applied",
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

    if total_reviews > 0:
        summary_df["percent"] = (summary_df["count"] / total_reviews * 100).round(2)
    else:
        summary_df["percent"] = np.nan

    print("\n========== ENHANCED RoBERTa REVIEW-LEVEL AUDIT SUMMARY ==========")
    print(summary_df.to_string(index=False))

    return summary_df

def get_roberta_rule_operation_text(rule_key, rule_catalog_df):
    matching_rule_df = rule_catalog_df[
        rule_catalog_df["rule_key"] == rule_key
    ]

    if matching_rule_df.empty:
        return str(rule_key) + " | rule detected"

    rule_row = matching_rule_df.iloc[0]

    marker = str(rule_row["marker"])
    description = str(rule_row["description"])

    if marker == "AFFIRMATIVE_INTERPRETATION":
        operation = "added affirmative interpretation"
    elif marker == "CONTEXT_SIGNAL":
        operation = "detected context signal"
    else:
        operation = "rule applied"

    return str(rule_key) + " | " + operation + " | " + description


def roberta_review_example_block(row, rule_catalog_df):
    rules_applied = row["rule_keys"]

    print("\nREVIEW INDEX:", row["review_index"])

    print("\nORIGINAL REVIEW:")
    print('"' + str(row["original_text"]) + '"')

    print("\nCHANGED REVIEW:")
    print('"' + str(row["enhanced_text"]) + '"')

    print("\nRULES APPLIED:")
    print(rules_applied)

    print("\nFULL OPERATIONS:")
    if len(rules_applied) == 0:
        print("- No rules applied")
    else:
        for rule_key in rules_applied:
            print("- " + get_roberta_rule_operation_text(
                rule_key=rule_key,
                rule_catalog_df=rule_catalog_df
            ))

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

def roberta_review_examples(title, selected_df, rule_catalog_df, number=0):
    if number > 0:
        sample_size = min(number, len(selected_df))
    else:
        sample_size = len(selected_df)

    print("\n========== " + title + " ==========")
    print("Count:", len(selected_df))

    if selected_df.empty:
        print("No examples found.")
        return

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
        roberta_review_example_block(
            row=row,
            rule_catalog_df=rule_catalog_df
        )

def print_all_roberta_affected_review_examples(
        audit_df,
        rule_catalog_df,
        number=0
):
    affected_df = audit_df[audit_df["number_of_rules"] > 0].copy()
    corrected_df = affected_df[affected_df["effect"] == "corrected"]
    harmed_df = affected_df[affected_df["effect"] == "harmed"]
    wrong_to_wrong_df = affected_df[affected_df["effect"] == "wrong_to_wrong"]
    stayed_wrong_df = affected_df[affected_df["effect"] == "stayed_wrong"]
    stayed_correct_df = affected_df[affected_df["effect"] == "stayed_correct"]

    roberta_review_examples(
        "CORRECTED AFFECTED REVIEWS",
        corrected_df,
        rule_catalog_df,
        number=number
    )
    roberta_review_examples(
        "HARMED AFFECTED REVIEWS",
        harmed_df,
        rule_catalog_df,
        number=number
    )
    roberta_review_examples(
        "WRONG TO WRONG AFFECTED REVIEWS",
        wrong_to_wrong_df,
        rule_catalog_df,
        number=number
    )
    roberta_review_examples(
        "STAYED WRONG AFFECTED REVIEWS",
        stayed_wrong_df,
        rule_catalog_df,
        number=number
    )
    roberta_review_examples(
        "STAYED CORRECT AFFECTED REVIEWS",
        stayed_correct_df,
        rule_catalog_df,
        number=number
    )
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
        dataloader_pin_memory=torch.cuda.is_available(),
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
# ENHANCED ROBERTA HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------- 
N_WORKERS = min(16, os.cpu_count())

print("Creating Enhanced RoBERTa Train Text...")
enhanced_text_train = pd.Series(
    process_map(
        add_affirmative_interpretation,
        text_train.tolist(),
        max_workers=N_WORKERS,
        chunksize=1000,
        desc="Enhanced RoBERTa Train Text"
    ),
    index=text_train.index
)

print("Creating Enhanced RoBERTa Validation Text...")
enhanced_text_val = pd.Series(
    process_map(
        add_affirmative_interpretation,
        text_val.tolist(),
        max_workers=N_WORKERS,
        chunksize=1000,
        desc="Enhanced RoBERTa Validation Text"
    ),
    index=text_val.index
)

print("Creating Enhanced RoBERTa Test Text...")
enhanced_text_test = pd.Series(
    process_map(
        add_affirmative_interpretation,
        text_test.tolist(),
        max_workers=N_WORKERS,
        chunksize=1000,
        desc="Enhanced RoBERTa Test Text"
    ),
    index=text_test.index
)

ROBERTA_OPTUNA_TRAIN_SIZE = min(90000, len(enhanced_text_train))

if ROBERTA_OPTUNA_TRAIN_SIZE < len(enhanced_text_train):
    enhanced_text_train_optuna, _, sentiment_train_num_optuna, _ = train_test_split(
        enhanced_text_train,
        sentiment_train_num,
        train_size=ROBERTA_OPTUNA_TRAIN_SIZE,
        stratify=sentiment_train_num,
        random_state=42
    )
else:
    enhanced_text_train_optuna = enhanced_text_train
    sentiment_train_num_optuna = sentiment_train_num

print("\n========== ENHANCED RoBERTa OPTUNA SUBSET ==========")
print("Optuna training rows:", len(enhanced_text_train_optuna))
print(sentiment_train_num_optuna.value_counts())

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def enhanced_roberta_optuna(trial):
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
        train_text=enhanced_text_train_optuna,
        train_sentiment=sentiment_train_num_optuna,
        tokenizer=optuna_tokenizer,
        model_name=optuna_model_name,
        roberta_params=optuna_params,
        output_dir="./tmp"
    )

    optuna_val_logits = predict_roberta_logits(
        trainer=optuna_trainer,
        predict_text=enhanced_text_val,
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
    enhanced_roberta_optuna,
    n_trials=20,
    n_jobs=1,
    gc_after_trial=True,
    show_progress_bar=True
)

enhanced_roberta_best = roberta_study.best_params

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("\nENHANCED RoBERTa BEST PARAMETERS: " + str(roberta_study.best_value))
print(enhanced_roberta_best)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# Temperature Scaling 
# -----------------------------------------------------------------------------
print("\n===== ENHANCED RoBERTa TEMPERATURE SCALING =====")

roberta_train_calibration_logits = np.zeros(
    (len(enhanced_text_train), 3)
)

for calibration_fold, (calibration_train_idx, calibration_val_idx) in enumerate(
        calibration_cv.split(enhanced_text_train, sentiment_train_num)
    ):
    print(f"\n----- TEMPERATURE SCALING FOLD {calibration_fold + 1}/{calibration_cv.n_splits} -----")

    text_calibration_train = enhanced_text_train.iloc[calibration_train_idx]
    text_calibration_val = enhanced_text_train.iloc[calibration_val_idx]

    sentiment_calibration_train = sentiment_train_num.iloc[calibration_train_idx]
    sentiment_calibration_val = sentiment_train_num.iloc[calibration_val_idx]

    calibration_trainer = train_roberta_model(
        train_text=text_calibration_train,
        train_sentiment=sentiment_calibration_train,
        tokenizer=tokenizer,
        model_name=MODEL,
        roberta_params=enhanced_roberta_best,
        output_dir="./tmp"
    )

    calibration_val_logits = predict_roberta_logits(
        trainer=calibration_trainer,
        predict_text=text_calibration_val,
        predict_sentiment=sentiment_calibration_val,
        tokenizer=tokenizer,
        max_length=enhanced_roberta_best["max_length"]
    )

    roberta_train_calibration_logits[calibration_val_idx] = calibration_val_logits

    cleanup_trainer(calibration_trainer)

enhanced_roberta_temperature_scaler = fit_temperature_scaler(
    logits=roberta_train_calibration_logits,
    sentiment=sentiment_train_num.to_numpy()
)

print("\nEnhanced RoBERTa Temperature:", enhanced_roberta_temperature_scaler.get_temperature())
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE ENHANCED ROBERTA
# -----------------------------------------------------------------------------
print("\n===== ENHANCED RoBERTa =====")

enhanced_roberta_trainer = train_roberta_model(
    train_text=enhanced_text_train,
    train_sentiment=sentiment_train_num,
    tokenizer=tokenizer,
    model_name=MODEL,
    roberta_params=enhanced_roberta_best,
    output_dir="./tmp"
)

enhanced_val_logits = predict_roberta_logits(
    trainer=enhanced_roberta_trainer,
    predict_text=enhanced_text_val,
    predict_sentiment=sentiment_val_num,
    tokenizer=tokenizer,
    max_length=enhanced_roberta_best["max_length"]
)

enhanced_roberta_test_logits = predict_roberta_logits(
    trainer=enhanced_roberta_trainer,
    predict_text=enhanced_text_test,
    predict_sentiment=sentiment_test_num,
    tokenizer=tokenizer,
    max_length=enhanced_roberta_best["max_length"]
)

enhanced_roberta_val_probabilities = apply_temperature_scaling(
    logits=enhanced_val_logits,
    temperature_scaler=enhanced_roberta_temperature_scaler
)

enhanced_roberta_test_probabilities = apply_temperature_scaling(
    logits=enhanced_roberta_test_logits,
    temperature_scaler=enhanced_roberta_temperature_scaler
)

enhanced_roberta_val_prediction_ids = np.argmax(
    enhanced_roberta_val_probabilities,
    axis=1
)

enhanced_roberta_test_prediction_ids = np.argmax(
    enhanced_roberta_test_probabilities,
    axis=1
)

enhanced_roberta_val_sentiment = [
    reverse_label_map[prediction_id]
    for prediction_id in enhanced_roberta_val_prediction_ids
]

enhanced_roberta_test_sentiment = [
    reverse_label_map[prediction_id]
    for prediction_id in enhanced_roberta_test_prediction_ids
]

print("\nBASE RoBERTa ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_roberta_val_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_val, base_roberta_val_sentiment, digits=4))
print("ENHANCED RoBERTa ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, enhanced_roberta_val_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_val, enhanced_roberta_val_sentiment, digits=4))

print("\nBASE RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, base_roberta_test_sentiment, digits=4))
print("ENHANCED RoBERTa ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_roberta_test_sentiment) * 100, 2)) + "%")
print(classification_report(sentiment_test, enhanced_roberta_test_sentiment, digits=4))

cleanup_trainer(enhanced_roberta_trainer)

print("\n===== ENHANCED RoBERTa OOF META FEATURES =====")

roberta_oof_probabilities = np.zeros(
    (len(enhanced_text_train), 3)
)

for fold, (train_idx, fold_val_idx) in enumerate(
        base_cv.split(enhanced_text_train, sentiment_train_num)
    ):
    print(
        f"\n----- OOF META FEATURES FOLD {fold + 1}/{base_cv.n_splits} -----"
    )

    text_fold_train = enhanced_text_train.iloc[train_idx]
    text_fold_val = enhanced_text_train.iloc[fold_val_idx]

    sentiment_fold_train = sentiment_train_num.iloc[train_idx]
    sentiment_fold_val = sentiment_train_num.iloc[fold_val_idx]

    fold_trainer = train_roberta_model(
        train_text=text_fold_train,
        train_sentiment=sentiment_fold_train,
        tokenizer=tokenizer,
        model_name=MODEL,
        roberta_params=enhanced_roberta_best,
        output_dir="./tmp"
    )

    fold_val_logits = predict_roberta_logits(
        trainer=fold_trainer,
        predict_text=text_fold_val,
        predict_sentiment=sentiment_fold_val,
        tokenizer=tokenizer,
        max_length=enhanced_roberta_best["max_length"]
    )

    calibrated_fold_val_probabilities = apply_temperature_scaling(
        logits=fold_val_logits,
        temperature_scaler=enhanced_roberta_temperature_scaler
    )

    roberta_oof_probabilities[fold_val_idx] = calibrated_fold_val_probabilities

    cleanup_trainer(fold_trainer)

enhanced_roberta_train_probabilities = roberta_oof_probabilities
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# ROBERTA AUDIT ON VALIDATION SET
# -----------------------------------------------------------------------------
roberta_rule_catalog_df = build_affirmative_roberta_rule_catalog()

roberta_train_rule_sets = text_train.apply(extract_affirmative_interpretation_rules)
roberta_val_rule_sets = text_val.apply(extract_affirmative_interpretation_rules)
roberta_test_rule_sets = text_test.apply(extract_affirmative_interpretation_rules)

roberta_rule_usage_df = build_roberta_rule_usage_table(
    rule_catalog_df=roberta_rule_catalog_df,
    train_rule_sets=roberta_train_rule_sets,
    val_rule_sets=roberta_val_rule_sets,
    test_rule_sets=roberta_test_rule_sets
)

roberta_rule_review_audit_df = create_roberta_rule_review_audit_df(
    original_text=text_val,
    enhanced_text=enhanced_text_val,
    true_labels=sentiment_val,
    base_sentiment=base_roberta_val_sentiment,
    enhanced_sentiment=enhanced_roberta_val_sentiment,
    rule_sets=roberta_val_rule_sets,
    base_probabilities=base_roberta_val_probabilities,
    enhanced_probabilities=enhanced_roberta_val_probabilities,
    reverse_label_map=reverse_label_map
)

print_roberta_rule_usage_table(roberta_rule_usage_df)

print_short_roberta_review_audit_summary(roberta_rule_review_audit_df)

print_roberta_all_rules_summary_table(roberta_rule_usage_df)

print_roberta_general_correction_summary(
    val_rule_sets=roberta_val_rule_sets,
    sentiment_true=sentiment_val,
    base_sentiment=base_roberta_val_sentiment,
    enhanced_sentiment=enhanced_roberta_val_sentiment
)

print_roberta_non_exclusive_all_rule_table(
    rule_catalog_df=roberta_rule_catalog_df,
    val_rule_sets=roberta_val_rule_sets,
    sentiment_true=sentiment_val,
    base_sentiment=base_roberta_val_sentiment,
    enhanced_sentiment=enhanced_roberta_val_sentiment
)

print_roberta_exclusive_rule_table(
    rule_catalog_df=roberta_rule_catalog_df,
    val_rule_sets=roberta_val_rule_sets,
    sentiment_true=sentiment_val,
    base_sentiment=base_roberta_val_sentiment,
    enhanced_sentiment=enhanced_roberta_val_sentiment
)

print_all_scoped_exclusive_roberta_rule_tables(
    rule_catalog_df=roberta_rule_catalog_df,
    val_rule_sets=roberta_val_rule_sets,
    sentiment_true=sentiment_val,
    base_sentiment=base_roberta_val_sentiment,
    enhanced_sentiment=enhanced_roberta_val_sentiment
)

print_all_roberta_affected_review_examples(
    audit_df=roberta_rule_review_audit_df,
    rule_catalog_df=roberta_rule_catalog_df,
    number=1
)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
output_folder = "Base_Learner/Results/RoBERTa/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_roberta_train_probabilities_df = pd.DataFrame(enhanced_roberta_train_probabilities).rename(columns={
    0: "enhanced_roberta_neg",
    1: "enhanced_roberta_neu",
    2: "enhanced_roberta_pos"
})

enhanced_roberta_val_probabilities_df = pd.DataFrame(enhanced_roberta_val_probabilities).rename(columns={
    0: "enhanced_roberta_neg",
    1: "enhanced_roberta_neu",
    2: "enhanced_roberta_pos"
})

enhanced_roberta_test_probabilities_df = pd.DataFrame(enhanced_roberta_test_probabilities).rename(columns={
    0: "enhanced_roberta_neg",
    1: "enhanced_roberta_neu",
    2: "enhanced_roberta_pos"
})

enhanced_roberta_train_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_train_probabilities.csv"),
    index=False
)

enhanced_roberta_val_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_val_probabilities.csv"),
    index=False
)

enhanced_roberta_test_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_test_probabilities.csv"),
    index=False
)

print("\nSaved RoBERTa Probabilities CSV Files to:", output_folder)

enhanced_roberta_val_sentiment_df = pd.DataFrame({
    "enhanced_roberta_sentiment": enhanced_roberta_val_sentiment
})

enhanced_roberta_test_sentiment_df = pd.DataFrame({
    "enhanced_roberta_sentiment": enhanced_roberta_test_sentiment
})

enhanced_roberta_val_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_val_sentiment.csv"),
    index=False
)

enhanced_roberta_test_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_test_sentiment.csv"),
    index=False
)

print("\nSaved RoBERTa Sentiment CSV Files to:", output_folder)

output_folder = "Base_Learner/Rule_Decisions/RoBERTa"
os.makedirs(output_folder, exist_ok=True)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_roberta_affected_review_examples(
        audit_df=roberta_rule_review_audit_df,
        rule_catalog_df=roberta_rule_catalog_df,
        number=0
    )
    text_output = output.getvalue()
with open(os.path.join(output_folder, "rule_affected_reviews.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_roberta_rule_usage_table(roberta_rule_usage_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "all_rule_usage_table.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_roberta_exclusive_rule_table(
        rule_catalog_df=roberta_rule_catalog_df,
        val_rule_sets=roberta_val_rule_sets,
        sentiment_true=sentiment_val,
        base_sentiment=base_roberta_val_sentiment,
        enhanced_sentiment=enhanced_roberta_val_sentiment
    )
text_output = output.getvalue()
with open(os.path.join(output_folder, "exclusive_rule_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_scoped_exclusive_roberta_rule_tables(
        rule_catalog_df=roberta_rule_catalog_df,
        val_rule_sets=roberta_val_rule_sets,
        sentiment_true=sentiment_val,
        base_sentiment=base_roberta_val_sentiment,
        enhanced_sentiment=enhanced_roberta_val_sentiment
    )
text_output = output.getvalue()
with open(os.path.join(output_folder, "scoped_exclusive_rule_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_short_roberta_review_audit_summary(roberta_rule_review_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "enhanced_roberta_audit_summary.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

print("Saved RoBERTa Audit Text Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/RoBERTa/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_roberta_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        enhanced_roberta_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_roberta_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        enhanced_roberta_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_roberta_val_report_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_validation_classification_report.csv"),
    index_label="class"
)

enhanced_roberta_test_report_df.to_csv(
    os.path.join(output_folder, "enhanced_roberta_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_roberta_val_report_df.values,
    rowLabels=enhanced_roberta_val_report_df.index,
    colLabels=enhanced_roberta_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_roberta_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_roberta_test_report_df.values,
    rowLabels=enhanced_roberta_test_report_df.index,
    colLabels=enhanced_roberta_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_roberta_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Enhanced RoBERTa Classification Report to:", output_folder)

output_folder = "Base_Learner/Results/RoBERTa/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_roberta_optuna_summary = pd.DataFrame([
    {
        "hyperparameter": "learning_rate",
        "search_range": "1e-6 to 5e-5, logarithmic",
        "best_value": enhanced_roberta_best["learning_rate"]
    },
    {
        "hyperparameter": "per_device_train_batch_size",
        "search_range": "8, 16",
        "best_value": enhanced_roberta_best[
            "per_device_train_batch_size"
        ]
    },
    {
        "hyperparameter": "num_train_epochs",
        "search_range": "1 to 3",
        "best_value": enhanced_roberta_best["num_train_epochs"]
    },
    {
        "hyperparameter": "weight_decay",
        "search_range": "0.0 to 0.1",
        "best_value": enhanced_roberta_best["weight_decay"]
    },
    {
        "hyperparameter": "max_length",
        "search_range": "64, 128, 256",
        "best_value": enhanced_roberta_best["max_length"]
    },
    {
        "hyperparameter": "lr_scheduler_type",
        "search_range": "linear, cosine",
        "best_value": enhanced_roberta_best["lr_scheduler_type"]
    },
    {
        "hyperparameter": "warmup_ratio",
        "search_range": "0.0 to 0.1",
        "best_value": enhanced_roberta_best["warmup_ratio"]
    }
])

enhanced_roberta_optuna_summary.to_csv(
    os.path.join(output_folder, "enhanced_roberta_optuna_parameters.csv"),
    index=False
)

print("Saved Enhanced RoBERTa Optuna Parameters to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================ END