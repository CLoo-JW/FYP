import pandas as pd
import numpy as np
from tqdm import tqdm 
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import re
import os
import io
import contextlib
# import nltk
# nltk.download("vader_lexicon")

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
base_vader_val_scores_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_val_scores.csv")

base_vader_val_scores = base_vader_val_scores_df[["neg", "neu", "pos", "compound"]].to_dict("records")

base_vader_val_sentiment_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_val_sentiment.csv")
base_vader_train_sentiment_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_train_sentiment.csv")
base_vader_test_sentiment_df = pd.read_csv("Base_Learner/Results/VADER/Base/base_vader_test_sentiment.csv")

base_vader_train_sentiment = base_vader_train_sentiment_df["base_vader_sentiment"]
base_vader_test_sentiment = base_vader_test_sentiment_df["base_vader_sentiment"]
base_vader_val_sentiment = base_vader_val_sentiment_df["base_vader_sentiment"]

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
# ENHANCED VADER
# ======================================================================================================================
# ----------------------------------------------------------------------------- START
# ENHANCED VADER SETUP
# -----------------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
raw_sia = SentimentIntensityAnalyzer()

sia.constants.NEGATE = set(sia.constants.NEGATE)
sia.constants.BOOSTER_DICT = dict(sia.constants.BOOSTER_DICT)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# DOMAIN-SPECIFIC SINGLE-WORD & PHRASE LEXICONS
# -----------------------------------------------------------------------------
DOMAIN_LEXICON = {
    # Negative words
    "unusable": -2.8,
    "flimsy": -2.0,
    "counterfeit": -2.8,
    "overpriced": -2.0,
    "underwhelming": -1.8,
    "unreliable": -2.4,
    "dented": -1.8,
    "bent": -1.7,
    "warped": -1.8,
    "fragile": -1.8,
    "knockoff": -2.5,
    "malfunctioning": -2.8,
    "unresponsive": -2.4,
    "overheating": -2.5,
    "glitchy": -2.2,
    "laggy": -1.9,
    "flickering": -2.0,
    "scratchy": -1.8,
    "ripped": -2.1,
    "stained": -2.0,
    "shrunken": -2.0,
    "pilling": -1.8,
    "faded": -1.7,
    "inaccurate": -1.8,
    "incomplete": -1.8,
    "misprinted": -2.2,
    "unreadable": -2.8,
    "incoherent": -2.3,
    "lifeless": -1.7,

    # Positive words
    "durable": 2.0,
    "sturdy": 1.8,
    "lightweight": 1.4,
    "affordable": 1.7,
    "authentic": 1.5,
    "adjustable": 1.2,
    "protective": 1.5,
    "convenient": 1.7,
    "smooth": 1.5,
    "sleek": 1.5,
    "premium": 1.6,
    "compatible": 1.2,
    "reliable": 2.0,
    "washable": 1.3,
    "breathable": 1.4,
    "stylish": 1.8,
    "soft": 1.5,
    "stretchy": 1.3,
    "insightful": 2.1,
    "informative": 1.8,
}

DOMAIN_LEXICON_TO_ADD = {
    word: value
    for word, value in DOMAIN_LEXICON.items()
    if word not in raw_sia.lexicon
}
DOMAIN_LEXICON_SKIPPED = {
    word: raw_sia.lexicon[word]
    for word in DOMAIN_LEXICON
    if word in raw_sia.lexicon
}
for word, value in DOMAIN_LEXICON_SKIPPED.items():
    print(f"Skipped '{word}': {value} (already in VADER lexicon)")

sia.lexicon.update(DOMAIN_LEXICON_TO_ADD)
CUSTOM_DOMAIN_WORDS = set(DOMAIN_LEXICON_TO_ADD.keys())

PHRASE_LEXICON = {
    # Positive phrases
    "vaderphrasenotbad": 1.3,
    "vaderphrasenocomplaints": 2.0,
    "vaderphrasenoissues": 2.0,
    "vaderphrasenoproblems": 1.9,
    "vaderphrasenoregrets": 2.1,
    "vaderphraseworksgreat": 2.4,
    "vaderphraseworksperfectly": 2.7,
    "vaderphraseworksasexpected": 1.8,
    "vaderphrasehighlyrecommend": 2.7,
    "vaderphrasewouldrecommend": 2.2,
    "vaderphrasewouldbuyagain": 2.4,
    "vaderphrasewortheverypenny": 2.5,
    "vaderphrasebetterthanexpected": 2.5,
    "vaderphraseexceededexpectations": 2.8,
    "vaderphrasecouldnotbehappier": 3.2,
    "vaderphrasecannotrecommendenough": 3.0,
    "vaderphrasetruetosize": 2.0,
    "vaderphrasecomfortablefit": 2.1,
    "vaderphrasegreatread": 2.4,
    "vaderphrasewellwritten": 2.3,
    "vaderphraseeasytofollow": 2.0,
    "vaderphrasehighlyinformative": 2.2,

    # Neutral / mixed phrases
    "vaderphraseneutralgoodbutnotgreat": 0.2,
    "vaderphraseneutralnotbadnotgreat": 0.1,
    "vaderphraseneutralokaybutissues": -0.3,
    "vaderphraseneutralworksbut": -0.2,
    "vaderphraseneutraldecentforprice": 0.3,
    "vaderphraseneutralaveragenothingspecial": 0.0,
    "vaderphraseneutralprosandcons": 0.0,
    "vaderphraseneutralsomewhatdisappointed": -0.5,
    "vaderphraseneutralexpectedmore": -0.4,

    # Negative phrases
    "vaderphrasenotworthit": -2.8,
    "vaderphrasewasteofmoney": -3.2,
    "vaderphraselowquality": -2.4,
    "vaderphrasecheaplymade": -2.5,
    "vaderphrasefellapart": -3.0,
    "vaderphrasenotlasting": -2.5,
    "vaderphrasenotasdescribed": -2.7,
    "vaderphrasewrongitem": -2.6,
    "vaderphrasemissingparts": -2.6,
    "vaderphraseneverreceived": -3.0,
    "vaderphrasenotdelivered": -3.0,
    "vaderphrasehadreturn": -2.5,
    "vaderphrasewantrefund": -2.5,
    "vaderphrasenotsatisfied": -2.2,
    "vaderphrasewouldnotrecommend": -2.8,
    "vaderphrasedoesnotwork": -3.2,
    "vaderphrasestoppedworking": -3.1,
    "vaderphrasenolongerworks": -3.0,
    "vaderphrasedeadonarrival": -3.2,
    "vaderphrasenotpoweringon": -3.2,
    "vaderphrasenotturningon": -3.2,
    "vaderphrasenotcharging": -3.0,
    "vaderphrasebatterydrainsfast": -2.6,
    "vaderphrasedoesnotholdcharge": -2.8,
    "vaderphrasekeepsdisconnecting": -2.4,
    "vaderphrasepoorconnection": -2.1,
    "vaderphraseoverheatsquickly": -2.6,
    "vaderphrasescreencracked": -2.7,
    "vaderphraseflickeringscreen": -2.3,
    "vaderphrasetouchscreennotresponsive": -2.7,
    "vaderphrasenotfitting": -2.6,
    "vaderphrasewrongsize": -2.2,
    "vaderphrasewrongcolor": -2.0,
    "vaderphraserunssmall": -1.8,
    "vaderphraserunslarge": -1.6,
    "vaderphrasepoorfit": -2.2,
    "vaderphraseseethrough": -2.1,
    "vaderphraseshrunkafterwash": -2.3,
    "vaderphrasecolorfaded": -1.9,
    "vaderphrasefabricfeelscheap": -2.2,
    "vaderphraseseamripped": -2.4,
    "vaderphrasezipperbroken": -2.5,
    "vaderphrasemissingpages": -2.8,
    "vaderphrasepoorlywritten": -2.4,
    "vaderphrasehardtofollow": -2.1,
    "vaderphrasebadtranslation": -2.2,
    "vaderphraseprintingerror": -2.2,
}

sia.lexicon.update(PHRASE_LEXICON)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# CUSTOM INTENSIFIERS AND DIMINISHERS
# -----------------------------------------------------------------------------
CUSTOM_INTENSIFIERS = {
    "very": 0.293, "really": 0.293, "extremely": 0.293, "incredibly": 0.293, "highly": 0.293,
    "super": 0.293, "ultra": 0.293, "absolutely": 0.293, "completely": 0.293, "totally": 0.293,
    "surprisingly": 0.293, "ridiculously": 0.293, "seriously": 0.293, "terribly": 0.293
}

CUSTOM_DIMINISHERS = {
    "slightly": -0.293, "somewhat": -0.293, "mildly": -0.293, "partly": -0.293,
    "partially": -0.293, "kinda": -0.293, "sorta": -0.293, "barely": -0.293, "hardly": -0.293,
    "almost": -0.293, "fairly": -0.293
}

CUSTOM_INTENSIFIERS_TO_ADD = {
    word: value
    for word, value in CUSTOM_INTENSIFIERS.items()
    if word not in raw_sia.constants.BOOSTER_DICT
}
CUSTOM_DIMINISHERS_TO_ADD = {
    word: value
    for word, value in CUSTOM_DIMINISHERS.items()
    if word not in raw_sia.constants.BOOSTER_DICT
}

sia.constants.BOOSTER_DICT.update(CUSTOM_INTENSIFIERS_TO_ADD)
sia.constants.BOOSTER_DICT.update(CUSTOM_DIMINISHERS_TO_ADD)

CUSTOM_INTENSIFIER_WORDS = set(CUSTOM_INTENSIFIERS_TO_ADD.keys())
CUSTOM_DIMINISHER_WORDS = set(CUSTOM_DIMINISHERS_TO_ADD.keys())
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# CUSTOM NEGATORS
# -----------------------------------------------------------------------------
CUSTOM_NEGATORS = {
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

CUSTOM_NEGATORS_TO_ADD = (CUSTOM_NEGATORS - raw_sia.constants.NEGATE)
sia.constants.NEGATE.update(CUSTOM_NEGATORS_TO_ADD)
CUSTOM_NEGATOR_WORDS = set(CUSTOM_NEGATORS_TO_ADD)
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# NEGATED AUXILIARY VERBS
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
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# PHRASE PATTERNS
# -----------------------------------------------------------------------------
NEGATIVE_PHRASE_PATTERNS = [
    {
        "type": "negative_phrase",
        "name": "not_worth_it",
        "pattern": re.compile(
            r"\bnot\s+" + OPTIONAL_DEGREE +
            r"worth\s+(?:it|the\s+money|the\s+price|buying|getting|keeping)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotworthit"
    },
    {
        "type": "negative_phrase",
        "name": "waste_of_money",
        "pattern": re.compile(
            r"\b(?:a\s+)?(?:complete\s+|total\s+|real\s+|absolute\s+)?"
            r"waste\s+of\s+(?:money|time)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewasteofmoney"
    },
    {
        "type": "product_quality",
        "name": "low_quality",
        "pattern": re.compile(
            r"\b(?:low|poor|bad|terrible|awful|horrible)\s+quality\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraselowquality"
    },
    {
        "type": "product_quality",
        "name": "cheaply_made",
        "pattern": re.compile(
            r"\b(?:cheaply\s+made|"
            r"(?:feel|feels|felt|feeling)\s+cheap|"
            r"(?:material|fabric|plastic|product|item)\s+"
            r"(?:feel|feels|felt|feeling)\s+cheap)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasecheaplymade"
    },
    {
        "type": "product_quality",
        "name": "fell_apart",
        "pattern": re.compile(
            r"\b(?:fell|came|comes|coming)\s+apart\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasefellapart"
    },
    {
        "type": "product_quality",
        "name": "not_lasting",
        "pattern": re.compile(
            r"\b(?:"
            r"(?:did\s+not|didn't|didnt)\s+last|"
            r"not\s+lasting|"
            r"only\s+lasted|"
            r"lasted\s+(?:only\s+)?(?:a\s+)?(?:day|week|month|few\s+days|few\s+weeks)|"
            r"broke\s+(?:after|within)"
            r")\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotlasting"
    },
    {
        "type": "product_description",
        "name": "not_as_described",
        "pattern": re.compile(
            r"\b(?:not|isn't|isnt|wasn't|wasnt|is\s+not|was\s+not)\s+"
            r"(?:as\s+)?described\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotasdescribed"
    },
    {
        "type": "order_issue",
        "name": "wrong_item",
        "pattern": re.compile(
            r"\bwrong\s+(?:item|product|model|version|book|charger|case)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewrongitem"
    },
    {
        "type": "order_issue",
        "name": "missing_parts",
        "pattern": re.compile(
            r"\bmissing\s+(?:parts?|pieces?|accessories|components|items?)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasemissingparts"
    },
    {
        "type": "delivery_issue",
        "name": "never_received",
        "pattern": re.compile(
            r"\b(?:never\s+received|"
            r"did\s+not\s+receive|didn't\s+receive|didnt\s+receive|"
            r"have\s+not\s+received|haven't\s+received|havent\s+received)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneverreceived"
    },
    {
        "type": "delivery_issue",
        "name": "not_delivered",
        "pattern": re.compile(
            r"\b(?:not|never|was\s+not|wasn't|wasnt|"
            r"has\s+not\s+been|hasn't\s+been|hasnt\s+been)"
            r"\s+delivered\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotdelivered"
    },
    {
        "type": "return_issue",
        "name": "had_to_return",
        "pattern": re.compile(
            r"\b(?:had\s+to\s+return|"
            r"returned\s+(?:it|this|the\s+item|the\s+product|the\s+book))\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasehadreturn"
    },
    {
        "type": "return_issue",
        "name": "want_refund",
        "pattern": re.compile(
            r"\b(?:want|wanted|need|needed|request(?:ed)?|asking\s+for)"
            r"\s+(?:a\s+)?refund\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewantrefund"
    },
    {
        "type": "negative_phrase",
        "name": "not_satisfied",
        "pattern": re.compile(
            r"\bnot\s+(?:very\s+|really\s+|fully\s+|completely\s+)?"
            r"(?:satisfied|happy)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotsatisfied"
    },
    {
        "type": "negative_phrase",
        "name": "would_not_recommend",
        "pattern": re.compile(
            r"\b(?:would\s+not|wouldn't|wouldnt)\s+recommend\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewouldnotrecommend"
    },

    # Electronics
    {
        "type": "electronics",
        "name": "does_not_work",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasedoesnotwork"
    },
    {
        "type": "electronics",
        "name": "stopped_working",
        "pattern": re.compile(
            r"\b(?:stopped|stop|stops|quit|quits)\s+working\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasestoppedworking"
    },
    {
        "type": "electronics",
        "name": "no_longer_works",
        "pattern": re.compile(
            r"\bno\s+longer\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenolongerworks"
    },
    {
        "type": "electronics",
        "name": "dead_on_arrival",
        "pattern": re.compile(
            r"\b(?:dead\s+on\s+arrival|doa)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasedeadonarrival"
    },
    {
        "type": "electronics",
        "name": "not_powering_on",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:power\s+on|powering\s+on)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotpoweringon"
    },
    {
        "type": "electronics",
        "name": "not_turning_on",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:turn\s+on|turning\s+on)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotturningon"
    },
    {
        "type": "electronics",
        "name": "not_charging",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+charg(?:e|es|ed|ing)\b|"
            r"\b(?:stopped|stop|stops|quit|quits)\s+charging\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotcharging"
    },
    {
        "type": "electronics",
        "name": "battery_drains_fast",
        "pattern": re.compile(
            r"\b(?:battery|batteries)\s+"
            r"(?:drain|drains|drained|die|dies|died)\s+"
            r"(?:too\s+)?(?:fast|quickly|rapidly)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasebatterydrainsfast"
    },
    {
        "type": "electronics",
        "name": "does_not_hold_charge",
        "pattern": re.compile(
            rf"\b(?:battery\s+)?(?:{NEG_AUX})\s+hold\s+"
            r"(?:a\s+|the\s+)?charge\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasedoesnotholdcharge"
    },
    {
        "type": "electronics",
        "name": "keeps_disconnecting",
        "pattern": re.compile(
            r"\b(?:keep|keeps|kept)\s+disconnecting\b|"
            r"\blosing\s+connection\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasekeepsdisconnecting"
    },
    {
        "type": "electronics",
        "name": "poor_connection",
        "pattern": re.compile(
            r"\b(?:poor|bad|weak|unstable)\s+(?:connection|signal)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasepoorconnection"
    },
    {
        "type": "electronics",
        "name": "overheats_quickly",
        "pattern": re.compile(
            r"\b(?:overheat|overheats|overheated|overheating)\s+"
            r"(?:too\s+)?(?:quickly|fast|easily)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseoverheatsquickly"
    },
    {
        "type": "electronics",
        "name": "screen_cracked",
        "pattern": re.compile(
            r"\b(?:screen\s+cracked|cracked\s+screen)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasescreencracked"
    },
    {
        "type": "electronics",
        "name": "flickering_screen",
        "pattern": re.compile(
            r"\b(?:screen|display)\s+(?:is\s+|was\s+|keeps\s+|kept\s+|started\s+)?"
            r"(?:flickering|flickers|flicker)\b|"
            r"\b(?:flickering|flickers|flicker)\s+(?:screen|display)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseflickeringscreen"
    },
    {
        "type": "electronics",
        "name": "touchscreen_not_responsive",
        "pattern": re.compile(
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:is\s+|was\s+)?not\s+responsive\b|"
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:does\s+not|doesn't|doesnt)\s+respond\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasetouchscreennotresponsive"
    },

    # Fashion
    {
        "type": "fashion",
        "name": "not_fitting",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+fit(?:s|ted|ting)?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotfitting"
    },
    {
        "type": "fashion",
        "name": "wrong_size",
        "pattern": re.compile(
            r"\bwrong\s+size\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewrongsize"
    },
    {
        "type": "fashion",
        "name": "wrong_color",
        "pattern": re.compile(
            r"\bwrong\s+(?:color|colour)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewrongcolor"
    },
    {
        "type": "fashion",
        "name": "runs_small",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?small\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraserunssmall"
    },
    {
        "type": "fashion",
        "name": "runs_large",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?(?:large|big)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraserunslarge"
    },
    {
        "type": "fashion",
        "name": "poor_fit",
        "pattern": re.compile(
            r"\b(?:poor|bad|terrible|awkward|weird)\s+fit\b|"
            r"\bfit(?:s|ted)?\s+(?:poorly|badly|terribly|awkwardly)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasepoorfit"
    },
    {
        "type": "fashion",
        "name": "see_through",
        "pattern": re.compile(
            r"\b(?:see\s*through|see-through|too\s+sheer|transparent)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseseethrough"
    },
    {
        "type": "fashion",
        "name": "shrunk_after_wash",
        "pattern": re.compile(
            r"\b(?:shrank|shrunk|shrinked)\s+"
            r"(?:after|following)\s+(?:a\s+)?(?:wash|washing)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseshrunkafterwash"
    },
    {
        "type": "fashion",
        "name": "color_faded",
        "pattern": re.compile(
            r"\b(?:color|colour|colors|colours)\s+(?:faded|fades|fade)\b|"
            r"\b(?:faded|fades|fade)\s+(?:color|colour|colors|colours)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasecolorfaded"
    },
    {
        "type": "fashion",
        "name": "fabric_feels_cheap",
        "pattern": re.compile(
            r"\b(?:fabric|material)\s+(?:feel|feels|felt|feeling)\s+cheap\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasefabricfeelscheap"
    },
    {
        "type": "fashion",
        "name": "seam_ripped",
        "pattern": re.compile(
            r"\bseam\s+(?:ripped|torn)\b|"
            r"\bstitching\s+(?:came\s+loose|undone|ripped|torn)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseseamripped"
    },
    {
        "type": "fashion",
        "name": "zipper_broken",
        "pattern": re.compile(
            r"\bzipper\s+(?:broken|stuck|jammed|broke)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasezipperbroken"
    },

    # Books
    {
        "type": "books",
        "name": "missing_pages",
        "pattern": re.compile(
            r"\bmissing\s+pages?\b|"
            r"\bpages?\s+(?:is\s+|are\s+|was\s+|were\s+)?missing\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasemissingpages"
    },
    {
        "type": "books",
        "name": "poorly_written",
        "pattern": re.compile(
            r"\b(?:poorly|badly|terribly)\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:poor|bad|terrible|awful)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasepoorlywritten"
    },
    {
        "type": "books",
        "name": "hard_to_follow",
        "pattern": re.compile(
            r"\b(?:hard|difficult|confusing)\s+to\s+follow\b|"
            r"\bnot\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasehardtofollow"
    },
    {
        "type": "books",
        "name": "bad_translation",
        "pattern": re.compile(
            r"\b(?:bad|poor|terrible|awful)\s+translation\b|"
            r"\btranslation\s+(?:is\s+|was\s+)?(?:bad|poor|terrible|awful)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasebadtranslation"
    },
    {
        "type": "books",
        "name": "printing_error",
        "pattern": re.compile(
            r"\b(?:printing|print)\s+errors?\b|"
            r"\b(?:misprint|misprinted|misprints)\b|"
            r"\bpages?\s+(?:printed|print)\s+incorrectly\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseprintingerror"
    },
]

POSITIVE_PHRASE_PATTERNS = [
    {
        "type": "positive_phrase",
        "name": "not_bad",
        "pattern": re.compile(
            r"\bnot\s+(?:too\s+|that\s+|so\s+|very\s+)?bad\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenotbad"
    },
    {
        "type": "positive_phrase",
        "name": "no_complaints",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?complaints?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenocomplaints"
    },
    {
        "type": "positive_phrase",
        "name": "no_issues",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?issues?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenoissues"
    },
    {
        "type": "positive_phrase",
        "name": "no_problems",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?problems?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenoproblems"
    },
    {
        "type": "positive_phrase",
        "name": "no_regrets",
        "pattern": re.compile(
            r"\bno\s+regrets?\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasenoregrets"
    },
    {
        "type": "positive_phrase",
        "name": "works_great",
        "pattern": re.compile(
            r"\bworks?\s+(?:really\s+|very\s+|so\s+)?great\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseworksgreat"
    },
    {
        "type": "positive_phrase",
        "name": "works_perfectly",
        "pattern": re.compile(
            r"\bworks?\s+(?:perfectly|flawlessly)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseworksperfectly"
    },
    {
        "type": "positive_phrase",
        "name": "works_as_expected",
        "pattern": re.compile(
            r"\bwork(?:s|ed)?\s+(?:exactly\s+)?as\s+expected\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseworksasexpected"
    },
    {
        "type": "positive_phrase",
        "name": "highly_recommend",
        "pattern": re.compile(
            r"\b(?:highly|strongly|definitely)\s+recommend\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasehighlyrecommend"
    },
    {
        "type": "positive_phrase",
        "name": "would_recommend",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+|highly\s+)?recommend\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewouldrecommend"
    },
    {
        "type": "positive_phrase",
        "name": "would_buy_again",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+)?buy\s+(?:it\s+|this\s+)?again\b|"
            r"\bbuy\s+(?:it\s+|this\s+)?again\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewouldbuyagain"
    },
    {
        "type": "positive_phrase",
        "name": "worth_every_penny",
        "pattern": re.compile(
            r"\bworth\s+every\s+(?:penny|cent)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewortheverypenny"
    },
    {
        "type": "positive_phrase",
        "name": "better_than_expected",
        "pattern": re.compile(
            r"\bbetter\s+than\s+(?:i\s+)?expected\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasebetterthanexpected"
    },
    {
        "type": "positive_phrase",
        "name": "exceeded_expectations",
        "pattern": re.compile(
            r"\bexceeded\s+(?:my\s+)?expectations\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseexceededexpectations"
    },
    {
        "type": "positive_phrase",
        "name": "could_not_be_happier",
        "pattern": re.compile(
            r"\b(?:could\s+not|couldn't|couldnt)\s+be\s+happier\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasecouldnotbehappier"
    },
    {
        "type": "positive_phrase",
        "name": "cannot_recommend_enough",
        "pattern": re.compile(
            r"\b(?:cannot|can\s+not|can't|cant)\s+recommend"
            r"(?:\s+(?:it|this|these|them))?\s+enough\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasecannotrecommendenough"
    },
    {
        "type": "positive_phrase",
        "name": "true_to_size",
        "pattern": re.compile(
            r"\btrue\s+to\s+size\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasetruetosize"
    },
    {
        "type": "positive_phrase",
        "name": "comfortable_fit",
        "pattern": re.compile(
            r"\bcomfortable\s+fit\b|"
            r"\bfit(?:s|ted)?\s+comfortably\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasecomfortablefit"
    },
    {
        "type": "positive_phrase",
        "name": "great_read",
        "pattern": re.compile(
            r"\bgreat\s+read\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasegreatread"
    },
    {
        "type": "positive_phrase",
        "name": "well_written",
        "pattern": re.compile(
            r"\bwell\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:excellent|great|clear)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasewellwritten"
    },
    {
        "type": "positive_phrase",
        "name": "easy_to_follow",
        "pattern": re.compile(
            r"\b(?:easy|clear)\s+to\s+follow\b|"
            r"\bclear\s+and\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseeasytofollow"
    },
    {
        "type": "positive_phrase",
        "name": "highly_informative",
        "pattern": re.compile(
            r"\b(?:highly|very|really)\s+informative\b|"
            r"\binformative\s+and\s+useful\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphrasehighlyinformative"
    },
]

NEUTRAL_PHRASE_PATTERNS = [
    {
        "type": "neutral_phrase",
        "name": "neutral_good_but_not_great",
        "pattern": re.compile(
            r"\b(?:good|decent|okay|ok|fine)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect|the\s+best)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralgoodbutnotgreat"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_not_bad_not_great",
        "pattern": re.compile(
            r"\bnot\s+(?:bad|terrible|awful)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralnotbadnotgreat"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_okay_but_issues",
        "pattern": re.compile(
            r"\b(?:okay|ok|fine|decent|good)\s+but\s+"
            r"(?:has|have|had|with)\s+(?:some\s+)?"
            r"(?:issues|problems|flaws|drawbacks|downsides)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralokaybutissues"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_works_but",
        "pattern": re.compile(
            r"\b(?:works|worked|work)\s+but\s+"
            r"(?:not\s+perfect|not\s+great|has\s+issues|could\s+be\s+better|"
            r"there\s+are\s+issues|with\s+some\s+problems)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralworksbut"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_decent_for_price",
        "pattern": re.compile(
            r"\b(?:decent|okay|ok|fine|acceptable|reasonable)\s+"
            r"(?:for|given)\s+(?:the\s+)?(?:price|money|cost)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutraldecentforprice"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_average_nothing_special",
        "pattern": re.compile(
            r"\b(?:average|mediocre|ordinary)\s+"
            r"(?:product|item|quality|book|read|purchase)\b|"
            r"\bnothing\s+(?:special|amazing|great|exceptional)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralaveragenothingspecial"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_pros_and_cons",
        "pattern": re.compile(
            r"\b(?:pros\s+and\s+cons|good\s+and\s+bad|"
            r"some\s+good\s+and\s+some\s+bad|mixed\s+feelings|mixed\s+review)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralprosandcons"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_somewhat_disappointed",
        "pattern": re.compile(
            r"\b(?:somewhat|slightly|a\s+little|kind\s+of|kinda)\s+"
            r"(?:disappointed|underwhelmed)\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralsomewhatdisappointed"
    },
    {
        "type": "neutral_phrase",
        "name": "neutral_expected_more",
        "pattern": re.compile(
            r"\b(?:expected|was\s+expecting)\s+"
            r"(?:a\s+)?(?:little\s+)?more\b|"
            r"\bnot\s+(?:quite|really)\s+what\s+i\s+expected\b",
            flags=re.IGNORECASE
        ),
        "replacement": "vaderphraseneutralexpectedmore"
    },
]
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAFE SENTENCE SEGMENTATION
# -----------------------------------------------------------------------------
SENTENCE_PATTERN = re.compile(
    r"[^.!?]+(?:[.!?]+|$)",
    flags=re.DOTALL
)

def transform_sentences(text, transformation):
    matches = list(SENTENCE_PATTERN.finditer(text))

    if not matches:
        return text, []

    output_parts = []
    operations = []
    previous_end = 0

    for match in matches:
        output_parts.append(text[previous_end:match.start()])
        sentence = match.group(0)
        transformed_sentence, sentence_operations = (transformation(sentence))
        output_parts.append(transformed_sentence)
        operations.extend(sentence_operations)
        previous_end = match.end()

    output_parts.append(text[previous_end:])

    return "".join(output_parts), operations
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# NOT ONLY ... BUT ALSO EXCEPTION
# -----------------------------------------------------------------------------
NOT_ONLY_PATTERN = re.compile(
    r"\bnot\s+only\b",
    flags=re.IGNORECASE
)
BUT_ALSO_PATTERN = re.compile(
    r"\bbut\s+also\b",
    flags=re.IGNORECASE
)
def handle_not_only_in_sentence(sentence):
    not_only_matches = list(NOT_ONLY_PATTERN.finditer(sentence))
    but_also_matches = list(BUT_ALSO_PATTERN.finditer(sentence))

    if len(not_only_matches) != 1 or len(but_also_matches) != 1:
        return sentence, []

    not_only_match = not_only_matches[0]
    but_also_match = but_also_matches[0]
    if but_also_match.start() <= not_only_match.end():
        return sentence, []

    middle_clause = sentence[not_only_match.end(): but_also_match.start()].strip(" ,;:-")
    second_clause = sentence[but_also_match.end():].strip(" ,;:-.!?")
    if not re.search(r"[A-Za-z]", middle_clause) or not re.search(r"[A-Za-z]", second_clause):
        return sentence, []

    updated_sentence = NOT_ONLY_PATTERN.sub("", sentence, count=1)
    updated_sentence = BUT_ALSO_PATTERN.sub("and", updated_sentence, count=1)
    updated_sentence = re.sub(r"[ \t]{2,}"," ", updated_sentence)

    return updated_sentence, [{
        "type": "negation_exception",
        "name": "not_only_but_also",
        "original_sentence": sentence,
        "rewritten_sentence": updated_sentence,
        "action": (
            "removed additive negation and "
            "replaced 'but also' with 'and'"
        )
    }]

def handle_not_only_exception(text):
    return transform_sentences(
        text,
        handle_not_only_in_sentence
    )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# CONCESSIVE CONTRAST NORMALISATION
# -----------------------------------------------------------------------------
LEADING_CONCESSIVE_PATTERN = re.compile(
    r"^\s*(although|though)\s+(.+?),\s+(.+?)\s*$",
    flags=re.IGNORECASE
)

TRAILING_CONCESSIVE_PATTERN = re.compile(
    r"^\s*(.+?),\s*(although|though)\s+(.+?)\s*$",
    flags=re.IGNORECASE
)

def normalise_concessive_in_sentence(sentence):
    leading_match = LEADING_CONCESSIVE_PATTERN.match(sentence)

    if leading_match:
        marker = leading_match.group(1)
        concessive_clause = leading_match.group(2).strip()
        main_clause = leading_match.group(3).strip()

        updated_sentence = concessive_clause + " but " + main_clause

        return updated_sentence, [{
            "type": "concessive_normalisation",
            "name": marker.lower(),
            "cue": marker,
            "concessive_clause": concessive_clause,
            "main_clause": main_clause,
            "original_sentence": sentence,
            "rewritten_sentence": updated_sentence,
            "action": "normalised leading concessive contrast to 'but'"
        }]

    trailing_match = TRAILING_CONCESSIVE_PATTERN.match(sentence)

    if trailing_match:
        main_clause = trailing_match.group(1).strip()
        marker = trailing_match.group(2)
        concessive_clause = trailing_match.group(3).strip()

        updated_sentence = concessive_clause + " but " + main_clause

        return updated_sentence, [{
            "type": "concessive_normalisation",
            "name": marker.lower(),
            "cue": marker,
            "concessive_clause": concessive_clause,
            "main_clause": main_clause,
            "original_sentence": sentence,
            "rewritten_sentence": updated_sentence,
            "action": "normalised trailing concessive contrast to 'but'"
        }]

    return sentence, []


def normalise_concessive_contrast(text):
    return transform_sentences(
        text,
        normalise_concessive_in_sentence
    )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# CONTRAST NORMALISATION
# -----------------------------------------------------------------------------
SAFE_CONTRAST_PATTERN = re.compile(
    r"\b(?:however|nevertheless|nonetheless)\b",
    flags=re.IGNORECASE
)
ALL_CONTRAST_PATTERN = re.compile(
    r"\b(?:"
    r"but|yet|although|though|"
    r"however|nevertheless|nonetheless"
    r")\b",
    flags=re.IGNORECASE
)

def non_neutral_vader_label(scores):
    compound = scores["compound"]

    if compound >= 0.05:
        return "pos"
    if compound <= -0.05:
        return "neg"

    return None

def normalise_contrast_in_sentence(sentence):
    safe_matches = list(SAFE_CONTRAST_PATTERN.finditer(sentence))
    all_contrast_matches = list(ALL_CONTRAST_PATTERN.finditer(sentence))

    if len(safe_matches) != 1 or len(all_contrast_matches) != 1:
        return sentence, []

    marker = safe_matches[0]
    left_clause = sentence[:marker.start()].strip(" \t,;:-")
    right_clause = sentence[marker.end():].strip(" \t,;:-.!?")

    if not re.search(r"[A-Za-z]", left_clause):
        return sentence, []
    if not re.search(r"[A-Za-z]", right_clause):
        return sentence, []

    left_scores = sia.polarity_scores(left_clause)
    right_scores = sia.polarity_scores(right_clause)
    left_polarity = non_neutral_vader_label(left_scores)
    right_polarity = non_neutral_vader_label(right_scores)

    if left_polarity is None or right_polarity is None:
        return sentence, []
    if left_polarity == right_polarity:
        return sentence, []

    left_output = sentence[:marker.start()].rstrip(" \t,;:")
    right_output = sentence[marker.end():].lstrip(" \t,;:")
    updated_sentence = (left_output + " but " + right_output)

    return updated_sentence, [{
        "type": "contrast_normalisation",
        "name": marker.group(0).lower(),
        "cue": marker.group(0),
        "left_clause": left_clause,
        "right_clause": right_clause,
        "left_polarity": left_polarity,
        "right_polarity": right_polarity,
        "original_sentence": sentence,
        "rewritten_sentence": updated_sentence,
        "action": "normalised safe within-sentence contrast marker to 'but'"
    }]



def normalise_contrast(text):
    return transform_sentences(
        text,
        normalise_contrast_in_sentence
    )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# APPLY EXACT PATTERN MATCHING
# -----------------------------------------------------------------------------
def apply_exact_patterns(text, pattern_definitions):
    processed_text = text
    operations = []

    for rule in pattern_definitions:
        matches = list(rule["pattern"].finditer(processed_text))

        if not matches:
            continue

        matched_texts = [match.group(0) for match in matches]
        before_text = processed_text
        processed_text = (rule["pattern"].sub(rule["replacement"],processed_text))
        operations.append({
            "type":
                rule["type"],
            "name":
                rule["name"],
            "matches":
                len(matches),
            "matched_texts":
                matched_texts,
            "replacement":
                rule["replacement"],
            "before_text":
                before_text,
            "after_text":
                processed_text,
            "action":
                "replaced exact phrase"
        })

    return processed_text, operations
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# COMPLETE RULE-BASED PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
def tokenize_for_rule_detection(text):
    normalized_text = (
        str(text)
        .lower()
        .replace("’", "'")
    )

    return set(
        re.findall(
            r"[a-z]+(?:'[a-z]+)?",
            normalized_text
        )
    )

def detect_dictionary_operations(text):
    tokens = tokenize_for_rule_detection(text)
    operations = []

    for word in sorted(tokens & CUSTOM_DOMAIN_WORDS):
        operations.append({
            "type": "domain_lexicon",
            "name": word,
            "action": (
                "used custom domain lexicon value"
            )
        })
    for word in sorted(tokens & CUSTOM_INTENSIFIER_WORDS):
        operations.append({
            "type": "intensifier",
            "name": word,
            "action": (
                "used custom intensifier"
            )
        })
    for word in sorted(tokens & CUSTOM_DIMINISHER_WORDS):
        operations.append({
            "type": "diminisher",
            "name": word,
            "action": (
                "used custom diminisher"
            )
        })
    for word in sorted(tokens & CUSTOM_NEGATOR_WORDS):
        operations.append({
            "type": "custom_negator",
            "name": word,
            "action": "used custom VADER negator"
        })

    return operations

def preprocess_for_rule_vader(text):
    original_text = str(text)
    processed_text = original_text
    operations = detect_dictionary_operations(original_text)
    pattern_groups = [
        NEUTRAL_PHRASE_PATTERNS,
        NEGATIVE_PHRASE_PATTERNS,
        POSITIVE_PHRASE_PATTERNS
    ]

    processed_text, not_only_operations = (
        handle_not_only_exception(
            processed_text
        )
    )
    operations.extend(not_only_operations)

    processed_text, concessive_operations = (
        normalise_concessive_contrast(
            processed_text
        )
    )
    operations.extend(concessive_operations)

    for pattern_group in pattern_groups:
        processed_text, group_operations = (
            apply_exact_patterns(
                processed_text,
                pattern_group
            )
        )
        operations.extend(group_operations)

    processed_text, contrast_operations = (
        normalise_contrast(
            processed_text
        )
    )
    operations.extend(contrast_operations)

    if not operations:
        return original_text, []

    return processed_text, operations
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# VADER LABEL
# -----------------------------------------------------------------------------
VADER_SCORE_KEYS = ["neg", "neu", "pos", "compound"]

def vader_label(compound):
    if compound >= 0.05:
        return "pos"

    if compound <= -0.05:
        return "neg"

    return "neu"
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SCORE & AUDIT ENHANCED VADER
# -----------------------------------------------------------------------------
def score_enhanced_vader_reviews(texts, keep_trace=False, split="Enhanced VADER"):
    scores = []
    audit = [] if keep_trace else None

    for text_value in tqdm(
            texts,
            total=len(texts),
            desc=split
    ):
        original_text = str(text_value)
        processed_text, operations = (
            preprocess_for_rule_vader(
                original_text
            )
        )

        result = sia.polarity_scores(processed_text)
        scores.append(result)

        if keep_trace:
            audit.append({
                "original_review": original_text,
                "processed_review": processed_text,
                "rule_applied": bool(operations),
                "operations": operations,
                "scores": result
            })

    return scores, audit

def extract_rule_keys(operations):
    rule_keys = []

    for operation in operations:
        rule_type = operation.get("type", "unknown")
        rule_name = operation.get("name", operation.get("cue", "unnamed"))
        rule_keys.append(f"{rule_type}:{str(rule_name).lower()}")

    return sorted(set(rule_keys))

def scores_are_different(raw_scores, enhanced_scores, tolerance=1e-6):
    raw_values = np.asarray([raw_scores[key] for key in VADER_SCORE_KEYS])
    enhanced_values = np.asarray([enhanced_scores[key] for key in VADER_SCORE_KEYS])

    return not np.allclose(raw_values, enhanced_values, atol=tolerance, rtol=0.0)

def true_class_margin(scores, true_label):
    true_score = scores[true_label]
    other_scores = [scores[label]
                    for label in ["neg", "neu", "pos"]
                    if label != true_label
                    ]

    return true_score - max(other_scores)

def create_enhanced_vader_audit(texts, true_labels, raw_scores, enhanced_scores, enhanced_audit):
    rows = []

    for index in range(len(texts)):
        original_text = str(texts.iloc[index])
        true_label = str(true_labels.iloc[index])

        raw_result = raw_scores[index]
        enhanced_result = enhanced_scores[index]

        raw_prediction = vader_label(raw_result["compound"])
        enhanced_prediction = vader_label(enhanced_result["compound"])

        raw_correct = (raw_prediction == true_label)
        enhanced_correct = (enhanced_prediction == true_label)

        if not raw_correct and enhanced_correct:
            effect = "corrected"
        elif raw_correct and not enhanced_correct:
            effect = "harmed"
        elif raw_correct and enhanced_correct:
            effect = "remained_correct"
        else:
            effect = "remained_wrong"

        audit_item = enhanced_audit[index]
        processed_text = audit_item["processed_review"]
        operations = audit_item["operations"]

        raw_margin = true_class_margin(raw_result, true_label)
        enhanced_margin = true_class_margin(enhanced_result, true_label)

        rows.append({
            "review_index":
                index,
            "original_review":
                original_text,
            "processed_review":
                processed_text,
            "true_label":
                true_label,
            "raw_prediction":
                raw_prediction,
            "enhanced_prediction":
                enhanced_prediction,
            "raw_neg":
                raw_result["neg"],
            "raw_neu":
                raw_result["neu"],
            "raw_pos":
                raw_result["pos"],
            "raw_compound":
                raw_result["compound"],
            "enhanced_neg":
                enhanced_result["neg"],
            "enhanced_neu":
                enhanced_result["neu"],
            "enhanced_pos":
                enhanced_result["pos"],
            "enhanced_compound":
                enhanced_result["compound"],
            "text_changed":
                (original_text != processed_text),
            "score_changed":
                scores_are_different(raw_result, enhanced_result),
            "prediction_changed":
                (raw_prediction != enhanced_prediction),
            "raw_correct":
                raw_correct,
            "enhanced_correct":
                enhanced_correct,
            "effect":
                effect,
            "raw_margin":
                raw_margin,
            "enhanced_margin":
                enhanced_margin,
            "margin_change":
                (enhanced_margin - raw_margin),
            "margin_improved":
                (enhanced_margin > raw_margin),
            "margin_worsened":
                (enhanced_margin < raw_margin),
            "operations":
                operations,
            "rule_keys":
                extract_rule_keys(operations)
        })

    return pd.DataFrame(rows)

def print_enhanced_vader_audit_summary(audit_df):
    rule_detected_mask = (audit_df["rule_keys"].apply(bool))
    rule_affected_df = audit_df[rule_detected_mask]

    corrected_count = (audit_df["effect"] == "corrected").sum()
    harmed_count = (audit_df["effect"] == "harmed").sum()

    print("\n========== ENHANCED VADER AUDIT SUMMARY ==========")
    print("Total reviews:", len(audit_df))
    print("Reviews where at least one rule was applied:", rule_detected_mask.sum())
    print("Reviews whose text was changed:", audit_df["text_changed"].sum())
    print("Reviews whose VADER scores changed:", audit_df["score_changed"].sum())
    print("Reviews whose predicted sentiment changed:", audit_df["prediction_changed"].sum())

    print("\n========== SENTIMENT CLASSIFICATION EFFECT ==========")
    print(audit_df["effect"].value_counts().reindex([
        "corrected", "harmed", "remained_correct", "remained_wrong"],
        fill_value=0)
    )
    print("\nCorrected:", corrected_count)
    print("Harmed:", harmed_count)
    print("Net corrections:", corrected_count - harmed_count)

    positive_correction = (corrected_count + harmed_count)
    if positive_correction > 0:
        print("Correction precision:", corrected_count / positive_correction)

    print("\nTrue-class margin improved:", audit_df["margin_improved"].sum())
    print("True-class margin worsened:", audit_df["margin_worsened"].sum())

    print("\nEffects among rule-detected reviews:")
    print(rule_affected_df["effect"].value_counts())

def print_audit_consistency_check(audit_df):
    total_reviews = len(audit_df)
    corrected = (audit_df["effect"] == "corrected").sum()
    harmed = (audit_df["effect"] == "harmed").sum()
    remained_correct = (audit_df["effect"] == "remained_correct").sum()
    remained_wrong = (audit_df["effect"] == "remained_wrong").sum()
    raw_correct = audit_df["raw_correct"].sum()
    enhanced_correct = (audit_df["enhanced_correct"].sum())
    prediction_changed = (audit_df["prediction_changed"].sum())
    wrong_to_wrong_changed = (audit_df["prediction_changed"]
                              & ~audit_df["raw_correct"]
                              & ~audit_df["enhanced_correct"]).sum()

    print("\n========== AUDIT CONSISTENCY CHECK ==========")
    print("Total reviews:", total_reviews)
    print("Raw correct:", raw_correct)
    print("Enhanced correct:", enhanced_correct)
    print("Difference in correct predictions:", enhanced_correct - raw_correct)
    print("Corrected minus harmed:", corrected - harmed)
    print("Total rows affected:", corrected + harmed + remained_correct + remained_wrong)
    print("Prediction changes:", prediction_changed)
    print("Wrong to different wrong:", wrong_to_wrong_changed)

    # Consistency Checks
    assert (
        (corrected + harmed + remained_correct + remained_wrong) == total_reviews
    ), "Total rows affected do not sum to total reviews!"
    assert (
        (enhanced_correct - raw_correct) == corrected - harmed
    ), "Net corrections do not match accuracy change!"
    assert (
        prediction_changed == (corrected + harmed + wrong_to_wrong_changed)
    ), "Prediction changes are inconsistent!"
    assert (
        audit_df["text_changed"] == (audit_df["original_review"] != audit_df["processed_review"])
    ).all(), "Some text_changed values are incorrect!"
    assert (
        ~audit_df["prediction_changed"] | audit_df["score_changed"]
    ).all(), "A prediction changed without the VADER scores changing!"

    print("\nAll audit consistency checks passed.")
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# RULES SUMMARY AND VALIDATION
# -----------------------------------------------------------------------------
def create_rule_summary(audit_df):
    exploded_df = (audit_df.explode("rule_keys"))
    exploded_df = exploded_df[exploded_df["rule_keys"].notna()].copy()

    if exploded_df.empty:
        return pd.DataFrame()

    summary_df = (
        exploded_df.groupby(
            "rule_keys",
            as_index=False
        ).agg(
            applications=(
                "review_index",
                "size"
            ),
            text_changes=(
                "text_changed",
                "sum"
            ),
            score_changes=(
                "score_changed",
                "sum"
            ),
            prediction_changes=(
                "prediction_changed",
                "sum"
            ),
            corrected=(
                "effect",
                lambda values:
                    (values == "corrected").sum()
            ),
            harmed=(
                "effect",
                lambda values:
                    (values == "harmed").sum()
            ),
            remained_correct=(
                "effect",
                lambda values:
                    (values == "remained_correct").sum()
            ),
            remained_wrong=(
                "effect",
                lambda values:
                    (values == "remained_wrong").sum()
            ),
            margin_improved=(
                "margin_improved",
                "sum"
            ),
            margin_worsened=(
                "margin_worsened",
                "sum"
            ),
            mean_margin_change=(
                "margin_change",
                "mean"
            )
        ).rename(
            columns={
                "rule_keys": "rule"
            }
        )
    )

    summary_df["net_corrections"] = (summary_df["corrected"] - summary_df["harmed"])
    decisive_cases = ( summary_df["corrected"] + summary_df["harmed"])
    summary_df["correction_precision"] = np.where(decisive_cases > 0, summary_df["corrected"] / decisive_cases, np.nan)

    summary_df = (
        summary_df
        .sort_values(
            [
                "net_corrections",
                "mean_margin_change",
                "applications"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return summary_df

def create_all_rule_usage_table(audit_df):
    registered_rules = sorted(get_registered_rule_keys())
    registered_df = pd.DataFrame({"rule": registered_rules})

    exploded_df = audit_df.explode("rule_keys")
    exploded_df = exploded_df[exploded_df["rule_keys"].notna()].copy()
    if exploded_df.empty:
        usage_df = registered_df.copy()
        usage_df["times_used"] = 0
        usage_df["used_in_reviews"] = 0
        usage_df["usage_status"] = "NEVER_USED"
        return usage_df

    exploded_df["rule_keys"] = (exploded_df["rule_keys"].astype(str).str.lower())
    used_counts_df = (exploded_df
                      .groupby("rule_keys", as_index=False)
                      .agg(
                          times_used=("rule_keys","size"), 
                          used_in_reviews=("review_index","nunique"))
                          .rename(columns={"rule_keys": "rule"}))

    usage_df = registered_df.merge(
        used_counts_df,
        on="rule",
        how="left"
    )

    usage_df["times_used"] = usage_df["times_used"].fillna(0).astype(int)
    usage_df["used_in_reviews"] = usage_df["used_in_reviews"].fillna(0).astype(int)

    usage_df["usage_status"] = np.where(
        usage_df["times_used"] > 0,
        "USED",
        "NEVER_USED"
    )

    usage_df = (usage_df.sort_values(
        ["times_used", "rule"],
        ascending=[False, True])
        .reset_index(drop=True))

    return usage_df

def get_registered_rule_keys():
    registered_rules = set()
    pattern_groups = [
        NEUTRAL_PHRASE_PATTERNS,
        NEGATIVE_PHRASE_PATTERNS,
        POSITIVE_PHRASE_PATTERNS  
    ]

    registered_rules.add("negation_exception:not_only_but_also")
    for pattern_group in pattern_groups:
        for rule in pattern_group:
            registered_rules.add(f"{rule['type']}:{rule['name']}".lower())
    for marker in ["however", "nevertheless", "nonetheless"]:
        registered_rules.add("contrast_normalisation:" + marker)
    for word in CUSTOM_DOMAIN_WORDS:
        registered_rules.add("domain_lexicon:" + word)
    for word in CUSTOM_INTENSIFIER_WORDS:
        registered_rules.add("intensifier:" + word)
    for word in CUSTOM_DIMINISHER_WORDS:
        registered_rules.add( "diminisher:" + word)
    for word in CUSTOM_NEGATOR_WORDS:
        registered_rules.add("custom_negator:" + word)

    return registered_rules

def decide_rule(row):
    if (row["applications"] < 1):
        return "UNUSED"
    if row["net_corrections"] < 0:
        return "CULL"
    if row["applications"] < 30 or row["decisive_changes"] < 10:
        return "REVIEW"
    if (row["net_corrections"] > 0
        and row["correction_precision"] >= 0.60
        and row["mean_margin_change"] >= 0
    ):
        return "KEEP"

    return "REVIEW"

def get_vader_rule_scope(rule_key):
    rule_key = str(rule_key).lower()

    if rule_key.startswith("domain_lexicon:"):
        return "domain_lexicon"

    if rule_key.startswith("custom_negator:"):
        return "negator"

    if rule_key.startswith("negator:"):
        return "negator"

    if rule_key.startswith("intensifier:"):
        return "intensifier"

    if rule_key.startswith("diminisher:"):
        return "diminisher"

    if rule_key.startswith("contrast_normalisation:"):
        return "contrast"

    if rule_key.startswith("contrast:"):
        return "contrast"

    if rule_key.startswith("negation_exception:"):
        return "negation_exception"

    return "phrase"

def get_rules_in_scope(rule_keys, target_scope):
    return [
        rule_key
        for rule_key in rule_keys
        if get_vader_rule_scope(rule_key) == target_scope
    ]

def create_scoped_exclusive_vader_rule_summary_df(
        audit_df,
        target_scope="phrase"
):
    scoped_df = audit_df.copy()

    scoped_df["scoped_rule_keys"] = scoped_df["rule_keys"].apply(
        lambda rule_keys: get_rules_in_scope(
            rule_keys,
            target_scope
        )
    )

    scoped_df["number_of_scoped_rules"] = scoped_df["scoped_rule_keys"].apply(len)

    scoped_exclusive_df = scoped_df[
        scoped_df["number_of_scoped_rules"] == 1
    ].copy()

    if scoped_exclusive_df.empty:
        return pd.DataFrame(columns=[
            "scope",
            "exclusive_rule",
            "applications",
            "corrected",
            "harmed",
            "score_changes",
            "prediction_changes",
            "decisive_changes",
            "net_corrections",
            "correction_precision",
            "mean_margin_change",
            "decision"
        ])

    scoped_exclusive_df["exclusive_rule"] = scoped_exclusive_df["scoped_rule_keys"].apply(
        lambda rule_keys: rule_keys[0]
    )

    scoped_summary_df = (
        scoped_exclusive_df
        .groupby("exclusive_rule")
        .agg(
            applications=(
                "review_index",
                "size"
            ),
            corrected=(
                "effect",
                lambda values: (values == "corrected").sum()
            ),
            harmed=(
                "effect",
                lambda values: (values == "harmed").sum()
            ),
            score_changes=(
                "score_changed",
                "sum"
            ),
            prediction_changes=(
                "prediction_changed",
                "sum"
            ),
            mean_margin_change=(
                "margin_change",
                "mean"
            )
        )
        .reset_index()
    )

    scoped_summary_df["decisive_changes"] = (
        scoped_summary_df["corrected"] + scoped_summary_df["harmed"]
    )

    scoped_summary_df["net_corrections"] = (
        scoped_summary_df["corrected"] + scoped_summary_df["harmed"]
    )

    scoped_summary_df["net_corrections"] = (
        scoped_summary_df["corrected"] - scoped_summary_df["harmed"]
    )

    scoped_summary_df["correction_precision"] = np.where(
        scoped_summary_df["decisive_changes"] > 0,
        scoped_summary_df["corrected"] / scoped_summary_df["decisive_changes"],
        np.nan
    )

    scoped_summary_df["scope"] = target_scope

    scoped_summary_df["decision"] = scoped_summary_df.apply(
        decide_rule,
        axis=1
    )

    scoped_summary_df = (
        scoped_summary_df
        .sort_values(
            [
                "net_corrections",
                "correction_precision",
                "applications"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    return scoped_summary_df[
        [
            "scope",
            "exclusive_rule",
            "applications",
            "corrected",
            "harmed",
            "decisive_changes",
            "net_corrections",
            "correction_precision",
            "mean_margin_change",
            "score_changes",
            "prediction_changes",
            "decision"
        ]
    ]

def print_scoped_exclusive_vader_rule_results(
        audit_df,
        target_scope="phrase"
):
    scoped_summary_df = create_scoped_exclusive_vader_rule_summary_df(
        audit_df=audit_df,
        target_scope=target_scope
    )

    print(
        "\n========== Scoped-Exclusive "
        + target_scope.upper()
        + " Rule Results =========="
    )

    if scoped_summary_df.empty:
        print("No scoped-exclusive rules found for scope:", target_scope)
        return scoped_summary_df

    print(scoped_summary_df.to_string(index=False))

    for decision_name in ["KEEP", "CULL", "REVIEW", "UNUSED"]:
        selected_rules = scoped_summary_df[
            scoped_summary_df["decision"] == decision_name
        ]

        print(
            "\n========== Scoped-Exclusive "
            + target_scope.upper()
            + " "
            + decision_name
            + " Rules =========="
        )

        if selected_rules.empty:
            print("No rules.")
        else:
            print(selected_rules[["exclusive_rule"]].to_string(index=False, header=False))

    return scoped_summary_df

def print_all_scoped_exclusive_vader_rule_results(audit_df):
    scoped_tables = {}

    for target_scope in [
        "phrase",
        "domain_lexicon",
        "negator",
        "intensifier",
        "diminisher",
        "contrast",
        "negation_exception"
    ]:
        scoped_tables[target_scope] = print_scoped_exclusive_vader_rule_results(
            audit_df=audit_df,
            target_scope=target_scope
        )

    return scoped_tables

def print_exclusive_rule_results(audit_df, unused_rules):
    exclusive_rule_df = (audit_df[audit_df["number_of_rules"] == 1].copy())
    exclusive_rule_df["exclusive_rule"] = (exclusive_rule_df["rule_keys"].str[0])

    exclusive_summary = (
        exclusive_rule_df.groupby(
            "exclusive_rule"
        ).agg(
            applications=(
                "review_index",
                "size"
            ),
            corrected=(
                "effect",
                lambda values:
                (values == "corrected").sum()
            ),
            harmed=(
                "effect",
                lambda values:
                (values == "harmed").sum()
            ),
            score_changes=(
                "score_changed",
                "sum"
            ),
            prediction_changes=(
                "prediction_changed",
                "sum"
            ),
            mean_margin_change=(
                "margin_change",
                "mean"
            )
        ).reset_index()
    )

    exclusive_summary["net_corrections"] = (exclusive_summary["corrected"] - exclusive_summary["harmed"])
    exclusive_summary = (exclusive_summary.sort_values("net_corrections", ascending=False))
    exclusive_summary["decisive_changes"] = (exclusive_summary["corrected"] + exclusive_summary["harmed"])
    exclusive_summary["correction_precision"] = np.where(exclusive_summary["decisive_changes"] > 0,
                                                         exclusive_summary["corrected"]
                                                         / exclusive_summary["decisive_changes"], np.nan)
    exclusive_summary["decision"] = exclusive_summary.apply(decide_rule, axis=1)
    exclusive_summary = (exclusive_summary.sort_values("net_corrections", ascending=False))

    print("\n========== STRICT EXCLUSIVE-RULE RESULTS ==========")
    print(exclusive_summary[
              [
                  "exclusive_rule",
                  "applications",
                  "corrected",
                  "harmed",
                  "decisive_changes",
                  "net_corrections",
                  "correction_precision",
                  "mean_margin_change",
                  "decision"
              ]
          ].to_string(index=False)
          )

    print("\n========== RULES NEVER USED ==========")
    if not unused_rules:
        print("Every registered rule was used at least once.")
    else:
        for rule in unused_rules:
            print(rule)

    for decision_name in ["KEEP", "CULL", "REVIEW"]:
        selected_rules = exclusive_summary[exclusive_summary["decision"] == decision_name]

        print(f"\n========== {decision_name} ==========")
        if selected_rules.empty:
            print("No rules.")
        else:
            for rule_name in selected_rules["exclusive_rule"]:
                print(rule_name)

def print_all_rule_usage_table(audit_df):
    rule_usage_df = create_all_rule_usage_table(audit_df)

    total_registered_rules = len(rule_usage_df)
    total_used_rules = (rule_usage_df["times_used"] > 0).sum()
    total_unused_rules = (rule_usage_df["times_used"] == 0).sum()

    print("\n========== ALL REGISTERED RULE USAGE ==========")
    print("Total registered rules:", total_registered_rules)
    print("Used rules:", total_used_rules)
    print("Never-used rules:", total_unused_rules)

    print("\n========== RULE USAGE TABLE ==========")
    print(
        rule_usage_df[
            [
                "rule",
                "times_used",
                "used_in_reviews",
                "usage_status"
            ]
        ].to_string(index=False)
    )

    return rule_usage_df
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# RESULT VERIFICATION & VALIDATION
# -----------------------------------------------------------------------------
def print_phrase_token_check(analyzer, phrase_lexicon):
    rows = []

    for token, expected_valence in (phrase_lexicon.items()):
        plain_scores = (analyzer.polarity_scores(token))
        period_scores = (analyzer.polarity_scores(token + "."))
        sentence_scores = (analyzer.polarity_scores("The product was " + token + "."))
        expected_sign = np.sign(expected_valence)
        plain_sign = np.sign(plain_scores["compound"])
        period_sign = np.sign(period_scores["compound"])
        sentence_sign = np.sign(sentence_scores["compound"])

        rows.append({
            "token":
                token,
            "expected_valence":
                expected_valence,
            "plain_compound":
                plain_scores["compound"],
            "period_compound":
                period_scores["compound"],
            "sentence_compound":
                sentence_scores["compound"],
            "plain_recognised":
                plain_sign == expected_sign,
            "period_recognised":
                period_sign == expected_sign,
            "sentence_recognised":
                sentence_sign == expected_sign
        })

    results = pd.DataFrame(rows)
    failed = results[~results[["plain_recognised", "period_recognised", "sentence_recognised"]].all(axis=1)]

    print("\n========== PHRASE TOKEN CHECK ==========")
    if failed.empty:
        print("All phrase tokens were recognised correctly.")
    else:
        print("Potentially broken phrase tokens:")
        print(failed.to_string(index=False))

    return results

def print_rule_detection_check(audit_df):
    detected_from_keys = (audit_df["rule_keys"].apply(bool))
    detected_from_operations = (audit_df["operations"].apply(lambda operations: len(operations) > 0))

    print("\n========== RULE DETECTION CHECK ==========")
    print("Detected from rule keys:", detected_from_keys.sum())
    print("Detected from operations:", detected_from_operations.sum())

    mismatch = (detected_from_keys != detected_from_operations)
    print("Detection mismatches:", mismatch.sum())
    assert (
        not mismatch.any()
    ), "rule_keys and operations disagree."

def print_text_change_check(audit_df):
    changed_df = audit_df[audit_df["text_changed"]]
    unchanged_df = audit_df[~audit_df["text_changed"]]
    false_changed = changed_df[changed_df["original_review"] == changed_df["processed_review"]]
    false_unchanged = unchanged_df[unchanged_df["original_review"] != unchanged_df["processed_review"]]

    print("\n========== TEXT CHANGE CHECK ==========")
    print("Reported changes:", len(changed_df))
    print("Changed but texts equal:", len(false_changed))
    print("Unchanged but texts differ:", len(false_unchanged))

    assert false_changed.empty
    assert false_unchanged.empty

def classify_rule_effect(row):
    if (not row["raw_correct"]) and row["enhanced_correct"]:
        return "corrected"

    if row["raw_correct"] and (not row["enhanced_correct"]):
        return "harmed"

    if ((not row["raw_correct"])
        and (not row["enhanced_correct"])
        and row["raw_prediction"] != row["enhanced_prediction"]
    ):
        return "wrong_to_wrong"

    if row["raw_correct"] and row["enhanced_correct"]:
        return "stayed_correct"

    return "stayed_wrong"


def format_operations_for_print(operations):
    if not operations:
        return "No operations recorded."

    operation_lines = []

    for operation in operations:
        rule_type = operation.get("type", "unknown")
        rule_name = operation.get("name", operation.get("cue", "unnamed"))
        action = operation.get("action", "no action recorded")
        matched = operation.get("matched_texts", operation.get("cue", None))
        replacement = operation.get("replacement", None)

        line = f"- {rule_type}:{rule_name} | {action}"
        if matched is not None:
            line += f" | matched={matched}"
        if replacement is not None:
            line += f" | replacement={replacement}"

        operation_lines.append(line)

    return "\n".join(operation_lines)

def print_rule_affected_reviews(audit_df, number=0):
    display_df = audit_df.copy()
    display_df["rule_effect"] = display_df.apply(classify_rule_effect, axis=1)
    display_df = display_df[display_df["rule_keys"].apply(bool)].copy()

    display_df["compound_change"] = (display_df["enhanced_compound"] - display_df["raw_compound"])
    display_df["neg_change"] = (display_df["enhanced_neg"] - display_df["raw_neg"])
    display_df["neu_change"] = (display_df["enhanced_neu"] - display_df["raw_neu"])
    display_df["pos_change"] = (display_df["enhanced_pos"] - display_df["raw_pos"])

    section_order = [
        ("corrected", "CORRECTED"),
        ("harmed", "HARMED"),
        ("wrong_to_wrong", "WRONG TO WRONG"),
        ("stayed_correct", "STAYED CORRECT"),
        ("stayed_wrong", "STAYED WRONG")
    ]

    for effect_key, heading in section_order:
        section_df = display_df[display_df["rule_effect"] == effect_key].copy()

        print(f"\n========== {heading} REVIEWS ==========")
        print("Count:", len(section_df))

        if section_df.empty:
            continue

        section_df["abs_margin_change"] = (section_df["margin_change"].abs())
        section_df = section_df.sort_values(
            by=[
                "prediction_changed",
                "abs_margin_change",
                "score_changed"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )

        sample_size = min(number, len(section_df))
        if number > 0:
            examples = section_df.head(sample_size)
        else:
            examples = section_df

        for _, row in examples.iterrows():
            print("\n" + "-" * 120)
            print("REVIEW INDEX:", row["review_index"])

            print("\nORIGINAL REVIEW:")
            print("\"" + row["original_review"] + "\"")

            print("\nCHANGED REVIEW:")
            print("\"" + row["processed_review"] + "\"")

            print("\nRULES APPLIED:")
            print(row["rule_keys"])

            print("\nFULL OPERATIONS:")
            print(format_operations_for_print(row["operations"]))

            print("\nSENTIMENT:")
            print("True label:", row["true_label"])
            print("Raw prediction:", row["raw_prediction"], "| Correct:", row["raw_correct"])
            print("Enhanced prediction:", row["enhanced_prediction"], "| Correct:", row["enhanced_correct"])

            print("\nSCORE CHANGE (Raw -> Enhanced):")
            print("Compound:", round(row["raw_compound"], 4), "->", round(row["enhanced_compound"], 4),
                  "| Change:", round(row["compound_change"], 4))
            print("Negative:", round(row["raw_neg"], 4), "->", round(row["enhanced_neg"], 4),
                  "| Change:", round(row["neg_change"], 4))
            print("Neutral:", round(row["raw_neu"], 4), "->", round(row["enhanced_neu"], 4),
                  "| Change:", round(row["neu_change"], 4))
            print("Positive:", round(row["raw_pos"], 4), "->", round(row["enhanced_pos"], 4),
                  "| Change:", round(row["pos_change"], 4))

            print("\nTRUE-CLASS MARGIN:")
            print("Raw margin:", round(row["raw_margin"], 4), "-> Enhanced margin:", round(row["enhanced_margin"], 4),
                  "| Margin change:", round(row["margin_change"], 4))

            print("\nFLAGS:")
            print("Text changed:", row["text_changed"], "| Score changed:", row["score_changed"],
                  "| Prediction changed:", row["prediction_changed"])
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE ENHANCED VADER
# -----------------------------------------------------------------------------
print()
enhanced_vader_val_scores, enhanced_vader_val_audit = score_enhanced_vader_reviews(text_val, keep_trace=True, split="Predicting Validation Scores With Enhanced VADER")
enhanced_vader_train_scores, _ = score_enhanced_vader_reviews(text_train, keep_trace=True, split="Predicting Train Scores With Enhanced VADER")
enhanced_vader_test_scores, _ = score_enhanced_vader_reviews(text_test, keep_trace=True, split="Predicting Test Scores With Enhanced VADER")

enhanced_vader_val_sentiment = [vader_label(score["compound"])
                    for score in enhanced_vader_val_scores
                    ]
enhanced_vader_train_sentiment = [vader_label(score["compound"])
                    for score in enhanced_vader_train_scores
                    ]
enhanced_vader_test_sentiment = [vader_label(score["compound"])
                    for score in enhanced_vader_test_scores
                    ]

enhanced_vader_val_probabilities = np.asarray([[score["neg"], score["neu"], score["pos"], score["compound"]]
                                  for score in enhanced_vader_val_scores
                                  ], dtype=np.float32)
enhanced_vader_train_probabilities = np.asarray([[score["neg"], score["neu"], score["pos"], score["compound"]]
                                  for score in enhanced_vader_train_scores
                                  ], dtype=np.float32)
enhanced_vader_test_probabilities = np.asarray([[score["neg"], score["neu"], score["pos"], score["compound"]]
                                  for score in enhanced_vader_test_scores
                                  ], dtype=np.float32)

enhanced_vader_audit_df = (
    create_enhanced_vader_audit(
        texts=text_val,
        true_labels=sentiment_val,
        raw_scores=base_vader_val_scores,
        enhanced_scores=enhanced_vader_val_scores,
        enhanced_audit=enhanced_vader_val_audit
    )
)

print("\n========== ENHANCED VADER VS BASE VADER ==========")
print("BASE VADER ON TRAIN: ACCURACY = " + str(round(accuracy_score(sentiment_train, base_vader_train_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_train, base_vader_train_sentiment, digits=4))
print("ENHANCED VADER ON TRAIN: ACCURACY = " + str(round(accuracy_score(sentiment_train, enhanced_vader_train_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_train, enhanced_vader_train_sentiment, digits=4))  # File

print("\nBASE VADER ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, base_vader_val_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_val, base_vader_val_sentiment, digits=4))
print("ENHANCED VADER ON VALIDATION: ACCURACY = " + str(round(accuracy_score(sentiment_val, enhanced_vader_val_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_val, enhanced_vader_val_sentiment, digits=4))  # File

print("\nBASE VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, base_vader_test_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_test, base_vader_test_sentiment, digits=4))
print("ENHANCED VADER ON TEST: ACCURACY = " + str(round(accuracy_score(sentiment_test, enhanced_vader_test_sentiment) * 100, 4)) + "%")
print(classification_report(sentiment_test, enhanced_vader_test_sentiment, digits=4))  # File
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# EVALUATE VADER
# -----------------------------------------------------------------------------
enhanced_vader_rule_summary_df = (create_rule_summary(enhanced_vader_audit_df))
registered_rules = (get_registered_rule_keys())
enhanced_vader_audit_df["number_of_rules"] = (enhanced_vader_audit_df["rule_keys"].apply(len))

if enhanced_vader_rule_summary_df.empty:
    used_rules = set()
else:
    used_rules = set(enhanced_vader_rule_summary_df["rule"])

unused_rules = sorted(registered_rules - used_rules)

print_enhanced_vader_audit_summary(enhanced_vader_audit_df)  # File

print_rule_detection_check(enhanced_vader_audit_df)

print_audit_consistency_check(enhanced_vader_audit_df)

all_rule_usage_df = print_all_rule_usage_table(enhanced_vader_audit_df)  # File

print_exclusive_rule_results(enhanced_vader_audit_df, unused_rules)  # File

vader_scoped_exclusive_tables = print_all_scoped_exclusive_vader_rule_results(enhanced_vader_audit_df)  #file

phrase_token_results = (print_phrase_token_check(sia, PHRASE_LEXICON))

print_text_change_check(enhanced_vader_audit_df)

print_rule_affected_reviews(enhanced_vader_audit_df, number=1)  # File
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE RESULTS
# -----------------------------------------------------------------------------
output_folder = "Base_Learner/Results/VADER/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_vader_train_probabilities_df = pd.DataFrame(enhanced_vader_train_probabilities).rename(columns={
    0: "enhanced_vader_neg",
    1: "enhanced_vader_neu",
    2: "enhanced_vader_pos",
    3: "enhanced_vader_compound"
})

enhanced_vader_val_probabilities_df = pd.DataFrame(enhanced_vader_val_probabilities).rename(columns={
    0: "enhanced_vader_neg",
    1: "enhanced_vader_neu",
    2: "enhanced_vader_pos",
    3: "enhanced_vader_compound"
})

enhanced_vader_test_probabilities_df = pd.DataFrame(enhanced_vader_test_probabilities).rename(columns={
    0: "enhanced_vader_neg",
    1: "enhanced_vader_neu",
    2: "enhanced_vader_pos",
    3: "enhanced_vader_compound"
})

enhanced_vader_train_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_train_probabilities.csv"),
    index=False
)

enhanced_vader_val_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_val_probabilities.csv"),
    index=False
)

enhanced_vader_test_probabilities_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_test_probabilities.csv"),
    index=False
)

print("\nSaved VADER Probabilities CSV Files to:", output_folder)

enhanced_vader_train_sentiment_df = pd.DataFrame({
    "enhanced_vader_sentiment": enhanced_vader_train_sentiment
})

enhanced_vader_val_sentiment_df = pd.DataFrame({
    "enhanced_vader_sentiment": enhanced_vader_val_sentiment
})

enhanced_vader_test_sentiment_df = pd.DataFrame({
    "enhanced_vader_sentiment": enhanced_vader_test_sentiment
})

enhanced_vader_train_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_train_sentiment.csv"),
    index=False
)

enhanced_vader_val_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_val_sentiment.csv"),
    index=False
)

enhanced_vader_test_sentiment_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_test_sentiment.csv"),
    index=False
)

print("Saved VADER Sentiment CSV Files to:", output_folder)

output_folder = "Base_Learner/Rule_Decisions/VADER"
os.makedirs(output_folder, exist_ok=True)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_rule_affected_reviews(enhanced_vader_audit_df, number=0)
text_output = output.getvalue()
with open(os.path.join(output_folder, "rule_affected_reviews.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_rule_usage_table(enhanced_vader_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "all_rule_usage_table.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_exclusive_rule_results(enhanced_vader_audit_df, unused_rules)
text_output = output.getvalue()
with open(os.path.join(output_folder, "exclusive_rule_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_scoped_exclusive_vader_rule_results(enhanced_vader_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "scoped_exclusive_rule_results.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_enhanced_vader_audit_summary(enhanced_vader_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "enhanced_vader_audit_summary.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

print("Saved VADER Audit Text Files to:", output_folder)

output_folder = "Base_Learner/Classification_Report/VADER/Enhanced"
os.makedirs(output_folder, exist_ok=True)

enhanced_vader_train_report_df = pd.DataFrame(
    classification_report(
        sentiment_train,
        enhanced_vader_train_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_vader_val_report_df = pd.DataFrame(
    classification_report(
        sentiment_val,
        enhanced_vader_val_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_vader_test_report_df = pd.DataFrame(
    classification_report(
        sentiment_test,
        enhanced_vader_test_sentiment,
        digits=4,
        output_dict=True
    )
).transpose().round(4)

enhanced_vader_train_report_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_train_classification_report.csv"),
    index_label="class"
)

enhanced_vader_val_report_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_validation_classification_report.csv"),
    index_label="class"
)

enhanced_vader_test_report_df.to_csv(
    os.path.join(output_folder, "enhanced_vader_test_classification_report.csv"),
    index_label="class"
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_vader_train_report_df.values,
    rowLabels=enhanced_vader_train_report_df.index,
    colLabels=enhanced_vader_train_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_vader_train_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_vader_val_report_df.values,
    rowLabels=enhanced_vader_val_report_df.index,
    colLabels=enhanced_vader_val_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_vader_validation_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=enhanced_vader_test_report_df.values,
    rowLabels=enhanced_vader_test_report_df.index,
    colLabels=enhanced_vader_test_report_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)

plt.savefig(
    os.path.join(output_folder, "enhanced_vader_test_classification_report.png"),
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("Saved Enhanced VADER Classification Report to:", output_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END
