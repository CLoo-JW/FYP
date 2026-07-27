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
import re
import os
import io
import contextlib
from tqdm.contrib.concurrent import process_map
from transformers import DataCollatorWithPadding

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
base_roberta_val_sentiment_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_val_sentiment.csv")
base_roberta_test_sentiment_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_test_sentiment.csv")

base_roberta_test_sentiment = base_roberta_test_sentiment_df["base_roberta_sentiment"].to_numpy()
base_roberta_val_sentiment = base_roberta_val_sentiment_df["base_roberta_sentiment"].to_numpy()

base_roberta_val_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_val_probabilities.csv")
base_roberta_test_probabilities_df = pd.read_csv("Base_Learner/Results/RoBERTa/Base/base_roberta_test_probabilities.csv")

base_roberta_val_probabilities = base_roberta_val_probabilities_df[["base_roberta_neg", "base_roberta_neu", "base_roberta_pos"]].to_numpy()
base_roberta_test_probabilities = base_roberta_test_probabilities_df[["base_roberta_neg", "base_roberta_neu", "base_roberta_pos"]].to_numpy()

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

NEGATIVE_STATE_AUX = (
    r"(?:"
    r"does\s+not|"
    r"did\s+not|"
    r"will\s+not|"
    r"would\s+not|"
    r"has\s+not|"
    r"had\s+not|"
    r"can\s+not"
    r")"
)

OPTIONAL_DEGREE = r"(?:really\s+|very\s+|so\s+|too\s+|quite\s+|at\s+all\s+)?"

NEGATION_CUE_PATTERN = re.compile(
    r"\b(?:"
    r"not|no|never|without|nothing|"
    r"can\s+not|do\s+not|does\s+not|did\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|"
    r"will\s+not|would\s+not|should\s+not|could\s+not|"
    r"has\s+not|have\s+not|had\s+not"
    r")\b",
    flags=re.IGNORECASE
)

def affirmative_rule(rule_key, pattern, interpretation):
    return {
        "rule_key": rule_key,
        "rule_group": "negation_phrase",
        "pattern": re.compile(
            pattern,
            flags=re.IGNORECASE
        ),
        "interpretation": interpretation
    }


AFFIRMATIVE_NEGATION_RULES = [
    affirmative_rule(
        "negation:not_bad_not_great",
        (
            r"\bnot\s+(?:bad|terrible|awful)\s+"
            r"but\s+not\s+"
            r"(?:great|amazing|excellent|perfect)\b"
        ),
        "the reviewed item is average"
    ),

    affirmative_rule(
        "negation:good_but_not_great",
        (
            r"\b(?:good|decent|okay|ok|fine)\s+"
            r"but\s+not\s+"
            r"(?:great|amazing|excellent|perfect|the\s+best)\b"
        ),
        "the reviewed item is good yet ordinary"
    ),

    affirmative_rule(
        "negation:works_but_not_great",
        (
            r"\b(?:works|worked|work)\s+but\s+not\s+"
            r"(?:perfect|great|excellent|amazing)\b"
        ),
        "the reviewed item works with limitations"
    ),

    affirmative_rule(
        "negation:nothing_special",
        (
            r"\bnothing\s+"
            r"(?:special|amazing|great|exceptional)\b"
        ),
        "the reviewed item is ordinary"
    ),

    affirmative_rule(
        "negation:not_what_expected",
        (
            r"\bnot\s+(?:quite\s+|really\s+)?"
            r"what\s+i\s+expected\b"
        ),
        "the reviewed item fell short of expectations"
    ),

    affirmative_rule(
        "negation:not_worth_it",
        (
            r"\bnot\s+"
            + OPTIONAL_DEGREE
            + r"worth\s+(?:it|the\s+money|the\s+price|"
              r"buying|getting|keeping)\b"
        ),
        "the reviewed item offers poor value"
    ),

    affirmative_rule(
        "negation:not_lasting",
        (
            r"\b(?:"
            r"did\s+not\s+last|"
            r"does\s+not\s+last|"
            r"has\s+not\s+lasted|"
            r"not\s+lasting"
            r")\b"
        ),
        "the reviewed item has a short lifespan"
    ),

    affirmative_rule(
        "negation:not_as_described",
        (
            r"\b(?:"
            r"is\s+not\s+as\s+described|"
            r"was\s+not\s+as\s+described|"
            r"not\s+as\s+described"
            r")\b"
        ),
        "the reviewed item differs from its description"
    ),

    # affirmative_rule(
    #     "negation:never_received",
    #     (
    #         r"\b(?:"
    #         r"never\s+received|"
    #         r"did\s+not\s+receive|"
    #         r"have\s+not\s+received|"
    #         r"has\s+not\s+arrived"
    #         r")\b"
    #     ),
    #     "the customer is still waiting for the order"
    # ),

    # affirmative_rule(
    #     "negation:not_delivered",
    #     (
    #         r"\b(?:"
    #         r"was\s+not|is\s+not|never|"
    #         r"has\s+not\s+been"
    #         r")\s+delivered\b"
    #     ),
    #     "the delivery remains pending"
    # ),

    affirmative_rule(
        "negation:not_satisfied",
        (
            r"\bnot\s+"
            r"(?:very\s+|really\s+|fully\s+|completely\s+)?"
            r"(?:satisfied|happy)\b"
        ),
        "the customer feels let down"
    ),

    affirmative_rule(
        "negation:would_not_recommend",
        (
            r"\b(?:would|will|do|does|did|could|should)"
            r"\s+not\s+recommend\b"
        ),
        "the customer gives the reviewed item a poor recommendation"
    ),

    affirmative_rule(
        "negation:not_bad",
        (
            r"\bnot\s+"
            r"(?:too\s+|that\s+|so\s+|very\s+)?bad\b"
        ),
        "the reviewed item is acceptable"
    ),

    affirmative_rule(
        "negation:no_complaints",
        (
            r"\bno\s+"
            r"(?:real\s+|major\s+|serious\s+)?"
            r"complaints?\b"
        ),
        "the customer is satisfied"
    ),

    affirmative_rule(
        "negation:no_issues",
        (
            r"\bno\s+"
            r"(?:real\s+|major\s+|serious\s+)?"
            r"issues?\b"
        ),
        "the customer reports a smooth experience"
    ),

    affirmative_rule(
        "negation:no_problems",
        (
            r"\bno\s+"
            r"(?:real\s+|major\s+|serious\s+)?"
            r"problems?\b"
        ),
        "the customer reports a smooth experience"
    ),

    affirmative_rule(
        "negation:no_regrets",
        r"\bno\s+regrets?\b",
        "the customer is pleased with the purchase"
    ),

    affirmative_rule(
        "negation:could_not_be_happier",
        r"\bcould\s+not\s+be\s+happier\b",
        "the customer is extremely happy"
    ),

    affirmative_rule(
        "negation:cannot_recommend_enough",
        (
            r"\bcan\s+not\s+recommend"
            r"(?:\s+(?:it|this|these|them))?"
            r"\s+enough\b"
        ),
        "the customer strongly recommends the reviewed item"
    ),

    affirmative_rule(
        "negation:never_worked",
        r"\bnever\s+work(?:ed|s|ing)?\b",
        "the reviewed item was defective from the beginning"
    ),

    affirmative_rule(
        "negation:does_not_work",
        (
            r"\b(?:"
            r"does\s+not|did\s+not|will\s+not|"
            r"has\s+not|had\s+not"
            r")\s+work(?:s|ed|ing)?\b"
            r"(?!\s+better\b)"
        ),
        "the reviewed item is defective"
    ),

    affirmative_rule(
        "negation:no_longer_works",
        r"\bno\s+longer\s+work(?:s|ed|ing)?\b",
        "the reviewed item became defective"
    ),

    affirmative_rule(
        "negation:not_powering_on",
        (
            rf"\b(?:{NEG_AUX})\s+"
            r"(?:power\s+on|powering\s+on)\b"
        ),
        "the power function is defective"
    ),

    affirmative_rule(
        "negation:not_turning_on",
        (
            rf"\b(?:{NEG_AUX})\s+"
            r"(?:turn\s+on|turning\s+on)\b"
        ),
        "the power function is defective"
    ),

    affirmative_rule(
         "negation:not_charging",
        (
            rf"\b(?:{NEGATIVE_STATE_AUX})\s+"
            r"charg(?:e|es|ed|ing)\b"
        ),
        "the charging function is defective"
    ),

    affirmative_rule(
        "negation:does_not_hold_charge",
        (
            rf"\b(?:battery\s+)?(?:{NEG_AUX})\s+hold\s+"
            r"(?:a\s+|the\s+)?charge\b"
        ),
        "the battery has poor charge retention"
    ),

    affirmative_rule(
        "negation:touchscreen_not_responsive",
        (
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:is\s+|was\s+)?not\s+responsive\b|"
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"does\s+not\s+respond\b"
        ),
        "the touchscreen responds poorly"
    ),

    affirmative_rule(
        "negation:not_fitting",
        rf"\b(?:{NEG_AUX})\s+fit(?:s|ted|ting)?\b",
        "the reviewed item fits poorly"
    ),

    affirmative_rule(
        "negation:not_easy_to_follow",
        r"\bnot\s+easy\s+to\s+follow\b",
        "the content is difficult to follow"
    ),

    affirmative_rule(
        "negation:not_good",
        rf"\b(?:{NEG_AUX})\s+good\b",
        "the evaluated aspect is poor"
    ),

    affirmative_rule(
        "negation:not_great",
        rf"\b(?:{NEG_AUX})\s+great\b",
        "the evaluated aspect is mediocre"
    ),

    affirmative_rule(
        "negation:not_useful",
        rf"\b(?:{NEG_AUX})\s+useful\b",
        "the reviewed item provides little practical value"
    ),

    affirmative_rule(
        "negation:not_clear",
        rf"\b(?:{NEG_AUX})\s+clear\b",
        "the content is confusing"
    ),

    affirmative_rule(
        "negation:not_durable",
        rf"\b(?:{NEG_AUX})\s+durable\b",
        "the reviewed item has poor durability"
    ),

    affirmative_rule(
        "negation:not_reliable",
        rf"\b(?:{NEG_AUX})\s+reliable\b",
        "the reviewed item has poor reliability"
    ),

    affirmative_rule(
        "negation:not_accurate",
        rf"\b(?:{NEG_AUX})\s+accurate\b",
        "the reviewed item has poor accuracy"
    ),

    affirmative_rule(
        "negation:not_impressed",
        rf"\b(?:{NEG_AUX})\s+impressed\b",
        "the customer feels let down"
    )
]

FORBIDDEN_EXPLICIT_NEGATION = re.compile(
    r"\b(?:"
    r"not|no|never|without|nothing|nowhere|"
    r"nobody|none|neither|"
    r"can\s+not|do\s+not|does\s+not|did\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|"
    r"will\s+not|would\s+not|should\s+not|"
    r"could\s+not|has\s+not|have\s+not|had\s+not"
    r")\b",
    flags=re.IGNORECASE
)


def validate_affirmative_negation_rules():
    invalid_rows = []
    seen_rule_keys = set()

    for rule in AFFIRMATIVE_NEGATION_RULES:
        rule_key = rule["rule_key"]
        interpretation = rule["interpretation"].strip()

        problems = []

        if rule_key in seen_rule_keys:
            problems.append("duplicate rule key")

        seen_rule_keys.add(rule_key)

        if not interpretation:
            problems.append("empty interpretation")

        if FORBIDDEN_EXPLICIT_NEGATION.search(
            interpretation
        ):
            problems.append(
                "interpretation contains explicit negation"
            )

        if problems:
            invalid_rows.append({
                "rule_key": rule_key,
                "interpretation": interpretation,
                "problems": "; ".join(problems)
            })

    if invalid_rows:
        raise ValueError(
            "Invalid affirmative interpretation rules:\n"
            + pd.DataFrame(
                invalid_rows
            ).to_string(index=False)
        )


validate_affirmative_negation_rules()
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
# NEGATION PHRASE EXTRACTION
# ----------------------------------------------------------------------------- 
def split_into_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text).strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def contains_negation(text):
    normalised_text = normalise_negation_contractions(text)

    return bool(
        NEGATION_CUE_PATTERN.search(normalised_text)
    )


def find_non_overlapping_affirmative_matches(text):
    candidate_matches = []

    for rule in AFFIRMATIVE_NEGATION_RULES:
        for match in rule["pattern"].finditer(text):
            candidate_matches.append({
                "rule_key": rule["rule_key"],
                "rule_group": rule["rule_group"],
                "matched_text": match.group(0),
                "interpretation": rule["interpretation"],
                "start": match.start(),
                "end": match.end(),
                "length": match.end() - match.start()
            })

    # Earlier matches are considered first.
    # For equal starts, the longest rule wins.
    candidate_matches = sorted(
        candidate_matches,
        key=lambda item: (
            item["start"],
            -item["length"]
        )
    )

    selected_matches = []
    occupied_ranges = []

    for candidate in candidate_matches:
        overlaps = any(
            candidate["start"] < occupied_end
            and candidate["end"] > occupied_start
            for occupied_start, occupied_end
            in occupied_ranges
        )

        if overlaps:
            continue

        selected_matches.append(candidate)

        occupied_ranges.append((
            candidate["start"],
            candidate["end"]
        ))

    return selected_matches


def find_affirmative_interpretations(
        text,
        max_interpretations=4
):
    interpretations = []
    applied_rules = []
    match_details = []

    seen_interpretations = set()
    seen_rules = set()

    for sentence in split_into_sentences(text):
        normalised_sentence = (
            normalise_negation_contractions(sentence)
        )

        if not NEGATION_CUE_PATTERN.search(
            normalised_sentence
        ):
            continue

        sentence_matches = (
            find_non_overlapping_affirmative_matches(
                normalised_sentence
            )
        )

        for sentence_match in sentence_matches:
            interpretation = sentence_match[
                "interpretation"
            ]

            rule_key = sentence_match[
                "rule_key"
            ]

            match_details.append({
                "source_sentence": sentence,
                "normalised_sentence": normalised_sentence,
                "matched_text": sentence_match[
                    "matched_text"
                ],
                "rule_key": rule_key,
                "interpretation": interpretation
            })

            if interpretation not in seen_interpretations:
                interpretations.append(
                    interpretation
                )
                seen_interpretations.add(
                    interpretation
                )

            if rule_key not in seen_rules:
                applied_rules.append(rule_key)
                seen_rules.add(rule_key)

            if len(interpretations) >= max_interpretations:
                break

        if len(interpretations) >= max_interpretations:
            break

    return (
        interpretations,
        applied_rules,
        match_details
    )


def create_affirmative_interpretation(
        text,
        max_interpretations=4
):
    interpretations, _, _ = (
        find_affirmative_interpretations(
            text,
            max_interpretations=max_interpretations
        )
    )

    if not interpretations:
        return ""

    return ". ".join(interpretations) + "."


def extract_affirmative_interpretation_rules(text):
    _, applied_rules, _ = (
        find_affirmative_interpretations(text)
    )

    return applied_rules


def extract_affirmative_interpretation_details(text):
    _, _, match_details = (
        find_affirmative_interpretations(text)
    )

    return match_details
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# AFFIRMATIVE INTERPRETATION CREATION
# ----------------------------------------------------------------------------- 

# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# RULE TABLES
# ----------------------------------------------------------------------------- 
def build_affirmative_roberta_rule_catalog():
    catalog_rows = []

    for rule in AFFIRMATIVE_NEGATION_RULES:
        catalog_rows.append({
            "rule_key": rule["rule_key"],
            "rule_group": "negation_phrase",
            "marker": "AFFIRMATIVE_INTERPRETATION",
            "description": rule["pattern"].pattern,
            "interpretation": rule["interpretation"],
            "polarity": None
        })

    return (
        pd.DataFrame(catalog_rows)
        .drop_duplicates(subset=["rule_key"])
        .sort_values("rule_key")
        .reset_index(drop=True)
    )

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

        if rule_group != "negation_phrase":
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
        affirmative_interpretations,
        match_details,
        true_labels,
        base_sentiment,
        enhanced_sentiment,
        rule_sets,
        base_probabilities,
        enhanced_probabilities,
        reverse_label_map
):
    original_text = (
        pd.Series(original_text)
        .reset_index(drop=True)
        .fillna("")
        .astype(str)
    )

    affirmative_interpretations = (
        pd.Series(affirmative_interpretations)
        .reset_index(drop=True)
        .fillna("")
        .astype(str)
    )

    match_details = (
        pd.Series(match_details)
        .reset_index(drop=True)
    )

    true_labels = (
        pd.Series(true_labels)
        .reset_index(drop=True)
    )

    base_sentiment = (
        pd.Series(base_sentiment)
        .reset_index(drop=True)
    )

    enhanced_sentiment = (
        pd.Series(enhanced_sentiment)
        .reset_index(drop=True)
    )

    rule_sets = (
        pd.Series(rule_sets)
        .reset_index(drop=True)
    )

    base_probabilities = np.asarray(
        base_probabilities
    )

    enhanced_probabilities = np.asarray(
        enhanced_probabilities
    )

    number_of_reviews = len(original_text)

    lengths = {
        "affirmative_interpretations":
            len(affirmative_interpretations),

        "match_details":
            len(match_details),

        "true_labels":
            len(true_labels),

        "base_sentiment":
            len(base_sentiment),

        "enhanced_sentiment":
            len(enhanced_sentiment),

        "rule_sets":
            len(rule_sets),

        "base_probabilities":
            len(base_probabilities),

        "enhanced_probabilities":
            len(enhanced_probabilities)
    }

    incorrect_lengths = {
        name: length
        for name, length in lengths.items()
        if length != number_of_reviews
    }

    if incorrect_lengths:
        raise ValueError(
            "Audit input lengths do not match: "
            + str(incorrect_lengths)
        )

    if base_probabilities.shape[1] != 3:
        raise ValueError(
            "Base probabilities must have 3 columns."
        )

    if enhanced_probabilities.shape[1] != 3:
        raise ValueError(
            "Enhanced probabilities must have 3 columns."
        )

    class_to_index = {
        class_label: class_id
        for class_id, class_label
        in reverse_label_map.items()
    }

    audit_rows = []

    for review_index, text in enumerate(
        original_text
    ):
        interpretation = (
            affirmative_interpretations
            .iloc[review_index]
            .strip()
        )

        details = match_details.iloc[
            review_index
        ]

        if not isinstance(details, list):
            details = []

        rules_applied = list(
            rule_sets.iloc[review_index]
        )

        matched_phrases = [
            detail["matched_text"]
            for detail in details
        ]

        source_sentences = list(dict.fromkeys(
            detail["source_sentence"]
            for detail in details
        ))

        detail_interpretations = []

        for detail in details:
            detail_interpretation = detail[
                "interpretation"
            ]

            if (
                detail_interpretation
                not in detail_interpretations
            ):
                detail_interpretations.append(
                    detail_interpretation
                )

        expected_interpretation = ""

        if detail_interpretations:
            expected_interpretation = (
                ". ".join(
                    detail_interpretations
                )
                + "."
            )

        interpretation_added = bool(
            interpretation
        )

        negation_detected = contains_negation(
            text
        )

        unsupported_negation = (
            negation_detected
            and not interpretation_added
        )

        true_label = true_labels.iloc[
            review_index
        ]

        base_prediction = base_sentiment.iloc[
            review_index
        ]

        enhanced_prediction = (
            enhanced_sentiment.iloc[
                review_index
            ]
        )

        base_margin = (
            get_roberta_true_class_margin(
                probabilities=(
                    base_probabilities[
                        review_index
                    ]
                ),
                reverse_label_map=(
                    reverse_label_map
                ),
                true_label=true_label
            )
        )

        enhanced_margin = (
            get_roberta_true_class_margin(
                probabilities=(
                    enhanced_probabilities[
                        review_index
                    ]
                ),
                reverse_label_map=(
                    reverse_label_map
                ),
                true_label=true_label
            )
        )

        margin_change = (
            enhanced_margin - base_margin
        )

        effect = classify_roberta_effect(
            true_label=true_label,
            base_prediction=base_prediction,
            enhanced_prediction=(
                enhanced_prediction
            )
        )

        true_class_id = class_to_index[
            true_label
        ]

        base_true_probability = (
            base_probabilities[
                review_index,
                true_class_id
            ]
        )

        enhanced_true_probability = (
            enhanced_probabilities[
                review_index,
                true_class_id
            ]
        )

        audit_rows.append({
            "review_index": review_index,

            "original_text": text,

            "affirmative_interpretation":
                interpretation,

            "input_mode": (
                "review_and_interpretation"
                if interpretation_added
                else "review_only"
            ),

            # Human-readable representation.
            # The tokenizer inserts the actual
            # RoBERTa separator tokens.
            "paired_input_preview": (
                text
                + " [SEP] "
                + interpretation
                if interpretation_added
                else text
            ),

            "negation_detected":
                negation_detected,

            "interpretation_added":
                interpretation_added,

            "unsupported_negation":
                unsupported_negation,

            "interpretation_matches_details": (
                interpretation
                == expected_interpretation
            ),

            "matched_phrases":
                matched_phrases,

            "source_sentences":
                source_sentences,

            "rule_keys":
                rules_applied,

            "number_of_rules":
                len(rules_applied),

            "number_of_matches":
                len(details),

            "true_label":
                true_label,

            "base_prediction":
                base_prediction,

            "enhanced_prediction":
                enhanced_prediction,

            "base_correct": (
                base_prediction == true_label
            ),

            "enhanced_correct": (
                enhanced_prediction
                == true_label
            ),

            "prediction_changed": (
                base_prediction
                != enhanced_prediction
            ),

            "effect":
                effect,

            "base_margin":
                base_margin,

            "enhanced_margin":
                enhanced_margin,

            "margin_change":
                margin_change,

            "base_true_class_probability":
                base_true_probability,

            "enhanced_true_class_probability":
                enhanced_true_probability,

            "true_class_probability_change": (
                enhanced_true_probability
                - base_true_probability
            ),

            "base_neg_probability":
                base_probabilities[
                    review_index, 0
                ],

            "base_neu_probability":
                base_probabilities[
                    review_index, 1
                ],

            "base_pos_probability":
                base_probabilities[
                    review_index, 2
                ],

            "enhanced_neg_probability":
                enhanced_probabilities[
                    review_index, 0
                ],

            "enhanced_neu_probability":
                enhanced_probabilities[
                    review_index, 1
                ],

            "enhanced_pos_probability":
                enhanced_probabilities[
                    review_index, 2
                ],

            "neg_probability_change": (
                enhanced_probabilities[
                    review_index, 0
                ]
                - base_probabilities[
                    review_index, 0
                ]
            ),

            "neu_probability_change": (
                enhanced_probabilities[
                    review_index, 1
                ]
                - base_probabilities[
                    review_index, 1
                ]
            ),

            "pos_probability_change": (
                enhanced_probabilities[
                    review_index, 2
                ]
                - base_probabilities[
                    review_index, 2
                ]
            ),

            "score_changed": not np.allclose(
                base_probabilities[
                    review_index
                ],
                enhanced_probabilities[
                    review_index
                ],
                atol=1e-12,
                rtol=0.0
            )
        })

    return pd.DataFrame(audit_rows)

def print_short_roberta_review_audit_summary(
        audit_df
):
    total_reviews = len(audit_df)

    negation_df = audit_df[
        audit_df["negation_detected"]
    ]

    interpreted_df = audit_df[
        audit_df["interpretation_added"]
    ]

    uninterpreted_df = audit_df[
        ~audit_df["interpretation_added"]
    ]

    unsupported_negation_df = audit_df[
        audit_df["unsupported_negation"]
    ]

    corrected_df = audit_df[
        audit_df["effect"] == "corrected"
    ]

    harmed_df = audit_df[
        audit_df["effect"] == "harmed"
    ]

    corrected_interpreted_df = interpreted_df[
        interpreted_df["effect"]
        == "corrected"
    ]

    harmed_interpreted_df = interpreted_df[
        interpreted_df["effect"]
        == "harmed"
    ]

    changed_interpreted_df = interpreted_df[
        interpreted_df["prediction_changed"]
    ]

    inconsistent_df = audit_df[
        ~audit_df[
            "interpretation_matches_details"
        ]
    ]

    net_corrections_total = (
        len(corrected_df)
        - len(harmed_df)
    )

    net_corrections_interpreted = (
        len(corrected_interpreted_df)
        - len(harmed_interpreted_df)
    )

    summary_rows = [
        {
            "metric": "Total reviews",
            "count": total_reviews
        },
        {
            "metric": "Reviews containing negation",
            "count": len(negation_df)
        },
        {
            "metric": "Reviews with an interpretation",
            "count": len(interpreted_df)
        },
        {
            "metric": "Reviews without an interpretation",
            "count": len(uninterpreted_df)
        },
        {
            "metric": (
                "Negated reviews without "
                "a supported interpretation"
            ),
            "count": len(
                unsupported_negation_df
            )
        },
        {
            "metric": (
                "Interpretation/detail "
                "consistency failures"
            ),
            "count": len(inconsistent_df)
        },
        {
            "metric": "Corrected reviews total",
            "count": len(corrected_df)
        },
        {
            "metric": "Harmed reviews total",
            "count": len(harmed_df)
        },
        {
            "metric": (
                "Corrected reviews with "
                "an interpretation"
            ),
            "count": len(
                corrected_interpreted_df
            )
        },
        {
            "metric": (
                "Harmed reviews with "
                "an interpretation"
            ),
            "count": len(
                harmed_interpreted_df
            )
        },
        {
            "metric": (
                "Prediction changes among "
                "interpreted reviews"
            ),
            "count": len(
                changed_interpreted_df
            )
        },
        {
            "metric": "Net corrections total",
            "count": net_corrections_total
        },
        {
            "metric": (
                "Net corrections among "
                "interpreted reviews"
            ),
            "count": (
                net_corrections_interpreted
            )
        }
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    if total_reviews > 0:
        summary_df["percent_of_total"] = (
            summary_df["count"]
            / total_reviews
            * 100
        ).round(2)
    else:
        summary_df["percent_of_total"] = (
            np.nan
        )

    print(
        "\n========== AFFIRMATIVE "
        "INTERPRETATION AUDIT SUMMARY =========="
    )

    print(
        summary_df.to_string(index=False)
    )

    if len(interpreted_df) > 0:
        base_accuracy = (
            interpreted_df[
                "base_correct"
            ].mean()
        )

        enhanced_accuracy = (
            interpreted_df[
                "enhanced_correct"
            ].mean()
        )

        print(
            "\nBaseline accuracy on "
            "interpreted reviews:",
            round(base_accuracy, 4)
        )

        print(
            "Enhanced accuracy on "
            "interpreted reviews:",
            round(enhanced_accuracy, 4)
        )

        print(
            "Accuracy change on "
            "interpreted reviews:",
            round(
                enhanced_accuracy
                - base_accuracy,
                4
            )
        )

    return summary_df

def get_roberta_rule_operation_text(
        rule_key,
        rule_catalog_df
):
    matching_rule_df = rule_catalog_df[
        rule_catalog_df["rule_key"]
        == rule_key
    ]

    if matching_rule_df.empty:
        return (
            str(rule_key)
            + " | rule detected"
        )

    rule_row = matching_rule_df.iloc[0]

    return (
        str(rule_key)
        + " | generated: "
        + str(rule_row["interpretation"])
    )

def roberta_review_example_block(
        row,
        rule_catalog_df
):
    rules_applied = row["rule_keys"]

    print(
        "\nREVIEW INDEX:",
        row["review_index"]
    )

    print("\nORIGINAL REVIEW:")
    print(
        '"'
        + str(row["original_text"])
        + '"'
    )

    print("\nINPUT MODE:")
    print(row["input_mode"])

    print("\nAFFIRMATIVE INTERPRETATION:")

    if row["interpretation_added"]:
        print(
            '"'
            + str(
                row[
                    "affirmative_interpretation"
                ]
            )
            + '"'
        )
    else:
        print(
            "No interpretation generated."
        )

    print("\nMATCHED NEGATION PHRASES:")

    if row["matched_phrases"]:
        for phrase in row[
            "matched_phrases"
        ]:
            print("- " + str(phrase))
    else:
        print(
            "- No supported phrase matched"
        )

    print("\nSOURCE SENTENCES:")

    if row["source_sentences"]:
        for sentence in row[
            "source_sentences"
        ]:
            print("- " + str(sentence))
    else:
        print(
            "- No interpreted sentence"
        )

    print("\nRULES APPLIED:")

    if not rules_applied:
        print("- No rules applied")
    else:
        for rule_key in rules_applied:
            print(
                "- "
                + get_roberta_rule_operation_text(
                    rule_key=rule_key,
                    rule_catalog_df=(
                        rule_catalog_df
                    )
                )
            )

    print("\nSENTIMENT:")

    print(
        "True label:",
        row["true_label"]
    )

    print(
        "Base prediction:",
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

    print(
        "\nSCORE CHANGE "
        "(Base -> Enhanced):"
    )

    print(
        "Negative:",
        round(
            row[
                "base_neg_probability"
            ],
            4
        ),
        "->",
        round(
            row[
                "enhanced_neg_probability"
            ],
            4
        ),
        "| Change:",
        round(
            row[
                "neg_probability_change"
            ],
            4
        )
    )

    print(
        "Neutral:",
        round(
            row[
                "base_neu_probability"
            ],
            4
        ),
        "->",
        round(
            row[
                "enhanced_neu_probability"
            ],
            4
        ),
        "| Change:",
        round(
            row[
                "neu_probability_change"
            ],
            4
        )
    )

    print(
        "Positive:",
        round(
            row[
                "base_pos_probability"
            ],
            4
        ),
        "->",
        round(
            row[
                "enhanced_pos_probability"
            ],
            4
        ),
        "| Change:",
        round(
            row[
                "pos_probability_change"
            ],
            4
        )
    )

    print("\nTRUE-CLASS MARGIN:")

    print(
        "Base margin:",
        round(row["base_margin"], 4),
        "-> Enhanced margin:",
        round(
            row["enhanced_margin"],
            4
        ),
        "| Change:",
        round(
            row["margin_change"],
            4
        )
    )

    print("\nFLAGS:")

    print(
        "Negation detected:",
        row["negation_detected"],
        "| Interpretation added:",
        row["interpretation_added"],
        "| Unsupported negation:",
        row["unsupported_negation"]
    )

    print(
        "Score changed:",
        row["score_changed"],
        "| Prediction changed:",
        row["prediction_changed"],
        "| Effect:",
        row["effect"]
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
class ReviewPairDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            texts,
            interpretations,
            labels,
            tokenizer,
            max_length
    ):
        self.texts = (
            pd.Series(texts)
            .reset_index(drop=True)
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.interpretations = (
            pd.Series(interpretations)
            .reset_index(drop=True)
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            pd.Series(labels)
            .reset_index(drop=True)
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        if not (
            len(self.texts)
            == len(self.interpretations)
            == len(self.labels)
        ):
            raise ValueError(
                "Texts, interpretations and labels "
                "must have equal lengths."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        review = self.texts[index]
        interpretation = (
            self.interpretations[index].strip()
        )

        encoding = self.tokenizer(
            text=review,
            text_pair=(
                interpretation
                if interpretation
                else None
            ),
            truncation="longest_first",
            max_length=self.max_length,
            padding=False
        )

        encoding["labels"] = self.labels[index]

        return encoding
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
        predict_interpretation,
        predict_sentiment,
        tokenizer,
        max_length
):
    predict_dataset = build_roberta_dataset(
        text_series=predict_text,
        interpretation_series=predict_interpretation,
        sentiment_series=predict_sentiment,
        tokenizer=tokenizer,
        max_length=max_length
    )

    return trainer.predict(
        predict_dataset
    ).predictions

def build_roberta_dataset(
        text_series,
        interpretation_series,
        sentiment_series,
        tokenizer,
        max_length
):
    return ReviewPairDataset(
        texts=text_series,
        interpretations=interpretation_series,
        labels=sentiment_series,
        tokenizer=tokenizer,
        max_length=max_length
    )

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
        train_interpretation,
        train_sentiment,
        tokenizer,
        model_name,
        roberta_params,
        output_dir
):
    train_dataset = build_roberta_dataset(
        text_series=train_text,
        interpretation_series=train_interpretation,
        sentiment_series=train_sentiment,
        tokenizer=tokenizer,
        max_length=roberta_params["max_length"]
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            num_labels=3
        )
    )

    training_args = make_roberta_training_args(
        output_dir=output_dir,
        roberta_params=roberta_params,
        number_of_training_rows=len(train_text)
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()

    return trainer
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# ENHANCED ROBERTA HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------- 
N_WORKERS = max(
    1,
    min(16, os.cpu_count() or 1)
)

def create_interpretation_series(
        texts,
        description
):
    return pd.Series(
        process_map(
            create_affirmative_interpretation,
            texts.tolist(),
            max_workers=N_WORKERS,
            chunksize=1000,
            desc=description
        ),
        index=texts.index,
        dtype="object"
    )


print("Creating train affirmative interpretations...")
affirmative_train = create_interpretation_series(
    text_train,
    "Train Affirmative Interpretations"
)

print("Creating validation affirmative interpretations...")
affirmative_val = create_interpretation_series(
    text_val,
    "Validation Affirmative Interpretations"
)

print("Creating test affirmative interpretations...")
affirmative_test = create_interpretation_series(
    text_test,
    "Test Affirmative Interpretations"
)

for split_name, texts, interpretations in [
    ("Train", text_train, affirmative_train),
    ("Validation", text_val, affirmative_val),
    ("Test", text_test, affirmative_test)
]:
    has_negation = texts.apply(contains_negation)

    has_interpretation = (
        interpretations
        .fillna("")
        .str.strip()
        .ne("")
    )

    print(f"\n{split_name}")
    print("Reviews containing negation:", int(has_negation.sum()))
    print("Reviews with interpretations:", int(has_interpretation.sum()))
    print(
        "Negated reviews without a supported rule:",
        int((has_negation & ~has_interpretation).sum())
    )

ROBERTA_OPTUNA_TRAIN_SIZE = min(60000, len(text_train))

enhanced_roberta_optuna_df = pd.DataFrame({
    "Text": text_train.reset_index(drop=True),

    "AffirmativeInterpretation": (
        affirmative_train.reset_index(drop=True)
    ),

    "Sentiment": (
        sentiment_train.reset_index(drop=True)
    ),

    "sentiment_num": (
        sentiment_train_num
        .astype(int)
        .reset_index(drop=True)
    ),

    "Stratify": (
        train_split_df["Stratify"]
        .reset_index(drop=True)
    )
})

if ROBERTA_OPTUNA_TRAIN_SIZE < len(text_train):
    enhanced_train_optuna, _ = train_test_split(
        enhanced_roberta_optuna_df,
        train_size=ROBERTA_OPTUNA_TRAIN_SIZE,
        stratify=enhanced_roberta_optuna_df["Stratify"],
        random_state=42
    )
    enhanced_train_optuna = (enhanced_train_optuna.reset_index(drop=True))

    optuna_text_train = (
        enhanced_train_optuna["Text"]
    )

    optuna_interpretation_train = (
        enhanced_train_optuna[
            "AffirmativeInterpretation"
        ]
    )

    sentiment_train_num_optuna = (
        enhanced_train_optuna[
            "sentiment_num"
        ]
    )
else:
    optuna_text_train = (
        enhanced_roberta_optuna_df["Text"]
    )

    optuna_interpretation_train = (
        enhanced_roberta_optuna_df[
            "AffirmativeInterpretation"
        ]
    )

    sentiment_train_num_optuna = (
        enhanced_roberta_optuna_df[
            "sentiment_num"
        ]
    )

print("\n========== ENHANCED RoBERTa OPTUNA SUBSET ==========")
print("Optuna training rows:", len(optuna_text_train))
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
        train_text=optuna_text_train,
        train_interpretation=optuna_interpretation_train,
        train_sentiment=sentiment_train_num_optuna,
        tokenizer=optuna_tokenizer,
        model_name=optuna_model_name,
        roberta_params=optuna_params,
        output_dir="./tmp"
    )

    optuna_val_logits = predict_roberta_logits(
        trainer=optuna_trainer,
        predict_text=text_val,
        predict_interpretation=affirmative_val,
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
    (len(text_train), 3)
)

for calibration_fold, (calibration_train_idx, calibration_val_idx) in enumerate(
        calibration_cv.split(text_train, sentiment_train_num)
    ):
    print(f"\n----- TEMPERATURE SCALING FOLD {calibration_fold + 1}/{calibration_cv.n_splits} -----")

    text_calibration_train = (text_train.iloc[calibration_train_idx])
    interpretation_calibration_train = (affirmative_train.iloc[calibration_train_idx])
    text_calibration_val = (text_train.iloc[calibration_val_idx])
    interpretation_calibration_val = (affirmative_train.iloc[calibration_val_idx])

    sentiment_calibration_train = (sentiment_train_num.iloc[calibration_train_idx])
    sentiment_calibration_val = (sentiment_train_num.iloc[calibration_val_idx])

    calibration_trainer = train_roberta_model(
        train_text=text_calibration_train,
        train_interpretation=(interpretation_calibration_train),
        train_sentiment=sentiment_calibration_train,
        tokenizer=tokenizer,
        model_name=MODEL,
        roberta_params=enhanced_roberta_best,
        output_dir="./tmp"
    )

    calibration_val_logits = predict_roberta_logits(
        trainer=calibration_trainer,
        predict_text=text_calibration_val,
        predict_interpretation=(interpretation_calibration_val),
        predict_sentiment=sentiment_calibration_val,
        tokenizer=tokenizer,
        max_length=enhanced_roberta_best["max_length"]
    )

    roberta_train_calibration_logits[
        calibration_val_idx
    ] = calibration_val_logits

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
    train_text=text_train,
    train_interpretation=affirmative_train,
    train_sentiment=sentiment_train_num,
    tokenizer=tokenizer,
    model_name=MODEL,
    roberta_params=enhanced_roberta_best,
    output_dir="./tmp"
)

enhanced_val_logits = predict_roberta_logits(
    trainer=enhanced_roberta_trainer,
    predict_text=text_val,
    predict_interpretation=affirmative_val,
    predict_sentiment=sentiment_val_num,
    tokenizer=tokenizer,
    max_length=enhanced_roberta_best["max_length"]
)

enhanced_roberta_test_logits = predict_roberta_logits(
    trainer=enhanced_roberta_trainer,
    predict_text=text_test,
    predict_interpretation=affirmative_test,
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
    (len(text_train), 3)
)

for fold, (train_idx, fold_val_idx) in enumerate(
        base_cv.split(text_train, sentiment_train_num)
    ):
    print(
        f"\n----- OOF META FEATURES FOLD {fold + 1}/{base_cv.n_splits} -----"
    )

    text_fold_train = text_train.iloc[train_idx]
    text_fold_val = text_train.iloc[fold_val_idx]

    interpretation_fold_train = (affirmative_train.iloc[train_idx])
    interpretation_fold_val = (affirmative_train.iloc[fold_val_idx])

    sentiment_fold_train = sentiment_train_num.iloc[train_idx]
    sentiment_fold_val = sentiment_train_num.iloc[fold_val_idx]

    fold_trainer = train_roberta_model(
        train_text=text_fold_train,
        train_interpretation=interpretation_fold_train,
        train_sentiment=sentiment_fold_train,
        tokenizer=tokenizer,
        model_name=MODEL,
        roberta_params=enhanced_roberta_best,
        output_dir="./tmp"
    )

    fold_val_logits = predict_roberta_logits(
        trainer=fold_trainer,
        predict_text=text_fold_val,
        predict_interpretation=interpretation_fold_val,
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

roberta_train_match_details = (
    text_train.apply(
        extract_affirmative_interpretation_details
    )
)

roberta_val_match_details = (
    text_val.apply(
        extract_affirmative_interpretation_details
    )
)

roberta_test_match_details = (
    text_test.apply(
        extract_affirmative_interpretation_details
    )
)

roberta_rule_usage_df = build_roberta_rule_usage_table(
    rule_catalog_df=roberta_rule_catalog_df,
    train_rule_sets=roberta_train_rule_sets,
    val_rule_sets=roberta_val_rule_sets,
    test_rule_sets=roberta_test_rule_sets
)

roberta_rule_review_audit_df = (
    create_roberta_rule_review_audit_df(
        original_text=text_val,
        affirmative_interpretations=(
            affirmative_val
        ),
        match_details=(
            roberta_val_match_details
        ),
        true_labels=sentiment_val,
        base_sentiment=(
            base_roberta_val_sentiment
        ),
        enhanced_sentiment=(
            enhanced_roberta_val_sentiment
        ),
        rule_sets=roberta_val_rule_sets,
        base_probabilities=(
            base_roberta_val_probabilities
        ),
        enhanced_probabilities=(
            enhanced_roberta_val_probabilities
        ),
        reverse_label_map=(
            reverse_label_map
        )
    )
)


roberta_test_rule_review_audit_df = (
    create_roberta_rule_review_audit_df(
        original_text=text_test,
        affirmative_interpretations=(
            affirmative_test
        ),
        match_details=(
            roberta_test_match_details
        ),
        true_labels=sentiment_test,
        base_sentiment=(
            base_roberta_test_sentiment
        ),
        enhanced_sentiment=(
            enhanced_roberta_test_sentiment
        ),
        rule_sets=roberta_test_rule_sets,
        base_probabilities=(
            base_roberta_test_probabilities
        ),
        enhanced_probabilities=(
            enhanced_roberta_test_probabilities
        ),
        reverse_label_map=(
            reverse_label_map
        )
    )
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
    print_short_roberta_review_audit_summary(roberta_rule_review_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "enhanced_roberta_audit_summary.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_short_roberta_review_audit_summary(roberta_test_rule_review_audit_df)
text_output = output.getvalue()
with open(os.path.join(output_folder, "enhanced_roberta_test_audit_summary.txt"), "w", encoding="utf-8") as file:
    file.write(text_output)

output = io.StringIO()
with contextlib.redirect_stdout(output):
    print_all_roberta_affected_review_examples(
        audit_df=roberta_test_rule_review_audit_df,
        rule_catalog_df=roberta_rule_catalog_df,
        number=0
    )
    text_output = output.getvalue()
with open(os.path.join(output_folder, "rule_affected_test_reviews.txt"), "w", encoding="utf-8") as file:
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